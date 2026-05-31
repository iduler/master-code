import logging

import numpy as np
from numpy.typing import NDArray

import porepy as pp
from export_deltas import ExportDeltasMixin
from boundary_conditions import PressureBoundaryConditions, MechanicalBoundaryConditions


from shedule_and_stabilization import ScheduleClippingTimeManager, PreinjectionStabilization
from material_properties import HeterogeneousProperties

from porepy.applications.initial_conditions.model_initial_conditions import (
    InitialConditionHydrostaticPressureValues,
)

from porepy.numerics.nonlinear import line_search

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ModelGeometry():
    """Geometry mixin defining the domain and fractures."""

    def domain_sizes(self) -> NDArray[np.float64]:
        """Return the domain dimensions in the x-, y-, z-directions.

        Returns:
            Array containing the domain size in the x-, y-, and z-directions.
        """
        return self.units.convert_units(
            self.params.get("domain_sizes", np.ones(3, dtype=float)), "m"
        )

    def set_domain(self) -> None:
        """Define the simulation domain.

        The domain is a 3D box with top at z=0 and bottom at z=-z_size.
        """
        x_size, y_size, z_size = self.domain_sizes()
        box = {
            "xmin": 0.0,
            "xmax": x_size,
            "ymin": 0.0,
            "ymax": y_size,
            "zmin": -z_size,
            "zmax": 0.0,
        }
        self._domain = pp.Domain(box)

    def set_fractures(self) -> None:
        """Define the fractures and the mesh constraints.

        Four planes are created:
          - the elliptic fracture we want to study,
          - two horizontal planes (sedimentary/crystalline interface and
            the interface between the two sedimentary layers), used only
            to constrain the mesh,
          - a strip on the north face marking the crystalline injection
            zone (also a mesh constraint).
        """
        dx, dy, _ = self.domain_sizes()
        frac = self.params["fracture_parameters"]
        injection = self.params["injection_params"]

        # Position the center elliptic fracture
        fracture_center = np.array(
            [
                dx - self.units.convert_units(frac["fracture_distance_east"], "m"),
                dy - self.units.convert_units(frac["fracture_distance_north"], "m"),
                -self.units.convert_units(frac["fracture_depth"], "m"),
            ]
        )

        elliptic_fracture = pp.EllipticFracture(
            center=fracture_center,
            strike_angle=frac["strike_angles"][0],
            dip_angle=frac["dip_angles"][0],
            major_axis=self.units.convert_units(frac["fracture_major_axes"][0], "m"),
            minor_axis=self.units.convert_units(frac["fracture_minor_axes"][0], "m"),
            major_axis_angle=frac["major_axis_angles"][0],
        )

        # Horizontal plane at z = -interface_depth, spanning the full xy plane.
        z_interface = -self.units.convert_units(
            self.params["layer_parameters"]["interface_depth"], "m"
        )
        interface_corners = np.array(
            [
                [0.0, dx, dx, 0.0],
                [0.0, 0.0, dy, dy],
                [z_interface, z_interface, z_interface, z_interface],
            ]
        )

        interface_fracture = pp.PlaneFracture(interface_corners)

        z_interface_2 = -self.units.convert_units(
            self.params["layer_parameters"]["depth_top_sedimentary"], "m"
        )
        interface_corners_2 = np.array(
            [
                [0.0, dx, dx, 0.0],
                [0.0, 0.0, dy, dy],
                [z_interface_2, z_interface_2, z_interface_2, z_interface_2],
            ]
        )
        interface_fracture_2 = pp.PlaneFracture(interface_corners_2)

        # Vertical strip on the north face marking the crystalline injection zone.
        z_strip_mid = -self.units.convert_units(injection["depth_crystalline_injection"], "m")
        half_height = 0.5 * self.units.convert_units(injection["crystalline_injection_thickness"], "m")

        strip_corners = np.array(
            [
                [0.0, dx, dx, 0.0],
                [dy, dy, dy, dy],
                [
                    z_strip_mid - half_height,
                    z_strip_mid - half_height,
                    z_strip_mid + half_height,
                    z_strip_mid + half_height,
                ],
            ]
        )
        strip_fracture = pp.PlaneFracture(strip_corners)

        # Order matters: only index 0 is treated as a real fracture, see
        # `fracture_indices` and `meshing_kwargs["constraints"]` in ModelParameters.

        self._fractures = [
            elliptic_fracture,
            interface_fracture,
            interface_fracture_2,
            strip_fracture,
        ]


class InjectionPoromechanicsModel(
    # Constitutive laws.
    pp.constitutive_laws.GravityForce,
    pp.constitutive_laws.CubicLawPermeability,

    # Boundary condition mixins.
    PressureBoundaryConditions,
    MechanicalBoundaryConditions,

    # Initial condition mixins.
    InitialConditionHydrostaticPressureValues,

    # Geometry and property mixins.
    HeterogeneousProperties,
    ModelGeometry,

    # Export mixins.
    ExportDeltasMixin,

    # Helper mixin for the line-search solution strategy.
    pp.models.solution_strategy.ContactIndicators,

    # Base class.
    pp.Poromechanics,
):
    """Three-dimensional poromechanics simulation model."""

    pass


class ModelParameters:
    """Container for model and solver parameter definitions."""

    def model_parameters(self) -> dict:
        """Return the model parameter dictionary."""

        # Long pre-injection settling time
        dt_init = 10000 * pp.YEAR

        # Time schedule:
        schedule = np.array(
            [
                0.0,
                dt_init - 1 * pp.YEAR,   # Stable state
                dt_init,                 # Constant injection starts 1990 
                dt_init + 20 * pp.YEAR,  # Reference state before increase in injection 2009
                dt_init + 21 * pp.YEAR,  # Variable injection starts here. 2010
                dt_init + 22 * pp.YEAR,
                dt_init + 23 * pp.YEAR,
                dt_init + 24 * pp.YEAR,
                dt_init + 25 * pp.YEAR,
                dt_init + 26 * pp.YEAR,
                dt_init + 27 * pp.YEAR,  # 2016
            ]
        )

        # Total well injection rate in sedimentary (baseline) at each schedule point [m^3/month].
        total_well_injection_rates = (np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.5, 1.25, 1.75, 1.5])) * 1e6

        # Constant background injection into the crystalline strip [m^3/month],
        # applied from dt_init. Set to 0 to verify pre-injection equilibration in isolation.
        crystalline_constant_rate = 0.25 * 1e6

        # Lateral injection profile along the north face: positions [m] and
        # the share of the total rate [m^3/month] at each position.
        injection_positions_from_east = (
            np.array([7.0, 6.125, 5.25, 3.5, 2.625, 1.75, 0.875, 0.0]) * 1e4
        )
        injection_rates_from_east = (
            np.array([0.0, 60.0, 80.0, 440.0, 380.0, 480.0, 20.0, 0.0]) * 1e3
        )

        # Layer depths, injection depths and layer thickness
        # All depths are given by positive values

        depth_top_sedimentary = 2000.0       # m, top of Arbuckle layer
        interface_depth = 3000.0        # m, sedimentary/crystalline boundary.
        sedimentary_thickness = interface_depth - depth_top_sedimentary

        # Crystalline injection target sits straight below the sedimentary layer
        # with the same thickness.
        crystalline_injection_thickness = sedimentary_thickness
        depth_crystalline_injection = interface_depth + 0.5 * sedimentary_thickness

        # Vertical injection profile: fractional depth through the layer
        # (0 = top, 1 = bottom) and the injection weight applied at each depth.
        injection_fractional_offsets = np.array(
            [0.0, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0]
        )
        injection_weights_depth = np.array([2.0, 10.0, 25.0, 35.0, 25.0, 10.0, 2.0])

        # The position weights are applied to fit the layer's thickness and height.
        injection_sedimentary_positions_depth = (
            injection_fractional_offsets * sedimentary_thickness + depth_top_sedimentary
        )
        injection_crystalline_positions_depth = (
            injection_fractional_offsets * crystalline_injection_thickness
            + (depth_crystalline_injection - 0.5 * crystalline_injection_thickness)
        )

        # Domain size, expressed via the length scale.
        length_scale = 1e3  # m
        cell_size = length_scale * 0.9
        domain_sizes = np.array([70.0, 40.0, 12.0]) * length_scale

        # Mesh sizes: coarser at the boundary, finer near the fracture.
        meshing_parameters = {
            "cell_size_boundary": cell_size * 7.0,
            "cell_size_fracture": cell_size * 2.0,
            "cell_size_min": cell_size * 1.5,
            "background_transition_multiplier": 6.0,
        }

        # The material parameters for the two layers.
        sedimentary_rock_parameters = {
            "biot_coefficient": 0.8,
            "density": 2680.0,         # kg/m^3
            "porosity": 0.275,
            "permeability": 1.0e-13,   # m^2
            "lame_lambda": 3.51e10,    # Pa
            "shear_modulus": 2.99e10,  # Pa
            "friction_coefficient": 0.75,
            "residual_aperture": 1e-3, # m
            "fracture_gap": 0,       # m
            "normal_permeability": 1.0e-7, # m^2

        }

        crystalline_rock_parameters = {
            "biot_coefficient": 0.47,
            "density": 2620.0,         # kg/m^3
            "porosity": 0.211,
            "permeability": 5.0e-18,     # m^2
            "lame_lambda": 4.62e10,    # Pa
            "shear_modulus": 3.08e10,  # Pa
            "friction_coefficient": 0.75,
            "residual_aperture": 1e-3, # m
            "fracture_gap":  0,       # m
            "normal_permeability": 1.0e-7,  # m^2
    
        }

        # Fracture geometry parameters for the fault we want to study. 
        fracture_parameters = {
            "num_fractures": 1,
            "fracture_major_axes": np.array([7000.0]),   # m
            "fracture_minor_axes": np.array([3500.0]),   # m
            "strike_angles": np.array([np.radians(48)]),
            "dip_angles": np.array([np.radians(70)]),
            "major_axis_angles": np.array([np.pi / 4]),
            "fracture_depth": 5750.0,            # m
            "fracture_distance_north": 20000.0,  # m
            "fracture_distance_east": 45000.0,   # m
        }


        return {
            # Per-layer geometry and per-layer material properties. 

            "layer_parameters": {
                "depth_top_sedimentary": depth_top_sedimentary,
                "interface_depth":  interface_depth,
                "sedimentary": pp.SolidConstants(**sedimentary_rock_parameters),
                "crystalline": pp.SolidConstants(**crystalline_rock_parameters),
            },

            "fracture_parameters": fracture_parameters,

            # Time manager
            "time_manager": ScheduleClippingTimeManager(
                schedule=schedule,
                dt_init=dt_init - 1 * pp.YEAR,
                constant_dt=False,
                dt_min_max=(0.1 * pp.HOUR, max(pp.YEAR, dt_init)),
                iter_optimal_range=(6, 10),
                iter_relax_factors=(0.5, 1.8),
            ),
            "lithostatic_stress_multipliers": np.array([1.2, 0.63, 1.0]),

            "stabilization_time": dt_init,  # Target time for the pre-injection stabilization check.

            # Injection parameters
            "injection_params": {
                "total_well_injection_rates": total_well_injection_rates,
                "injection_positions_from_east": injection_positions_from_east,
                "injection_rates_from_east": injection_rates_from_east,
                "injection_sedimentary_positions_depth": injection_sedimentary_positions_depth,
                "injection_crystalline_positions_depth": injection_crystalline_positions_depth,
                "injection_weights_depth": injection_weights_depth,
                "depth_crystalline_injection": depth_crystalline_injection,
                "crystalline_injection_thickness": crystalline_injection_thickness,
                "start_varible_injection": dt_init + 21 * pp.YEAR,  # Time at which the varible injection starts.
                "start_constant_injection": dt_init, # Time at which the stabilisation is finished and the constant injection in crystalline layer begins.
                "varible_injection_in_sedimentary": True,  # If False, varible injection into the crystalline layer instead.
                "crystalline_constant_rate": crystalline_constant_rate,
            },


            # Per-layer material properties live in `layer_parameters` above; `solid` here is a fallback.
            # Used only if something reads `self.solid` directly. We point
            # it at the crystalline parameters because that layer is the
            # largest by volume.
            "material_constants": {
                "fluid": pp.FluidComponent(**pp.fluid_values.water),
                "solid": pp.SolidConstants(**crystalline_rock_parameters),
                "numerical": pp.NumericalConstants(
                    characteristic_displacement=1e-2
                ),
            },
            "reference_variable_values": pp.ReferenceVariableValues(
                pressure=pp.ATMOSPHERIC_PRESSURE,
                temperature=25.0,
            ),  
            # kg 1.0e5 makes better convergence for some reason
            "units": pp.Units(m=1.0, kg=1.0e5, K=1.0),
            "grid_type": "simplex",
            "meshing_arguments": meshing_parameters,
            # Indices 1, 2, 3 (interface plane 1, interface plane 2, crystalline strip) are mesh
            # constraints only — they shape the grid but are not real fractures.
            # num_processors speed up the meshing
            "meshing_kwargs": {"constraints": np.array([1, 2, 3]), "num_processors": 4,},
            "fracture_indices": np.array([0]),  # Only the elliptic fracture is a real fracture.
            "domain_sizes": domain_sizes,
            "adaptive_indicator_scaling": 1,
            "folder_name": "fluid_injection_baseline",
        }

    def solver_parameters(self) -> dict:
        """Return the solver parameter dictionary."""
        return {
            "prepare_simulation": True,
            "nl_max_iterations": 40,
            "nl_convergence_inc_atol": 1e-8,  # Increment norm tolerance.
            "nl_convergence_res_atol": 1e-3,  # Residual norm tolerance.
            "nl_divergence_inc_atol": 1e20,
            "nl_divergence_res_atol": 1e20,
            "nonlinear_solver": line_search.ConstraintLineSearchNonlinearSolver,
            "global_line_search": 0,
            "local_line_search": 1,
        }


if __name__ == "__main__":

    params = ModelParameters()
    model = InjectionPoromechanicsModel(params.model_parameters())

    pp.run_time_dependent_model(model, params.solver_parameters())

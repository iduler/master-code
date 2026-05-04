import logging

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import porepy as pp
from material_properties import HeterogeneousProperties
from porepy.applications.boundary_conditions.model_boundary_conditions import (
    BoundaryConditionsMechanicsNeumann,
)
from porepy.applications.initial_conditions.model_initial_conditions import (
    InitialConditionHydrostaticPressureValues,
)
from porepy.viz.data_saving_model_mixin import FractureDeformationExporting
from porepy.numerics.nonlinear import line_search

from Grid import CombinedGeometry

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

Scalar = pp.ad.Scalar

# Approximate number of seconds in one calendar month (30.44 days).
_SECONDS_PER_MONTH: float = 30.44 * 24 * 3600


class ModelGeometry():

    """Geometry mixin defining the domain and fracture configuration."""

    def domain_sizes(self) -> NDArray[np.float64]:
        """Return the domain dimensions in model units.

        Returns:
            Array containing the domain size in the x-, y-, and z-directions.
        """
        return self.units.convert_units(
            self.params.get("domain_sizes", np.ones(3, dtype=float)), "m"
        )

    def set_domain(self) -> None:
        """Define the simulation domain.

        The domain is a 3D box offset vertically by the depth of the top boundary.
        """
        top_depth = self.units.convert_units(
            self.params["layer_parameters"]["depth_top_domain"], "m"
        )
        x_size, y_size, z_size = self.domain_sizes()
        box = {
            "xmin": 0.0,
            "xmax": x_size,
            "ymin": 0.0,
            "ymax": y_size,
            "zmin": -(z_size + top_depth),  # Bottom of domain.
            "zmax": -top_depth,             # Top of domain.
        }
        self._domain = pp.Domain(box)

    def set_fractures(self) -> None:
        """Define the elliptic fracture geometry.

        The fracture center is positioned relative to the domain extents using
        the east and north offset distances provided in the fracture parameters.
        """
        dx, dy, _ = self.domain_sizes()
        frac = self.params["fracture_parameters"]

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

        # Horizontal plane fracture at the sediment-crystalline interface.
        interface_depth = self.units.convert_units(
            self.params["layer_parameters"]["interface_depth"], "m"
        )

        horizontal_fracture =  pp.PlaneFracture(np.array(
            [
                [0.0, dx,  dx, 0.0],
                [0.0, 0.0, dy,  dy],
                [-interface_depth] * 4,
            ]
        ))

        self._fractures = [elliptic_fracture]


    def depth(self, points: np.ndarray) -> np.ndarray:
        """Compute depth (positive downward) from point coordinates.

        Parameters:
            points: Array of point coordinates, shape (nd, num_points).

        Returns:
            Depth values for each point.
        """
        return -points[self.nd - 1, :]


class Contact_mech(pp.PorePyModel):
    
    def friction_bound(self, sd: list[pp.Grid]) -> pp.ad.Operator:
        """Friction bound [-].

        Dimensionless, since fracture deformation equations consider non-dimensional
        tractions. In this class, the bound is given by
       .. math::
 
            - F t_n

        where :math:`F` is the friction coefficient and :math:`t_n` is the normal
        component of the contact traction.

        Parameters:
            subdomains: List of fracture subdomains.

        Returns:
            Cell-wise friction bound operator [-].

        """
        c = 0.0

        t_n: pp.ad.Operator = self.normal_component(sd) @ self.contact_traction(sd)
        bound: pp.ad.Operator = (
            Scalar(-1.0) * self.friction_coefficient(sd) * t_n + Scalar(c)
        )
        bound.set_name("friction_bound")
        return bound


class PressureBoundaryConditions(pp.PorePyModel):
    """Pressure boundary condition mixin for Darcy flux and pressure values."""

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Define Dirichlet boundary condition types for Darcy flux.

        Dirichlet conditions are applied on the west, east, south, and bottom
        boundaries. All remaining boundaries use the default zero-Neumann condition.

        Parameters:
            sd: Subdomain grid.

        Returns:
            Boundary condition object.
        """
        sides = self.domain_boundary_sides(sd)
        dirichlet_faces = sides.west + sides.bottom + sides.east + sides.south
        return pp.BoundaryCondition(sd, dirichlet_faces, "dir")

    def well_injection(self, points: np.ndarray) -> np.ndarray:
        """Compute spatially distributed injection flux for the given x-coordinates.

        The injection profile is derived by interpolating the prescribed well rates
        onto the supplied cell x-coordinates, then scaling the result so that the
        total flux matches the time-interpolated target rate.

        Parameters:
            points: Array of point coordinates, shape (nd, num_points).

        Returns:
            Mass flux values for each supplied coordinate.
            Returns zeros when no injection is scheduled at the current time.
        """

        target_rates = self.params["well_injection_rates"]
        
        total_flux = self.get_well_value(
            target_rates, self.time_manager.schedule, self.time_manager.time
        )

        # No injection during the pre-injection phase. [0] to get rigth size of the array.
        if total_flux == 0 and self.time_manager.time <= 3 * pp.YEAR:
            return np.zeros_like(points[0])


        x_coords = points[0]
        z_coords = points[2]


        # Interpolate prescribed rates (m^3/month) onto the cell x-coordinates.
        well_positions_from_east = self.units.convert_units(
            self.params["injection_positions_from_east"], "m"
        )
        start_rates = self.params["injection_rates_from_east"]  # m^3/month
        positions_from_west = self.domain_sizes()[0] - well_positions_from_east
        rates_at_cells_x = np.interp(x_coords, positions_from_west, start_rates)



        # Interpolate prescribed rates (m^3/month) onto the cell z-coordinates.
        positive_injection_positions_depth = self.units.convert_units(
            self.params["injection_positions_depth"], "m"
        )
        injection_weights_depth = self.params["injection_weights_depth"] 

        rates_at_cells_z = np.interp(-z_coords, positive_injection_positions_depth, injection_weights_depth)

        # Combine x and z profiles by multiplying the rates.
        rates_at_cells = rates_at_cells_x * rates_at_cells_z

        total_unscaled = np.sum(rates_at_cells)
        if total_unscaled == 0:
            # No injection cells within this subdomain. [0] to get rigth size of the array.
            return np.zeros_like(points[0])

        scaling = total_flux / total_unscaled

        # Sign convention: inflow is negative. Factor of 1/2 for model symmetry.
        # Convert m^3/month to m^3/s, then multiply by viscosity so that the 1/mu
        # PorePy applies later via the upwind mobility recovers the prescribed
        # volumetric flux. Resulting unit: m^3 * Pa.
        viscosity = self.fluid.reference_component.viscosity
        volumetric_flux = self.units.convert_units(
            -scaling / (2.0 * _SECONDS_PER_MONTH) * rates_at_cells,
            "m^3 * s^-1",
        )
        return volumetric_flux * viscosity

    def bc_values_darcy_flux(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Compute Darcy flux values on the boundary.

        Injection is applied on the north face within a depth window centred on
        the well. All other boundary cells receive zero flux.

        Parameters:
            bg: Boundary grid.

        Returns:
            Array of Darcy flux values for each boundary cell.
        """
        flux_values = np.zeros(bg.num_cells)
        domain_sides = self.domain_boundary_sides(bg)


        interface = -self.units.convert_units(self.params["layer_parameters"]["interface_depth"], "m")
        
        inflow_mask = domain_sides.north.copy()
        
        # Restrict injection to north-face cells within the top rock layer.
        inflow_mask  &=bg.cell_centers[2] > interface
        
        flux_values[inflow_mask] = self.well_injection(bg.cell_centers[:, inflow_mask])

        #  Plot of the injection profile.
        x_km = bg.cell_centers[0, inflow_mask] / 1e3
        flux_plot = -flux_values[inflow_mask]
        sort_idx = np.argsort(x_km)
        plt.plot(x_km[sort_idx], flux_plot[sort_idx], "o-")
        plt.xlabel("Distance from west [km]")
        plt.ylabel("Flux per cell")
        plt.title(f"Injection profile at t={self.time_manager.time:.2e} s")
        #plt.show()

        return flux_values

    def get_well_value(
        self,
        values: np.ndarray,
        times: np.ndarray,
        current_time: float,
    ) -> float:
        """Return the linearly interpolated well value at the current time.

        Clamps to the first or last value when outside the schedule range.

        Parameters:
            values: Time-dependent well values.
            times: Time points corresponding to the well values.
            current_time: Current simulation time.

        Returns:
            Interpolated well value at the current time.
        """
        if current_time <= times[0]:
            return float(values[0])
        if current_time >= times[-1]:
            return float(values[-1])
        return float(np.interp(current_time, times, values))

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Compute hydrostatic pressure values on the boundary. 

        Parameters:
            bg: Boundary grid.

        Returns:
            Array of pressure values for each boundary cell.
        """
        domain_sides = self.domain_boundary_sides(bg)
        values = np.zeros(bg.num_cells)
        depth = self.depth(bg.cell_centers)

        values[domain_sides.east] = self.hydrostatic_pressure(depth[domain_sides.east])
        values[domain_sides.south] = self.hydrostatic_pressure(depth[domain_sides.south]) 
        values[domain_sides.bottom] = self.hydrostatic_pressure(depth[domain_sides.bottom])
        values[domain_sides.west] = self.hydrostatic_pressure(depth[domain_sides.west])

        return values


class MechanicalBoundaryConditions(BoundaryConditionsMechanicsNeumann, pp.PorePyModel):
    """Mechanical boundary condition mixin for displacement and stress."""

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Define mechanical boundary condition types.

        Neumann conditions are used on all exterior faces by default. A roller
        condition (u_x = 0) is applied on the north boundary. 

        Parameters:
            sd: Subdomain grid.

        Returns:
            Vectorial boundary condition object.
        """
        domain_sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, domain_sides.all_bf, "neu")

        # For subdomains with dimension less than the model dimension, we keep the default zero-Neumann conditions.
        if sd.dim < self.nd:
            return bc

        # Prevent independent rigid motion of fracture faces.
        bc.internal_to_dirichlet(sd)

        # North boundary: roller condition with u_x = 0.
        bc.is_dir[0, domain_sides.north] = True
        bc.is_neu[0, domain_sides.north] = False

        # Anchor constraints to eliminate remaining rigid-body modes.
        dirichlet_masks = [
            np.array([True, True, True]),    # Fix x, y, z on face 1.
            np.array([False, False, True]),   # Fix z on face 2.
            np.array([True, False, True]),    # Fix x and z on face 3.
        ]
        for face, mask in zip(self.faces_to_fix(sd), dirichlet_masks):
            bc.is_dir[:, face] = mask
            bc.is_neu[:, face] = ~mask

        return bc

    @property
    def lithostatic_stress_multipliers(self) -> np.ndarray:
        """Stress multipliers for the three principal directions (x, y, z).

        Returns:
            Array of multipliers; defaults to ones if not set in params.
        """
        return self.params.get("lithostatic_stress_multipliers", np.ones(3))

    def bulk_gravity_force_magnitude_boundary(self) -> np.ndarray:
        """Compute the bulk gravity force magnitude for each rock layer.

        The bulk gravity combines fluid and solid contributions weighted by
        porosity: g * (phi * rho_fluid + (1 - phi) * rho_solid).

        Returns:
            Array of shape (2,) with gravity force magnitudes for the
            sedimentary (upper) and crystalline (lower) layers.
        """
        mat = self.params["material_parameters"]
        sed = mat["sedimentary"]
        cryst = mat["crystalline"]

        sed_density = self.units.convert_units(sed.density, "kg*m^-3")
        cryst_density = self.units.convert_units(cryst.density, "kg*m^-3")
        g = self.units.convert_units(pp.GRAVITY_ACCELERATION, "m * s^-2")

        # Use constant fluid density for now
        fluid_density = self.fluid.reference_component.density

        gravity_sed = g * (
            fluid_density * sed.porosity + (1 - sed.porosity) * sed_density
        )
        gravity_cryst = g * (
            fluid_density * cryst.porosity + (1 - cryst.porosity) * cryst_density
        )
        return np.array([gravity_sed, gravity_cryst])

    def bc_values_stress(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Compute lithostatic stress values on the boundary.

        The stress is computed as a depth-integrated gravity load, accounting
        for the two-layer structure of the subsurface. The initial stress state
        is set to zero for numerical stability.

        Parameters:
            bg: Boundary grid.

        Returns:
            Flattened array of stress values.
        """
        values = np.zeros((3, bg.num_cells))

        # Zero initial stress for numerical stability at t = 0.
        if self.time_manager.time < 1e-5:
            return values.ravel("F")

        # One value per layer
        gravity = self.bulk_gravity_force_magnitude_boundary()
        multipliers = self.lithostatic_stress_multipliers
        domain_sides = self.domain_boundary_sides(bg)
        depth = self.depth(bg.cell_centers)

        interface_depth = self.units.convert_units(
            self.params["layer_parameters"]["interface_depth"], "m"
        )

        # Loop over the three spatial components (x, y, z). For each, the negative
        # side (west/south/bottom) has outward normal pointing inward, so sign=+1,
        # and the positive side (east/north/top) has outward normal pointing outward,
        # so sign=-1.
        for i, sides in enumerate(
            [["west", "east"], ["south", "north"], ["bottom", "top"]]
        ):
            for side, sign in zip(sides, [1, -1]):
                ind = getattr(domain_sides, side)
                # Split boundary cells into the two rock layers.
                upper = ind & (depth <= interface_depth)  # Sedimentary layer.
                lower = ind & (depth > interface_depth)   # Crystalline layer.

                if np.any(upper):
                    # Stress increases linearly with depth (single layer above interface). 
                    # Here we assume that the density for layers over sedimenary rock 
                    # is the same as for the sedimentary layer.
                    values[i, upper] = (
                        multipliers[i]
                        * gravity[0]
                        * depth[upper]
                        * sign
                        * bg.cell_volumes[upper]
                    )

                if np.any(lower):
                    # Stress is the sedimentary column down to the interface, plus the
                    # crystalline column from the interface to the cell depth.
                    values[i, lower] = (
                        multipliers[i]
                        * (
                            gravity[0] * interface_depth
                            + gravity[1] * (depth[lower] - interface_depth)
                        )
                        * sign
                        * bg.cell_volumes[lower]
                    )
        return values.ravel("F")


class ExportMixin(FractureDeformationExporting, pp.PorePyModel):
    """Mixin for exporting fracture-related quantities and injection-induced changes."""

    def data_to_export(self) -> list:
        """Return fracture-specific quantities to include in model export.

        Returns:
            List of (subdomain, name, values) tuples to be exported.
        """
        data = super().data_to_export() 
        # --- Coulomb slip condition of fractures ---¨

        sds = self.mdg.subdomains(dim=2)
        cell_offsets_nd = np.cumsum([0] + [sd.num_cells * self.nd for sd in sds])
        cell_offsets = np.cumsum([0] + [sd.num_cells for sd in sds])

        friction_bound = self.evaluate_and_scale(sds, "friction_bound", "Pa")

        traction = self.evaluate_and_scale(sds, "contact_traction", "-").reshape(
            (self.nd, -1), order="F"
        )

        rhs = friction_bound

        # all rows exept normal
        lhs = np.linalg.norm(traction[:-1], axis=0)

        slip_indicator = np.where(lhs >= rhs, 1.0, 0.0)

        for i, sd in enumerate(sds):
            data += [(sd, "slip_indicator", slip_indicator[cell_offsets[i]:cell_offsets[i + 1]])]


        # --- 3-D subdomain quantities (delta relative to injection start) ---
        injection_reference_time = 3 * pp.YEAR

        bulk_subdomains = self.mdg.subdomains(dim=3)
        pressure_offsets = np.cumsum([0] + [sd.num_cells for sd in bulk_subdomains])
        displacement_offsets = np.cumsum(
            [0] + [sd.num_cells * self.nd for sd in bulk_subdomains]
        )

        pressure = self.evaluate_and_scale(bulk_subdomains, "pressure", "Pa")
        displacement = self.evaluate_and_scale(bulk_subdomains, "displacement", "m")

        # Setting reference values. hasattr(self, "p_ref") checks if there exists a reference pressure state. 
        if self.time_manager.time >= injection_reference_time and not hasattr(self, "p_ref"):
            self.p_ref = pressure.copy()
            self.u_ref = displacement.copy()

        # Compute delta relative to injection start.
        if hasattr(self, "p_ref"):
            delta_pressure = pressure - self.p_ref
            delta_displacement = displacement - self.u_ref
        
        # If the reference state has not been set yet.
        else:
            delta_pressure = np.zeros_like(pressure)
            delta_displacement = np.zeros_like(displacement)

        for i, sd in enumerate(bulk_subdomains):
            data += [
                (
                    sd,
                    "delta_pressure_from_injection",
                    delta_pressure[pressure_offsets[i] : pressure_offsets[i + 1]],
                ),
                (
                    sd,
                    "delta_displacement_from_injection",
                    delta_displacement[
                        displacement_offsets[i] : displacement_offsets[i + 1]
                    ],
                ),
            ]

        return data


class InjectionPoromechanicsModel(
    # Constitutive laws.
    pp.constitutive_laws.GravityForce,
    pp.constitutive_laws.CubicLawPermeability,
    Contact_mech,
    # Boundary condition mixins.
    PressureBoundaryConditions,
    MechanicalBoundaryConditions,
    # Initial condition mixins.
    InitialConditionHydrostaticPressureValues,
    # Geometry and property mixins.
    HeterogeneousProperties,
    ModelGeometry,
    # Export mixin.
    ExportMixin,
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
        # Delay injection to allow the model to reach a steady initial state.
        dt_init = 3 * pp.YEAR

        schedule = np.array(
            [
                0.0,
                dt_init,
                dt_init + 1 * pp.YEAR,
                dt_init + 2 * pp.YEAR,
                dt_init + 3 * pp.YEAR,
                dt_init + 4 * pp.YEAR,
                dt_init + 5 * pp.YEAR,
                dt_init + 6 * pp.YEAR,
                dt_init + 7 * pp.YEAR,
            ]
        )

        # Total injection rates per time step [m^3/month].
        total_well_injection_rates = (
            np.array([0.0, 0.0, 0.25, 0.25, 0.25, 0.5, 1.25, 1.75, 1.5]) * 1e6
        )

        # Spatial injection profile: positions [m] and rates [m^3/month] from east.
        injection_positions_from_east = (
            np.array([7.0, 6.125, 5.25, 3.5, 2.625, 1.75, 0.875, 0.0]) * 1e4
        )
        injection_rates_from_east = (
            np.array([0.0, 60.0, 80.0, 440.0, 280.0, 380.0, 360.0, 0.0]) * 1e3
        )

        # Depth points for well injection profile and corresponding weights
        injection_positions_depth =   np.array([2000.0, 2100.0, 2200.0, 2250.0, 2300.0, 2400.0, 2500.0])

        injection_weights_depth =    np.array([2.0, 10.0, 20.0, 36.0, 20.0, 10.0, 2.0])

        schedule_length = schedule.size

        schedule = schedule[:schedule_length]
        total_well_injection_rates = total_well_injection_rates[:schedule_length]

        length_scale = 1e3  # m
        domain_sizes = np.array([70, 40, 10]) * length_scale

        meshing_parameters = {
            "cell_size_boundary": length_scale*4,
            "cell_size_fracture": length_scale*2,
            #"cell_size_min": length_scale*0.5,
        }

        sedimentary_rock_parameters = {
            "biot_coefficient": 0.8,
            "density": 2680.0,         # kg/m^3
            "porosity": 0.275,
            "permeability": 1.0e-12,   # m^2
            "lame_lambda": 3.51e10,    # Pa
            "shear_modulus": 2.99e10,  # Pa
            "friction_coefficient": 0.75,
            "residual_aperture": 1e-4, # m
            "fracture_gap": 0.0,       # m
            "normal_permeability": 1.0e-8, # m^2
            #"cohesion": 0  # Pa
        }

        crystalline_rock_parameters = {
            "biot_coefficient": 0.47,
            "density": 2620.0,         # kg/m^3
            "porosity": 0.211,
            "permeability": 1e-18,     # m^2
            "lame_lambda": 4.62e10,    # Pa
            "shear_modulus": 3.08e10,  # Pa
            "friction_coefficient": 0.65,
            "residual_aperture": 1e-4, # m
            "fracture_gap": 0.0,       # m
            "normal_permeability": 1.0e-8,
            #"cohesion": 0   # m^2
        }

        fracture_parameters = {
            "num_fractures": 3,
            "fracture_major_axes": np.array([7000]),   # m
            "fracture_minor_axes": np.array([3000]),   # m
            "strike_angles": np.array([np.radians(48)]),
            "dip_angles": np.array([np.radians(70)]),
            "major_axis_angles": np.array([np.pi / 4]),
            "fracture_depth": 7000,            # m
            "fracture_distance_north": 20000,  # m
            "fracture_distance_east": 45000,   # m
        }

        return {
            "layer_parameters": {
                "depth_top_domain": 2000.0,  # m
                "interface_depth":  2500.0,   # m
            },
            "fracture_parameters": fracture_parameters,
            "time_manager": pp.TimeManager(
                schedule=schedule,
                dt_init=dt_init,
                constant_dt=False,
                dt_min_max=(0.1 * pp.HOUR, max(pp.YEAR, dt_init)),
                iter_optimal_range=(6, 10),
                iter_relax_factors=(0.5, 1.8),
            ),
            "lithostatic_stress_multipliers": np.array([1.2, 0.63, 1.0]),
            "well_injection_rates": total_well_injection_rates,
            "injection_positions_from_east": injection_positions_from_east,
            "injection_rates_from_east": injection_rates_from_east,
            "injection_positions_depth": injection_positions_depth,
            "injection_weights_depth": injection_weights_depth,


            "material_parameters": {
                "sedimentary": pp.SolidConstants(**sedimentary_rock_parameters),
                "crystalline": pp.SolidConstants(**crystalline_rock_parameters),
                "fluid": pp.FluidComponent(**pp.fluid_values.water),
                "numerical_parameters": pp.NumericalConstants(
                    characteristic_displacement=1e-2
                ),
            },
            "material_constants": {
                "fluid": pp.FluidComponent(**pp.fluid_values.water),
            },
            "reference_variable_values": pp.ReferenceVariableValues(
                pressure=pp.ATMOSPHERIC_PRESSURE,
                temperature=25.0,
            ),  # type: ignore[arg-type]
            "units": pp.Units(m=1.0, kg=1.0e5, K=1.0),
            "grid_type": "simplex",
            "meshing_arguments": meshing_parameters,
           # "meshing_kwargs": {"constraints": np.array([0])},
            #"fracture_indices": np.array([0]),
            "domain_sizes": domain_sizes,
            "adaptive_indicator_scaling": 1,
            "folder_name": "fluid_injection_3D",
        }

    def solver_parameters(self) -> dict:
        """Return the solver parameter dictionary."""
        return {
            "prepare_simulation": True,
            "nl_max_iterations": 40,
            "nl_convergence_inc_atol": 1e-5,  # Increment norm tolerance.
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

    #model.prepare_simulation()
    #model.exporter.write_pvd()

    pp.run_time_dependent_model(model, params.solver_parameters())

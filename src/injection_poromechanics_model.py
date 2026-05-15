import logging

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import porepy as pp
from material_properties import HeterogeneousProperties
from export_deltas import ExportDeltasMixin
from porepy.applications.boundary_conditions.model_boundary_conditions import (
    BoundaryConditionsMechanicsNeumann,
)
from porepy.applications.initial_conditions.model_initial_conditions import (
    InitialConditionHydrostaticPressureValues,
)
from porepy.viz.data_saving_model_mixin import FractureDeformationExporting
from porepy.numerics.nonlinear import line_search
from porepy.applications.discretizations.flux_discretization import FluxDiscretization


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

Scalar = pp.ad.Scalar

# Approximate number of seconds in one calendar month (30.44 days).
_SECONDS_PER_MONTH: float = 30.44 * 24 * 3600

# Time tolerance used to detect "t = 0" — any time below this counts as the
# initial step.
_INITIAL_TIME_TOL: float = 1e-5


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
        injection = self.params["injection_params"]

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

        self._fractures = [
            elliptic_fracture,
            interface_fracture,
            strip_fracture,
        ]


    def depth(self, points: np.ndarray) -> np.ndarray:
        """Compute depth (positive downward) from point coordinates.

        Parameters:
            points: Array of point coordinates, shape (nd, num_points).

        Returns:
            Depth values for each point.
        """
        return -points[self.nd - 1, :]


class PressureBoundaryConditions():
    """Pressure boundary condition mixin for Darcy flux and pressure values."""

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Define Dirichlet boundary condition types for Darcy flux.

        Dirichlet conditions are applied on the west, east, south, and bottom
        boundaries. All remaining boundaries use the default zero-Neumann condition (North and top).

        Parameters:
            sd: Subdomain grid.

        Returns:
            Boundary condition object.
        """
        sides = self.domain_boundary_sides(sd)
        dirichlet_faces = sides.west + sides.bottom + sides.east + sides.south

        return pp.BoundaryCondition(sd, dirichlet_faces, "dir")

    def well_injection(self, points: np.ndarray, if_sedimentary: bool) -> np.ndarray:
        """Compute the boundary injection BC value at the given points.

        The injection profile is built as the product of an x-profile and a
        z-profile (depth-weighted), each interpolated from prescribed values.
        The result is scaled so the cell-wise sum equals the time-interpolated
        target rate (``well_injection_rates``), then converted from m^3/month to
        the BC unit expected by PorePy's Darcy-flux discretization
        (m^3 * Pa, i.e. volumetric flux multiplied by viscosity).

        Parameters:
            points: Array of point coordinates, shape (nd, num_points).

        Returns:
            Boundary flux value per supplied point, in m^3 * Pa (model units).
            Returns zeros when no injection is scheduled at the current time.
        """

        injection = self.params["injection_params"]
        target_rates = injection["total_well_injection_rates"]

        total_flux = self.interpolate_well_value_at_time(
            target_rates, self.time_manager.schedule, self.time_manager.time
        )

        # No injection during the pre-injection phase. points[0] gives the right shape.
        if total_flux == 0:
            return np.zeros_like(points[0])


        x_coords = points[0]
        z_coords = points[2]


        # Interpolate prescribed rates (m^3/month) onto the cell x-coordinates.
        well_positions_from_east = self.units.convert_units(
            injection["injection_positions_from_east"], "m"
        )
        start_rates = injection["injection_rates_from_east"]  # m^3/month
        positions_from_west = self.domain_sizes()[0] - well_positions_from_east
        rates_at_cells_x = np.interp(x_coords, positions_from_west, start_rates)

        if if_sedimentary:
            injection_positions_depth = self.units.convert_units(
                injection["injection_sedimentary_positions_depth"], "m"
            )
        else:
            injection_positions_depth = self.units.convert_units(
                injection["injection_crystalline_positions_depth"], "m"
            )

        injection_weights_depth = injection["injection_weights_depth"]

        # z_coords is negative (z = 0 at surface), but injection_positions_depth is
        # stored as positive depths. Flip the sign so the it matches.
        rates_at_cells_z = np.interp(
            -z_coords, injection_positions_depth, injection_weights_depth
        )

        # Combine x and z profiles by multiplying the rates.
        rates_at_cells = rates_at_cells_x * rates_at_cells_z

        total_unscaled = np.sum(rates_at_cells)
        if total_unscaled == 0:
            # No injection cells within this subdomain. points[0] gives the right shape.
            return np.zeros_like(points[0])

        scaling = total_flux / total_unscaled

        # Sign convention: inflow is negative. Factor of 1/2 for model symmetry.
        # Convert m^3/month to m^3/s, then multiply by viscosity so that the 1/mu
        # PorePy applies later via the upwind mobility recovers the prescribed
        # volumetric flux. Resulting unit: m^3 * Pa.
        viscosity =  1.002e-3
        volumetric_flux = self.units.convert_units(
            -scaling / (2.0 * _SECONDS_PER_MONTH) * viscosity * rates_at_cells,
            "m^3 * Pa",
        )
            
        return volumetric_flux

    def bc_values_darcy_flux(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Compute Darcy flux values on the boundary.

        Injection is applied to north-face cells whose centres lie above the
        sediment-crystalline interface (i.e. within the upper rock layer). All
        other boundary cells receive zero flux.

        Parameters:
            bg: Boundary grid.

        Returns:
            Array of Darcy flux BC values for each boundary cell, in the units
            expected by the discretization (m^3 * Pa).
        """
        flux_values = np.zeros(bg.num_cells)
        domain_sides = self.domain_boundary_sides(bg)
        inflow_mask = domain_sides.north.copy()
        injection = self.params["injection_params"]
        if_sedimentary = injection["injection_in_sedimentary"]

        if if_sedimentary:
            interface = -self.units.convert_units(
                self.params["layer_parameters"]["interface_depth"], "m"
            )
            inflow_mask &= bg.cell_centers[2] > interface

        else:
            depth = -self.units.convert_units(injection["depth_crystalline_injection"], "m")
            half_height = 0.5 * self.units.convert_units(
                injection["crystalline_injection_thickness"], "m"
            )
            inflow_mask &= bg.cell_centers[2] >= depth - half_height
            inflow_mask &= bg.cell_centers[2] <= depth + half_height

        
        flux_values[inflow_mask] = self.well_injection(bg.cell_centers[:, inflow_mask], if_sedimentary)


        return flux_values

    def interpolate_well_value_at_time(
        self,
        values: np.ndarray,
        times: np.ndarray,
        current_time: float,
    ) -> float:
        """Return the linearly interpolated well value at ``current_time``.

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


class MechanicalBoundaryConditions(BoundaryConditionsMechanicsNeumann):
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

        faces_to_fix = self.faces_to_fix(sd)

        # Anchor constraints to eliminate remaining rigid-body modes.
        dir = [
            np.array([True, True, True]),    # Fix x, y, z on face 1.
            np.array([False, False, True]),   # Fix z on face 2.
            np.array([True, False, True]),    # Fix x and z on face 3.
        ]

        for i, face in enumerate(faces_to_fix):
            bc.is_dir[:, face] = dir[i]
            bc.is_neu[:, face] = ~dir[i]  # Negate for Neumann

        return bc

    @property
    def lithostatic_stress_multipliers(self) -> np.ndarray:
        """Stress multipliers for the three principal directions (x, y, z).

        Returns:
            Array of multipliers; defaults to ones if not set in params.
        """
        return self.params.get("lithostatic_stress_multipliers", np.ones(3))

    def bulk_specific_weight_per_layer(self) -> np.ndarray:
        """Compute the bulk specific weight (rho_bulk * g) for each rock layer.

        Bulk density combines fluid and solid contributions weighted by
        porosity: rho_bulk = phi * rho_fluid + (1 - phi) * rho_solid. The
        returned quantity is rho_bulk * g, with units Pa/m.

        Returns:
            Array of shape (2,) with the specific weight of the sedimentary
            (upper) and crystalline (lower) layers, in model units.
        """
        layers = self.params["layer_parameters"]

        sed = layers["sedimentary"]
        cryst = layers["crystalline"]

        sed_density = self.units.convert_units(sed.density, "kg*m^-3")
        cryst_density = self.units.convert_units(cryst.density, "kg*m^-3")
        g = self.units.convert_units(pp.GRAVITY_ACCELERATION, "m * s^-2")

        # Use constant fluid density from the reference value. Same for the porosity. 
        fluid_density = self.fluid.reference_component.density

        gravity_sed = g * (
            fluid_density * sed.porosity + (1 - sed.porosity) * sed_density
        )
        gravity_cryst = g * (
            fluid_density * cryst.porosity + (1 - cryst.porosity) * cryst_density
        )
        return np.array([gravity_sed, gravity_cryst])

    def bc_values_stress(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Compute lithostatic stress BC values on the boundary.

        The lithostatic stress is computed as a depth-integrated gravity load using the
        two-layer specific weight from :meth:`bulk_specific_weight_per_layer`,
        then multiplied by the face area (``bg.cell_volumes``). At t = 0 the
        BC is set to zero for numerical stability.

        Parameters:
            bg: Boundary grid.

        Returns:
            Flattened array of integrated traction values,
            in model units. Shape (3 * bg.num_cells,), ordered Fortran-style.
        """
        values = np.zeros((3, bg.num_cells))

        # Zero initial stress for numerical stability at t = 0.
        if self.time_manager.time < _INITIAL_TIME_TOL:
            return values.ravel("F")

        # One value for bulk gravity load per layer. 
        gravity = self.bulk_specific_weight_per_layer()
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
                    # Here we assume that the density for layers over sedimentary rock
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


class PreinjectionStabilization():

    def calculate_time_to_stabilization(self, tolerance: float = 1e-4) -> float:
        dt = self.params["stabilization_time"]
        
        if self.time_manager.time < dt:
            self.displacement_start = self.evaluate_and_scale(self.mdg.subdomains(dim=self.nd), "displacement", "m")
        
        if self.time_manager.time >= dt and not hasattr(self, "displacement_ref"):
            self.displacement_ref = self.evaluate_and_scale(self.mdg.subdomains(dim=self.nd), "displacement", "m")
            change_displacement = self.displacement_ref - self.displacement_start
            tot_change = np.linalg.norm(change_displacement)
            tot_displacement_now = np.linalg.norm(self.displacement_ref)
            prosent_change = tot_change / tot_displacement_now if tot_displacement_now > 0 else 0.0

            if prosent_change <= tolerance:
                print(f"Stabilization achieved at stabilization time {dt} with total displacement change {tot_change}.")
            else:
                return print(f"Not yet stabilized, "f"{prosent_change:.3e} m"
                )


class ExportMixin(ExportDeltasMixin, FractureDeformationExporting):
    pass


class InjectionPoromechanicsModel(
    #FluxDiscretization,
    #pp.models.poromechanics.TpsaPoromechanicsMixin,
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
    ExportMixin,
    # Helper mixin for the line-search solution strategy.
    pp.models.solution_strategy.ContactIndicators,
    # Base class.
    pp.Poromechanics,
):
    """Three-dimensional poromechanics simulation model."""

    pass


class ScheduleClippingTimeManager(pp.TimeManager):
    """TimeManager that fix timeiteration for the schedule. written by claude
    """

    def compute_time_step(self, *args, **kwargs):
        dt = super().compute_time_step(*args, **kwargs)
        if dt is None:
            return None
        if self._scheduled_idx < len(self.schedule):
            next_target = self.schedule[self._scheduled_idx]
            remaining = next_target - self.time
            if self.dt > remaining > 0:
                self.dt = remaining
        return self.dt


class ModelParameters():
    """Container for model and solver parameter definitions."""

    def model_parameters(self) -> dict:
        """Return the model parameter dictionary."""
        # Delay injection to allow the model to reach a steady initial state.
        dt_init = 100000 * pp.YEAR 

        schedule = np.array(


            [
                0.0,
                dt_init - 1 * pp.YEAR, # Used to calculate changes in displacment because of stabilization.
                dt_init, # Simulation of what we are itrested in starts. Before that it is just stabilization.
                dt_init + 1 * pp.YEAR, # injection starts
                dt_init + 2 * pp.YEAR, 
                dt_init + 3 * pp.YEAR,
                dt_init + 4 * pp.YEAR,
                dt_init + 5 * pp.YEAR,
                dt_init + 6 * pp.YEAR,
                dt_init + 7 * pp.YEAR,
            ]
        )

        # Total injection rates per time step [m^3/month].
        total_well_injection_rates = np.array([0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.5, 1.25, 1.75, 1.5])* 1e6 
        

        # Spatial injection profile: positions [m] and rates [m^3/month] from east.
        injection_positions_from_east = (
            np.array([7.0, 6.125, 5.25, 3.5, 2.625, 1.75, 0.875, 0.0]) * 1e4
        )
        injection_rates_from_east = (
            np.array([0.0, 60.0, 80.0, 440.0, 380.0, 480.0, 20.0, 0.0]) * 1e3
        )

        # Layer depths. Three layers, top-down: overburden (acts as seal),
        # sedimentary (injection target), crystalline (basement).

        depth_top_domain = 2000.0  # m, overburden/sedimentary boundary
        interface_depth = 3000.0        # m, sedimentary/crystalline boundary
        sedimentary_thickness = interface_depth - depth_top_domain
        crystalline_injection_thickness = sedimentary_thickness
        depth_crystalline_injection = interface_depth + 0.5 * sedimentary_thickness # stright bellow sedimentary layer with the same thickness.

        # Depth points for the well injection profile, given as fractional
        # offsets through the sedimentary layer (0 = sed top, 1 = sed bottom)
        # and converted to absolute depths.
        injection_fractional_offsets = np.array(
            [0.0, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0]
        )
        injection_weights_depth = np.array([2.0, 10.0, 25.0, 35.0, 25.0, 10.0, 2.0])
        
        injection_sedimentary_positions_depth = (
             injection_fractional_offsets * sedimentary_thickness + depth_top_domain
        )

        injection_crystalline_positions_depth = (
             injection_fractional_offsets * crystalline_injection_thickness + (depth_crystalline_injection - 0.5 * crystalline_injection_thickness)
        )


        schedule_length = schedule.size

        schedule = schedule[:schedule_length]
        total_well_injection_rates = total_well_injection_rates[:schedule_length]     
        
        length_scale = 1e3  # m
        domain_sizes = np.array([70.0, 40.0, 10.0]) * length_scale

        meshing_parameters = {
            "cell_size_boundary": length_scale*6,
            "cell_size_fracture": length_scale*2,
            "cell_size_min": length_scale*2,
        }

        sedimentary_rock_parameters = {
            "biot_coefficient": 0.8,
            "density": 2680.0,         # kg/m^3
            "porosity": 0.275,
            "permeability": 1.0e-13,   # m^2
            "lame_lambda": 3.51e10,    # Pa
            "shear_modulus": 2.99e10,  # Pa
            "friction_coefficient": 0.7,
            "residual_aperture": 1e-3, # m
            "fracture_gap": 0,       # m
            "normal_permeability": 1.0e-7, # m^2

        }

        crystalline_rock_parameters = {
            "biot_coefficient": 0.47,
            "density": 2620.0,         # kg/m^3
            "porosity": 0.211,
            "permeability": 5e-18,     # m^2
            "lame_lambda": 4.62e10,    # Pa
            "shear_modulus": 3.08e10,  # Pa
            "friction_coefficient": 0.7,
            "residual_aperture": 1e-3, # m
            "fracture_gap":  0,       # m
            "normal_permeability": 1.0e-7,
    
        }

        fracture_parameters = {
            "num_fractures": 1,
            "fracture_major_axes": np.array([7000.0]),   # m
            "fracture_minor_axes": np.array([3000.0]),   # m
            "strike_angles": np.array([np.radians(48)]),
            "dip_angles": np.array([np.radians(70)]),
            "major_axis_angles": np.array([np.pi / 4]),
            "fracture_depth": 5500.0,            # m
            "fracture_distance_north": 20000.0,  # m
            "fracture_distance_east": 45000.0,   # m
        }

        return {
            "darcy_flux_discretization": "tpfa",
            # Per-layer geometry and per-layer material properties. The
            # sedimentary and crystalline SolidConstants are consumed by
            # HeterogeneousProperties (per-cell assignment based on depth),
            # not by PorePy's `self.solid`.
            "layer_parameters": {
                "depth_top_domain": depth_top_domain,
                "interface_depth":       interface_depth,
                "sedimentary": pp.SolidConstants(**sedimentary_rock_parameters),
                "crystalline": pp.SolidConstants(**crystalline_rock_parameters),
            },

            "fracture_parameters": fracture_parameters,
            "time_manager": ScheduleClippingTimeManager(
                schedule=schedule,
                dt_init=dt_init - 1 * pp.YEAR, 
                constant_dt=False,
                dt_min_max=(0.1 * pp.HOUR, max(pp.YEAR, dt_init)),
                iter_optimal_range=(6, 10),
                iter_relax_factors=(0.5, 1.8),
            ),
            "lithostatic_stress_multipliers": np.array([1.2, 0.63, 1.0]),
            "stabilization_time": dt_init, # Time to reach steady state before injection starts.
            "injection_params": {
                "total_well_injection_rates": total_well_injection_rates,
                "injection_positions_from_east": injection_positions_from_east,
                "injection_rates_from_east": injection_rates_from_east,
                "injection_sedimentary_positions_depth": injection_sedimentary_positions_depth,
                "injection_crystalline_positions_depth": injection_crystalline_positions_depth,
                "injection_weights_depth": injection_weights_depth,
                "depth_crystalline_injection": depth_crystalline_injection, # m, depth of injection target within crystalline layer (if injection_in_sedimentary=False)
                "crystalline_injection_thickness": crystalline_injection_thickness, # m, thickness of injection target within crystalline layer (if injection_in_sedimentary=False)
                "start_simulation":dt_init, # Time at which the simulation we are intrested in starts.
                "injection_in_sedimentary": True, # If False, injection is applied in the crystalline layer instead.
            },


            # Standard PorePy material schema. `numerical` carries the scaling
            # constants that the contact-mechanics line search uses
            # (characteristic_displacement). Per-layer solids live in
            # `layer_parameters` above; `solid` here is a fallback used only if
            # something in the framework reads `self.solid` directly. We point
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
            ),  # type: ignore[arg-type]
            "units": pp.Units(m=1.0, kg=1.0e5, K=1.0),
            "grid_type": "simplex",
            "meshing_arguments": meshing_parameters,
            "meshing_kwargs": {"constraints": np.array([1, 2]), "num_processors": 4,},
            "fracture_indices": np.array([0]),
            "domain_sizes": domain_sizes,
            "adaptive_indicator_scaling": 1,
            "folder_name": "fluid_injection_3D",
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

    #model.prepare_simulation()
    #model.exporter.write_pvd()

    pp.run_time_dependent_model(model, params.solver_parameters())

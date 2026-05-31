import numpy as np
import porepy as pp

# Approximate number of seconds in one calendar month (30.44 days).
_SECONDS_PER_MONTH: float = 30.44 * 24 * 3600

# Time tolerance used to detect "t = 0" — any time below this counts as the
# initial step.
_INITIAL_TIME_TOL: float = 1e-5

class PressureBoundaryConditions():
    """Pressure boundary condition mixin for Darcy flux and pressure values."""

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Define Dirichlet boundary condition types for Darcy flux.

        Dirichlet conditions are applied on the west, east, south, top and bottom
        boundaries. The remaining north boundary is default zero-Neumann condition.

        Parameters:
            sd: Subdomain grid.

        Returns:
            Boundary condition object.
        """
        sides = self.domain_boundary_sides(sd)
        dirichlet_faces = sides.west + sides.bottom + sides.east + sides.south + sides.top

        return pp.BoundaryCondition(sd, dirichlet_faces, "dir")

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Compute hydrostatic pressure values on the boundary where we have Dirichlet boundary conditions.

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
        values[domain_sides.top] = self.hydrostatic_pressure(depth[domain_sides.top])

        return values

    def well_injection(self, points: np.ndarray, if_sedimentary: bool, constant_flux_in_crystalline: float, areas: np.ndarray) -> np.ndarray:
        """Compute the boundary injection BC value at the given points.

        The injection profile is the product of an x-profile on the north face and a depth-weighted
        z-profile, weighted by face area. Each profile is interpolated from prescribed values, then scaled
        so the cell-wise sum equals the target rate, then converted from m^3/month
        to the BC unit expected by PorePy's Darcy-flux discretization (m^3 * Pa).

        Parameters:
            points: Array of point coordinates, shape (nd, num_points).
            if_sedimentary: If True, use the sedimentary depth profile;
                otherwise use the crystalline one.
            constant_flux_in_crystalline: If given, calculates the injection for the constant background injection.
            areas: Array of cell areas for the boundry points where injection is applied.

        Returns:
            Boundary flux value per supplied point, in m^3 * Pa (model units).
        """
        injection = self.params["injection_params"]

        # Target rate: time-interpolated schedule rate, or the constant override if given.
        if constant_flux_in_crystalline is None:
            target_rates = injection["total_well_injection_rates"]
            total_flux = self.interpolate_well_value_at_time(
                target_rates, self.time_manager.schedule, self.time_manager.time
            )
        else:
            total_flux = constant_flux_in_crystalline

        # Nothing to inject at the current time.
        if total_flux == 0:
            return np.zeros_like(points[0])

        x_coords = points[0]
        z_coords = points[2]

        # x-profile: interpolate prescribed rates (m^3/month) onto the cell x-coordinates.
        well_positions_from_east = self.units.convert_units(
            injection["injection_positions_from_east"], "m"
        )
        start_rates = injection["injection_rates_from_east"]  # m^3/month
        positions_from_west = self.domain_sizes()[0] - well_positions_from_east
        rates_at_cells_x = np.interp(x_coords, positions_from_west, start_rates)

        # z-profile: pick the depth points for the active layer.
        if if_sedimentary:
            injection_positions_depth = self.units.convert_units(
                injection["injection_sedimentary_positions_depth"], "m"
            )
        else:
            injection_positions_depth = self.units.convert_units(
                injection["injection_crystalline_positions_depth"], "m"
            )

        injection_weights_depth = injection["injection_weights_depth"]

        # z_coords is negative (z = 0 at surface), but injection_positions_depth
        # is stored as positive depths, so flip the sign to match.
        rates_at_cells_z = np.interp(
            -z_coords, injection_positions_depth, injection_weights_depth
        )

        # Combined cell-wise rate from the two profiles and area weigth
        rates_at_cells = rates_at_cells_x * rates_at_cells_z * areas

        total_unscaled = np.sum(rates_at_cells)
        if total_unscaled == 0:
            # No injection cells within this subdomain.
            return np.zeros_like(points[0])

        # Rescale so the cell-wise sum matches the target rate.
        scaling = total_flux / total_unscaled

        # Sign convention: inflow is negative. Factor of 1/2 accounts for model symmetry.
        # Convert m^3/month -> m^3/s, then multiply by viscosity so the 1/mu PorePy
        # applies later recovers the prescribed volumetric
        # flux. Result is in m^3 * Pa.
        viscosity = 1.002e-3
        volumetric_flux = self.units.convert_units(
            -scaling / (2.0 * _SECONDS_PER_MONTH) * viscosity * rates_at_cells,
            "m^3 * Pa",
        )

        return volumetric_flux

    def bc_values_darcy_flux(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Compute Darcy flux values on the boundary.

        Injection is applied on the north face. The scheduled well rate goes
        into either the sedimentary or the crystalline strip depending on
        ``varible_injection_in_sedimentary``. A constant background rate is added to
        the crystalline strip in either case.

        Parameters:
            bg: Boundary grid.

        Returns:
            Array of Darcy flux BC values for each boundary cell, in the units
            expected by the discretization (m^3 * Pa).
        """

        flux_values = np.zeros(bg.num_cells)
        domain_sides = self.domain_boundary_sides(bg)
        injection = self.params["injection_params"]

        # True if the variable injection schedule applies to the sedimentary strip.
        # False if it applies to the crystalline strip.
        if_sedimentary = injection["varible_injection_in_sedimentary"]

        # Time points for the start of variable and constant injection.
        start_varible_injection = injection["start_varible_injection"]
        start_constant_injection = injection["start_constant_injection"]

        # Sedimentary strip mask: north-face cells above the interface.
        interface = -self.units.convert_units(
            self.params["layer_parameters"]["interface_depth"], "m"
        )

        depth_top_sedimentary = -self.units.convert_units(
            self.params["layer_parameters"]["depth_top_sedimentary"], "m"
        )

        sed_mask = (domain_sides.north
            & (bg.cell_centers[2] < depth_top_sedimentary)
            & (bg.cell_centers[2] > interface))


        # Crystalline strip mask: north-face cells centred on depth_crystalline_injection.
        depth = -self.units.convert_units(injection["depth_crystalline_injection"], "m")
        half_height = 0.5 * self.units.convert_units(
            injection["crystalline_injection_thickness"], "m"
        )
        cryst_mask = (
            domain_sides.north
            & (bg.cell_centers[2] > depth - half_height)
            & (bg.cell_centers[2] < depth + half_height)
        )

        if self.time_manager.time >= start_varible_injection:
            # Apply the scheduled injection to either the sedimentary or the crystalline strip.
            if if_sedimentary:
                flux_values[sed_mask] = self.well_injection(
                    bg.cell_centers[:, sed_mask], if_sedimentary, None, bg.cell_volumes[sed_mask]
                )
            else:
                flux_values[cryst_mask] = self.well_injection(
                    bg.cell_centers[:, cryst_mask], if_sedimentary, None, bg.cell_volumes[cryst_mask]
                )

        # Constant background flux into the crystalline strip, applied at all
        # times after the stabilisation.
        if self.time_manager.time >= start_constant_injection:
            constant_rate = injection["crystalline_constant_rate"]
            flux_values[cryst_mask] += self.well_injection(
                bg.cell_centers[:, cryst_mask],
                False,
                constant_flux_in_crystalline=constant_rate,
                areas=bg.cell_volumes[cryst_mask],
            )

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


class MechanicalBoundaryConditions():
    """Mechanical boundary condition mixin for displacement and stress."""

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Define mechanical boundary condition types.

        Defaults to Neumann on all exterior faces, then adds:
          - a roller (u_y = 0) on the north boundary,
          - Dirichlet on fracture internal faces to prevent independent motion,
          - two anchor points at the top to remove rigid-body motions.

        Parameters:
            sd: Subdomain grid.

        Returns:
            Vectorial boundary condition object.
        """
        domain_sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, domain_sides.all_bf, "neu")

        # Lower-dimensional subdomains keep the default zero-Neumann BCs.
        if sd.dim < self.nd:
            return bc

        # Prevent independent rigid motion of fracture faces.
        bc.internal_to_dirichlet(sd)

        # Roller condition on the north face: lock the y-component, leave x and z free (Neumann).
        bc.is_dir[1, domain_sides.north] = True
        bc.is_neu[1, domain_sides.north] = False

        # Function that gives the anchor points
        faces_to_fix = self.faces_to_fix(sd)

        # Anchor two top-north faces to remove the remaining rigid-body modes
        # (one full Dirichlet, one Dirichlet in y and z only).
        dir = [
            np.array([True, True, True]),    # East anchor: fix x, y, z.
            np.array([False, True, True]),   # West anchor: fix y, z.
        ]

        # Apply the per-anchor Dirichlet mask.
        for i, face in enumerate(faces_to_fix):
            bc.is_dir[:, face] = dir[i]
            bc.is_neu[:, face] = ~dir[i]

        return bc

    def faces_to_fix(self, sd: pp.Grid) -> list[np.int64]:
        """Return two north-face indices to anchor against rigid-body motions.

        Picks the north-boundary face closest to the top-east corner and the
        one closest to the top-west corner. Both target points are nudged
        half a cell inward so they fall on a face centre instead of an edge.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            ``[east_anchor_point, west_anchor_point]``.
        """
        domain_sides = self.domain_boundary_sides(sd)
        box = self.domain.bounding_box

        # Typical north-face cell size; used to nudge the target points inward.
        h = np.mean(np.sqrt(sd.face_areas[domain_sides.north]))

        # East anchor: top-east corner of the north face.
        y = box["ymax"]
        z_max = box["zmax"] - 0.5 * h
        x_max = box["xmax"] - 0.5 * h
        point_1 = np.array([x_max, y, z_max])
        pts_1 = sd.face_centers[:, domain_sides.north]
        ind_1 = domain_sides.north.nonzero()[0][
            np.argmin(pp.distances.point_pointset(point_1, pts_1))
        ]

        # West anchor: top-west corner of the north face.
        x_min = box["xmin"] + 0.5 * h
        point_2 = np.array([x_min, y, z_max])
        pts_2 = sd.face_centers[:, domain_sides.north]
        ind_2 = domain_sides.north.nonzero()[0][
            np.argmin(pp.distances.point_pointset(point_2, pts_2))
        ]
        return [ind_1, ind_2]

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
            (upper) and crystalline (lower) layers. The two sedimentary sub-layers
            share density and porosity (only permeability differs), so a single
            sedimentary specific weight covers both — hence two values, not three.

        """
        layers = self.params["layer_parameters"]

        sed = layers["sedimentary"]
        cryst = layers["crystalline"]

        sed_density = self.units.convert_units(sed.density, "kg*m^-3")
        cryst_density = self.units.convert_units(cryst.density, "kg*m^-3")
        g = self.units.convert_units(pp.GRAVITY_ACCELERATION, "m * s^-2")

        # Constant fluid density taken from the reference component.
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
            in model units. Shape (3 * bg.num_cells,).
        """
        values = np.zeros((3, bg.num_cells))

        # Zero initial stress for numerical stability at t = 0.
        if self.time_manager.time < _INITIAL_TIME_TOL:
            return values.ravel("F")

        # rho_bulk * g per layer (sedimentary, crystalline)
        gravity = self.bulk_specific_weight_per_layer()
        multipliers = self.lithostatic_stress_multipliers
        domain_sides = self.domain_boundary_sides(bg)
        depth = self.depth(bg.cell_centers)

        interface_depth = self.units.convert_units(
            self.params["layer_parameters"]["interface_depth"], "m"
        )

        # Loop over the three spatial components (x, y, z). For each component,
        # the negative side (west/south/bottom) has outward normal pointing in
        # the -axis direction (sign=+1), and the positive side (east/north/top)
        # has outward normal pointing in the +axis direction (sign=-1).
        for i, sides in enumerate(
            [["west", "east"], ["south", "north"], ["bottom", "top"]]
        ):
            for side, sign in zip(sides, [1, -1]):
                ind = getattr(domain_sides, side)

                # Split boundary cells into the two rock layers.
                upper = ind & (depth <= interface_depth)  # Sedimentary layer.
                lower = ind & (depth > interface_depth)   # Crystalline layer.

                if np.any(upper):
                    # Single sedimentary column down to the cell depth. Overburden
                    # above the model top is assumed to have the same density.
                    values[i, upper] = (
                        multipliers[i]
                        * gravity[0]
                        * depth[upper]
                        * sign
                        * bg.cell_volumes[upper]
                    )

                if np.any(lower):
                    # Sedimentary column down to the interface, plus crystalline
                    # column from the interface to the cell depth.
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

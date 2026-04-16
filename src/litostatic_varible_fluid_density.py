import porepy as pp
import numpy as np


def bulk_gravity_force_magnitude_boundary(self, bg: pp.BoundaryGrid) -> np.ndarray:
    """Compute the bulk gravity force magnitude for each boundary cell.

    The bulk gravity combines fluid and solid contributions weighted by
    porosity: g * (phi * rho_fluid + (1 - phi) * rho_solid).
    Returns a spatially-varying array (one value per boundary cell).

    Parameters:
        bg: Boundary grid.

    Returns:
        Array of gravity force magnitudes, shape (bg.num_cells,).
    """
    # Get AD operators for density and porosity
    density_solid_op = self.solid_density([bg])
    density_fluid_op = self.rho([bg])

   
    porosity_op = self.porosity([bg])  # This is the problem!!!

    # Evaluate AD operators to get numerical arrays
    density_solid = self.equation_system.evaluate(density_solid_op)
    density_fluid = self.equation_system.evaluate(density_fluid_op)
    porosity = self.equation_system.evaluate(porosity_op)

    g = self.units.convert_units(pp.GRAVITY_ACCELERATION, "m * s^-2")

    # Compute bulk gravity
    gravity = g * (
        density_fluid * porosity + (1 - porosity) * density_solid
    )

    return gravity


def bc_values_stress(self, bg: pp.BoundaryGrid) -> np.ndarray:
    """Compute lithostatic stress values on the boundary.

    The stress is computed as a depth-integrated gravity load with spatially-varying
    gravity based on local material properties. The initial stress state is set to 
    zero for numerical stability.

    Parameters:
        bg: Boundary grid.

    Returns:
        Flattened array of stress values.
    """
    values = np.zeros((3, bg.num_cells))

    # Zero initial stress for numerical stability at t = 0.
    if self.time_manager.time < 1e-5:
        return values.ravel("F")

    # Get per-cell gravity values
    gravity = self.bulk_gravity_force_magnitude_boundary(bg)
    multipliers = self.lithostatic_stress_multipliers
    domain_sides = self.domain_boundary_sides(bg)
    depth = self.depth(bg.cell_centers)

    # Loop over the three spatial components (x, y, z)
    for i, sides in enumerate(
        [["west", "east"], ["south", "north"], ["bottom", "top"]]
    ):
        for side, sign in zip(sides, [1, -1]):
            ind = getattr(domain_sides, side)
            
            if np.any(ind):
                # Compute stress = gravity * depth for all boundary cells on this side
                values[i, ind] = (
                    multipliers[i]
                    * gravity[ind]
                    * depth[ind]
                    * sign
                    * bg.cell_volumes[ind]
                )

    return values.ravel("F")

import porepy as pp
import numpy as np


class HeterogeneousProperties():
    """Mixin assigning per-cell heterogeneous material properties based on depth.

    Overrides the standard PorePy material methods (density, porosity,
    permeability, elastic moduli, ...) so each cell's value is taken from one
    of two ``pp.SolidConstants`` objects (``sedimentary`` and ``crystalline``)
    depending on whether the cell centre lies above or below the
    sediment-crystalline interface stored in ``params["layer_parameters"]``.

    Permeability uses an extra seal layer above ``depth_top_sedimentary`` with
    a low (cap-rock) value; all other properties share the sedimentary
    values within the seal sub-layer.
    """

    def make_heterogeneous(self, subdomains: list[pp.Grid], property_name: str) -> np.ndarray:
        """Return a per-cell array assigning sed/cryst values by depth.

        For each cell across all supplied subdomains, the value of
        ``property_name`` is taken from ``layers["sedimentary"]`` if the cell
        centre is above the interface (z > interface_z) and from
        ``layers["crystalline"]`` otherwise.

        Parameters:
            subdomains: List of subdomains.
            property_name: Attribute name on ``pp.SolidConstants`` (e.g.
                ``"density"``, ``"shear_modulus"``).

        Returns:
            1D numpy array.
        """
        # interface_depth is stored as a positive depth; flip the sign so it
        # matches the z-coordinate convention (z = 0 at surface, z < 0 below).
        interface = -self.units.convert_units(
            self.params["layer_parameters"]["interface_depth"], "m"
        )
        interface_2 = -self.units.convert_units(
            self.params["layer_parameters"]["depth_top_sedimentary"], "m"
        )

        layers = self.params["layer_parameters"]

        sed = layers["sedimentary"]
        cryst = layers["crystalline"]



        # Pull the requested property from each SolidConstants object.
        value_1 = getattr(sed, property_name)
        value_2 = getattr(cryst, property_name)
        value_3 = 1.0e-18 # m^2 permeability for the top layer.

        vals = []
        for sd in subdomains:
            z = sd.cell_centers[2]
            if property_name == "permeability":
                # Three layers for permeability: seal (top) / sedimentary / crystalline.
                heterogeneous_values = np.select(
                    [z > interface_2, z > interface],
                    [value_3, value_1],
                    default=value_2,
                )
            else:
                # Two layers: sedimentary above the interface, crystalline below.
                heterogeneous_values = np.where(z > interface, value_1, value_2)

            vals.append(heterogeneous_values)
        if len(vals) == 0:
            return np.array([])

        return np.hstack(vals)
       
    def solid_density(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Per-cell solid density (kg/m^3)."""
        vals = self.make_heterogeneous(subdomains,"density")
        vals = self.units.convert_units(vals, "kg*m^-3")
        return pp.wrap_as_dense_ad_array(vals, "density")


    def reference_porosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Per-cell reference porosity (dimensionless)."""
        vals = self.make_heterogeneous(subdomains, "porosity")
        return pp.wrap_as_dense_ad_array(vals, "reference_porosity")

    def friction_coefficient(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Per-cell friction coefficient (dimensionless)."""
        vals = self.make_heterogeneous(subdomains, "friction_coefficient")
        return pp.wrap_as_dense_ad_array(vals, "friction_coefficient")


    def biot_coefficient(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Per-cell Biot coefficient (dimensionless)."""
        vals = self.make_heterogeneous(subdomains, "biot_coefficient")
        return pp.wrap_as_dense_ad_array(vals, "biot_coefficient")

    def permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Per-cell isotropic permeability tensor (m^2)."""
        vals = self.make_heterogeneous(subdomains, "permeability")
        vals = self.units.convert_units(vals, "m^2")

        permeability_array = pp.wrap_as_dense_ad_array(vals,"permeability")

        return self.isotropic_second_order_tensor(subdomains, permeability_array)


    def lame_lambda(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Per-cell Lamé's first parameter (Pa)."""
        vals = self.make_heterogeneous(subdomains, "lame_lambda")
        vals = self.units.convert_units(vals, "Pa")

        return pp.wrap_as_dense_ad_array(vals, "lame_lambda")


    def shear_modulus(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Per-cell shear modulus (Pa)."""
        vals = self.make_heterogeneous(subdomains, "shear_modulus")
        vals = self.units.convert_units(vals, "Pa")
        return pp.wrap_as_dense_ad_array(vals, "shear_modulus")


    def residual_aperture(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Per-cell residual aperture (m)."""
        vals = self.make_heterogeneous(subdomains, "residual_aperture")
        vals = self.units.convert_units(vals, "m")
        return pp.wrap_as_dense_ad_array(vals, "residual_aperture")


    def fracture_gap(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Per-cell fracture gap (m)."""
        vals = self.make_heterogeneous(subdomains, "fracture_gap")
        vals = self.units.convert_units(vals, "m")
        return pp.wrap_as_dense_ad_array(vals, "fracture_gap")


    def normal_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Per-cell normal permeability (m^2)."""
        vals = self.make_heterogeneous(subdomains, "normal_permeability")
        vals = self.units.convert_units(vals, "m^2")
        return pp.wrap_as_dense_ad_array(vals, "normal_permeability")


    def biot_tensor(self, subdomains: list[pp.Grid]) -> pp.SecondOrderTensor:
        """Per-cell Biot tensor (dimensionless)."""
        biot_values = self.make_heterogeneous(subdomains,"biot_coefficient")

        return pp.SecondOrderTensor(biot_values)

    def youngs_modulus(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Compute Young's modulus for heterogeneous rock layers.

        Returns Young's modulus values in Pa for sedimentary and crystalline layers
        based on cell center depths relative to the interface.
        Calculated from shear modulus and Lamé lambda: E = μ(3λ + 2μ)/(λ + μ)
        """
        mu = self.make_heterogeneous(subdomains, "shear_modulus")
        lam = self.make_heterogeneous(subdomains, "lame_lambda")

        E = mu * (3 * lam + 2 * mu) / (lam + mu)

        # Convert to proper units (Pa). 
        E = self.units.convert_units(E, "Pa")

        # PorePy multiplies E by the contact traction vector. That vector has
        # 2 * nd entries per fracture cell (nd components on each of 2 sides),
        # so its total length is n_fracture_cells * 2 * nd.
        #
        # The default E is a single number that multiplies every entry. Our
        # heterogeneous E has one value per cell, so we repeat each value
        # 2 * nd times to line it up cell-by-cell with the traction vector.

        E = np.repeat(E, 2 * self.nd)

        return pp.wrap_as_dense_ad_array(E, "youngs_modulus")


    def bulk_modulus(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Compute bulk modulus for heterogeneous rock layers.
        
        Returns the drained bulk modulus values in Pa for sedimentary and crystalline layers
        based on cell center depths relative to the interface.
        Calculated as K = λ + (2/3)μ
        """
        shear_modulus = self.shear_modulus(subdomains)
        lame_lambda = self.lame_lambda(subdomains)

        val = lame_lambda + shear_modulus * 2 / 3
        # Units are already correct (Pa) since shear_modulus and lame_lambda are converted
        return val

    def stiffness_tensor(self, subdomains: pp.Grid) -> pp.FourthOrderTensor:
        """Compute stiffness tensor for heterogeneous rock layers.

        Returns fourth-order stiffness tensor in Pa for sedimentary and crystalline layers
        based on cell center depths relative to the interface.
        """
        mu = self.make_heterogeneous([subdomains], "shear_modulus")
        lmbda = self.make_heterogeneous([subdomains], "lame_lambda")

        mu = self.units.convert_units(mu, "Pa")
        lmbda = self.units.convert_units(lmbda, "Pa")

        return pp.FourthOrderTensor(mu, lmbda)

    def grid_aperture(self,  subdomains: pp.Grid) -> np.ndarray:
        """Compute grid aperture for subdomains.
        
        Returns aperture values in m. For matrix grids, returns unit values scaled by length units.
        For fracture grids, returns residual aperture values. For well grids, returns well radius.
        """
        aperture = np.ones(subdomains.num_cells)
        if subdomains.dim < self.nd:
            if self.is_well_grid(subdomains):
                # This is a well. The aperture is the well radius.
                aperture *= self.solid.well_radius
            else:
                aperture = self.make_heterogeneous([subdomains], "residual_aperture")
                aperture = self.units.convert_units(aperture, "m")
        else:
            # For the matrix, the aperture is one, but needs to be scaled by the
            # length units.
            aperture = self.units.convert_units(aperture, "m")
        return aperture

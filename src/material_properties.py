import porepy as pp
import numpy as np


class HeterogeneousProperties(pp.PorePyModel):

    def make_heterogeneous(self, subdomains: list[pp.Grid], property_name: str) -> np.ndarray:
        
        # Get the interfacee depth where the properties change. The interface depth is negative, so we take the negative of it to get the positive depth.
        # Also we get the material constants for the two materials.
        interface = -self.params["layer_parameters"]["interface_depth"]
        materials = self.params["material_parameters"]

        # For sedimentary and crystalline rock.
        sed = materials["sedimentary"]
        crys = materials["crystalline"]

        
        value_1 = getattr(sed, property_name)
        value_2 = getattr(crys, property_name)

        vals = []

        # Loop over the subdomains and assign the property values based on the depth of the cell centers. If the cell center is above the interface, we assign value_1, otherwise we assign value_2.
        for sd in subdomains:
            z = sd.cell_centers[2]
            heterogeneous_values = np.where(z > interface, value_1, value_2)
            vals.append(heterogeneous_values)
        
        # If there are no subdomains, we return an empty array. hstack does not work with an empty list.
        if len(vals) == 0:
            return np.array([])
        else:
            return np.hstack(vals)
       
    def solid_density(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        vals = self.make_heterogeneous(subdomains,"density")
        vals = self.units.convert_units(vals, "kg*m^-3")
        return pp.wrap_as_dense_ad_array(vals, "density")
        
        
    def reference_porosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        vals = self.make_heterogeneous(subdomains, "porosity")
        return pp.wrap_as_dense_ad_array(vals, "reference_porosity")

    def friction_coefficient(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        vals = self.make_heterogeneous(subdomains, "friction_coefficient")
        return pp.wrap_as_dense_ad_array(vals, "friction_coefficient")


    def biot_coefficient(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        vals = self.make_heterogeneous(subdomains, "biot_coefficient")
        return pp.wrap_as_dense_ad_array(vals, "biot_coefficient")

    # One emty sd because of no wells
    def permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        vals = self.make_heterogeneous(subdomains, "permeability")
        vals = self.units.convert_units(vals, "m^2")

        permeability_array = pp.wrap_as_dense_ad_array(vals,"permeability")

        return self.isotropic_second_order_tensor(subdomains, permeability_array)


    def lame_lambda(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        vals = self.make_heterogeneous(subdomains, "lame_lambda")
        vals = self.units.convert_units(vals, "kg*m^-1*s^-2")
        return pp.wrap_as_dense_ad_array(vals, "lame_lambda")


    def shear_modulus(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        vals = self.make_heterogeneous(subdomains, "shear_modulus")
        vals = self.units.convert_units(vals, "kg*m^-1*s^-2")
        return pp.wrap_as_dense_ad_array(vals, "shear_modulus")


    def residual_aperture(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        vals = self.make_heterogeneous(subdomains, "residual_aperture")
        vals = self.units.convert_units(vals, "m")
        return pp.wrap_as_dense_ad_array(vals, "residual_aperture")


    def fracture_gap(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        vals = self.make_heterogeneous(subdomains, "fracture_gap")
        vals = self.units.convert_units(vals, "m")
        return pp.wrap_as_dense_ad_array(vals, "fracture_gap")


    def normal_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        vals = self.make_heterogeneous(subdomains, "normal_permeability")
        vals = self.units.convert_units(vals, "m^2")
        return pp.wrap_as_dense_ad_array(vals, "normal_permeability")


    def biot_tensor(self, subdomains: list[pp.Grid]) -> pp.SecondOrderTensor:
        biot_values = self.make_heterogeneous(subdomains,"biot_coefficient")

        return pp.SecondOrderTensor(biot_values)

    # used in characteristic_contact_traction, characteristic_displacement, used for scaling.
    def youngs_modulus(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        mu = self.make_heterogeneous(subdomains, "shear_modulus")
        lam = self.make_heterogeneous(subdomains, "lame_lambda")

        E = mu * (3 * lam + 2 * mu) / (lam + mu)

        # Repeat to get same size as nondim_traction.
        E = np.repeat(E, 6)

        return pp.wrap_as_dense_ad_array(E, "youngs_modulus")


    def bulk_modulus(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        shear_modulus = self.shear_modulus(subdomains)
        lame_lambda = self.lame_lambda(subdomains)

        val = lame_lambda + shear_modulus * 2 / 3
        return val

    # Used in solution strategy. 
    def stiffness_tensor(self, subdomains: pp.Grid) -> pp.FourthOrderTensor:
        shear_modulus = self.make_heterogeneous([subdomains],"shear_modulus")
        lame_lambda = self.make_heterogeneous([subdomains],"lame_lambda")

        lmbda = lame_lambda 
        mu = shear_modulus 
        return pp.FourthOrderTensor(mu, lmbda)

    def grid_aperture(self,  subdomains: pp.Grid) -> np.ndarray:
        aperture = np.ones(subdomains.num_cells)
        if subdomains.dim < self.nd:
            if self.is_well_grid(subdomains):
                # This is a well. The aperture is the well radius.
                aperture *= self.solid.well_radius
            else:
                aperture = self.make_heterogeneous([subdomains], "residual_aperture")
        else:
            # For the matrix, the aperture is one, but needs to be scaled by the
            # length units.
            aperture = self.units.convert_units(aperture, "m")
        return aperture
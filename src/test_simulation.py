from simulation import Simulation3D

from simulation import parameters

import numpy as np

import porepy as pp


# set up the model with correct parameters
def test_model():

    params = parameters()
    model_params = params.model_paragrams()
    model = Simulation3D(model_params)
    model.prepare_simulation()

    return model


def test_heterogeneous_parameters(parameter_method_name, boundary, expected_value_1, expected_value_2):
    print("Running test:", parameter_method_name)

    model = test_model()

    # get the subdomains of the model. 
    subdomains = model.mdg.subdomains()

    for sd in subdomains:
        # Get the values used in the model
        parameter_method = getattr(model, parameter_method_name)

        model_values = parameter_method([sd]).value(model.equation_system) # value in array

        # depth of the cell centers
        z = sd.cell_centers[2]

        # Wanted values
        expected_values = np.where(z > -boundary, expected_value_1, expected_value_2) # calculated as array

        # For now
        if parameter_method_name == "permeability":
            expected_tensor = model.isotropic_second_order_tensor([sd], expected_values) #Ad.operator
            expected_values = expected_tensor.value(model.equation_system) #to array

        # Check if the values are close enough.
        assert np.allclose(model_values, expected_values)

model = test_model()
parameters_solid_1 = model.params["material_constants"]["solid"]



#boundary = 3500 
#test_heterogeneous_parameters("solid_density", boundry, 2400.0, 2680.0)
#test_heterogeneous_parameters("biot_coefficient", boundry, 0.8, 0.47)
#test_heterogeneous_parameters("porosity", boundry, 0.2, 0.01)
#test_heterogeneous_parameters("permeability", boundry, 1e-13,  5.0e-13)
#test_heterogeneous_parameters("lame_lambda", boundry, 3.3e9, 1.5e10)
#test_heterogeneous_parameters("shear_modulus", boundry, 8.9e9, 1.5e10)
#test_heterogeneous_parameters("friction_coefficient", boundry, 0.7, 0.65)
#test_heterogeneous_parameters("dilation_angle", boundry, 0.1, 0.10)
#test_heterogeneous_parameters("residual_aperture", boundry, 1e-3, 1e-3)
#test_heterogeneous_parameters("fracture_gap", boundry, 1e-4, 1e-4)
#test_heterogeneous_parameters("normal_permeability", boundry, 1.0e-10, 1.0e-10)
    

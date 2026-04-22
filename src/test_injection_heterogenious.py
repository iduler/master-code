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


def test_injection_flux():
    ...


def test_lithostatic_stress():
    ...

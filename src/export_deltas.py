import numpy as np


class ExportDeltasMixin:
    """Mixin for exporting fracture-related quantities and injection-induced changes.

    Designed to be combined with ``FractureDeformationExporting`` via::

        class ExportMixin(FractureDeformationExporting, ExportDeltasMixin):
            pass

    Then put ``ExportMixin`` in the model's mixin chain. The
    ``super().data_to_export()`` call chains into PorePy's
    ``DataSavingMixin.data_to_export`` via the MRO, so this class does not
    need to inherit ``pp.PorePyModel`` explicitly.

    Bulk subdomain (3D) fields appended:
      - ``delta_pressure``      change in pressure since ``start_simulation``.
      - ``delta_displacement``  change in displacement since ``start_simulation``.
      - ``permeability``        per-cell isotropic permeability.

    Fracture subdomain (nd - 1) fields appended:
      - ``delta_fracture_pressure``  change in fracture pore pressure.
      - ``delta_slip_tendency``      change in the Mohr-Coulomb yield ratio.
      - ``delta_traction``           change in contact traction (local frame).
    """

    def data_to_export(self) -> list:
        """Return fracture-specific quantities to include in model export.

        Returns:
            List of (subdomain, name, values) tuples to be exported.
        """
        data = super().data_to_export() 


        # --- 3-D subdomain quantities (delta relative to injection start) ---
        # Use the same source of truth as the schedule
        # (params["injection_params"]["start_simulation"]) so the reference
        # state is captured at the moment injection actually starts.
        injection_reference_time = self.units.convert_units(
            self.params["injection_params"]["start_simulation"], "s"
        )

        bulk_subdomains = self.mdg.subdomains(dim=3)
        fracture_subdomains = self.mdg.subdomains(dim=2)

        bulk_scalar_offsets = np.cumsum([0] + [sd.num_cells for sd in bulk_subdomains])
        bulk_vector_offsets = np.cumsum([0] + [sd.num_cells * self.nd for sd in bulk_subdomains])

        fracture_scalar_offsets = np.cumsum([0] + [sd.num_cells for sd in fracture_subdomains])
        fracture_vector_offsets = np.cumsum([0] + [sd.num_cells * self.nd for sd in fracture_subdomains])


        pressure = self.evaluate_and_scale(bulk_subdomains, "pressure", "Pa")
        displacement = self.evaluate_and_scale(bulk_subdomains, "displacement", "m")
        traction = self.evaluate_and_scale(fracture_subdomains, "contact_traction", "Pa")
        pressure_fracture = self.evaluate_and_scale(fracture_subdomains, "pressure", "Pa")
        friction_coefficient = self.evaluate_and_scale(fracture_subdomains, "friction_coefficient", "")

        slip_tendency = self.compute_slip_tendency(traction.reshape((self.nd, -1), order="F"), friction_coefficient)



        # Setting reference values. hasattr(self, "p_ref") checks if there exists a reference pressure state. 
        if self.time_manager.time >= injection_reference_time and not hasattr(self, "p_ref"):
            self.p_ref = pressure.copy()
            self.u_ref = displacement.copy()
            self.traction_ref = traction.copy()
            self.slip_tendency_ref = slip_tendency.copy()
            self.fracture_pressure_ref = pressure_fracture.copy()

        # Compute delta relative to injection start.
        if hasattr(self, "p_ref"):
            delta_pressure = pressure - self.p_ref
            delta_displacement = displacement - self.u_ref
            delta_traction = traction - self.traction_ref
            delta_slip_tendency = slip_tendency - self.slip_tendency_ref
            delta_fracture_pressure = pressure_fracture - self.fracture_pressure_ref

        # If the reference state has not been set yet.
        else:
            delta_pressure = np.zeros_like(pressure)
            delta_displacement = np.zeros_like(displacement)
            delta_traction = np.zeros_like(traction)
            delta_slip_tendency = np.zeros_like(slip_tendency)
            delta_fracture_pressure = np.zeros_like(pressure_fracture)

        # Per-cell permeability (scalar, m^2). Computed from the layer-wise
        # values in ``HeterogeneousProperties`` so visualisations can show
        # which cells are sedimentary vs crystalline at a glance.
        permeability = self.units.convert_units(
            self.make_heterogeneous(bulk_subdomains, "permeability"), "m^2"
        )

    

        for i, sd in enumerate(bulk_subdomains):
            data.append(
                (
                    sd,
                    "delta_pressure",
                    delta_pressure[bulk_scalar_offsets[i] : bulk_scalar_offsets[i + 1]]
                ))
            data.append(
                (
                    sd,
                    "delta_displacement",
                    delta_displacement[bulk_vector_offsets[i] : bulk_vector_offsets[i + 1]],
                ))
            data.append(
                (
                    sd,
                    "permeability",
                    permeability[bulk_scalar_offsets[i] : bulk_scalar_offsets[i + 1]],
                ))
            
        for i, sd in enumerate(fracture_subdomains):
            data.append(
                (
                    sd,
                    "delta_fracture_pressure",
                    delta_fracture_pressure[fracture_scalar_offsets[i] : fracture_scalar_offsets[i + 1]]
                ))
            data.append(
                (
                    sd,
                    "delta_slip_tendency",
                    delta_slip_tendency[fracture_scalar_offsets[i] : fracture_scalar_offsets[i + 1]],
                ))
            data.append(
                (
                    sd,
                    "delta_traction",
                    delta_traction[fracture_vector_offsets[i] : fracture_vector_offsets[i + 1]],
                ))

        return data


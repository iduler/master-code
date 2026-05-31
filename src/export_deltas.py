import numpy as np
import porepy as pp

from porepy.viz.data_saving_model_mixin import FractureDeformationExporting

class ExportDeltasMixin(FractureDeformationExporting):
    """Export injection-induced changes in pressure, displacement, and stress.

    All deltas are differences from a reference state captured at ``start_varible_injection - 1 year``.

    Bulk (3D) fields:
      - delta_pressure: pressure change since the reference state.
      - delta_displacement: displacement change since the reference state.
      - delta_mean_volume_stress: K · Δ(∇·u).
      - delta_mean_stress: K · Δ(∇·u) - α · Δp
      - delta_mean_stress_abs / _neg / _pos: absolute / compressive-only /
        dilation-only magnitudes of delta_mean_stress.
      - permeability: per-cell permeability, exported to visualise heterogeneity.

    Fracture (2D) fields:
      - delta_fracture_pressure: fracture pore pressure change.
      - delta_slip_tendency: change in |τ|/(σ_n·μ); 1.0 means at the friction limit.
    """

    def data_to_export(self) -> list:
        """
        Returns:
            List of (subdomain, name, values) tuples to be exported.
        """

        # Start with exports from parent classes including FractureDeformationExporting.
        data = super().data_to_export()

        # Reference time: 1 year before the injection starts
        injection_reference_time = self.units.convert_units(
            self.params["injection_params"]["start_varible_injection"], "s") - 1 * pp.YEAR

        bulk_subdomains = self.mdg.subdomains(dim=self.nd)
        fracture_subdomains = self.mdg.subdomains(dim=self.nd-1)

        # Offsets into the combined cell arrays (scalar: 1 value per cell, vector: nd values per cell).
        bulk_scalar_offsets = np.cumsum([0] + [sd.num_cells for sd in bulk_subdomains])
        bulk_vector_offsets = np.cumsum([0] + [sd.num_cells * self.nd for sd in bulk_subdomains])

        fracture_scalar_offsets = np.cumsum([0] + [sd.num_cells for sd in fracture_subdomains])
        fracture_vector_offsets = np.cumsum([0] + [sd.num_cells * self.nd for sd in fracture_subdomains])

        pressure = self.evaluate_and_scale(bulk_subdomains, "pressure", "Pa")
        displacement = self.evaluate_and_scale(bulk_subdomains, "displacement", "m")

        traction = self.evaluate_and_scale(fracture_subdomains, "contact_traction", "Pa")
        pressure_fracture = self.evaluate_and_scale(fracture_subdomains, "pressure", "Pa")
        friction_coefficient = self.evaluate_and_scale(fracture_subdomains, "friction_coefficient", "")
        alpha = self.evaluate_and_scale(bulk_subdomains, "biot_coefficient", "")

        # Biot-scaled volumetric strain (α·∇·u).
        div_u_alpha = self.evaluate_and_scale(bulk_subdomains, "displacement_divergence", "")

        # Undo Biot scaling to get plain ∇·u for the stress formulas below.
        div_u = div_u_alpha / alpha

        # Calculate from functions in FractureDeformationExporting
        slip_tendency = self.compute_slip_tendency(traction.reshape((self.nd, -1), order="F"), friction_coefficient)

        # Store reference state once
        if self.time_manager.time >= injection_reference_time and not hasattr(self, "p_ref"):
            self.p_ref = pressure.copy()
            self.u_ref = displacement.copy()
            self.traction_ref = traction.copy()
            self.slip_tendency_ref = slip_tendency.copy()
            self.fracture_pressure_ref = pressure_fracture.copy()
            self.div_u_ref = div_u.copy()

        if hasattr(self, "p_ref"):
            delta_pressure = pressure - self.p_ref
            delta_displacement = displacement - self.u_ref
            delta_slip_tendency = slip_tendency - self.slip_tendency_ref
            delta_fracture_pressure = pressure_fracture - self.fracture_pressure_ref
            delta_div_u = div_u - self.div_u_ref

        # Reference not set yet: report zeros.
        else:
            delta_pressure = np.zeros_like(pressure)
            delta_displacement = np.zeros_like(displacement)
            delta_slip_tendency = np.zeros_like(slip_tendency)
            delta_fracture_pressure = np.zeros_like(pressure_fracture)
            delta_div_u = np.zeros_like(div_u)

        # Mean volume stress change (Pa) = K · Δ(∇·u).
        bulk_modulus = self.evaluate_and_scale(bulk_subdomains, "bulk_modulus", "Pa")
        delta_mean_volume_stress = bulk_modulus * delta_div_u

        # Mean total stress change (Pa) = K · Δ(∇·u) − α · Δp.
        delta_mean_stress = bulk_modulus * delta_div_u - alpha * delta_pressure
        delta_mean_stress_abs = np.abs(delta_mean_stress)
        delta_mean_stress_neg = np.maximum(-delta_mean_stress, 0.0)
        delta_mean_stress_pos = np.maximum(delta_mean_stress, 0.0)

        # Per-cell permeability (scalar, m^2)
        permeability = self.units.convert_units(
            self.make_heterogeneous(bulk_subdomains, "permeability"), "m^2"
        )

    
        # Append the values to the data exporter
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
            data.append(
                (
                    sd,
                    "delta_mean_volume_stress",
                    delta_mean_volume_stress[bulk_scalar_offsets[i] : bulk_scalar_offsets[i + 1]],
                ))
            data.append(
                (
                    sd,
                    "delta_mean_stress",
                    delta_mean_stress[bulk_scalar_offsets[i] : bulk_scalar_offsets[i + 1]],
                ))
            data.append(
                (
                    sd,
                    "delta_mean_stress_abs",
                    delta_mean_stress_abs[bulk_scalar_offsets[i] : bulk_scalar_offsets[i + 1]],
                ))
            data.append(
                (
                    sd,
                    "delta_mean_stress_neg",
                    delta_mean_stress_neg[bulk_scalar_offsets[i] : bulk_scalar_offsets[i + 1]],
                ))
            data.append(
                (
                    sd,
                    "delta_mean_stress_pos",
                    delta_mean_stress_pos[bulk_scalar_offsets[i] : bulk_scalar_offsets[i + 1]],
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
                    delta_slip_tendency[fracture_scalar_offsets[i] : fracture_scalar_offsets[i + 1]]
                ))

        return data


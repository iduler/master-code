"""3D grid construction for the two-layer (crystalline + sedimentary) model.

Adapted from code by Eirik Keilegavlen and Ingrid Kristine Jacobsen,
with help from Ivar Stefansson.
"""

import numpy as np
import scipy.sparse as sps

import porepy as pp
from grid_manipulation import glue_grids, extract_surface_grid


class CombinedGeometry:
    """Crystalline (bottom) + sedimentary (top) glued into a single 3D grid.

    The bottom layer is meshed by PorePy as a 3D simplex grid. The top face of
    that grid is extracted, extruded with uniform layers, and pasted onto the
    bottom layer. The result is wrapped in a `MixedDimensionalGrid`.

    Layer geometry is read from ``self.params``:
      * ``layer_parameters["depth_top_domain"]`` — depth of the top boundary [m]
      * ``layer_parameters["interface_depth"]`` — sediment/crystalline interface [m]
      * ``layer_parameters["n_sedimentary_layers"]`` — uniform extrusion layers
        in the sedimentary block.
      * ``domain_sizes`` — ``[x_size, y_size, z_size]`` in m; the bottom depth
        is ``depth_top_domain + z_size``.
    """

    def top_depth(self) -> float:
        """Depth of the top boundary, in m (positive = downward)."""
        return float(self.params["layer_parameters"]["depth_top_domain"])

    def interface_depth(self) -> float:
        """Depth of the sediment-crystalline interface, in m."""
        return float(self.params["layer_parameters"]["interface_depth"])

    def bottom_depth(self) -> float:
        """Depth of the bottom boundary, in m (= top depth + vertical extent)."""
        return self.top_depth() + float(self.params["domain_sizes"][2])

    def n_sed_layers(self) -> int:
        """Number of uniform layers in the sedimentary extrusion."""
        return int(self.params["layer_parameters"]["n_sedimentary_layers"])

    def box_2d(self) -> dict:
        """Return the horizontal extent of the domain in model units."""
        dx, dy, _ = self.params["domain_sizes"]
        dx = self.units.convert_units(dx, "m")
        dy = self.units.convert_units(dy, "m")
        return {"xmin": 0.0, "xmax": dx, "ymin": 0.0, "ymax": dy}

    def set_domain(self) -> None:
        box = self.box_2d()
        box.update({
            "zmin": self.units.convert_units(-self.bottom_depth(), "m"),
            "zmax": self.units.convert_units(-self.top_depth(), "m"),
        })
        self._domain = pp.Domain(box)

    def create_mdg(self) -> None:
        z_interface = self.units.convert_units(-self.interface_depth(), "m")
        z_top       = self.units.convert_units(-self.top_depth(), "m")
        z_bottom    = self.units.convert_units(-self.bottom_depth(), "m")

    
        # --- Crystalline (bottom) layer: standard 3D simplex mesh ---
       
        fractures = getattr(self, "_fractures", [])
        box_bottom = self.box_2d()
        box_bottom.update({"zmin": z_bottom, "zmax": z_interface})

        fn_bottom = pp.create_fracture_network(fractures, domain=pp.Domain(box_bottom))
        mdg_bottom = pp.create_mdg("simplex", self.meshing_arguments(), fn_bottom)
        # self.mdg = mdg_bottom

        g_3d_bottom = mdg_bottom.subdomains(dim=3)[0]
        g_2d =  mdg_bottom.subdomains(dim=2)[0]

        target_faces_bottom = np.where(
            np.isclose(g_3d_bottom.face_centers[2, :], z_interface)
        )[0]


        # --- Sedimentary (top) layer: uniform extrusion of the interface surface ---
        sed_thickness = z_top - z_interface  # positive (500 m)
        z_layers = np.linspace(0.0, sed_thickness, self.n_sed_layers() + 1)

        g_2d_top = extract_surface_grid.extract(g_3d_bottom, target_faces_bottom)
        g_2d_top.compute_geometry()

        g_3d_top, *_ = pp.grid_extrusion.extrude_grid(g_2d_top, z_layers)
        g_3d_top.nodes[2] += z_interface
        g_3d_top.compute_geometry()

        # --- Glue the two layers together ---

        plane_coefficients = np.array([0, 0, 1]).reshape((3, 1))

        g = glue_grids.paste_3d_simplex_grids(
            g_3d_bottom, g_3d_top,
            plane_coefficients=plane_coefficients,
            offset=z_interface,
        )
        

        g.compute_geometry()
        self.mdg = pp.MixedDimensionalGrid()
        self.mdg.add_subdomains([g, g_2d])
        intf_bottom = mdg_bottom.interfaces(dim=2)[0]

        # Get old bottom map in csc.
        primary_secondary_map_bottom = mdg_bottom.interface_data(intf_bottom)["face_cells"]
        # Add zeros for the faces of the top grid, which are not part of the interface.
        num_new_faces = g.num_faces - g_3d_bottom.num_faces
        # Create sparse zeros for the new faces in the top grid.
        zeros_top = sps.csr_matrix(( primary_secondary_map_bottom.shape[0], num_new_faces))
    
        # Concatenate the old map with the zeros to create the new map for the bottom interface.
        primary_secondary_map = sps.hstack([primary_secondary_map_bottom, zeros_top], format="csc")
        pp.fracs.meshing.create_interfaces(self.mdg,{(g, g_2d): primary_secondary_map})

        # self.mdg.add_interface(intf_bottom, [g, g_2d], primary_secondary_map)
        self.mdg.compute_geometry()
        self.mdg.set_boundary_grid_projections()
        self.nd: int = self.mdg.dim_max()

        # Create projections between local and global coordinates for fracture grids.
        pp.set_local_coordinate_projections(self.mdg)
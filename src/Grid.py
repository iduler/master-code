import numpy as np
import scipy.sparse as sps

import porepy as pp
from grid_manipulation import glue_grids, extract_surface_grid


class CombinedGeometry:
    """Crystalline (bottom) + sedimentary (top) glued into a single 3D grid.

    The bottom layer is meshed by PorePy as a 3D simplex grid. The top face of
    that grid is extracted, extruded with uniform layers, and pasted onto the
    bottom layer. The result is wrapped in a `MixedDimensionalGrid`.
    """

    _top_depth: float       = 2000.0    # m  depth of top boundary
    _interface_depth: float = 2500.0    # m  sedimentary / crystalline interface
    _bottom_depth: float    = 12000.0   # m  depth of bottom boundary
    _n_sed_layers: int      = 2         # number of uniform sedimentary layers

    def box_2d(self) -> dict:
        dx = self.units.convert_units(70000.0, "m")
        dy = self.units.convert_units(40000.0, "m")
        return {"xmin": 0.0, "xmax": dx, "ymin": 0.0, "ymax": dy}

    def set_domain(self) -> None:
        box = self.box_2d()
        box.update({
            "zmin": self.units.convert_units(-self._bottom_depth, "m"),
            "zmax": self.units.convert_units(-self._top_depth, "m"),
        })
        self._domain = pp.Domain(box)

    def create_mdg(self) -> None:
        z_interface = self.units.convert_units(-self._interface_depth, "m")
        z_top       = self.units.convert_units(-self._top_depth, "m")
        z_bottom    = self.units.convert_units(-self._bottom_depth, "m")

    
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
        z_layers = np.linspace(0.0, sed_thickness, self._n_sed_layers + 1)

        g_2d_top = extract_surface_grid.extract(g_3d_bottom, target_faces_bottom)
        g_2d_top.compute_geometry()

        g_3d_top, *_ = pp.grid_extrusion.extrude_grid(g_2d_top, z_layers)
        g_3d_top.nodes[2] += z_interface
        g_3d_top.compute_geometry()

        # --- Glue the two layers together ---

        plane_coefficients = np.array([0, 0, 1]).reshape((3, 1))
        
        g = glue_grids.paste_3d_simplex_grids(
            g_3d_bottom, g_3d_top, plane_coefficients=plane_coefficients, offset=-2500.0)
        

        g.compute_geometry()

        # Wrap the pasted grid in a MixedDimensionalGrid so PorePy's model
        # machinery sees the right type of object.
        # full_box =  self.box_2d()
        # full_box.update({"zmin": z_bottom, "zmax":  z_top})
        # fn_box = pp.create_fracture_network(fractures, domain=pp.Domain(full_box))
        # self.mdg = pp.create_mdg("simplex", self.meshing_arguments(), fn_box)
        # self.mdg.remove_subdomain(self.mdg.subdomains(dim=3)[0])

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
        primary_secondary_map = sps.hstack([primary_secondary_map_bottom, zeros_top], format="csr")
        intf = fsdkløfkøadskodfawpp.MortarGrid(2)
        self.mdg.add_interface(intf_bottom, [g, g_2d], primary_secondary_map)
        self.mdg.compute_geometry()
        self.mdg.set_boundary_grid_projections()

        self.nd: int = self.mdg.dim_max()

        # Create projections between local and global coordinates for fracture grids.
        pp.set_local_coordinate_projections(self.mdg)

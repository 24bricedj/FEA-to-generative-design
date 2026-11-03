"""
Read ANY mesh meshio understands (Gmsh, Abaqus, VTK, ...) into the arrays the
solver needs, plus named node sets from the mesh's physical groups.
"""
import numpy as np
import meshio


def read_mesh(fname):
    """Return nodes (n,2), tris (ne,3), and {group_name: node_indices}."""
    m = meshio.read(fname)
    nodes = m.points[:, :2].copy()           # drop z for 2D

    tris = m.cells_dict["triangle"].astype(int)

    # node sets from physical groups on points (vertex) and edges (line)
    node_sets = {}
    for name, by_type in m.cell_sets_dict.items():
        if name.startswith("gmsh:"):
            continue
        ids = []
        for ctype, idx in by_type.items():
            if ctype in ("vertex", "line"):
                ids.append(m.cells_dict[ctype][idx].ravel())
        if ids:
            node_sets[name] = np.unique(np.concatenate(ids))

    # drop nodes not used by any triangle (keeps K non-singular)
    used = np.unique(tris)
    remap = -np.ones(len(nodes), dtype=int)
    remap[used] = np.arange(len(used))
    nodes = nodes[used]
    tris = remap[tris]
    node_sets = {k: remap[v][remap[v] >= 0] for k, v in node_sets.items()}

    # fix orientation: every triangle counter-clockwise (positive area)
    x, y = nodes[:, 0], nodes[:, 1]
    a = tris
    signed = ((x[a[:,1]]-x[a[:,0]])*(y[a[:,2]]-y[a[:,0]]) -
              (x[a[:,2]]-x[a[:,0]])*(y[a[:,1]]-y[a[:,0]]))
    flip = signed < 0
    tris[flip] = tris[flip][:, [0, 2, 1]]

    return nodes, tris, node_sets


def read_mesh_3d(fname):
    """3D version: nodes (n,3), tets (ne,4), named node sets, and named line
    segments (for distributing line loads along tagged edges)."""
    m = meshio.read(fname)
    nodes = m.points.copy()
    tets = m.cells_dict["tetra"].astype(int)

    node_sets, segments = {}, {}
    for name, by_type in m.cell_sets_dict.items():
        if name.startswith("gmsh:"):
            continue
        ids = []
        for ctype, idx in by_type.items():
            if ctype in ("vertex", "line", "triangle"):
                cells = m.cells_dict[ctype][idx]
                ids.append(cells.ravel())
                if ctype == "line":
                    segments[name] = cells
        if ids:
            node_sets[name] = np.unique(np.concatenate(ids))

    # drop unused nodes
    used = np.unique(tets)
    remap = -np.ones(len(nodes), dtype=int)
    remap[used] = np.arange(len(used))
    nodes, tets = nodes[used], remap[tets]
    node_sets = {k: remap[v][remap[v] >= 0] for k, v in node_sets.items()}
    segments = {k: remap[v] for k, v in segments.items()}

    # fix orientation: positive volume for every tet
    a, b, c, d = (nodes[tets[:, i]] for i in range(4))
    vol6 = np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))
    flip = vol6 < 0
    tets[flip] = tets[flip][:, [0, 2, 1, 3]]

    return nodes, tets, node_sets, segments
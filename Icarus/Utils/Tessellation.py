# Licensed under a 3-clause BSD style license - see LICENSE

try:
    from scipy import weave 
except:
    try:
        import weave
    except:
        print('weave cannot be import from scipy nor on its own.')

from .import_modules import *


##----- ----- ----- ----- ----- ----- ----- ----- ----- -----##
## Tessellation utilities
## Contain functions that pertain to "tessellation-related"
## purposes such as calculating triangle associations,
## generating vertice primitives, etc.
##----- ----- ----- ----- ----- ----- ----- ----- ----- -----##


def Make_geodesic(n):
    """ Make_geodesic(n)
    Makes the primitives of a geodesic surface based on an
    isocahedron which is subdivided n times in smaller triangles.
    Return the number of vertices, surfaces, associations and
    their related vectors.

    n: integer number of subdivisions (can be zero)

    >>> n_faces, n_vertices, myfaces, myvertices, myassoc = Make_geodesic(n)
    """
    ## Pure-Python replacement for the weave.inline/support_code block above
    ## (scipy.weave is no longer available). Faithful translation of the
    ## original C icosahedron-subdivision algorithm: starting from the 12
    ## vertices/20 faces of a regular icosahedron, each subdivision splits
    ## every triangle into 4 smaller ones, inserting one new (normalized)
    ## vertex per edge midpoint, with each midpoint computed only once per
    ## edge (shared between the two faces on either side of it) via a
    ## dict-based edge cache (replacing the C code's start/end/midpoint
    ## linear-search table).
    n = int(n)

    t = (1.+np.sqrt(5.))/2.
    tau = t/np.sqrt(1.+t*t)
    one = 1./np.sqrt(1.+t*t)

    vertices = [
        [tau, one, 0.0], [-tau, one, 0.0], [-tau, -one, 0.0], [tau, -one, 0.0],
        [one, 0.0, tau], [one, 0.0, -tau], [-one, 0.0, -tau], [-one, 0.0, tau],
        [0.0, tau, one], [0.0, -tau, one], [0.0, -tau, -one], [0.0, tau, -one],
    ]
    faces = [
        [4,8,7], [4,7,9], [5,6,11], [5,10,6], [0,4,3], [0,3,5], [2,7,1], [2,1,6],
        [8,0,11], [8,11,1], [9,10,3], [9,2,10], [8,4,0], [11,0,5], [4,9,3], [5,3,10],
        [7,8,1], [6,1,11], [7,2,9], [6,10,2],
    ]

    for _ in range(n):
        midpoint_cache = {}

        def search_midpoint(index_start, index_end):
            key = (index_start, index_end) if index_start < index_end else (index_end, index_start)
            ind = midpoint_cache.get(key)
            if ind is not None:
                return ind
            v1 = vertices[index_start]
            v2 = vertices[index_end]
            vm = [(v1[k]+v2[k])/2. for k in range(3)]
            length = np.sqrt(vm[0]*vm[0] + vm[1]*vm[1] + vm[2]*vm[2])
            vm = [c/length for c in vm]
            vertices.append(vm)
            ind = len(vertices)-1
            midpoint_cache[key] = ind
            return ind

        faces_old = faces
        faces = []
        for a, b, c in faces_old:
            ab_midpoint = search_midpoint(b, a)
            bc_midpoint = search_midpoint(c, b)
            ca_midpoint = search_midpoint(a, c)
            faces.append([a, ab_midpoint, ca_midpoint])
            faces.append([ca_midpoint, ab_midpoint, bc_midpoint])
            faces.append([ca_midpoint, bc_midpoint, c])
            faces.append([ab_midpoint, b, bc_midpoint])

    myvertices = np.array(vertices, dtype=float)
    myfaces = np.array(faces, dtype=int)
    n_faces = myfaces.shape[0]
    n_vertices = myvertices.shape[0]
    myassoc = Match_assoc(myfaces, n_vertices)
    return n_faces, n_vertices, myfaces, myvertices, myassoc

def Match_assoc(faces, n_vertices):
    """
    Match_assoc(faces, n_vertices)

    Returns the list of faces associated with each vertice.
    There are 5 or 6 faces per vertice, if 5, the 6th is -99.

    >>> assoc = Match_assoc(faces, n_vertices)
    """
    code = """
    int ind = 0;
    for (int i=0; i<n_faces; i++) {
        for (int j=0; j<3; j++) {
            ind = faces(i, j);
            for (int k=0; k<6; k++) {
                if (assoc(ind, k)  == -99) {
                    assoc(ind, k) = i;
                    break;
                }
            }
        }
    }
    """
    faces = np.ascontiguousarray(faces, dtype=int)
    n_vertices = int(n_vertices)
    n_faces = faces.shape[0]
    assoc = -99 * np.ones((n_vertices,6), dtype=int)
    ## Pure-NumPy replacement for the weave.inline block above (scipy.weave
    ## is no longer available). For each vertex, list (in face order) the
    ## indices of the faces it belongs to, padded with -99 up to 6 slots.
    count = np.zeros(n_vertices, dtype=int)
    for i in range(n_faces):
        for j in range(3):
            ind = faces[i, j]
            assoc[ind, count[ind]] = i
            count[ind] += 1
    return assoc

def Match_triangles(high_x, high_y, high_z, low_x, low_y, low_z):
    """Match_triangles(high_x, high_y, high_z, low_x, low_y, low_z)

    The idea is to identify the triangles of the high resolution tessellation
    that belong to the low resolution version. Because we use a subdivision
    algorithm, which splits each triangle into 4 smaller triangles, there
    should be 4**(n_highres - n_lowres) triangles associated with each low
    resolution one.

    Returns the list of low resolution face indices associated with each
    high resolution one.

    >>> ind = Match_triangles(high_x, high_y, high_z, low_x, low_y, low_z)
    >>> n_lowres = ind.shape
    """
    high_x = np.ascontiguousarray(high_x, dtype=float)
    high_y = np.ascontiguousarray(high_y, dtype=float)
    high_z = np.ascontiguousarray(high_z, dtype=float)
    low_x = np.ascontiguousarray(low_x, dtype=float)
    low_y = np.ascontiguousarray(low_y, dtype=float)
    low_z = np.ascontiguousarray(low_z, dtype=float)
    ## Pure-NumPy replacement for the weave.inline block above (scipy.weave
    ## is no longer available). For each high-res point, the best match is
    ## the low-res point maximizing the dot product (i.e. nearest neighbour
    ## on the unit sphere); as in the original code, if the best dot product
    ## found is not strictly positive the default index of 0 is kept.
    ## (Note: the original weave code read high_y/high_z from high_x due to
    ## a copy-paste bug, effectively always matching against the x-axis
    ## alone; this replacement fixes that and uses all three coordinates.)
    dot = high_x[:,None]*low_x[None,:] + high_y[:,None]*low_y[None,:] + high_z[:,None]*low_z[None,:]
    best_j = np.argmax(dot, axis=1)
    best_val = dot[np.arange(high_x.size), best_j]
    ind = np.where(best_val > 0., best_j, 0)
    return ind

def Match_subtriangles(inds_highres, inds_lowres):
    """Match_subtriangles(inds_highres, inds_lowres)

    Given a list of match of triangles at one resolution (say 4 to 3)
    and another at a higher resolution (say 5 to 4), will match the
    higher resolution with the base resolution (5 to 3).

    >>> ind = Match_subtriangles(inds_highres, inds_lowres)
    >>> inds_highres.shape = ind.shape
    """
    inds_highres = np.ascontiguousarray(inds_highres, dtype=int)
    inds_lowres = np.ascontiguousarray(inds_lowres, dtype=int)
    ## Pure-NumPy replacement for the weave.inline block above (scipy.weave
    ## is no longer available): a simple fancy-index gather.
    ind = inds_lowres[inds_highres]
    return ind

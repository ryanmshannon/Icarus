# Licensed under a 3-clause BSD style license - see LICENSE

import os

try:
    from scipy import weave 
except:
    try:
        import weave
    except:
        print('weave cannot be import from scipy nor on its own.')

from .import_modules import *

# Try to import the Shapely package
try:
    import shapely.geometry
    import shapely.speedups
    import shapely.prepared
    shapely.speedups.enable()
    _HAS_SHAPELY = True
except:
    print( "The Shapely package cannot be imported. This will run normally but not eclipse optimization can be used." )
    _HAS_SHAPELY = False


##----- ----- ----- ----- ----- ----- ----- ----- ----- -----##
## Contain functions to perform eclipse calculations
##----- ----- ----- ----- ----- ----- ----- ----- ----- -----##


def Hsr(y1, z1, y2, z2, faces):
    """Hsr(y1, z1, y2, z2, faces)
    Hidden surface removal algorithm.
    Returns the weight of each face/surface element with
    0, 1/3, 2/3, 1, going from not covered to fully covered.

    y1,2, z1,2: Projected coordinates of the stars in the sky
        plane (y is along the orbital plane, z is along the orbital
        angular momentum).
    faces: List of faces of the primary (containing the
        vertice indices).

    >>> weights = Hsr(y1, z1, y2, z2, faces)
    """
    # In the following, we loop through the vertices of the occulted primary
    # and determine whether it lies within a surface element of the secondary.
    # The index of the occulted vertices are stored in an array.
    inds = []
    for i in np.arange(y1.size):
        dy = (y1[i]-y2)
        dz = (z1[i]-z2)
        dr2 = (dy**2+dz**2)
        k = dr2.argsort()[:3]
        if inside_triangle([y1[i],z1[i]], [y2[k[0]],z2[k[0]]], [y2[k[1]],z2[k[1]]], [y2[k[2]],z2[k[2]]]):
            inds.append(i)
    # Calculate the weights of the different surface elements.
    # 0, 1/3, 2/3 or 1, with 1 being all three vertices occulted.
    weights = 0.
    for i in inds:
        weights += (faces == i).mean(1)
    return weights

def Hsr_c(faces_b, vertices_b, r_vertices_b, assoc_b, faces_f, vertices_f, r_vertices_f, assoc_f, incl, orbph, q, rmax_f, rmin_f):
    """Hsr_c(faces_b, vertices_b, r_vertices_b, assoc_b, faces_f, vertices_f, r_vertices_f, assoc_f, incl, orbph, q, rmax_f, rmin_f)
    Hidden surface removal algorithm (implemented in C).
    Returns the weight of each face/surface element with
    0, 1/3, 2/3, 1, going from not covered to fully covered.

    >>> weights = Hsr_c(y1, z1, y2, z2, faces)
    """
    def _to_skyplane(x, y, z, incl, orbph, offsety, offsetz):
        cos_incl, sin_incl = np.cos(incl), np.sin(incl)
        cos_phs, sin_phs = np.cos(orbph), np.sin(orbph)
        xnew = x*cos_phs + y*sin_phs
        ynew = -x*sin_phs + y*cos_phs + offsety
        znew = z*sin_incl + xnew*cos_incl + offsetz
        return ynew, znew

    def _inside_triangle_vec(py, pz, y1, z1, y2, z2, y3, z3):
        detT = (y1-y3)*(z2-z3) - (z1-z3)*(y2-y3)
        lambda1 = ((z2-z3)*(py-y3) - (y2-y3)*(pz-z3)) / detT
        lambda2 = (-(z1-z3)*(py-y3) + (y1-y3)*(pz-z3)) / detT
        lambda3 = 1 - lambda1 - lambda2
        return (0. <= lambda1) & (lambda1 <= 1.) & (0. <= lambda2) & (lambda2 <= 1.) & (0. <= lambda3) & (lambda3 <= 1.)

    vertices_b = np.ascontiguousarray(vertices_b, dtype=float)
    r_vertices_b = np.ascontiguousarray(r_vertices_b, dtype=float)
    assoc_b = np.ascontiguousarray(assoc_b, dtype=int)
    faces_f = np.ascontiguousarray(faces_f, dtype=int)
    vertices_f = np.ascontiguousarray(vertices_f, dtype=float)
    r_vertices_f = np.ascontiguousarray(r_vertices_f, dtype=float)
    assoc_f = np.ascontiguousarray(assoc_f, dtype=int)
    n_faces_b = faces_b.shape[0]
    incl = float(incl)
    q = float(q)
    rmax_f = float(rmax_f)
    rmin_f = float(rmin_f)
    weight = np.zeros(n_faces_b, dtype=float)
    ## Pure-NumPy replacement for the weave.inline/support_code block above
    ## (scipy.weave is no longer available). Sky-plane projection is
    ## vectorized over all vertices; the three mutually-exclusive per-vertex
    ## cases (clearly outside / fully hidden / needs the nearest-vertex +
    ## point-in-triangle test) are handled with boolean masks, restricting
    ## the expensive nearest-vertex search and triangle test to only the
    ## subset of eclipsed-star vertices that actually need it. The "first
    ## matching associated face wins" early-exit behavior of the original
    ## C loop over up to 6 candidate faces is preserved via a `found` mask
    ## updated across a (fixed, <=6 iteration) loop. Weight contributions
    ## are applied via a scatter-add (np.add.at), since a face can be
    ## reached from more than one vertex.
    PI = 4*np.arctan(1.)
    phs_b = orbph*2*PI
    phs_f = (orbph+0.5)*2*PI

    offsety_f, offsetz_f = _to_skyplane(1/(1+q), 0., 0., incl, phs_f, 0., 0.)
    Vy, Vz = _to_skyplane(vertices_f[:,0]*r_vertices_f, vertices_f[:,1]*r_vertices_f, vertices_f[:,2]*r_vertices_f,
                           incl, phs_f, offsety_f, offsetz_f)

    offsety_b, offsetz_b = _to_skyplane(q/(1+q), 0., 0., incl, phs_b, 0., 0.)
    y, z = _to_skyplane(vertices_b[:,0]*r_vertices_b, vertices_b[:,1]*r_vertices_b, vertices_b[:,2]*r_vertices_b,
                         incl, phs_b, offsety_b, offsetz_b)

    dr2 = (y-offsety_f)**2 + (z-offsetz_f)**2
    mask_outside = dr2 > rmax_f**2
    mask_hidden = (~mask_outside) & (dr2 < rmin_f**2)
    mask_test = (~mask_outside) & (~mask_hidden)

    is_hidden = mask_hidden.copy()

    if mask_test.any():
        y_t, z_t = y[mask_test], z[mask_test]
        dist2 = (y_t[:,None]-Vy[None,:])**2 + (z_t[:,None]-Vz[None,:])**2
        nearest_id = np.argmin(dist2, axis=1)

        n_test = y_t.size
        found = np.zeros(n_test, dtype=bool)
        for k in range(6):
            active = ~found
            if not active.any():
                break
            active_idx = np.nonzero(active)[0]
            id_surface_f = assoc_f[nearest_id[active_idx], k]
            valid = id_surface_f >= 0
            valid_idx = active_idx[valid]
            if valid_idx.size == 0:
                continue
            sf = id_surface_f[valid]
            v0, v1, v2 = faces_f[sf,0], faces_f[sf,1], faces_f[sf,2]
            inside = _inside_triangle_vec(y_t[valid_idx], z_t[valid_idx], Vy[v0], Vz[v0], Vy[v1], Vz[v1], Vy[v2], Vz[v2])
            found[valid_idx[inside]] = True

        test_idx_global = np.nonzero(mask_test)[0]
        is_hidden[test_idx_global[found]] = True

    np.add.at(weight, assoc_b[is_hidden, :5].ravel(), 1.)
    assoc_b6 = assoc_b[is_hidden, 5]
    valid6 = assoc_b6 != -99
    np.add.at(weight, assoc_b6[valid6], 1.)

    weight = 1 - weight/3.
    return weight

def Inside_triangle(p, a, b, c):
    """ inside_triangle(p, a, b, c)
    p: point (x,y)
    a, b, c: vertices of the triangle (x,y)

    >>> inside_triangle(p, a, b, c)
    """
    detT = (a[0]-c[0])*(b[1]-c[1]) - (a[1]-c[1])*(b[0]-c[0])
    lambda1 = ((b[1]-c[1])*(p[0]-c[0]) - (b[0]-c[0])*(p[1]-c[1])) / detT
    lambda2 = (-(a[1]-c[1])*(p[0]-c[0]) + (a[0]-c[0])*(p[1]-c[1])) / detT
    lambda3 = 1 - lambda1 - lambda2
    return (0 < lambda1 < 1) and (0 < lambda2 < 1) and (0 < lambda3 < 1)

def Occultation_approx(vertices, r_vertices, assoc, n_faces, incl, orbph, q, ntheta, radii):
    """Occultation_approx(vertices, r_vertices, assoc, n_faces, incl, orbph, q, ntheta, radii)

    Hidden surface removal algorithm.
    Returns the weight of each face/surface element with
    0, 1, 2, 3; going from not covered to fully covered.
    """
    def _to_skyplane(x, y, z, incl, orbph, offsety, offsetz):
        cos_incl, sin_incl = np.cos(incl), np.sin(incl)
        cos_phs, sin_phs = np.cos(orbph), np.sin(orbph)
        xnew = x*cos_phs + y*sin_phs
        ynew = -x*sin_phs + y*cos_phs + offsety
        znew = z*sin_incl + xnew*cos_incl + offsetz
        return ynew, znew

    vertices = np.ascontiguousarray(vertices, dtype=float)
    assoc = np.ascontiguousarray(assoc, dtype=int)
    r_vertices = np.ascontiguousarray(r_vertices, dtype=float)
    radii = np.ascontiguousarray(radii, dtype=float)
    incl = float(incl)
    orbph = float(orbph)
    q = float(q)
    ntheta = int(ntheta)
    weight = np.zeros(n_faces, dtype=float)
    ## Pure-NumPy replacement for the weave.inline/support_code block above
    ## (scipy.weave is no longer available). Vectorized sky-plane
    ## projection and occultation test over all vertices at once, followed
    ## by a scatter-add (np.add.at, needed since a face can be reached via
    ## more than one vertex) of the weight contributions.
    tmp_y, tmp_z = _to_skyplane(-1./(1.+q), 0., 0., incl, orbph, 0., 0.)
    offsety, offsetz = _to_skyplane(q/(1.+q), 0., 0., incl, orbph, 0., 0.)
    offsety -= tmp_y
    offsetz -= tmp_z

    vx, vy, vz = vertices[:,0], vertices[:,1], vertices[:,2]
    y, z = _to_skyplane(vx*r_vertices, vy*r_vertices, vz*r_vertices, incl, orbph, offsety, offsetz)

    theta_over_ntheta = np.arctan2(z, y)/ntheta
    pos = theta_over_ntheta.astype(int)  # C's (int) cast truncates toward zero, like astype(int)
    w = theta_over_ntheta - pos
    r = radii[pos]*(1-w) + radii[pos+1]*w

    mask = (y**2+z**2) < r**2
    np.add.at(weight, assoc[mask, :5].ravel(), 1.)
    assoc6 = assoc[mask, 5]
    valid6 = assoc6 != -99
    np.add.at(weight, assoc6[valid6], 1.)

    return weight

def Occultation_shapely(vertices, faces_ind, incl, orbph, q, ntheta, radii):
    """Occultation_shapely(vertices, faces_ind, incl, orbph, q, ntheta, radii)

    Hidden surface removal algorithm.
    Returns the weight of each face/surface element (i.e.
    fractional area uncovered).

    vertices (array (3,n_vertices)): Array of vertices making the faces of the
        star located in front.
    faces_ind (array (n_faces,3)): Array providing the vertice indices of the
        faces of the star located in front.
    """
    # Making sure that shapely is installed
    if not _HAS_SHAPELY:
        print( "You must install the Shapely package to run this function." )
        return

    #print( "orbph: {}".format(orbph) )
    import time
    T = []
    T.append(time.time())

    # Defining the front star polygon
    theta = np.arange(ntheta, dtype=float)/ntheta * cts.TWOPI
    xoff, yoff = Observer_2Dprojection(1./(1+q), 0., 0., incl, orbph+0.5)
    x_front = radii * np.cos(theta) + xoff
    y_front = radii * np.sin(theta) + yoff
    star_front = shapely.geometry.Polygon(np.c_[x_front, y_front].copy())
    prepared_star_front = shapely.prepared.prep(star_front)
    T.append(time.time())
    #print( "T{}: {} ({})".format(len(T), T[-1]-T[0], T[-1]-T[-2]) )

    # Defining the faces of the back star
    x_back, y_back = Observer_2Dprojection(vertices[0], vertices[1], vertices[2], incl, orbph, xoffset=q/(1.+q))
    x_back = x_back[faces_ind]
    y_back = y_back[faces_ind]
    faces = np.array([shapely.geometry.Polygon(zip(*xy)) for xy in zip(x_back,y_back)])
    T.append(time.time())
    #print( "T{}: {} ({})".format(len(T), T[-1]-T[0], T[-1]-T[-2]) )

    # Calculating the indices of overlapping, partially hidden and fully hidden faces
    overlap = np.array([prepared_star_front.intersects(f) for f in faces])
    partial = overlap.copy()
    if overlap.any():
        hidden = np.array([prepared_star_front.contains(f) for f in faces[overlap]])
        if hidden.any():
            partial[overlap] = ~hidden
            hidden = overlap - partial
        else:
            hidden = np.zeros_like(overlap)
    T.append(time.time())
    #print( "T{}: {} ({})".format(len(T), T[-1]-T[0], T[-1]-T[-2]) )

    # Calculating the weights (fractional hidden area
    weights = np.ones_like(overlap, dtype=float)
    weights[overlap] = 0.
    if partial.any():
        partial_weight = 1 - np.array( [ star_front.intersection(face).area/face.area for face in faces[partial] ] )
        weights[partial] = partial_weight
    T.append(time.time())
    #print( "T{}: {} ({})".format(len(T), T[-1]-T[0], T[-1]-T[-2]) )

    # Calculating the total area in two different ways
    area_geo = np.array([face.area for face in faces])
    #area_ica = np.abs(star2.area * star2.cosx)

    # Printing useful information
    #print( "area_geo.sum() {}".format(area_geo.sum()) )
    #print( "area_ica.sum() {}".format(area_ica.sum()) )
    #print( "fraction eclipse {}".format((weights*area_geo).sum()/area_geo.sum()) )
    #print( "predicted fraction eclipse {}".format( 1 - (star1.Radius()/star2.Radius())**2 ) )
    T.append(time.time())
    #print( "T{}: {} ({})".format(len(T), T[-1]-T[0], T[-1]-T[-2]) )

    # Plotting
    #from Pgplot import *
    #nextplotpage()
    #plotxy(y_back.flat, x_back.flat, line=None, symbol=1, aspect=1, rangey=[-1.1,1.1], rangex=[-1.1,1.1])
    #x, y = star_front.exterior.xy
    #plotxy(y, x, color=2)
    #plotxy([-2.,2.],[0.,0.])
    #plotxy([0.,0.],[-2.,2.])

    return weights

def Observer_2Dprojection(x, y, z, incl, orbph, xoffset=None):
    """ Observer_2Dprojection(x, y, z, incl, orbph, xoffset=None)
    x, y, z: cartesian coordinates
    incl: orbital inclination (radians)
    orbph: orbital phase (0-1)
    xoffset (None): x offset, due to star not located at the
        origin of the coordinate system

    >>> new_y,new_z = Observer_2Dprojection(x, y, z, incl, orbph, xoffset=None)
    """
    orbph = orbph%1
    cos_incl = np.cos(incl)
    sin_incl = np.sin(incl)
    cos_phs = np.cos(orbph*cts.TWOPI)
    sin_phs = np.sin(orbph*cts.TWOPI)
    xnew = x*cos_phs + y*sin_phs
    ynew = -x*sin_phs + y*cos_phs
    znew = z*sin_incl + xnew*cos_incl
    # We want to allow for a shift so that we translate the coordinate system from
    # the star center to the barycenter.
    if xoffset is not None:
        yoff, zoff = Observer_2Dprojection(xoffset, 0., 0., incl, orbph)
        ynew += yoff
        znew += zoff
    return ynew, znew

def Observer_3Dprojection(x, y, z, incl, orbph, xoffset=None):
    """ Observer_3Dprojection(x, y, z, incl, orbph, xoffset=None)
    x, y, z: cartesian coordinates
    incl: orbital inclination (radians)
    orbph: orbital phase (0-1)
    xoffset (None): x offset, due to star not located at the
        origin of the coordinate system

    >>> new_x,new_y,new_z = Observer_3Dprojection(x, y, z, incl, orbph, xoffset=None)
    """
    orbph = orbph%1
    cos_incl = np.cos(incl)
    sin_incl = np.sin(incl)
    cos_phs = np.cos(orbph*cts.TWOPI)
    sin_phs = np.sin(orbph*cts.TWOPI)
    xnew = x*cos_phs + y*sin_phs
    ynew = -x*sin_phs + y*cos_phs
    znew = z
    z = znew*sin_incl + xnew*cos_incl
    x = -znew*cos_incl + xnew*sin_incl
    y = ynew
    # We want to allow for a shift so that we translate the coordinate system from
    # the star center to the barycenter.
    if xoffset is not None:
        xoff, yoff, zoff = Observer_3Dprojection(xoffset, 0., 0., incl, orbph)
        x += xoff
        y += yoff
        z += zoff
    return x, y, z

def Overlap(y1, z1, y2, z2):
    """ overlap(y1, z1, y2, z2)
    """
    y1min = y1.min()
    y1max = y1.max()
    y2min = y2.min()
    y2max = y2.max()
    r1 = (y1max-y1min)
    r2 = (y2max-y2min)
    ycenter1 = (y1min+y1max)*0.5
    ycenter2 = (y2min+y2max)*0.5
    z1min = z1.min()
    z1max = z1.max()
    z2min = z2.min()
    z2max = z2.max()
    zcenter1 = (z1min+z1max)*0.5
    zcenter2 = (z2min+z2max)*0.5
    # Determining which points of star 1 and 2 are potentially overlapping
    # We approximate that the stars are confined within circles
    inds1 = np.sqrt((y1 - ycenter2)**2 + (z1 - zcenter2)**2) < r2
    inds2 = np.sqrt((y2 - ycenter1)**2 + (z2 - zcenter1)**2) < r1
    return inds1, inds2

def System_2Dprojection(x1, y1, z1, x2, y2, z2, incl, orbph, q):
    """ system_2Dprojection(x1, y1, z1, x2, y2, z2, incl, orbph, q)
    x1,2, y1,2, z1,2: cartesian coordinates
    incl: orbital inclination (radians)
    orbph: orbital phase of primary (0-1)
    mass ratio: M2/M1, used to calculate the x offset

    >>> new_y1,new_z1,new_y2,new_z2 = system_2Dprojection(x1, y1, z1, x2, y2, z3, incl, orbph, q)
    """
    y1, z1 = Observer_2Dprojection(x1, y1, z1, incl, orbph, -q/(1+q))
    y2, z2 = Observer_2Dprojection(x2, y2, z2, incl, orbph+0.5, 1/(1+q))
    return y1, z1, y2, z2

def Weights_transit(inds_highres, weight_highres, n_lowres):
    """Weights_transit(inds_highres, weight_highres, n_lowres)

    """
    inds_highres = np.ascontiguousarray(inds_highres, dtype=int)
    weight_highres = np.ascontiguousarray(weight_highres, dtype=float)
    n_lowres = int(n_lowres)
    weight_lowres = np.zeros(n_lowres, dtype=float)
    ## Pure-NumPy replacement for the weave.inline block above (scipy.weave
    ## is no longer available): a scatter-add. np.add.at (not plain fancy
    ## indexing) is required since the same low-res index can be targeted
    ## by more than one high-res element and those contributions must
    ## accumulate rather than overwrite.
    np.add.at(weight_lowres, inds_highres, weight_highres)
    return weight_lowres

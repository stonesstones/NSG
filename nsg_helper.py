import torch
import numpy as np
import imageio
import json
from matplotlib import pyplot as plt
from models import NeRFEncoding, NeRFField

# Misc utils

def img2mse(x, y): return torch.mean(torch.square(x - y))

def mse2psnr(x): return -10*torch.log10(x)/torch.log10(torch.tensor(10.))

def to8b(x): return (255*torch.clip(x, 0, 1)).to(torch.uint8)


def latentReg(z, reg): return torch.sum([1/reg * torch.norm(latent_i) for latent_i in z])

def get_embedder(multires, i=0, input_dims=3):
    if i == -1:
        return torch.nn.Identity, input_dims

    encoder = NeRFEncoding(input_dims, multires, 0, multires-1, include_input=True)
    return encoder, encoder.get_out_dim()


def init_nerf_model(pos_enc, dir_enc, base_mlp_num_layers=8, base_mlp_layer_width=256, head_mlp_num_layers=4, head_mlp_layer_width=128, skips=(4,), use_viewdirs=False, trainable=True):

    model = NeRFField(position_encoding=pos_enc, 
                      direction_encoding=dir_enc, 
                      base_mlp_num_layers=base_mlp_num_layers,
                      base_mlp_layer_width=base_mlp_layer_width,
                      head_mlp_num_layers=head_mlp_num_layers,
                      head_mlp_layer_width=head_mlp_layer_width,
                      skip_connections=skips)
    return model


def init_latent_vector(latent_size, name=None):
    initializer = tf.random_normal_initializer(mean=0., stddev=0.01)

    return tf.Variable(initializer(shape=[latent_size], dtype=tf.float32),
                       trainable=True,
                       validate_shape=True,
                       name=name)


# Ray helpers
def get_rays(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    # Torch Version
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return torch.tensor(rays_o), torch.tensor(rays_d)


def get_rays_np(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    # Numpy Version
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """Normalized device coordinate rays.

    Space such that the canvas is a cube with sides [-1, 1] in each axis.

    Args:
      H: int. Height in pixels.
      W: int. Width in pixels.
      focal: float. Focal length of pinhole camera.
      near: float or array of shape[batch_size]. Near depth bound for the scene.
      rays_o: array of shape [batch_size, 3]. Camera origin.
      rays_d: array of shape [batch_size, 3]. Ray direction.

    Returns:
      rays_o: array of shape [batch_size, 3]. Camera origin in NDC.
      rays_d: array of shape [batch_size, 3]. Ray direction in NDC.
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1./(H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1./(W/(2.*focal)) * \
        (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
    d1 = -1./(H/(2.*focal)) * \
        (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = tf.stack([o0, o1, o2], -1)
    rays_d = tf.stack([d0, d1, d2], -1)

    return rays_o, rays_d


# Hierarchical sampling helper

def sample_pdf(bins, weights, N_samples, det=False):

    # Get pdf
    weights += 1e-5  # prevent nans
    pdf = weights / tf.reduce_sum(weights, -1, keepdims=True)
    cdf = tf.cumsum(pdf, -1)
    cdf = tf.concat([tf.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = tf.linspace(0., 1., N_samples)
        u = tf.broadcast_to(u, list(cdf.shape[:-1]) + [N_samples])
    else:
        u = tf.random.uniform(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    inds = tf.searchsorted(cdf, u, side='right')
    below = tf.maximum(0, inds-1)
    above = tf.minimum(cdf.shape[-1]-1, inds)
    inds_g = tf.stack([below, above], -1)
    cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = tf.where(denom < 1e-5, tf.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples


# Plane-Ray intersection helper
def plane_pts(rays, planes, id_planes, near, method='planes'):
    """ Ray-Plane intersection for given planes in the scene

    Args:
        rays: ray origin and directions
        planes: first plane position, plane normal and distance between planes
        id_planes: ids of used planes
        near: distance between camera pose and first intersecting plane
        method: Method used

    Returns:
        pts: [N_rays, N_samples+N_importance] - intersection points of rays and selected planes
        z_vals: position of the point along each ray respectively
    """
    # Extract ray and plane definitions
    rays_o, rays_d = rays
    N_rays = rays_o.shape[0]
    plane_bds, plane_normal, delta = planes

    # Get amount of all planes
    n_planes = torch.ceil(torch.linalg.vector_norm(plane_bds[:, -1] - plane_bds[:, 0]) / delta) + 1

    # Calculate how far the ray_origins lies apart from each plane
    d_ray_first_plane = torch.matmul(plane_bds[:, 0]-rays_o, plane_normal[:, None])
    d_ray_first_plane = torch.maximum(-d_ray_first_plane, -near)

    # Get the ids of the planes in front of each ray starting from near distance upto the far plane
    start_id = torch.ceil((d_ray_first_plane+near)/delta)
    plane_id = start_id + id_planes
    if method == 'planes':
        plane_id = torch.cat([plane_id[:, :-1], torch.repeat_interleave(n_planes, N_rays)[:, None]], dim=1)
    elif method == 'planes_plus':
        # Experimental setup, that got discarded due to lower or the same quality
        plane_id = torch.cat([plane_id[:, :1],
                              torch.repeat_interleave(id_planes[None, 1:-1], N_rays, dim=0),
                              torch.repeat_interleave(n_planes, N_rays)[:, None]], dim=1)

    # [N_samples, N_rays, xyz]
    z_planes = plane_normal[None, None, :] * torch.transpose(plane_id*delta, 0, 1)[..., None]
    relevant_plane_origins = plane_bds[:, 0][None, None, :]+z_planes

    # Distance between each ray's origin and associated planes
    d_plane_pose = relevant_plane_origins - rays_o[None, :, :]

    n = torch.matmul(d_plane_pose, plane_normal[..., None])
    z = torch.matmul(rays_d, plane_normal[..., None])

    z_vals = torch.transpose(torch.squeeze(n / z), 0, 1)

    pts = rays_o[..., None, :] + rays_d[..., None, :] *\
              z_vals[..., None]

    return pts, z_vals


def rotate_yaw(p, yaw):
    """Rotates p with yaw in the given coord frame with y being the relevant axis and pointing downwards

    Args:
        p: 3D points in a given frame [N_pts, N_frames, 3]/[N_pts, N_frames, N_samples, 3]
        yaw: Rotation angle

    Returns:
        p: Rotated points [N_pts, N_frames, N_samples, 3]
    """
    # p of size [batch_rays, n_obj, samples, xyz]
    if len(p.shape) < 4:
        p = p[..., None, :]

    c_y = torch.cos(yaw)[..., None]
    s_y = torch.sin(yaw)[..., None]

    p_x = c_y * p[..., 0] - s_y * p[..., 2]
    p_y = p[..., 1]
    p_z = s_y * p[..., 0] + c_y * p[..., 2]

    return torch.cat([p_x[..., None], p_y[..., None], p_z[..., None]], axis=-1)


def scale_frames(p, sc_factor, inverse=False):
    """Scales points given in N_frames in each dimension [xyz] for each frame or rescales for inverse==True

    Args:
        p: Points given in N_frames frames [N_points, N_frames, N_samples, 3]
        sc_factor: Scaling factor for new frame [N_points, N_frames, 3]
        inverse: Inverse scaling if true, bool

    Returns:
        p_scaled: Points given in N_frames rescaled frames [N_points, N_frames, N_samples, 3]
    """
    device = p.device
    # Take 150% of bbox to include shadows etc.
    dim = torch.tensor([1., 1., 1.], dtype=torch.float32).to(device) * sc_factor
    # dim = tf.constant([0.1, 0.1, 0.1]) * sc_factor

    half_dim = dim / 2
    scaling_factor = (1 / (half_dim + 1e-9))[:, :, None, :]

    if not inverse:
        p_scaled = scaling_factor * p
    else:
        p_scaled = (1/scaling_factor) * p

    return p_scaled


def world2object(pts, dirs, pose, theta_y, dim=None, inverse=False):
    """Transform points given in world frame into N_obj object frames

    Object frames are scaled to [[-1.,1], [-1.,1], [-1.,1]] inside the 3D bounding box given by dim

    Args:
        pts: N_pts times 3D points given in world frame, [N_pts, 3]
        dirs: Corresponding 3D directions given in world frame, [N_pts, 3]
        pose: object position given in world frame, [N_pts, N_obj, 3]/if inverse: [N_pts, 3]
        theta_y: Yaw of objects around world y axis, [N_pts, N_obj]/if inverse: [N_pts]
        dim: Object bounding box dimensions, [N_pts, N_obj, 3]/if inverse: [N_pts, 3] 幅、高さ、奥行き
        inverse: if true pts and dirs should be given in object frame and are transofmed back into world frame, bool
            For inverse: pts, [N_pts, N_obj, 3]; dirs, [N_pts, N_obj, 3]

    Returns:
        pts_w: 3d points transformed into object frame (world frame for inverse task)
        dir_w: unit - 3d directions transformed into object frame (world frame for inverse task)
    """
    device = pts.device
    #  Prepare args if just one sample per ray-object or world frame only
    if len(pts.shape) == 3: #TODO あとで
        # [batch_rays, n_obj, samples, xyz]
        n_sample_per_ray = pts.shape[1]

        pose = torch.repeat_interleave(pose, n_sample_per_ray, dim=0)
        theta_y = torch.repeat_interleave(theta_y, n_sample_per_ray, dim=0)
        if dim is not None:
            dim = torch.repeat_interleave(dim, n_sample_per_ray, dim=0)
        if len(dirs.shape) == 2:
            dirs = torch.repeat_interleave(dirs, n_sample_per_ray, dim=0)

        pts = torch.reshape(pts, [-1, 3])

    # Shift the object reference point to the middle of the bbox (vkitti2 specific)
    y_shift = (torch.tensor([0., -1., 0.], dtype=torch.float32)[None, :].to(device) if inverse else
               torch.tensor([0., -1., 0.], dtype=torch.float32)[None, None, :]).to(device) * \
              (dim[..., 1] / 2)[..., None]
    pose_w = pose + y_shift

    # Describes the origin of the world system w in the object system o
    t_w_o = rotate_yaw(-pose_w, theta_y)

    if not inverse:
        N_obj = theta_y.shape[1]
        pts_w = torch.repeat_interleave(pts[:, None, ...], N_obj, dim=1)
        dirs_w = torch.repeat_interleave(dirs[:, None, ...], N_obj, dim=1)

        # Rotate coordinate axis
        # TODO: Generalize for 3d roaations
        pts_o = rotate_yaw(pts_w, theta_y) + t_w_o
        dirs_o = rotate_yaw(dirs_w, theta_y)

        # Scale rays_o_v and rays_d_v for box [[-1.,1], [-1.,1], [-1.,1]]
        if dim is not None:
            pts_o = scale_frames(pts_o, dim)
            dirs_o = scale_frames(dirs_o, dim)

        # Normalize direction
        dirs_o = dirs_o / torch.linalg.vector_norm(dirs_o, dim=3)[..., None, :]
        return [pts_o, dirs_o]

    else:
        pts_o = pts[None, :, None, :]
        dirs_o = dirs
        if dim is not None:
            pts_o = scale_frames(pts_o, dim[None, ...], inverse=True)
            if dirs is not None:
                dirs_o = scale_frames(dirs_o, dim, inverse=True)

        pts_o = pts_o - t_w_o
        pts_w = rotate_yaw(pts_o, -theta_y)[0, :]

        if dirs is not None:
            dirs_w = rotate_yaw(dirs_o, -theta_y)
            # Normalize direction TODO: Check if necessary
            dirs_w = dirs_w / torch.linalg.vector_norm(dirs_w, dim=-1)[..., None, :]
        else:
            dirs_w = None

        return [pts_w, dirs_w]


def object2world(pts, dirs, pose, theta_y, dim=None, inverse=True):
    """Transform points given in world frame into N_obj object frames

    Object frames are scaled to [[-1.,1], [-1.,1], [-1.,1]] inside the 3D bounding box given by dim

    Args:
        pts: N_pts times 3D points given in N_obj object frames, [N_pts, N_obj, 3]
        dirs: Corresponding 3D directions given in N_obj object frames, [N_pts, N_obj, 3]
        pose: object position given in world frame, [N_pts, N_obj, 3]/if inverse: [N_pts, 3]
        theta_y: Yaw of objects around world y axis, [N_pts, N_obj]/if inverse: [N_pts]
        dim: Object bounding box dimensions, [N_pts, N_obj, 3]/if inverse: [N_pts, 3]

    Returns:
        pts_w: 3d points transformed into world frame
        dir_w: unit - 3d directions transformed into world frame
    """
    device = pts.device
    #  Prepare args if just one sample per ray-object
    if len(pts.shape) == 3:
        # [N_rays, N_obj, N_obj_samples, xyz]
        n_sample_per_ray = pts.shape[1]

        pose = torch.repeat_interleave(pose, n_sample_per_ray, dim=0)
        theta_y = torch.repeat_interleave(theta_y, n_sample_per_ray, dim=0)
        if dim is not None:
            dim = torch.repeat_interleave(dim, n_sample_per_ray, dim=0)
        if len(dirs.shape) == 2:
            dirs = torch.repeat_interleave(dirs, n_sample_per_ray, dim=0)

        pts = torch.reshape(pts, [-1, 3])

    # Shift the object reference point to the middle of the bbox (vkitti2 specific)
    y_shift = torch.tensor([0., -1., 0.], dtype=torch.float32)[None, :].to(device) * (dim[..., 1] / 2)[..., None]
    pose_w = pose + y_shift

    # Describes the origin of the world system w in the object system o
    t_w_o = rotate_yaw(-pose_w, theta_y)

    pts_o = pts[None, :, None, :]
    dirs_o = dirs
    if dim is not None:
        pts_o = scale_frames(pts_o, dim[None, ...], inverse=True)
        if dirs is not None:
            dirs_o = scale_frames(dirs_o, dim, inverse=True)

    pts_o = pts_o - t_w_o
    pts_w = rotate_yaw(pts_o, -theta_y)[0, :]

    if dirs is not None:
        dirs_w = rotate_yaw(dirs_o, -theta_y)
        # Normalize direction
        dirs_w = dirs_w / torch.linalg.vector_norm(dirs_w, dim=-1)[..., None, :]
    else:
        dirs_w = None

    return [pts_w, dirs_w]


def ray_box_intersection(ray_o, ray_d, aabb_min=None, aabb_max=None):
    """Returns 1-D intersection point along each ray if a ray-box intersection is detected

    If box frames are scaled to vertices between [-1., -1., -1.] and [1., 1., 1.] aabbb is not necessary

    Args:
        ray_o: Origin of the ray in each box frame, [rays, boxes, 3]
        ray_d: Unit direction of each ray in each box frame, [rays, boxes, 3]
        (aabb_min): Vertex of a 3D bounding box, [-1., -1., -1.] if not specified
        (aabb_max): Vertex of a 3D bounding box, [1., 1., 1.] if not specified

    Returns:
        z_ray_in:
        z_ray_out:
        intersection_map: Maps intersection values in z to their ray-box intersection
    """
    # Source: https://medium.com/@bromanz/another-view-on-the-classic-ray-aabb-intersection-algorithm-for-bvh-traversal-41125138b525
    # https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
    if aabb_min is None:
        aabb_min = torch.ones_like(ray_o) * -1. # tf.constant([-1., -1., -1.])
    if aabb_max is None:
        aabb_max = torch.ones_like(ray_o) # tf.constant([1., 1., 1.])

    inv_d = torch.reciprocal(ray_d)

    t_min = (aabb_min - ray_o) * inv_d
    t_max = (aabb_max - ray_o) * inv_d

    t0 = torch.minimum(t_min, t_max)
    t1 = torch.maximum(t_min, t_max)

    t_near = torch.maximum(torch.maximum(t0[..., 0], t0[..., 1]), t0[..., 2])
    t_far = torch.minimum(torch.minimum(t1[..., 0], t1[..., 1]), t1[..., 2])

    # Check if rays are inside boxes TODO　あってるかわからない
    intersection_map = torch.where(t_far > t_near)
    # Check that boxes are in front of the ray origin TODO　あってるかわからない
    positive_far = torch.where(t_far[intersection_map] > 0)
    intersection_map = (intersection_map[0][positive_far[0]], intersection_map[1][positive_far[0]])

    if not intersection_map[0].shape[0] == 0:
        z_ray_in = t_near[intersection_map]
        z_ray_out = t_far[intersection_map]
    else:
        return None, None, None

    return z_ray_in, z_ray_out, intersection_map


def box_pts(rays, pose, theta_y, dim=None, one_intersec_per_ray=False):
    """gets ray-box intersection points in world and object frames in a sparse notation

    Args:
        rays: ray origins and directions, [[N_rays, 3], [N_rays, 3]]
        pose: object positions in world frame for each ray, [N_rays, N_obj, 3]
        theta_y: rotation of objects around world y axis, [N_rays, N_obj]
        dim: object bounding box dimensions [N_rays, N_obj, 3]
        one_intersec_per_ray: If True only the first interesection along a ray will lead to an
        intersection point output

    Returns:
        pts_box_w: box-ray intersection points given in the world frame
        viewdirs_box_w: view directions of each intersection point in the world frame
        pts_box_o: box-ray intersection points given in the respective object frame
        viewdirs_box_o: view directions of each intersection point in the respective object frame
        z_vals_w: integration step in the world frame
        z_vals_o: integration step for scaled rays in the object frame
        intersection_map: mapping of points, viewdirs and z_vals to the specific rays and objects at the intersection

    """
    rays_o, rays_d = rays
    # Transform each ray into each object frame
    rays_o_o, dirs_o = world2object(rays_o, rays_d, pose, theta_y, dim)
    rays_o_o = torch.squeeze(rays_o_o)
    dirs_o = torch.squeeze(dirs_o)

    # Get the intersection with each Bounding Box
    z_ray_in_o, z_ray_out_o, intersection_map = ray_box_intersection(rays_o_o, dirs_o)

    if z_ray_in_o is not None:
        # Calculate the intersection points for each box in each object frame
        pts_box_in_o = rays_o_o[intersection_map] + \
                       z_ray_in_o[:, None] * dirs_o[intersection_map]

        # Transform the intersection points for each box in world frame
        pts_box_in_w, _ = world2object(pts_box_in_o,
                                    None,
                                    pose[intersection_map],
                                    theta_y[intersection_map],
                                    dim[intersection_map],
                                    inverse=True)
        pts_box_in_w_new, _ = object2world(pts_box_in_o,
                                       None,
                                       pose[intersection_map],
                                       theta_y[intersection_map],
                                       dim[intersection_map],)
        pts_box_in_w = torch.squeeze(pts_box_in_w)

        # Get all intersecting rays in unit length and the corresponding z_vals TODO waht is unit?
        rays_o_in_w = torch.repeat_interleave(rays_o[:, None, :], pose.shape[1], dim=1)[intersection_map]
        rays_d_in_w = torch.repeat_interleave(rays_d[:, None, :], pose.shape[1], dim=1)[intersection_map]
        # Account for non-unit length rays direction
        z_vals_in_w = torch.linalg.vector_norm(pts_box_in_w - rays_o_in_w, dim=-1) / torch.linalg.vector_norm(rays_d_in_w, dim=-1)

        if one_intersec_per_ray:
            # Get just nearest object point on a single ray　TODO check below function
            z_vals_in_w, intersection_map, first_in_only = get_closest_intersections(z_vals_in_w,
                                                                                     intersection_map,
                                                                                     N_rays=rays_o.shape[0],
                                                                                     N_obj=theta_y.shape[1])
            # Get previous calculated values just for first intersections TODO check below function
            z_ray_in_o = z_ray_in_o[first_in_only]
            z_ray_out_o = z_ray_out_o[first_in_only]
            pts_box_in_o = pts_box_in_o[first_in_only]
            pts_box_in_w = pts_box_in_w[first_in_only]
            rays_o_in_w = rays_o_in_w[first_in_only]
            rays_d_in_w = rays_d_in_w[first_in_only]

        # Get the far intersection points and integration steps for each ray-box intersection in world and object frames
        pts_box_out_o = rays_o_o[intersection_map] + \
                        z_ray_out_o[:, None] * dirs_o[intersection_map]
        pts_box_out_w, _ = world2object(pts_box_out_o, #TODO why pts_box_out_o?
                                       None,
                                       pose[intersection_map],
                                       theta_y[intersection_map],
                                       dim[intersection_map],
                                       inverse=True)

        pts_box_out_w_new, _ = object2world(pts_box_out_o,
                                        None,
                                        pose[intersection_map],
                                        theta_y[intersection_map],
                                        dim[intersection_map],)
        pts_box_out_w = torch.squeeze(pts_box_out_w)
        z_vals_out_w = torch.linalg.vector_norm(pts_box_out_w - rays_o_in_w, dim=1) / torch.linalg.vector_norm(rays_d_in_w, dim=-1)

        # Get viewing directions for each ray-box intersection
        viewdirs_box_o = dirs_o[intersection_map]
        viewdirs_box_w = 1 / torch.linalg.vector_norm(rays_d_in_w, dim=1)[:, None] * rays_d_in_w

    else:
        # In case no ray intersects with any object return empty lists
        z_vals_in_w = z_vals_out_w = []
        pts_box_in_w = pts_box_in_o = []
        viewdirs_box_w = viewdirs_box_o = []
        z_ray_out_o = z_ray_in_o = []
    return pts_box_in_w, viewdirs_box_w, z_vals_in_w, z_vals_out_w, \
           pts_box_in_o, viewdirs_box_o, z_ray_in_o, z_ray_out_o, \
           intersection_map


def get_closest_intersections(z_vals_w, intersection_map, N_rays, N_obj):
    """Reduces intersections given by z_vals and intersection_map to the first intersection along each ray

    Args:
        z_vals_w: All integration steps for all ray-box intersections in world coordinates [n_intersections,]
        intersection_map: Mapping from flat array to ray-box intersection matrix [n_intersections, 2]
        N_rays: Total number of rays
        N_obj: Total number of objects

    Returns:
        z_vals_w: Integration step for the first ray-box intersection per ray in world coordinates [N_rays,]
        intersection_map: Mapping from flat array to ray-box intersection matrix [N_rays, 2]
        id_first_intersect: Mapping from all intersection related values to first intersection only [N_rays,1]

    """
    # Flat to dense indices
    # Create matching ray-object intersectin matrix with index for all z_vals
    id_z_vals = torch.zeros([N_rays, N_obj], dtype=torch.int32)
    id_z_vals[intersection_map] = torch.arange(z_vals_w.shape[0], dtype=torch.int32)
    # Create ray-index array
    id_ray = torch.arange(N_rays).to(torch.int64)

    # Flat to dense values
    # Scatter z_vals in world coordinates to ray-object intersection matrix
    z_scatterd = torch.zeros([N_rays, N_obj])
    z_scatterd[intersection_map] = z_vals_w
    # Set empty intersections to 1e10
    z_scatterd_nz = torch.where(z_scatterd == 0, torch.ones_like(z_scatterd) * 1e10, z_scatterd)

    # Get minimum values along each ray and corresponding ray-box intersection id
    id_min = torch.argmin(z_scatterd_nz, dim=1)
    id_reduced = (id_ray, id_min)
    z_vals_w_reduced = z_scatterd[id_reduced]

    # Remove all rays w/o intersections (min(z_vals_reduced) == 0)
    id_non_zeros = torch.where(z_vals_w_reduced != 0)
    if len(id_non_zeros) != N_rays:
        z_vals_w_reduced = z_vals_w_reduced[id_non_zeros]
        id_reduced = (id_reduced[0][id_non_zeros], id_reduced[1][id_non_zeros])

    # Get intersection map only for closest intersection to the ray origin
    intersection_map_reduced = id_reduced
    id_first_intersect = id_z_vals[id_reduced]

    return z_vals_w_reduced, intersection_map_reduced, id_first_intersect


def combine_z(z_vals_bckg, z_vals_obj_w, intersection_map, N_rays, N_samples, N_obj, N_samples_obj=1):
    """Combines and sorts background node and all object node intersections along a ray

    Args:
        z_vals_bckg: integration step along each ray [N_rays, N_samples]
        z_vals_obj_w:  integration step of ray-box intersection in the world frame [n_intersects, N_samples_obj
        intersection_map: mapping of points, viewdirs and z_vals to the specific rays and objects at ray-box intersection
        N_rays: Amount of rays
        N_samples: Amount of samples along each ray
        N_obj: Maximum number of objects
        N_samples_obj: Number of samples per object

    Returns:
        z_vals:  [N_rays, N_samples + N_samples_obj*N_obj, 4]
        id_z_vals_bckg:
        id_z_vals_obj:
    """
    device = z_vals_bckg.device
    if z_vals_obj_w is None:
        z_vals_obj_w_sparse = torch.zeros([N_rays, N_obj * N_samples_obj]).to(device)
    else:
        z_vals_obj_w_sparse = torch.zeros([N_rays, N_obj, N_samples_obj]).to(device)
        z_vals_obj_w_sparse[intersection_map] = z_vals_obj_w
        z_vals_obj_w_sparse = torch.reshape(z_vals_obj_w_sparse, [N_rays, N_samples_obj * N_obj])

    sample_range = torch.arange(0, N_rays)
    obj_range = torch.repeat_interleave(torch.repeat_interleave(sample_range[:, None, None], N_obj, dim=1), N_samples_obj, dim=2)

    # Get ids to assign z_vals to each model
    if z_vals_bckg is not None:
        if len(z_vals_bckg.shape) < 2:
            z_vals_bckg = z_vals_bckg[None]
        # Combine and sort z_vals along each ray
        z_vals = torch.sort(torch.cat([z_vals_obj_w_sparse, z_vals_bckg], dim=1), dim=1).values

        bckg_range = torch.repeat_interleave(sample_range[:, None, None], N_samples, dim=1)
        id_z_vals_bckg = (bckg_range[...,0], torch.searchsorted(z_vals, z_vals_bckg.contiguous()))
    else:
        z_vals = torch.sort(z_vals_obj_w_sparse, axis=1)
        id_z_vals_bckg = None

    # id_z_vals_obj = tf.concat([obj_range, tf.searchsorted(z_vals, z_vals_obj_w_sparse)], axis=2)
    id_z_vals_obj = (obj_range, torch.reshape(torch.searchsorted(z_vals, z_vals_obj_w_sparse.contiguous()), [N_rays, N_obj, N_samples_obj]))

    return z_vals, id_z_vals_bckg, id_z_vals_obj


# def render_mot_scene(pts, viewdirs, network_fn, network_query_fn,
#                      inputs, viewdirs_obj, z_vals_in_o, n_intersect, object_idx, object_y, obj_pose,
#                      unique_classes, class_id, latent_vector_dict, object_network_fn_dict,
#                      N_rays,N_samples, N_obj, N_samples_obj,
#                      obj_only=False):
#
#     # Prepare raw output array
#     raw = tf.zeros([N_rays, N_samples + N_obj * N_samples_obj, 4]) if not obj_only else tf.zeros([N_rays, N_obj * N_samples_obj, 4])
#     raw_sh = raw.shape
#
#     if not obj_only:
#         # Predict RGB and density from background
#         raw_bckg = network_query_fn(pts, viewdirs, network_fn)
#         raw += tf.scatter_nd(id_z_vals_bckg, raw_bckg, raw_sh)
#
#     # Check for object intersections
#     if z_vals_in_o is not None:
#         # Loop for one model per object and no latent representations
#         if latent_vector_dict is None:
#             obj_id = tf.reshape(object_idx, obj_pose[..., 4].shape)
#             for k, track_id in enumerate(object_y):
#                 if track_id >= 0:
#                     input_indices = tf.where(tf.equal(obj_id, k))
#                     input_indices = tf.reshape(input_indices, [-1, N_samples_obj, 2])
#                     model_name = 'model_obj_' + str(np.array(track_id).astype(np.int32))
#                     # print('Hit', model_name, n_intersect, 'times.')
#                     if model_name in object_network_fn_dict:
#                         obj_network_fn = object_network_fn_dict[model_name]
#
#                         inputs_obj_k = tf.gather_nd(inputs, input_indices)
#                         viewdirs_obj_k = tf.gather_nd(viewdirs_obj,
#                                                       input_indices[..., None, 0]) if N_samples_obj == 1 else \
#                             tf.gather_nd(viewdirs_obj, input_indices[..., None, 0, 0])
#
#                         # Predict RGB and density from object model
#                         raw_k = network_query_fn(inputs_obj_k, viewdirs_obj_k, obj_network_fn)
#
#                         if n_intersect is not None:
#                             # Arrange RGB and denisty from object models along the respective rays
#                             raw_k = tf.scatter_nd(input_indices[:, :], raw_k, [n_intersect, N_samples_obj,
#                                                                                4])  # Project the network outputs to the corresponding ray
#                             raw_k = tf.scatter_nd(intersection_map[:, :2], raw_k, [N_rays, N_obj, N_samples_obj,
#                                                                                    4])  # Project to rays and object intersection order
#                             raw_k = tf.scatter_nd(id_z_vals_obj, raw_k, raw_sh)  # Reorder along z and ray
#                         else:
#                             raw_k = tf.scatter_nd(input_indices[:, 0][..., None], raw_k, [N_rays, N_samples, 4])
#
#                         # Add RGB and density from object model to the background and other object predictions
#                         raw += raw_k
#         # Loop over classes c and evaluate each models f_c for all latent object describtor
#         else:
#             for c, class_type in enumerate(unique_classes.y):
#                 # Ignore background class
#                 if class_type >= 0:
#                     input_indices = tf.where(tf.equal(class_id, c))
#                     input_indices = tf.reshape(input_indices, [-1, N_samples_obj, 2])
#                     model_name = 'model_class_' + str(int(np.array(class_type))).zfill(5)
#
#                     if model_name in object_network_fn_dict:
#                         obj_network_fn = object_network_fn_dict[model_name]
#
#                         inputs_obj_c = tf.gather_nd(inputs, input_indices)
#
#                         # Legacy version 2
#                         # latent_vector = tf.concat([
#                         #         latent_vector_dict['latent_vector_' + str(int(obj_id)).zfill(5)][None, :]
#                         #         for obj_id in np.array(tf.gather_nd(obj_pose[..., 4], input_indices)).astype(np.int32).flatten()],
#                         #         axis=0)
#                         # latent_vector = tf.reshape(latent_vector, [inputs_obj_k.shape[0], inputs_obj_k.shape[1], -1])
#                         # inputs_obj_k = tf.concat([inputs_obj_k, latent_vector], axis=-1)
#
#                         # viewdirs_obj_k = tf.gather_nd(viewdirs_obj,
#                         #                               input_indices[..., 0]) if N_samples_obj == 1 else \
#                         #     tf.gather_nd(viewdirs_obj, input_indices)
#
#                         viewdirs_obj_c = tf.gather_nd(viewdirs_obj, input_indices[..., None, 0])[:, 0, :]
#
#                         # Predict RGB and density from object model
#                         raw_k = network_query_fn(inputs_obj_c, viewdirs_obj_c, obj_network_fn)
#
#                         if n_intersect is not None:
#                             # Arrange RGB and denisty from object models along the respective rays
#                             raw_k = tf.scatter_nd(input_indices[:, :], raw_k, [n_intersect, N_samples_obj,
#                                                                                4])  # Project the network outputs to the corresponding ray
#                             raw_k = tf.scatter_nd(intersection_map[:, :2], raw_k, [N_rays, N_obj, N_samples_obj,
#                                                                                    4])  # Project to rays and object intersection order
#                             raw_k = tf.scatter_nd(id_z_vals_obj, raw_k,
#                                                   raw_sh)  # Reorder along z in  positive ray direction
#                         else:
#                             raw_k = tf.scatter_nd(input_indices[:, 0][..., None], raw_k,
#                                                   [N_rays, N_samples, 4])
#
#                         # Add RGB and density from object model to the background and other object predictions
#                         raw += raw_k
#                     else:
#                         print('No model ', model_name, ' found')
#
#     return raw


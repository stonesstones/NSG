import os
import random
import time
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from nsg_helper import *
# from data_loader.load_vkitti import load_vkitti_data
from data_loader.load_kitti import load_kitti_data, plot_kitti_poses, tracking2txt
from nsg_prepare_input_helper import *
from nsg_manipulation import *
from nsg_log import Logger
import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2
import torch
from collections import OrderedDict
from models import MyRaySamples, Frustums
from nerfstudio.field_components.field_heads import FieldHeadNames
def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs):
        fields_outputs = [fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)]
        fields_outputs = [torch.cat([fields_output[FieldHeadNames.RGB], fields_output[FieldHeadNames.DENSITY]], dim=-1) for fields_output in fields_outputs]
        return torch.cat(fields_outputs, 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, embedobj_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'."""
    inputs_flat = torch.reshape(inputs[..., :3], [-1, 3])
    
    # embedded = embed_fn(inputs_flat)
    if inputs.shape[-1] > 3:
        if inputs.shape[-1] == 4:
            # NeRF + T w/o embedding
            time_st = torch.reshape(inputs[..., 3], [inputs_flat.shape[0], -1])
            embedded = torch.cat([embedded, time_st], -1)
        else:
            # NeRF + Latent Code
            inputs_latent = torch.reshape(inputs[..., 3:], [inputs_flat.shape[0], -1])
            embedded = torch.cat([embedded, inputs_latent], -1)

    if viewdirs is not None:
        input_dirs = torch.broadcast_to(viewdirs[:, None, :3], inputs[..., :3].shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        # embedded_dirs = embeddirs_fn(input_dirs_flat)
        # embedded = torch.cat([embedded, embedded_dirs], -1)
        

        if viewdirs.shape[-1] > 3:
            # Use global locations of objects
            input_obj_pose = torch.broadcast_to(viewdirs[:, None, 3:],
                                             size=(inputs[..., :3].shape[0], inputs[..., :3].shape[1], 3))
            input_obj_pose_flat = torch.reshape(input_obj_pose, [-1, input_obj_pose.shape[-1]]) # [N_rays * N_samples, 3]
            # embedded_obj = embedobj_fn(input_obj_pose_flat)
            # embedded = torch.cat([embedded, embedded_obj], -1)
            input_dirs_flat = torch.cat([input_dirs_flat, input_obj_pose_flat], -1)
    frustums = Frustums(positions=inputs_flat, directions=input_dirs_flat)
    sample_rays = MyRaySamples(frustums=frustums)
    outputs_flat = batchify(fn, netchunk)(sample_rays)
    outputs = torch.reshape(outputs_flat, list(
        inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                N_samples_obj,
                retraw=False,
                perturb=1.,
                N_importance=0,
                network_fine=None,
                object_network_fn_dict=None,
                latent_vector_dict=None,
                N_obj=None,
                obj_only=False,
                obj_transparency=True,
                white_bkgd=False,
                raw_noise_std=0.,
                sampling_method=None,
                use_time=False,
                plane_bds=None,
                plane_normal=None,
                delta=0.,
                id_planes=0,
                verbose=False,
                obj_location=True,
                device=None,
                train=True):
    """Volumetric rendering.

    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      object_network_fn_dict: dictinoary of functions. Model for predicting RGB and density at each point in
        object frames
      latent_vector_dict: Dictionary of latent codes
      N_obj: Maximumn amount of objects per ray
      obj_only: bool. If True, only run models from object_network_fn_dict
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      sampling_mehtod: string. Select how points are sampled in space
      plane_bds: array of shape [2, 3]. If sampling method planes, descirbing the first and last plane in space.
      plane_normal: array of shape [3]. Normal of all planes
      delta: float. Distance between adjacent planes.
      id_planes: array of shape [N_samples]. Preselected planes for sampling method planes and a given sampling distribution
      verbose: bool. If True, print more debugging info.

    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """

    def raw2outputs(raw, z_vals, rays_d):
        """Transforms model's predictions to semantically meaningful values.

        Args:
          raw: [num_rays, num_samples along ray, 4]. Prediction from model.
          z_vals: [num_rays, num_samples along ray]. Integration time.
          rays_d: [num_rays, 3]. Direction of each ray.

        Returns:
          rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
          disp_map: [num_rays]. Disparity map. Inverse of depth map.
          acc_map: [num_rays]. Sum of weights along each ray.
          weights: [num_rays, num_samples]. Weights assigned to each sampled color.
          depth_map: [num_rays]. Estimated distance to object.
        """
        # Function for computing density from model prediction. This value is
        # strictly between [0, 1].
        def raw2alpha(raw, dists, act_fn=torch.nn.ReLU()): 
            return 1.0 - torch.exp(-act_fn(raw) * dists)

        # Compute 'distance' (in time) between each integration time along a ray.
        dists = z_vals[..., 1:] - z_vals[..., :-1]

        # The 'distance' from the last integration time is infinity.
        dists = torch.cat(
            [dists, torch.broadcast_to(torch.tensor(1e10).to(device), dists[..., :1].shape)],
            dim=-1)  # [N_rays, N_samples]

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        dists = dists * torch.linalg.vector_norm(rays_d[..., None, :], dim=-1)

        # Extract RGB of each sample position along each ray.
        rgb = raw[..., :3]  # [N_rays, N_samples, 3]

        # Add noise to model's predictions for density. Can be used to 
        # regularize network during training (prevents floater artifacts).
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn(raw[..., 3].shape).to(device) * raw_noise_std

        # Predict density of each sample along each ray. Higher values imply
        # higher likelihood of being absorbed at this point.
        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]

        # Compute weight for RGB of each sample along each ray.  A cumprod() is
        # used to express the idea of the ray not having reflected up to this
        # sample yet.
        # [N_rays, N_samples]
        
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(device), 1.-alpha + 1e-10], -1), -1)[:, :-1]

        # Computed weighted color of each sample along each ray.
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
        # Estimated depth map is expected distance.
        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map /  torch.max(torch.tensor(1e-10), torch.sum(weights, -1)))
        acc_map = torch.sum(weights, -1)
        # Disparity map is inverse depth.
        disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map /  torch.max(torch.tensor(1e-10), torch.sum(weights, -1)))

        # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
        acc_map = torch.sum(weights, -1)

        # To composite onto a white background, use the accumulated alpha map.
        if white_bkgd:
            rgb_map = rgb_map + (1.-acc_map[..., None])

        return rgb_map, disp_map, acc_map, weights, depth_map


    def sample_along_ray(near, far, N_samples, N_rays, sampling_method, perturb):
        # Sample along each ray given one of the sampling methods. Under the logic, all rays will be sampled at
        # the same times.
        t_vals = tf.linspace(0., 1., N_samples)
        if sampling_method == 'squareddist':
            z_vals = near * (1. - np.square(t_vals)) + far * (np.square(t_vals))
        elif sampling_method == 'lindisp':
            # Sample linearly in inverse depth (disparity).
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))
        else:
            # Space integration times linearly between 'near' and 'far'. Same
            # integration points will be used for all rays.
            z_vals = near * (1.-t_vals) + far * (t_vals)
            if sampling_method == 'discrete':
                perturb = 0

        # Perturb sampling time along each ray. (vanilla NeRF option)
        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = tf.concat([mids, z_vals[..., -1:]], -1)
            lower = tf.concat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = tf.random.uniform(z_vals.shape)
            z_vals = lower + (upper - lower) * t_rand

        return tf.broadcast_to(z_vals, [N_rays, N_samples]), perturb


    ###############################
    # batch size
    N_rays = int(ray_batch.shape[0])

    # Extract ray origin, direction.
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each

    # Extract unit-normalized viewing direction.
    viewdirs = ray_batch[:, 8:11] if ray_batch.shape[-1] > 8 else None

    # Extract lower, upper bound for ray distance.
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    if use_time:
        time_stamp = ray_batch[:, 11][:, None]

    # Extract object position, dimension and label
    if N_obj:
        obj_pose = ray_batch[:, 11:]
        # [N_rays, N_obj, 8] with 3D position, y rot angle, track_id, (3D dimension - length, height, width)
        obj_pose = torch.reshape(obj_pose, [N_rays, N_obj, obj_pose.shape[-1] // N_obj])
        if N_importance > 0:
            obj_pose_fine = torch.repeat_interleave(obj_pose[:, None, ...], N_importance + N_samples, dim=1)
    else:
        obj_pose = obj_pose_fine = None

    if not obj_only:
        # For training object models only sampling close to the objects is performed
        if (sampling_method == 'planes' or sampling_method == 'planes_plus') and plane_bds is not None:
            # Sample at ray plane intersection (Neural Scene Graphs)
            pts, z_vals = plane_pts([rays_o, rays_d], [plane_bds, plane_normal, delta], id_planes, near,
                                    method=sampling_method)
            N_importance = 0
        else:
            # Sample along ray (vanilla NeRF)
            z_vals, perturb = sample_along_ray(near, far, N_samples, N_rays, sampling_method, perturb)

            pts = rays_o[..., None, :] + rays_d[..., None, :] * \
                z_vals[..., :, None]  # [N_rays, N_samples, 3]

    ####### DEBUG Sampling Points
    # print('TURN OFF IF NOT DEBUGING!')
    # axes_ls = plt.figure(1).axes
    # for i in range(rays_o.shape[0]):
    #     plt.arrow(np.array(rays_o)[i, 0], np.array(rays_o)[i, 2],
    #               np.array(30 * rays_d)[i, 0],
    #               np.array(30 * rays_d)[i, 2], color='red')
    #
    # plt.sca(axes_ls[1])
    # for i in range(rays_o.shape[0]):
    #     plt.arrow(np.array(rays_o)[i, 2], np.array(rays_o)[i, 1],
    #               np.array(30 * rays_d)[i, 2],
    #               np.array(30 * rays_d)[i, 1], color='red')
    #
    # plt.sca(axes_ls[2])
    # for i in range(rays_o.shape[0]):
    #     plt.arrow(np.array(rays_o)[i, 0], np.array(rays_o)[i, 1],
    #               np.array(30 * rays_d)[i, 0],
    #               np.array(30 * rays_d)[i, 1], color='red')
    ####### DEBUG Sampling Points

    # Choose input options
    if not N_obj:
        # No objects
        if use_time:
            # Time parameter input
            time_stamp_fine = torch.repeat_interleave(time_stamp[:, None], N_importance + N_samples,
                                        dim=1) if N_importance > 0 else None
            time_stamp = torch.repeat_interleave(time_stamp[:, None], N_samples, dim=1)
            pts = torch.cat([pts, time_stamp], dim=-1)
            raw = network_query_fn(pts, viewdirs, network_fn)
        else:
            raw = network_query_fn(pts, viewdirs, network_fn)
    else:
        n_intersect = None
        if not obj_pose.shape[-1] > 5:
            # If no object dimension is given all points in the scene given in object coordinates will be used as an input to each object model
            pts_obj, viewdirs_obj = world2object(pts, viewdirs, obj_pose[..., :3], obj_pose[..., 3],
                                                 dim=obj_pose[..., 5:8] if obj_pose.shape[-1] > 5 else None)

            pts_obj = torch.transpose(torch.reshape(pts_obj, [N_rays, N_samples, N_obj, 3]), [0, 2, 1, 3])

            inputs = torch.cat([pts_obj, torch.repeat(obj_pose[..., None, :3], N_samples, axis=2)], axis=3)
        else:
            # If 3D bounding boxes are given get intersecting rays and intersection points in scaled object frames
            pts_box_w, viewdirs_box_w, z_vals_in_w, z_vals_out_w,\
            pts_box_o, viewdirs_box_o, z_vals_in_o, z_vals_out_o, \
            intersection_map = box_pts(
                [rays_o, rays_d], obj_pose[..., :3], obj_pose[..., 3], dim=obj_pose[..., 5:8],
                one_intersec_per_ray=not obj_transparency)

            if z_vals_in_o is None or len(z_vals_in_o) == 0:
                if obj_only:
                    # No computation necesary if rays are not intersecting with any objects and no background is selected
                    raw = torch.zeros([N_rays, 1, 4])
                    z_vals = torch.zeros([N_rays, 1])

                    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
                        raw, z_vals, rays_d)

                    rgb_map = torch.ones([N_rays, 3])
                    disp_map = torch.ones([N_rays])*1e10
                    acc_map = torch.zeros([N_rays])
                    depth_map = torch.zeros([N_rays])

                    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
                    if retraw:
                        ret['raw'] = raw
                    return ret
                else:
                    # TODO: Do not return anything for no intersections.
                    z_vals_obj_w = torch.zeros(1)
                    intersection_map = (torch.zeros(1).to(dtype=torch.int32), torch.zeros(1).to(dtype=torch.int32))

            else:
                n_intersect = z_vals_in_o.shape[0]

                obj_pose = obj_pose[intersection_map]
                obj_pose = torch.repeat_interleave(obj_pose[:, None, :], N_samples_obj, dim=1)
                # Get additional model inputs for intersecting rays
                if N_samples_obj > 1:
                    z_vals_box_o = torch.repeat_interleave(torch.linspace(0., 1., N_samples_obj)[None, :].to(device), n_intersect, dim=0) * \
                                   (z_vals_out_o - z_vals_in_o)[:, None]
                else:
                    z_vals_box_o = torch.repeat_interleave(torch.tensor(1/2)[None,None].to(device), n_intersect, dim=0) * \
                                   (z_vals_out_o - z_vals_in_o)[:, None]

                pts_box_samples_o = pts_box_o[:, None, :] + viewdirs_box_o[:, None, :] \
                                        * z_vals_box_o[..., None]
                # pts_box_samples_o = pts_box_samples_o[:, None, ...]
                # pts_box_samples_o = tf.reshape(pts_box_samples_o, [-1, 3])

                obj_pose_transform = torch.reshape(obj_pose, [-1, obj_pose.shape[-1]])

                pts_box_samples_w, _ = world2object(torch.reshape(pts_box_samples_o, [-1, 3]), None,
                                                    obj_pose_transform[..., :3],
                                                    obj_pose_transform[..., 3],
                                                    dim=obj_pose_transform[..., 5:8] if obj_pose.shape[-1] > 5 else None,
                                                    inverse=True)

                pts_box_samples_w = torch.reshape(pts_box_samples_w, [n_intersect, N_samples_obj, 3])

                z_vals_obj_w = torch.linalg.vector_norm(pts_box_samples_w - rays_o[intersection_map[0]][:, None, :], dim=-1)

                # else:
                #     z_vals_obj_w = z_vals_in_w[:, None]
                #     pts_box_samples_o = pts_box_o[:, None, :]
                #     pts_box_samples_w = pts_box_w[:, None, :]

                #####
                # print('TURN OFF IF NOT DEBUGING!')
                # axes_ls = plt.figure(1).axes
                # plt.sca(axes_ls[0])
                #
                # pts = np.reshape(pts_box_samples_w, [-1, 3])
                # plt.scatter(pts[:, 0], pts[:, 2], color='red')
                ####

                # Extract objects
                obj_ids = obj_pose[..., 4]
                object_y, object_idx = torch.unique(torch.reshape(obj_pose[..., 4], [-1]), sorted=False, return_inverse=True)
                # Extract classes
                obj_class = obj_pose[..., 8]
                unique_classes = torch.unique(torch.reshape(obj_class, [-1]), sorted=False, return_inverse=True) #TODO check 
                class_id = torch.reshape(unique_classes[1], obj_class.shape)

                inputs = pts_box_samples_o

                if latent_vector_dict is not None:
                    latent_vector_inputs = None
                    # TODO　確認
                    for y, obj_id in enumerate(object_y):
                        indices = torch.where(object_idx == torch.tensor(y))
                        latent_vector = latent_vector_dict['latent_vector_obj_' + str(int(obj_id)).zfill(5)][None, :]
                        latent_vector = torch.repeat_interleave(latent_vector, indices.shape[0], dim=0)
                        tmp_latent_vector_inputs = torch.zeros([n_intersect*N_samples_obj, latent_vector.shape[-1]])
                        tmp_latent_vector_inputs[indices] = latent_vector

                        if latent_vector_inputs is None:
                            latent_vector_inputs = tmp_latent_vector_inputs
                        else:
                            latent_vector_inputs += tmp_latent_vector_inputs

                    latent_vector_inputs = torch.reshape(latent_vector_inputs, [n_intersect, N_samples_obj, -1])
                    inputs = torch.cat([inputs, latent_vector_inputs], dim=2)

                # inputs = tf.concat([inputs, obj_pose[..., :3]], axis=-1)

                # objdirs = tf.concat([tf.cos(obj_pose[:, 0, 3, None]), tf.sin(obj_pose[:, 0, 3, None])], axis=1)
                # objdirs = objdirs / tf.reduce_sum(objdirs, axis=1)[:, None]
                # viewdirs_obj = tf.concat([viewdirs_box_o, obj_pose[..., :3][:, 0, :], objdirs], axis=1)
                if obj_location:
                    viewdirs_obj = torch.cat([viewdirs_box_o, obj_pose[..., :3][:, 0, :]], dim=1)
                else:
                    viewdirs_obj = viewdirs_box_o

        if not obj_only:
            # Get integration step for all models
            z_vals, id_z_vals_bckg, id_z_vals_obj = combine_z(z_vals,
                                                              z_vals_obj_w if z_vals_in_o is not None else None,
                                                              intersection_map,
                                                              N_rays,
                                                              N_samples,
                                                              N_obj,
                                                              N_samples_obj, )
        else:
            z_vals, _, id_z_vals_obj = combine_z(None, z_vals_obj_w, intersection_map, N_rays, N_samples, N_obj,
                                                 N_samples_obj)


        if not obj_only:
            # Run background model
            raw = torch.zeros([N_rays, N_samples + N_obj*N_samples_obj, 4]).to(device)
            raw_sh = raw.shape
            # Predict RGB and density from background TODO
            raw_bckg = network_query_fn(pts, viewdirs, network_fn)
            raw[id_z_vals_bckg] += raw_bckg
        else:
            raw = torch.zeros([N_rays, N_obj*N_samples_obj, 4])
            raw_sh = raw.shape

        if z_vals_in_o is not None and len(z_vals_in_o) != 0:
            # Loop for one model per object and no latent representations
            if latent_vector_dict is None:
                obj_id = torch.reshape(object_idx, obj_pose[..., 4].shape)
                for k, track_id in enumerate(object_y):
                    if track_id >= 0:
                        input_indices = torch.where(obj_id == k)
                        input_indices = (torch.reshape(input_indices[0], [-1, N_samples_obj]), torch.reshape(input_indices[1], [-1, N_samples_obj]))
                        model_name = 'model_obj_' + str(np.array(track_id.item()).astype(np.int32))
                        # print('Hit', model_name, n_intersect, 'times.')
                        if model_name in object_network_fn_dict:
                            obj_network_fn = object_network_fn_dict[model_name]

                            inputs_obj_k = inputs[input_indices]
                            viewdirs_obj_k = viewdirs_obj[input_indices[0]] if N_samples_obj == 1 else \
                                viewdirs_obj[input_indices[0][:,0]]

                            # Predict RGB and density from object model
                            raw_k = network_query_fn(inputs_obj_k, viewdirs_obj_k, obj_network_fn)

                            if n_intersect is not None:
                                #TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO check
                                # Arrange RGB and denisty from object models along the respective rays
                                tmp_raw_k = torch.zeros([n_intersect, N_samples_obj, 4]).to(device)
                                tmp_raw_k[input_indices] = raw_k # Project the network outputs to the corresponding ray 
                                raw_k = tmp_raw_k
                                tmp_raw_k = torch.zeros([N_rays, N_obj, N_samples_obj, 4]).to(device)
                                tmp_raw_k[intersection_map] = raw_k # Project to rays and object intersection order
                                raw_k = tmp_raw_k
                                tmp_raw_k = torch.zeros(raw_sh).to(device)
                                tmp_raw_k[id_z_vals_obj] = raw_k # Reorder along z and ray
                                raw_k = tmp_raw_k
                            else:
                                tmp_raw_k = torch.zeros([N_rays, N_samples, 4]).to(device)
                                tmp_raw_k[input_indices[:, 0][..., None]] = raw_k
                                raw_k = tmp_raw_k

                            # Add RGB and density from object model to the background and other object predictions
                            raw += raw_k
            # Loop over classes c and evaluate each models f_c for all latent object describtor
            else:
                for c, class_type in enumerate(unique_classes.y):
                    # Ignore background class
                    if class_type >= 0:
                        input_indices = torch.where(class_id == c)
                        input_indices = torch.reshape(input_indices, [-1, N_samples_obj, 2])
                        model_name = 'model_class_' + str(int(np.array(class_type))).zfill(5)

                        if model_name in object_network_fn_dict:
                            obj_network_fn = object_network_fn_dict[model_name]

                            inputs_obj_c = inputs[input_indices]

                            # Legacy version 2
                            # latent_vector = tf.concat([
                            #         latent_vector_dict['latent_vector_' + str(int(obj_id)).zfill(5)][None, :]
                            #         for obj_id in np.array(tf.gather_nd(obj_pose[..., 4], input_indices)).astype(np.int32).flatten()],
                            #         axis=0)
                            # latent_vector = tf.reshape(latent_vector, [inputs_obj_k.shape[0], inputs_obj_k.shape[1], -1])
                            # inputs_obj_k = tf.concat([inputs_obj_k, latent_vector], axis=-1)

                            # viewdirs_obj_k = tf.gather_nd(viewdirs_obj,
                            #                               input_indices[..., 0]) if N_samples_obj == 1 else \
                            #     tf.gather_nd(viewdirs_obj, input_indices)

                            viewdirs_obj_c = tf.gather_nd(viewdirs_obj, input_indices[..., None, 0])[:,0,:]

                            # Predict RGB and density from object model
                            raw_k = network_query_fn(inputs_obj_c, viewdirs_obj_c, obj_network_fn)

                            if n_intersect is not None:
                                # Arrange RGB and denisty from object models along the respective rays
                                raw_k = tf.scatter_nd(input_indices[:, :], raw_k, [n_intersect, N_samples_obj,
                                                                                   4])  # Project the network outputs to the corresponding ray
                                raw_k = tf.scatter_nd(intersection_map[:, :2], raw_k, [N_rays, N_obj, N_samples_obj,
                                                                                       4])  # Project to rays and object intersection order
                                raw_k = tf.scatter_nd(id_z_vals_obj, raw_k, raw_sh)  # Reorder along z in  positive ray direction
                            else:
                                raw_k = tf.scatter_nd(input_indices[:, 0][..., None], raw_k,
                                                      [N_rays, N_samples, 4])

                            # Add RGB and density from object model to the background and other object predictions
                            raw += raw_k
                        else:
                            print('No model ', model_name,' found')



    # raw_2 = render_mot_scene(pts, viewdirs, network_fn, network_query_fn,
    #                  inputs, viewdirs_obj, z_vals_in_o, n_intersect, object_idx, object_y, obj_pose,
    #                  unique_classes, class_id, latent_vector_dict, object_network_fn_dict,
    #                  N_rays,N_samples, N_obj, N_samples_obj,
    #                  obj_only=obj_only)

    # TODO: Reduce computation by removing 0 entrys
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw, z_vals, rays_d)

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        if sampling_method == 'planes' or sampling_method == 'planes_plus':
            pts, z_vals = plane_pts([rays_o, rays_d], [plane_bds, plane_normal, delta], id_planes, near,
                                    method=sampling_method)
        else:
            # Obtain additional integration times to evaluate based on the weights
            # assigned to colors in the coarse model.
            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(
                z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.))
            z_samples = tf.stop_gradient(z_samples)

            # Obtain all points to evaluate color, density at.
            z_vals = tf.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[..., None, :] + rays_d[..., None, :] * \
                z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]

        # Make predictions with network_fine.
        if use_time:
            pts = torch.cat([pts, time_stamp_fine], dim=-1)

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, rays_d)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        if not sampling_method == 'planes':
            ret['z_std'] = tf.math.reduce_std(z_samples, -1)  # [N_rays]
    # if latent_vector_dict is not None:
    #     ret['latent_loss'] = tf.reshape(latent_vector, [N_rays, N_samples_obj, -1])

    for k in ret:
        is_nan = torch.sum(torch.isnan(ret[k])).item()
        assert is_nan == 0, 'NaN output in {}'.format(k)

    return ret


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = {}
    device = kwargs['device']
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs) #TODO render and train
        for k in ret: 
            if k not in all_ret:
                all_ret[k] = []
            if kwargs["train"]:
                all_ret[k].append(ret[k])
            else:
                all_ret[k].append(ret[k].cpu())

            
    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H,
           W,
           focal,
           chunk=1024*32,
           rays=None,
           c2w=None,
           obj=None,
           time_stamp=None,
           near=0.,
           far=1.,
           use_viewdirs=False,
           c2w_staticcam=None,
           **kwargs):
    """Render rays

    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      obj: array of shape [batch_size, max_obj, n_obj_nodes]. Scene object's pose and propeties for each
      example in the batch
      time_stamp: bool. If True the frame will be taken into account as an additional input to the network
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.

    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    device = kwargs['device']
    if c2w is not None:
        # special case to render full image
        # rays = tf.random.shuffle(tf.concat([get_rays(H, W, focal, c2w)[0], get_rays(H, W, focal, c2w)[1]], axis=-1))
        # rays_o = rays[..., :3]
        # rays_d = rays[..., 3:]
        rays_o, rays_d = get_rays(H, W, focal, c2w)
        rays_o = rays_o.to(device)
        rays_d = rays_d.to(device)
        if obj is not None:
            obj = torch.repeat_interleave(obj[None, ...], H*W, dim=0)
        if time_stamp is not None:
            time_stamp = torch.repeat_interleave(time_stamp[None, ...], H*W, dim=0)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)

        # Make all directions unit magnitude.
        # shape: [batch_size, 3]
        viewdirs = viewdirs / torch.linalg.vector_norm(viewdirs, dim=-1, keepdims=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).to(torch.float32)

    sh = rays_d.shape  # [..., 3]

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).to(torch.float32)
    rays_d = torch.reshape(rays_d, [-1, 3]).to(torch.float32)
    near, far = near * \
        torch.ones_like(rays_d[..., :1]).to(device), far * torch.ones_like(rays_d[..., :1]).to(device)

    # (ray origin, ray direction, min dist, max dist) for each ray
    rays = torch.cat([rays_o, rays_d, near, far], dim=-1)
    if use_viewdirs:
        # (ray origin, ray direction, min dist, max dist, normalized viewing direction)
        rays = torch.cat([rays, viewdirs], dim=-1)

    if time_stamp is not None:
        time_stamp = torch.tensor(torch.reshape(time_stamp, [len(rays), -1]), dtype=torch.float32)
        rays = torch.cat([rays, time_stamp], dim=-1)

    if obj is not None:
        # (ray origin, ray direction, min dist, max dist, normalized viewing direction, scene objects)
        # obj = tf.cast(tf.reshape(obj, [obj.shape[0], obj.shape[1]*obj.shape[2]]), dtype=tf.float32)
        obj = torch.reshape(obj, [obj.shape[0], obj.shape[1] * obj.shape[2]])
        rays = torch.concat([rays, obj], dim=-1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)
        # all_ret[k] = tf.reshape(all_ret[k], [k_sh[0], k_sh[1], -1])

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, chunk, render_kwargs, obj=None, obj_meta=None, gt_imgs=None, savedir=None,
                render_factor=0, render_manipulation=None, rm_obj=None, time_stamp=None):

    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factorobj_nodes

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(render_poses):
        print(i, time.time() - t)

        if time_stamp is not None:
            time_st = time_stamp[i]
        else:
            time_st = None

        if obj is None:
            rgb, disp, acc, _ = render(
                H, W, focal, chunk=chunk, c2w=c2w[:3, :4], obj=None, time_stamp=time_st, **render_kwargs)

            rgbs.append(rgb.numpy())
            disps.append(disp.numpy())

            if i == 0:
                print(rgb.shape, disp.shape)

            if gt_imgs is not None and render_factor == 0:
                p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
                print(p)

            if savedir is not None:
                rgb8 = to8b(rgbs[-1])
                filename = os.path.join(savedir, '{:03d}.png'.format(i))
                imageio.imwrite(filename, rgb8)

            print(i, time.time() - t)

        else:
            # Manipulate scene graph edges
            # rm_obj = [3, 4, 8, 5, 12]
            render_set = manipulate_obj_pose(render_manipulation, np.array(obj), obj_meta, i, rm_obj=rm_obj)


            # Load manual generated scene graphs
            if render_manipulation is not None and 'handcraft' in render_manipulation:
                if str(i).zfill(3) + '.txt' in os.listdir(savedir):
                    print('Reloading', str(i).zfill(3) + '.txt')
                    render_set.pop()
                    loaded_obj_i = []
                    loaded_objs = np.loadtxt(os.path.join(savedir, str(i).zfill(3) + '.txt'))[:, :6]
                    loaded_objs[:, 5] = 0
                    loaded_objs[:, 4] = np.array([np.where(np.equal(obj_meta[:, 0], loaded_objs[j, 4])) for j in range(len(loaded_objs))])[:, 0, 0]
                    loaded_objs = torch.tensor(loaded_objs, dtype=torch.float32)
                    loaded_obj_i.append(loaded_objs)
                    render_set.append(loaded_obj_i)
                if '02' in render_manipulation:
                    c2w = render_poses[36]
                if '03' in render_manipulation:
                    c2w = render_poses[20]
                if '04' in render_manipulation or '05' in render_manipulation:
                    c2w = render_poses[20]

            render_kwargs['N_obj'] = len(render_set[0][0])

            steps = len(render_set)
            for r, render_set_i in enumerate(render_set):
                t = time.time()
                j = steps * i + r
                obj_i = render_set_i[0]

                if obj_meta is not None:
                    obj_i_metadata = obj_meta[torch.tensor(obj_i[:, 4], dtype=torch.int32)]
                    batch_track_id = obj_i_metadata[..., 0]

                    print("Next Frame includes Objects: ")
                    if batch_track_id.shape[0] > 1:
                        for object_tracking_id in np.array(torch.squeeze(batch_track_id)).astype(np.int32):
                            if object_tracking_id >= 0:
                                print(object_tracking_id)

                    obj_i_dim = obj_i_metadata[:, 1:4]
                    obj_i_label = obj_i_metadata[:, 4][:, None]
                    # xyz + roty
                    obj_i = obj_i[..., :4]

                    obj_i = torch.cat([obj_i, batch_track_id[..., None], obj_i_dim, obj_i_label], dim=-1)

                # obj_i = np.array(obj_i)
                # rm_ls_0 = [0, 1, 2,]
                # rm_ls_1 = [0, 1, 2]
                # rm_ls_2 = [0, 1, 2, 3, 5]
                # rm_ls = [rm_ls_0, rm_ls_1, rm_ls_2]
                # for k in rm_ls[i]:
                #     obj_i[k] = np.ones([9]) * -1

                rgb, disp, acc, _ = render(
                    H, W, focal, chunk=chunk, c2w=c2w[:3, :4], obj=obj_i, **render_kwargs)
                rgbs.append(rgb.numpy())
                disps.append(disp.numpy())

                if j == 0:
                    print(rgb.shape, disp.shape)

                if gt_imgs is not None and render_factor == 0:
                    p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
                    print(p)

                if savedir is not None:
                    rgb8 = to8b(rgbs[-1])
                    filename = os.path.join(savedir, '{:03d}.png'.format(j))
                    imageio.imwrite(filename, rgb8)
                    if render_manipulation is not None:
                        if 'handcraft' in render_manipulation:
                            filename = os.path.join(savedir, '{:03d}.txt'.format(j))
                            np.savetxt(filename, np.array(obj_i), fmt='%.18e %.18e %.18e %.18e %.1e %.18e %.18e %.18e %.1e')


                print(j, time.time() - t)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args, device):
    """Instantiate NeRF's MLP model."""
    if args.obj_detection:
        trainable = False
    else:
        trainable = True

    pos_enc, input_ch = get_embedder(args.multires, args.i_embed)
    pos_enc.to(device)

    if args.use_time:
        input_ch += 1

    input_ch_views = 0
    dir_enc = None
    if args.use_viewdirs:
        dir_enc, input_ch_views = get_embedder(
            args.multires_views, args.i_embed)
        dir_enc.to(device)
    output_ch = 4
    skips = (4,)   
    model = init_nerf_model(
        pos_enc=pos_enc, dir_enc=dir_enc,
        base_mlp_num_layers=args.netdepth, base_mlp_layer_width=args.netwidth,
        head_mlp_num_layers=args.netdepth//2, head_mlp_layer_width=args.netwidth//2, skips=skips,
        trainable=trainable)
    models = {'model': model.to(device)}
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = init_nerf_model(
        pos_enc=pos_enc, dir_enc=dir_enc,
        base_mlp_num_layers=args.netdepth, base_mlp_layer_width=args.netwidth,
        head_mlp_num_layers=args.netdepth//2, head_mlp_layer_width=args.netwidth//2, skips=skips,
        trainable=trainable)
        grad_vars += list(model.parameters())
        models['model_fine'] = model_fine.to(device)

    models_dynamic_dict = None
    obj_dir_enc = None
    latent_vector_dict = None if args.latent_size < 1 else {}
    latent_encodings = None if args.latent_size < 1 else {}
    if args.use_object_properties and not args.bckg_only:
        models_dynamic_dict = {}
        obj_dir_enc, input_ch_obj = get_embedder(
            args.multires_obj, -1 if args.multires_obj == -1 else args.i_embed, input_dims=6)
        obj_dir_enc.to(device)

        # Version a: One Network per object
        if args.latent_size < 1:
            input_ch = input_ch
            input_ch_color_head = input_ch_views
            # Don't add object location input for setting 1
            if args.object_setting != 1:
                input_ch_color_head += input_ch_obj
            # TODO: Change to number of objects in Frames
            for object_i in args.scene_objects:

                model_name = 'model_obj_' + str(int(object_i)) # .zfill(5)

                model_obj = init_nerf_model(
                    pos_enc=pos_enc, dir_enc=obj_dir_enc,
                    base_mlp_num_layers=args.netdepth, base_mlp_layer_width=args.netwidth,
                    head_mlp_num_layers=args.netdepth//2, head_mlp_layer_width=args.netwidth//2, skips=skips,
                    trainable=trainable)
                    # latent_size=args.latent_size)

                grad_vars += list(model_obj.parameters())
                models[model_name] = model_obj.to(device)
                models_dynamic_dict[model_name] = model_obj.to(device)

        # Version b: One Network for all similar objects of the same class
        else:
            input_ch = input_ch + args.latent_size
            input_ch_color_head = input_ch_views
            # Don't add object location input for setting 1
            if args.object_setting != 1:
                input_ch_color_head += input_ch_obj

            for obj_class in args.scene_classes:
                model_name = 'model_class_' + str(int(obj_class)).zfill(5)

                model_obj = init_nerf_model(
                    D=args.netdepth_fine, W=args.netwidth_fine,
                    input_ch=input_ch, output_ch=output_ch, skips=skips,
                    input_ch_color_head=input_ch_color_head,
                    # input_ch_shadow_head=input_ch_obj,
                    use_viewdirs=args.use_viewdirs, trainable=trainable)
                    # use_shadows=args.use_shadows,
                    # latent_size=args.latent_size)

                grad_vars += list(model_obj.parameters())
                models[model_name] = model_obj.to(device)
                models_dynamic_dict[model_name] = model_obj.to(device)

            for object_i in args.scene_objects:
                name = 'latent_vector_obj_'+str(int(object_i)).zfill(5)
                latent_vector_obj = init_latent_vector(args.latent_size, name)
                grad_vars.append(latent_vector_obj)

                latent_encodings[name] = latent_vector_obj.to(device)
                latent_vector_dict[name] = latent_vector_obj.to(device)

    # TODO: Remove object embedding function
    def network_query_fn(inputs, viewdirs, network_fn): return run_network(
        inputs, viewdirs, network_fn,
        embed_fn=pos_enc,
        embeddirs_fn=dir_enc,
        embedobj_fn=obj_dir_enc,
        netchunk=args.netchunk)

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'N_samples_obj': args.N_samples_obj,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'object_network_fn_dict': models_dynamic_dict,
        'latent_vector_dict': latent_vector_dict if latent_vector_dict is not None else None,
        'N_obj': args.max_input_objects if args.use_object_properties and not args.bckg_only else False,
        'obj_only': args.obj_only,
        'obj_transparency': not args.obj_opaque,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
        'sampling_method': args.sampling_method,
        'use_time': args.use_time,
        'obj_location': False if args.object_setting == 1 else True,
        'device': device,
        "train": True,
    }

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    render_kwargs_test["train"] = False
    # render_kwargs_test['obj_only'] = False

    start = 0
    basedir = args.basedir
    expname = args.expname
    weights_path = None

    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    elif args.model_library is not None and args.model_library != 'None':
        obj_ckpts = {}
        ckpts = []
        for f in sorted(os.listdir(args.model_library)):
            if 'model_' in f and 'fine' not in f and 'optimizer' not in f and 'obj' not in f:
                ckpts.append(os.path.join(args.model_library, f))
            if 'obj' in f and float(f[10:][:-11]) in args.scene_objects:
                obj_ckpts[f[:-11]] = (os.path.join(args.model_library, f))
    elif args.obj_only:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 ('_obj_' in f)]
    else:
        ckpts = [os.path.join(basedir, expname, "models", f) for f in sorted(os.listdir(os.path.join(basedir, expname, "models"))) if
                 ('model_' in f and 'fine' not in f and 'optimizer' not in f and 'obj' not in f and 'class' not in f)]
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload and (not args.obj_only or args.model_library):
        ft_weights = ckpts[-1]
        print('Reloading from', ft_weights)
        model.load_state_dict(torch.load(ft_weights))
        start = int(ft_weights[-10:-4]) + 1
        print('Resetting step to', start)

        if model_fine is not None:
            ft_weights_fine = '{}_fine_{}'.format(
                ft_weights[:-11], ft_weights[-10:])
            print('Reloading fine from', ft_weights_fine)
            model_fine.load_state_dict(torch.load(ft_weights_fine))
        if models_dynamic_dict is not None:
            for model_dyn_name, model_dyn in models_dynamic_dict.items():
                if args.model_library:
                    ft_weights_obj = obj_ckpts[model_dyn_name]
                else:
                    ft_weights_obj = '{}'.format(ft_weights[:-16]) + \
                                     model_dyn_name + '_{}'.format(ft_weights[-10:])
                print('Reloading model from', ft_weights_obj, 'for', model_dyn_name[6:])
                model_dyn.load_state_dict(torch.load(ft_weights_obj))

        if latent_vector_dict is not None:
            for latent_vector_name, latent_vector in latent_vector_dict.items():
                ft_weights_obj = '{}'.format(ft_weights[:-16]) + \
                                     latent_vector_name + '_{}'.format(ft_weights[-10:])
                print('Reloading objects latent vector from', ft_weights_obj)
                latent_vector.load_state_dict(torch.load(ft_weights_obj))

    elif len(ckpts) > 0 and args.obj_only:
        ft_weights = ckpts[-1]
        start = int(ft_weights[-10:-4]) + 1
        ft_weights_obj_dir = os.path.split(ft_weights)[0]
        for model_dyn_name, model_dyn in models_dynamic_dict.items():
            ft_weights_obj = os.path.join(ft_weights_obj_dir, model_dyn_name + '_{}'.format(ft_weights[-10:]))
            print('Reloading model from', ft_weights_obj, 'for', model_dyn_name[6:])
            model_dyn.load_state_dict(torch.load(ft_weights_obj))

        if latent_vector_dict is not None:
            for latent_vector_name, latent_vector in latent_vector_dict.items():
                ft_weights_obj = os.path.join(ft_weights_obj_dir, latent_vector_name + '_{}'.format(ft_weights[-10:]))
                print('Reloading objects latent vector from', ft_weights_obj)
                latent_vector.load_state_dict(torch.load(ft_weights_obj))

        weights_path = ft_weights

    if args.model_library:
        start = 0

    return render_kwargs_train, render_kwargs_test, start, grad_vars, models, latent_encodings, weights_path

def train():
    from config_parser import config_parser
    parser = config_parser()
    args = parser.parse_args()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if args.random_seed is not None:
        print('Fixing random seed', args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

    if args.obj_only and args.bckg_only:
        print('Object and background can not set as train only at the same time.')
        return

    if args.bckg_only or args.obj_only:
        # print('Deactivating object models to increase performance for training the background model only.')
        args.use_object_properties = True

    # Support first and last frame int
    starts = args.first_frame.split(',')
    ends = args.last_frame.split(',')
    if len(starts) != len(ends):
        print('Number of sequences is not defined. Using the first sequence')
        args.first_frame = int(starts[0])
        args.last_frame = int(ends[0])
    else:
        args.first_frame = [int(val) for val in starts]
        args.last_frame = [int(val) for val in ends]

    if args.dataset_type == 'kitti':
        # tracking2txt('../../CenterTrack/results/default_0006_results.json')

        images, poses, render_poses, hwf, i_split, visible_objects, objects_meta, render_objects, bboxes, \
        kitti_obj_metadata, time_stamp, render_time_stamp = \
            load_kitti_data(args.datadir,
                            selected_frames=[args.first_frame, args.last_frame] if args.last_frame else None,
                            use_obj=True,
                            row_id=True,
                            remove=args.remove_frame,
                            use_time=args.use_time,
                            exp=True if 'exp' in args.expname else False)
        print('Loaded kitti', images.shape,
              #render_poses.shape,
              hwf,
              args.datadir)

        if visible_objects is not None:
            args.max_input_objects = visible_objects.shape[1]
        else:
            args.max_input_objects = 0

        if args.render_only:
            visible_objects = render_objects

        i_train, i_val, i_test = i_split

        near = args.near_plane
        far = args.far_plane

        # Fix all persons at one position
        fix_ped_pose = False
        if fix_ped_pose:
            print('Pedestrians are fixed!')
            ped_poses = np.pad(visible_objects[np.where(visible_objects[..., 3] == 4)][:, 7:11], [[0, 0], [7, 3]])
            visible_objects[np.where(visible_objects[..., 3] == 4)] -= ped_poses
            visible_objects[np.where(visible_objects[..., 3] == 4)] += ped_poses[20]

        # Get SRN Poses, Images and Poses
        # work_dir = os.path.dirname(os.path.abspath(args.basedir))
        # srn_data_dir = os.path.join(work_dir, 'srn_data_' + args.expname)
        # srn_data_dir_train = srn_data_dir+'_train'
        # srn_data_dir_val = srn_data_dir + '_train_val'
        # if not os.path.exists(srn_data_dir_train):
        #     os.makedirs(srn_data_dir_train)
        #     os.makedirs(srn_data_dir_val)
        #     for dir in ['rgb', 'pose', 'intrinsics']:
        #         os.makedirs(os.path.join(srn_data_dir_train, dir))
        #         os.makedirs(os.path.join(srn_data_dir_val, dir))
        #
        # f = hwf[2]
        # c_x = hwf[1]/2.
        # c_y = hwf[0]/2.
        # id_val = 0
        #
        # for i, img in enumerate(images):
        #     frame_id = str(i).zfill(5)
        #     im = Image.fromarray((img*255).astype(np.uint8))
        #     im.save(os.path.join(srn_data_dir_train, 'rgb/'+frame_id+'.png'))
        #     np.savetxt(os.path.join(srn_data_dir_train, 'pose/' + frame_id + '.txt'),
        #                np.reshape(poses[i], [-1])[None], fmt='%.16f')
        #     np.savetxt(os.path.join(srn_data_dir_train, 'intrinsics/' + frame_id + '.txt'),
        #                np.array([f, 0.0, c_x, 0.0, f, c_y, 0.0, 0.0, 1.0], np.float32)[None], fmt='%.1f')
        #     if not i % 10:
        #         frame_id_val = str(id_val).zfill(5)
        #         im.save(os.path.join(srn_data_dir_val, 'rgb/' + frame_id_val + '.png'))
        #         np.savetxt(os.path.join(srn_data_dir_val, 'pose/' + frame_id_val + '.txt'),
        #                    np.reshape(poses[i], [-1])[None], fmt='%.16f')
        #         np.savetxt(os.path.join(srn_data_dir_val, 'intrinsics/' + frame_id_val + '.txt'),
        #                    np.array([f, 0.0, c_x, 0.0, f, c_y, 0.0, 0.0, 1.0], np.float32)[None], fmt='%.1f')
        #         id_val += 1
        #
        # print('Stored Image set for SRNs')


        # Get COLMAP formated poses
        # colmap_poses = poses[:, :3, :]
        # colmap_poses = np.concatenate([colmap_poses, np.repeat(np.array(hwf)[None], len(poses), axis=0)[..., None]], axis=2)
        # colmap_poses = np.reshape(colmap_poses, [-1,15])
        # colmap_poses = np.concatenate([colmap_poses, np.repeat(np.array([near, far])[None], len(poses), axis=0)], axis=1)
        # np.save(os.path.join(args.basedir, args.expname) +'/poses_bounds.npy', colmap_poses)

    elif args.dataset_type == 'vkitti':
        # TODO: Class by integer instead of hot-one-encoding for latent encoding in visible object
        images, instance_segm, poses, frame_id, render_poses, hwf, i_split, visible_objects, objects_meta, render_objects, bboxes = \
            load_vkitti_data(args.datadir,
                             selected_frames=[args.first_frame[0], args.last_frame[0]] if args.last_frame[0] >= 0 else -1,
                             use_obj=args.use_object_properties,
                             row_id=True if args.object_setting == 0 or args.object_setting == 1 else False,)
        render_time_stamp = None

        print('Loaded vkitti', images.shape,
              #render_poses.shape,
              hwf,
              args.datadir)
        if visible_objects is not None:
            args.max_input_objects = visible_objects.shape[1]
        else:
            args.max_input_objects = 0

        i_train, i_val, i_test = i_split

        near = args.near_plane
        far = args.far_plane

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Ploting Options for Debugging the Scene Graph
    plot_poses = False
    if args.debug_local and plot_poses:
        plot_kitti_poses(args, poses, visible_objects)

    # Cast intrinsics to right types
    np.linalg.norm(poses[:1, [0, 2], 3] - poses[1:, [0, 2], 3])
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Extract objects positions and labels
    if args.use_object_properties or args.bckg_only:
        obj_nodes, add_input_rows, obj_meta_ls, scene_objects, scene_classes = \
            extract_object_information(args, visible_objects, objects_meta)

        # obj_track_id_list = False if args.single_obj == None else [args.single_obj] #[4., 9.,, 3.] # [9.]

        if args.single_obj is not None:
            # Train only a single object
            args.scene_objects = [args.single_obj]
        else:
            args.scene_objects = scene_objects

        args.scene_classes = scene_classes

        n_input_frames = obj_nodes.shape[0]

        # Prepare object nodes [n_images, n_objects, H, W, add_input_rows, 3]
        obj_nodes = np.reshape(obj_nodes, [n_input_frames, args.max_input_objects * add_input_rows, 3])

        obj_meta_tensor = torch.tensor(np.array(obj_meta_ls), dtype=torch.float32)

        if args.render_test:
            render_objects = obj_nodes[i_test]

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    model_log_dir = os.path.join(basedir, expname, "models")
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    os.makedirs(model_log_dir, exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf representation models
    render_kwargs_train, render_kwargs_test, start, grad_vars, models, latent_encodings, weights_path = create_nerf(
        args, device)

    if args.obj_only:
        print('removed bckg model for obj training')
        del grad_vars[:len(models['model'].parameters())]
        models.pop('model')

    if args.ft_path is not None and args.ft_path != 'None':
        start = 0

    # Set bounds for point sampling along a ray
    if not args.sampling_method == 'planes' and not args.sampling_method == 'planes_plus':
        bds_dict = {
            'near': torch.tensor(near, dtype=torch.float32).to(device),
            'far': torch.tensor(far, dtype=torch.float32).to(device),
        }
    else:
        # TODO: Generalize for non front-facing scenarios
        plane_bds, plane_normal, plane_delta, id_planes, near, far = plane_bounds(
            poses, args.plane_type, near, far, args.N_samples)

        # planes = [plane_origin, plane_normal]
        bds_dict = {
            'near': torch.tensor(near, dtype=torch.float32).to(device),
            'far': torch.tensor(far, dtype=torch.float32).to(device),
            'plane_bds': torch.tensor(plane_bds, dtype=torch.float32).to(device),
            'plane_normal': torch.tensor(plane_normal, dtype=torch.float32).to(device),
            'id_planes': torch.tensor(id_planes, dtype=torch.float32).to(device),
            'delta': torch.tensor(plane_delta, dtype=torch.float32).to(device)
        }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Short circuit if np.argwhere(n[:1,:,0]>0)only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        if args.render_test:
            # render_test switches to test poses
            images = images[i_test]
        else:
            # Default is smoother render_poses path
            images = None

        testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format(
            'test' if args.render_test else 'path', start))
        if args.manipulate is not None:
            testsavedir = testsavedir + '_' + args.manipulate

        os.makedirs(testsavedir, exist_ok=True)
        print('test poses shape', render_poses.shape)

        # Select random from render_poses
        render_poses = render_poses[np.random.randint(0, len(render_poses) - 1, np.minimum(3, len(render_poses)))]
        with torch.no_grad():
            rgbs, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test,
                                obj=obj_nodes if args.use_object_properties and not args.bckg_only else None,
                                obj_meta=obj_meta_tensor if args.use_object_properties else None,
                                gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor,
                                render_manipulation=args.manipulate, rm_obj=args.remove_obj,
                                time_stamp=render_time_stamp)
        print('Done rendering', testsavedir)
        if args.dataset_type == 'vkitti':
            rgbs = rgbs[:, 1:, ...]
            macro_block_size = 2
        else:
            macro_block_size = 16

        imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'),
                         to8b(rgbs), fps=30, quality=10, macro_block_size=macro_block_size)

        return

    # Create optimizer
    lrate = args.lrate
    optimizer = torch.optim.Adam(lr=lrate, params=grad_vars)
    if args.lrate_decay > 0:
        lr_lambda = lambda step: 0.1**(step/(args.lrate_decay * 1000))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    models['optimizer'] = optimizer

    global_step = start


    N_rand = args.N_rand
    # For random ray batching.
    #
    # Constructs an array 'rays_rgb' of shape [N*H*W, 3, 3] where axis=1 is
    # interpreted as,
    #   axis=0: ray origin in world space
    #   axis=1: ray direction in world space
    #   axis=2: observed RGB color of pixel
    print('get rays')
    # get_rays_np() returns rays_origin=[H, W, 3], rays_direction=[H, W, 3]
    # for each pixel in the image. This stack() adds a new dimension.
    rays = [get_rays_np(H, W, focal, p) for p in poses[:, :3, :4]]
    rays = np.stack(rays, axis=0)  # [N, ro+rd, H, W, 3]
    print('done, concats')
    # [N, ro+rd+rgb, H, W, 3]
    rays_rgb = np.concatenate([rays, images[:, None, ...]], 1)

    if not args.use_object_properties:
        # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
        rays_rgb = np.stack([rays_rgb[i]
                             for i in i_train], axis=0)  # train images only
        # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
        rays_rgb = rays_rgb.astype(np.float32)
        if args.use_time:
            time_stamp_train = np.stack([time_stamp[i]
                                   for i in i_train], axis=0)
            time_stamp_train = np.repeat(time_stamp_train[:, None, :], H*W, axis=0).astype(np.float32)
            rays_rgb = np.concatenate([rays_rgb, time_stamp_train], axis=1)

    else:
        print("adding object nodes to each ray")
        rays_rgb_env = rays_rgb
        input_size = 0

        obj_nodes = np.repeat(obj_nodes[:, :, np.newaxis, ...], W, axis=2)
        obj_nodes = np.repeat(obj_nodes[:, :, np.newaxis, ...], H, axis=2)

        obj_size = args.max_input_objects * add_input_rows
        input_size += obj_size
        # [N, ro+rd+rgb+obj_nodes, H, W, 3]
        rays_rgb_env = np.concatenate([rays_rgb_env, obj_nodes], 1)

        # [N, H, W, ro+rd+rgb+obj_nodes*max_obj, 3]
        # with obj_nodes [(x+y+z)*max_obj + (track_id+is_training+0)*max_obj]
        rays_rgb_env = np.transpose(rays_rgb_env, [0, 2, 3, 1, 4])
        rays_rgb_env = np.stack([rays_rgb_env[i]
                                 for i in i_train], axis=0)  # train images only
        # [(N-1)*H*W, ro+rd+rgb+ obj_pose*max_obj, 3]
        rays_rgb_env = np.reshape(rays_rgb_env, [-1, 3+input_size, 3])

        rays_rgb = rays_rgb_env.astype(np.float32)
        del rays_rgb_env

        # get all rays intersecting objects
        if (args.bckg_only or args.obj_only or args.model_library is not None or args.use_object_properties): #and not args.debug_local:
            bboxes = None
            print(rays_rgb.shape)

            if args.use_inst_segm:
                # Ray selection from segmentation (early experiments)
                print('Using segmentation map')
                if not args.scene_objects:
                    rays_on_obj = np.where(instance_segm.flatten() > 0)[0]

                else:
                    # B) Single object per scene
                    rays_on_obj = []
                    for obj_track_id in args.scene_objects:
                        rays_on_obj.append(np.where(instance_segm.flatten() == obj_track_id+1)[0])
                    rays_on_obj = np.concatenate(rays_on_obj)
            elif bboxes is not None:
                # Ray selection from 2D bounding boxes (early experiments)
                print('Using 2D bounding boxes')
                rays_on_obj = get_bbox_pixel(bboxes, i_train, hwf)
            else:
                # Preferred option
                print('Using Ray Object Node intersections')
                rays_on_obj, rays_to_remove = get_all_ray_3dbox_intersection(rays_rgb, obj_meta_tensor,
                                                                             args.netchunk, local=args.debug_local,
                                                                             obj_to_remove=args.remove_obj)

            # Create Masks for background and objects to subsample the training batches
            obj_mask = np.zeros(len(rays_rgb), bool)
            obj_mask[rays_on_obj] = 1

            bckg_mask = np.ones(len(rays_rgb), bool)
            bckg_mask[rays_on_obj] = 0

            # Remove predefined objects from the scene
            if len(rays_to_remove) > 0 and args.remove_obj is not None:
                print('Removing obj ', args.remove_obj)
                # Remove rays from training set
                remove_mask = np.zeros(len(rays_rgb), np.bool)
                remove_mask[rays_to_remove] = 1
                obj_mask[remove_mask] = 0
                # Remove objects from graph
                rays_rgb = remove_obj_from_set(rays_rgb, np.array(obj_meta_ls), args.remove_obj)
                obj_nodes = np.reshape(np.transpose(obj_nodes, [0, 2, 3, 1, 4]), [-1, args.max_input_objects*2, 3])
                obj_nodes = remove_obj_from_set(obj_nodes, np.array(obj_meta_ls), args.remove_obj)
                obj_nodes = np.reshape(obj_nodes, [len(images), H, W, args.max_input_objects*2, 3])
                obj_nodes = np.transpose(obj_nodes, [0, 3, 1, 2, 4])

            # Debugging options to display selected rays/pixels
            debug_pixel_selection = False
            if args.debug_local and debug_pixel_selection:
                for i_smplimg in range(len(i_train)):
                    rays_rgb_debug = np.array(rays_rgb)
                    rays_rgb_debug[rays_on_obj, :] += np.random.rand(3) #0.
                    # rays_rgb_debug[remove_mask, :] += np.random.rand(3)
                    plt.figure()
                    img_sample = np.reshape(rays_rgb_debug[(H * W) * i_smplimg:(H * W) * (i_smplimg + 1), 2, :],
                                            [H, W, 3])
                    plt.imshow(img_sample)

                    # white_canvas = np.ones_like(rays_rgb_debug)
                    # white_canvas[rays_on_obj, :] = np.array([0., 0., 1.])
                    # white_sample = np.reshape(white_canvas[(H * W) * i_smplimg:(H * W) * (i_smplimg + 1), 2, :],
                    #                     [H, W, 3])
                    # white_sample = np.concatenate([white_sample, np.zeros([H, W, 1])], axis=2)
                    # white_sample = (white_sample*255).astype(np.uint8)
                    # white_sample[..., 3][np.where(white_sample[..., 1] < 1.)] = 255.
                    # plt.figure()
                    # plt.imshow(white_sample)
                    # Image.fromarray(white_sample).save('/home/julian/Desktop/sample.png')
                    # plt.arrow(0, H / 2, W, 0, color='red')
                    # plt.arrow(W / 2, 0, 0, H, color='red')
                    # plt.savefig('/home/julian/Desktop/debug_kitti_box/01/Figure_'+str(i_smplimg),)
                    # plt.close()

            if args.bckg_only:
                print('Removing objects from scene.')
                rays_rgb = rays_rgb[bckg_mask]
                print(rays_rgb.shape)
            elif args.obj_only and args.model_library is None or args.debug_local:
                print('Extracting objects from background.')
                rays_bckg = None
                rays_rgb = rays_rgb[obj_mask]
                print(rays_rgb.shape)
            else:
                rays_bckg = rays_rgb[bckg_mask]
                rays_rgb = rays_rgb[obj_mask]

            # Get Intersections per object and additional rays to have similar rays/object distributions VVVVV
            if not args.bckg_only:
                # # print(rays_rgb.shape)
                with torch.no_grad(): # TODO 必要？
                    rays_rgb = resample_rays(rays_rgb, rays_bckg, obj_meta_tensor, objects_meta,
                                            args.scene_objects, scene_classes, args.chunk, local=args.debug_local)
            # Get Intersections per object and additional rays to have similar rays/object distributions AAAAA

    print('shuffle rays')
    np.random.shuffle(rays_rgb)
    print('done')
    i_batch = 0

    N_iters = 1000000
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    logger = Logger(os.path.join(basedir, 'summaries', expname), model_log_dir, global_step=global_step)
    for i in range(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        batch_obj = None

        # Random over all images
        if not args.use_object_properties:
            # No object specific representations
            batch = rays_rgb[i_batch:i_batch+N_rand]  # [B, 2+1, 3*?]
            batch = torch.transpose(batch, [1, 0, 2])

            # batch_rays[i, n, xyz] = ray origin or direction, example_id, 3D position
            # target_s[n, rgb] = example_id, observed color.
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                np.random.shuffle(rays_rgb)
                i_batch = 0

        batch = rays_rgb[i_batch:i_batch + N_rand]  # [B, 2+1+max_obj, 3*?]
        batch = np.transpose(batch, [1, 0, 2]) # [2+1+max_obj, B, 3*?]
        batch = torch.tensor(batch).to(device)
        obj_meta_tensor = obj_meta_tensor.to(device)

        # batch_rays[i, n, xyz] = ray origin or direction, example_id, 3D position
        # target_s[n, rgb] = example_id, observed color.
        batch_rays, target_s, batch_dyn = batch[:2], batch[2], batch[3:]

        if args.use_time:
            batch_time = batch_dyn
        else:
            batch_time = None

        if args.use_object_properties:
            # batch_obj[N_rand, max_obj, properties+0]
            batch_obj_dyn = torch.reshape(torch.permute(
                batch_dyn, (1, 0, 2)), [batch.shape[1], args.max_input_objects, add_input_rows*3])


            # xyz + roty
            batch_obj = batch_obj_dyn[..., :4]

            # [N_rand, max_obj, trackID + label + model + color + Dimension]
            # Extract static nodes and edges (latent node, id, box size) for each object at each ray
            batch_obj_metadata = obj_meta_tensor[batch_obj_dyn[:, :, 4].to(torch.int32)]
            batch_track_id = batch_obj_metadata[:, :, 0]
            # TODO: For generalization later Give track ID in the beginning and change model name to track ID 意味がわからない何をしている
            batch_obj = torch.cat([batch_obj, batch_track_id[..., None]], dim=-1)
            batch_dim = batch_obj_metadata[:, :, 1:4]
            batch_label = batch_obj_metadata[:, :, 4][..., None]

            batch_obj = torch.cat([batch_obj, batch_dim, batch_label], dim=-1)


            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                np.random.shuffle(rays_rgb)
                i_batch = 0


        #####  Core optimization loop  #####

            # Make predictions for color, disparity, accumulated opacity.
            rgb, disp, acc, extras = render(
                H, W, focal, chunk=args.chunk, rays=batch_rays, obj=batch_obj, time_stamp=batch_time,
                verbose=i < 10, retraw=True, **render_kwargs_train)

            # Compute MSE loss between predicted and true RGB.
            img_loss = img2mse(rgb, target_s)
            trans = extras['raw'][..., -1]
            loss = img_loss
            psnr = mse2psnr(img_loss)

            # Add MSE loss for coarse-grained model
            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target_s)
                loss += img_loss0
                psnr0 = mse2psnr(img_loss0)

            # Add loss for latent code
            if args.latent_size > 0:
                reg = 1/args.latent_balance    # 1/0.01
                latent_reg = latentReg(list(render_kwargs_train['latent_vector_dict'].values()), reg)
                loss += latent_reg

        # torch
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        dt = time.time()-time0
        #####           end            #####

        # Rest is logging
        scalar_logs = OrderedDict()
        histogram_logs = OrderedDict()
        image_logs = OrderedDict()
        if logger.should_record(args.i_weights):
            logger.save_weights(models, i)

        if i < 10 or logger.should_record(args.i_print):
            print("\n##############################################")
            print(f"expname: {expname}")
            print(f"global step: {logger.global_step}")
            print('iter time {:.05f}'.format(dt))
            if logger.should_record(args.i_print):
                scalar_logs.update({
                    "train_loss": loss.item(),
                    "train_psnr": psnr.item(),
                })
                if args.latent_size > 0:
                    for latent_vector_sum in list(render_kwargs_train['latent_vector_dict'].values()):
                        histogram_logs.update({
                            latent_vector_sum.name: latent_vector_sum.value(),
                        })

            if logger.should_record(args.i_img) and not i == 0: # and not args.debug_local:
                print("\n##############################################")
                # Log a rendered validation view to Tensorboard
                img_i = np.random.choice(i_val)
                target = torch.tensor(images[img_i])
                pose = poses[img_i, :3, :4]
                time_st = time_stamp[img_i] if args.use_time else None

                if args.use_object_properties:
                    obj_i = obj_nodes[img_i, :, 0, 0, ...]
                    obj_i = torch.tensor(obj_i, dtype=torch.float32).to(device)
                    obj_i = torch.reshape(obj_i, [args.max_input_objects, obj_i.shape[0] // args.max_input_objects * 3])
                    indices = obj_i[:, 4].clone().detach().to(torch.int32)
                    obj_i_metadata = obj_meta_tensor[indices]
                    batch_track_id = obj_i_metadata[..., 0]
                    obj_i_dim = obj_i_metadata[:, 1:4]
                    obj_i_label = obj_i_metadata[:, 4][:, None]

                    # xyz + roty
                    obj_i = obj_i[..., :4]
                    obj_i = torch.cat([obj_i, batch_track_id[..., None], obj_i_dim, obj_i_label], dim=-1)
                    with torch.no_grad():
                        rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose, obj=obj_i,
                                                        **render_kwargs_test)
                else:
                    with torch.no_grad():
                        rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose, time_stamp=time_st,
                                                        **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                # Save out the validation image for Tensorboard-free monitoring
                testimgdir = os.path.join(basedir, expname, 'tboard_val_imgs')
                if i==0 or not os.path.exists(testimgdir):
                    os.makedirs(testimgdir, exist_ok=True)
                # imageio.imwrite(os.path.join(testimgdir, '{:06d}.png'.format(i)), to8b(rgb))

                if logger.should_record(args.i_img):
                    image_logs.update({
                        'eval_rgb':to8b(rgb)[None],
                        'eval_disp':to8b(disp[None, ..., None]),
                        'eval_acc': to8b(acc[None, ..., None]),
                        'eval_rgb_holdout':target[None],
                        })
                    scalar_logs.update({'psnr_holdout': psnr.item()})

                if args.N_importance > 0:

                    if logger.should_record(args.i_img):
                        image_logs.update({
                            'eval_rgb0': to8b(extras['rgb0'])[None],
                            'eval_disp0': extras['disp0'][None, ..., None],
                            'eval_z_std': extras['z_std'][None, ..., None],
                            })
        logger.add_scalars(scalar_logs)
        logger.add_histograms(histogram_logs)
        logger.add_images(image_logs)
        logger.flush()
        logger.add_global_step()


if __name__ == '__main__':
    train()

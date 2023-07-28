def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str,
                        default='./data/llff/fern', help='input data directory')
    parser.add_argument("--obj_meta_dir", type=str, default='./data/meta_data.pth',
                        help='where to load meta data')

    # training options
    parser.add_argument("--use_obj_meta", action='store_true',
                        help='use objects meta data')
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int,
                        default=8, help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float,
                        default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000s)')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    # Disabled and not implemented for Neural Scene Graphs
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--model_library", type=str, default=None,
                        help='specific weights npy file to load pretrained background and foreground models')
    parser.add_argument("--random_seed", type=int, default=None,
                        help='fix random seed for repeatability')
    parser.add_argument("--sampling_method", type=str, default=None,
                        help='method to sample points along the ray options: None / lindisp / squaredist / plane')
    parser.add_argument("--crop_size", type=int, default=16,
                        help='size of crop image for second stage deblurring')
    parser.add_argument("--bckg_only", action='store_true',
                        help='removes rays associated with objects from the training set to train just the background model.')
    parser.add_argument("--obj_only", action='store_true',
                        help='Train object models on rays close to the objects only.')
    parser.add_argument("--use_inst_segm", action='store_true',
                        help='Use an instance segmentation map to select a subset from all sampled rays')
    parser.add_argument("--latent_size", type=int, default=0,
                        help='Size of the latent vector representing each of object of a class. If 0 no latent vector '
                             'is applied and a single representation per object is used.')
    parser.add_argument("--latent_balance", type=float, default=0.01,
                        help="Balance between image loss and latent loss")

    # pre-crop options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')    

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_samples_obj", type=int, default=3,
                        help='number of samples per ray and object')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--use_shadows", action='store_true',
                        help='use pose of an object to predict shadow opacity')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--multires_obj", type=int, default=4,
                        help='log2 of max freq for positional encoding (3D object location + heading)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--use_time", action='store_true',
                        help='time parameter for nerf baseline version')
    parser.add_argument("--remove_frame", type=int, default=-1,
                        help="Remove the ith frame from the training set")
    parser.add_argument("--remove_obj", type=int, default=None,
                        help="Option to remove all pixels of an object from the training")

    # render flags
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')

    parser.add_argument("--manipulate", type=str, default=None,
                        help='Renderonly manipulation argument')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels / vkitti')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--training_factor", type=int, default=0,
                        help='downsample factor for all images during training')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # vkitti/kitti flags
    parser.add_argument("--first_frame", type=str, default=0,
                        help='specifies the beginning of a sequence if not the complete scene is taken as Input')
    parser.add_argument("--last_frame", type=str, default=None,
                        help='specifies the end of a sequence')
    parser.add_argument("--use_object_properties", action='store_true',
                        help='use pose and properties of visible objects as an Input')
    parser.add_argument("--object_setting", type=int, default=0,
                        help='specify which properties are used')
    parser.add_argument("--max_input_objects", type=int, default=20,
                        help='Max number of object poses considered by the network, will be set automatically')
    parser.add_argument("--scene_objects", type=list,
                        help='List of all objects in the trained sequence')
    parser.add_argument("--scene_classes", type=list,
                        help='List of all unique classes in the trained sequence')
    parser.add_argument("--obj_opaque", action='store_true',
                        help='Ray does stop after intersecting with the first object bbox if true')
    parser.add_argument("--single_obj", type=float, default=None,
                        help='Specify for sequential training.')
    parser.add_argument("--box_scale", type=float, default=1.0,
                        help="Maximum scale for boxes to include shadows")
    parser.add_argument("--plane_type", type=str, default='uniform',
                        help='specifies how the planes are sampled')
    parser.add_argument("--near_plane", type=float, default=0.5,
                        help='specifies the distance from the last pose to the far plane')
    parser.add_argument("--far_plane", type=float, default=150.,
                        help='specifies the distance from the last pose to the far plane')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=1000,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')

    # Object Detection through rendering
    parser.add_argument("--obj_detection", action='store_true',
                        help='Debug local')
    parser.add_argument("--frame_number", type=int, default=0,
                        help='Frame of the datadir which should be detected')

    # Local Debugging
    parser.add_argument("--debug_local", action='store_true',
                        help='Debug local')

    return parser
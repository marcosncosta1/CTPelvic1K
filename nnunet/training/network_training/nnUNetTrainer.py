import matplotlib
from nnunet.training.network_training.network_trainer import NetworkTrainer
from nnunet.network_architecture.neural_network import SegmentationNetwork
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.utilities.nd_softmax import softmax_helper
import torch
import numpy as np
from nnunet.utilities.tensor_utilities import sum_tensor
from torch.optim import lr_scheduler
from nnunet.training.dataloading.dataset_loading import load_dataset, DataLoader3D, DataLoader2D, unpack_dataset, DataLoader3D_oversampleJoint
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss, DC_and_CE_Exclusion_loss, DC_and_CE_Exclusion_loss_DeepS
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from torch import nn
from nnunet.training.data_augmentation.default_data_augmentation import default_3D_augmentation_params, \
    default_2D_augmentation_params, get_default_augmentation, get_patch_size
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
from nnunet.evaluation.evaluator import aggregate_scores
from multiprocessing import Pool
from nnunet.evaluation.metrics import ConfusionMatrix

matplotlib.use("agg")
from collections import OrderedDict

class nnUNetTrainer(NetworkTrainer):
    """
    self.__init__:            set some self arguments as a spare
    self.initialize:          init the trainer
        self.load_plans_file:
        self.process_plans:    set trainer information, but not init network architecture
        self.setup_DA_params:  set dataaug informations

        if traiing:
            self.get_basic_generators: load dataset, do split, and return dataloader
            (unpack_dataset to npy)
            (package dataloader with augmentation)

        self.initialize_network_optimizer_and_scheduler: network, optimizer(Adam),lr_scheduler.ReduceLROnPlateau
                                                            and set "self.network.inference_apply_nonlin = softmax_helper"

    self.run_training: patient class method
        empty cuda cache
        plot_network_architecture(pass)
        cudnn.benchmark and cudnn.deterministic !=True
        if not self.was_initialized:

        run_interation is mainbody of this method. This method mainly process some loss results and save checkpoint


    """
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, network_dims='3d'):
        """
        :param deterministic:
        :param fold: can be either
            [0 ... 5) for cross-validation,
            'all' to train on all available training data or
            None if you wish to load some checkpoint and do inference only

        :param plans_file: the pkl file generated by preprocessing. This file will determine all design choices
        :param subfolder_with_preprocessed_data:
            must be a subfolder of dataset_directory (just the name of the folder, not the entire path).
            This is where the preprocessed data lies that will be used for network training.
            We made this explicitly available so that differently preprocessed data can coexist and the user can choose what to use.
            Can be None if you are doing inference only.

        :param output_folder: where to store parameters, plot progress and to the validation
        :param dataset_directory:
            the parent directory in which the preprocessed Task data is stored. This is required because the split information is stored in this directory.
            For running prediction only this input is not required and may be set to None

        :param batch_dice: compute dice loss for each sample and average over all samples in the batch or pretend the
        batch is a pseudo volume?
        :param stage: The plans file may contain several stages (used for lowres / highres / pyramid). Stage must be
        specified for training:
        if stage 1 exists then stage 1 is the high resolution stage, otherwise it's 0
        :param unpack_data: if False, npz preprocessed data will not be unpacked to npy. This consumes less space but
            is considerably slower!
            Running unpack_data=False with 2d should never be done! ### ?????? why ?????? should open files more times than 3D???

        IMPORTANT: If you inherit from nnunetTrainer and the init args change then you need to redefine self.init_args
        in your init accordingly. Otherwise checkpoints won't load properly!
        """
        super(nnUNetTrainer, self).__init__(deterministic, fp16)
        self.unpack_data = unpack_data
        self.init_args = (plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic, fp16)
        self.stage = stage
        self.experiment_name = self.__class__.__name__
        self.plans_file = plans_file
        self.dataset_directory = dataset_directory
        self.output_folder = output_folder
        self.output_folder_base = self.output_folder
        self.fold = fold

        self.refresh_lowrespred = False
        ### exclusion loss
        self.exclusion_mark = True
        self.ex_rate = 2
        ### exclusion loss
        self.deep_supervision = True
        self.network_dims = network_dims
        if network_dims == '2d':
            self.deep_pool_kernal = [[2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]] # in plans_2D
        else:
            self.deep_pool_kernal = [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]  # in plans_3D
        self.DSweights = [1, 0.5, 0.25, 0.1] # weights for different feature level loss

        self.plans = None

        if self.dataset_directory is not None and isdir(self.dataset_directory):
            self.gt_niftis_folder = join(self.dataset_directory, "gt_segmentations")
        else:
            self.gt_niftis_folder = None

        self.folder_with_preprocessed_data = None

        # set in self.initialize()

        self.dl_tr = self.dl_val = None

        self.num_input_channels = \
            self.num_classes = \
            self.net_pool_per_axis = \
            self.patch_size = \
            self.batch_size = \
            self.threeD = \
            self.base_num_features = \
            self.intensity_properties = \
            self.normalization_schemes = \
            self.net_num_pool_op_kernel_sizes = \
            self.net_conv_kernel_sizes = None

        self.basic_generator_patch_size = \
            self.data_aug_params = \
            self.transpose_forward = \
            self.transpose_backward = None

        self.batch_dice = batch_dice

        if self.deep_supervision:
            if self.exclusion_mark:
                self.loss = DC_and_CE_Exclusion_loss_DeepS({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'weights': self.DSweights},
                                                           {'weights': self.DSweights},
                                                           rate=self.ex_rate, deepspool=self.deep_pool_kernal)
            else:
                raise NotImplementedError
        else:
            if self.exclusion_mark:
                self.loss = DC_and_CE_Exclusion_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {}, rate=self.ex_rate)
            else:
                self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'square': False}, {})



        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []

        self.classes = \
            self.do_dummy_2D_aug = \
            self.use_mask_for_norm = \
            self.only_keep_largest_connected_component = \
            self.min_region_size_per_class = \
            self.min_size_per_class = None

        self.inference_pad_border_mode = "constant"
        self.inference_pad_kwargs = {'constant_values': 0}

        self.update_fold(fold)

        self.pad_all_sides = None

        self.lr_scheduler_eps = 1e-3
        self.lr_scheduler_patience = 30
        self.initial_lr = 3e-4
        self.weight_decay = 3e-5

        self.oversample_foreground_percent = 0.33


    def update_fold(self, fold):
        """
        used to swap between folds for inference (ensemble of models from cross-validation)
        DO NOT USE DURING TRAINING AS THIS WILL NOT UPDATE THE DATASET SPLIT AND THE DATA AUGMENTATION GENERATORS
        :param fold:
        :return:
        """
        if fold is not None:
            if isinstance(fold, str):
                assert fold == "all", "if self.fold is a string then it must be \'all\'"
                if self.output_folder.endswith("%s" % str(self.fold)):
                    self.output_folder = self.output_folder_base
                self.output_folder = join(self.output_folder, "%s" % str(fold))
            else:
                if self.output_folder.endswith("fold_%s" % str(self.fold)):
                    self.output_folder = self.output_folder_base
                self.output_folder = join(self.output_folder, "fold_%s" % str(fold))
            self.fold = fold

    def setup_DA_params(self):
        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")

                self.data_aug_params["elastic_deform_alpha"] = default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            print('self.patch_size: ',self.patch_size)
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi,
                                                                 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params

        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))

            patch_size_for_spatialtransform = self.patch_size[1:]
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size,
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])

            patch_size_for_spatialtransform = self.patch_size

        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = patch_size_for_spatialtransform

        # when there are weight heatmap existing in npy data for CE loss map(before reducing(mean or sum) operation). False by default
        self.data_aug_params['move_last_few_data_chanels_to_data'] = False
        self.data_aug_params['last_few_data_channels'] = [-1]

    def initialize(self, training=True, force_load_plans=False):
        """
        For prediction of test cases just set training=False,
        this will prevent loading of training data and
        dataloader with augmentation & network & optimizer
        training batchgenerator initialization
        :param training:
        :return:
        """

        maybe_mkdir_p(self.output_folder)

        if force_load_plans or (self.plans is None):
            self.load_plans_file()
        self.process_plans(self.plans)

        self.setup_DA_params()

        self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] + "_stage%d" % self.stage)

        if training:
            self.dl_tr, self.dl_val = self.get_basic_generators()

            if self.unpack_data:
                self.print_to_log_file("unpacking dataset")
                unpack_dataset(self.folder_with_preprocessed_data)
                self.print_to_log_file("done")
            else:
                self.print_to_log_file(
                    "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                    "will wait all winter for your model to finish!")

            self.tr_gen, self.val_gen = get_default_augmentation(self.dl_tr,
                                                                 self.dl_val,
                                                                 self.data_aug_params['patch_size_for_spatialtransform'],
                                                                 self.data_aug_params)

            self.print_to_log_file("TRAINING KEYS:\n %s" % (str(list(self.dataset_tr.keys())[::20])), also_print_to_console=True)
            self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(list(self.dataset_val.keys())[::20])), also_print_to_console=True)
        else:
            pass

        self.initialize_network_optimizer_and_scheduler()
        self.was_initialized = True

    def initialize_network_optimizer_and_scheduler(self):
        """
        This is specific to the U-Net and must be adapted for other network architectures
        :return:
        """
        net_numpool = len(self.net_num_pool_op_kernel_sizes)

        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d
        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}

        if self.deep_supervision:
            self.network = Generic_UNet(self.num_input_channels,
                                        self.base_num_features,
                                        self.num_classes,
                                        net_numpool,
                                        2, 2, conv_op,
                                        norm_op, norm_op_kwargs,
                                        dropout_op, dropout_op_kwargs,
                                        net_nonlin, net_nonlin_kwargs,
                                        True, False, lambda x: x,
                                        # False, False, lambda x: x,   ### deep supervision
                                        InitWeights_He(1e-2),
                                        self.net_num_pool_op_kernel_sizes,
                                        self.net_conv_kernel_sizes,
                                        False, True, True)
        else:
            self.network = Generic_UNet(self.num_input_channels,
                                        self.base_num_features,
                                        self.num_classes,
                                        net_numpool,
                                        2, 2, conv_op,
                                        norm_op, norm_op_kwargs,
                                        dropout_op, dropout_op_kwargs,
                                        net_nonlin, net_nonlin_kwargs,
                                        # True, False, lambda x: x,
                                        False, False, lambda x: x,
                                        InitWeights_He(1e-2),
                                        self.net_num_pool_op_kernel_sizes,
                                        self.net_conv_kernel_sizes,
                                        False, True,
                                        True)

        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                          self.initial_lr,
                                          weight_decay=self.weight_decay,
                                          amsgrad=True)
        #self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2,
         #                                                  patience=self.lr_scheduler_patience,
          #                                                 verbose=True, threshold=self.lr_scheduler_eps,
           #                                                threshold_mode="abs")

        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2,
                                                            patience=self.lr_scheduler_patience,
                                                            threshold=self.lr_scheduler_eps,
                                                            threshold_mode="abs")
        self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def load_plans_file(self):
        """
        This is what actually configures the entire experiment. The plans file is generated by experiment planning
        :return:
        """
        self.plans = load_pickle(self.plans_file)

    def process_plans(self, plans):
        """
        These arguments start with None, being set in this func when init the trainer.
        :param plans:
        :return:
        """
        if self.stage is None:
            assert len(list(plans['plans_per_stage'].keys())) == 1, \
                "If self.stage is None then there can be only one stage in the plans file. That seems to not be the " \
                "case. Please specify which stage of the cascade must be trained"
            self.stage = list(plans['plans_per_stage'].keys())[0]
        self.plans = plans

        stage_plans = self.plans['plans_per_stage'][self.stage]
        self.batch_size = stage_plans['batch_size']
        self.net_pool_per_axis = stage_plans['num_pool_per_axis']
        self.patch_size = np.array(stage_plans['patch_size']).astype(int)
        self.do_dummy_2D_aug = stage_plans['do_dummy_2D_data_aug']
        self.net_num_pool_op_kernel_sizes = stage_plans['pool_op_kernel_sizes'] #[[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        self.net_conv_kernel_sizes = stage_plans['conv_kernel_sizes']

        self.pad_all_sides = None  # self.patch_size
        self.intensity_properties = plans['dataset_properties']['intensityproperties']
        self.normalization_schemes = plans['normalization_schemes']
        self.base_num_features = plans['base_num_features']

        self.num_input_channels = 15 if self.network_dims=='2d' else plans['num_modalities']
        print(f"self.num_input_channels: {self.num_input_channels}. (we modify 2d model to 2.5d model)")

        self.num_classes = plans['num_classes'] + 1  # background is no longer in num_classes
        self.classes = plans['all_classes']
        self.use_mask_for_norm = plans['use_mask_for_norm']
        self.only_keep_largest_connected_component = plans['keep_only_largest_region']
        self.min_region_size_per_class = plans['min_region_size_per_class']
        self.min_size_per_class = None  # DONT USE THIS. plans['min_size_per_class']

        if plans.get('transpose_forward') is None or plans.get('transpose_backward') is None:
            print("WARNING! You seem to have data that was preprocessed with a previous version of nnU-Net. You should rerun preprocessing. "
                  "We will proceed and assume that both transpose_foward and transpose_backward are [0, 1, 2]. If that is not correct then weird things will happen!")
            plans['transpose_forward'] = [0, 1, 2]
            plans['transpose_backward'] = [0, 1, 2]
        self.transpose_forward = plans['transpose_forward']
        self.transpose_backward = plans['transpose_backward']

        if len(self.patch_size) == 2:
            self.threeD = False
        elif len(self.patch_size) == 3:
            self.threeD = True
        else:
            raise RuntimeError("invalid patch size in plans file: %s" % str(self.patch_size))

    def load_dataset(self):
        self.dataset = load_dataset(self.folder_with_preprocessed_data)

    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()

        if self.threeD:
            dl_tr = DataLoader3D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 False, oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides)
            dl_val = DataLoader3D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
                                  False, oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides)
        else:
            dl_tr = DataLoader2D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 transpose=None,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 pseudo_3d_slices=self.num_input_channels,
                                 pad_mode="constant",
                                 pad_sides=self.pad_all_sides)
            dl_val = DataLoader2D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
                                  transpose=None,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pseudo_3d_slices=self.num_input_channels,
                                  pad_mode="constant",
                                  pad_sides=self.pad_all_sides)
        return dl_tr, dl_val

    """
    run training
    """
    def plot_network_architecture(self):
        try:
            from batchgenerators.utilities.file_and_folder_operations import join
            import hiddenlayer as hl
            g = hl.build_graph(self.network, torch.rand((1, self.num_input_channels, *self.patch_size)).cuda(),
                               transforms=None)
            g.save(join(self.output_folder, "network_architecture.pdf"))
            del g
        except Exception as e:
            self.print_to_log_file("Unable to plot network architecture:")
            self.print_to_log_file(e)
        finally:
            torch.cuda.empty_cache()

    def run_training(self):
        dct = OrderedDict()
        for k in self.__dir__():
            if not k.startswith("__"):
                if not callable(getattr(self, k)):
                    dct[k] = str(getattr(self, k))
        del dct['plans']
        del dct['intensity_properties']
        del dct['dataset']
        del dct['dataset_tr']
        del dct['dataset_val']
        save_json(dct, join(self.output_folder, "debug.json"))

        import shutil
        shutil.copy(self.plans_file, join(self.output_folder_base, "plans.pkl"))

        super(nnUNetTrainer, self).run_training()

    def save_checkpoint(self, fname, save_optimizer=True):
        super(nnUNetTrainer, self).save_checkpoint(fname, save_optimizer)
        info = OrderedDict()
        info['init'] = self.init_args
        info['name'] = self.__class__.__name__
        info['class'] = str(self.__class__)
        info['plans'] = self.plans

        write_pickle(info, fname + ".pkl")

    def finish_online_evaluation(self):
        self.online_eval_tp = np.sum(self.online_eval_tp, 0)
        self.online_eval_fp = np.sum(self.online_eval_fp, 0)
        self.online_eval_fn = np.sum(self.online_eval_fn, 0)

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                           zip(self.online_eval_tp, self.online_eval_fp, self.online_eval_fn)]
                               if not np.isnan(i)]
        self.all_val_eval_metrics.append(np.mean(global_dc_per_class))

        self.print_to_log_file("Val glob dc per class:", str(global_dc_per_class))

        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []

    def run_online_evaluation(self, output, target):
        with torch.no_grad():
            num_classes = output.shape[1]
            output_softmax = softmax_helper(output)
            output_seg = output_softmax.argmax(1)
            target = target[:, 0]
            axes = tuple(range(1, len(target.shape)))
            tp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fn_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            for c in range(1, num_classes):
                tp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target == c).float(), axes=axes)
                fp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target != c).float(), axes=axes)
                fn_hard[:, c - 1] = sum_tensor((output_seg != c).float() * (target == c).float(), axes=axes)

            tp_hard = tp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fp_hard = fp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fn_hard = fn_hard.sum(0, keepdim=False).detach().cpu().numpy()

            self.online_eval_foreground_dc.append(list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)))
            self.online_eval_tp.append(list(tp_hard))
            self.online_eval_fp.append(list(fp_hard))
            self.online_eval_fn.append(list(fn_hard))

    """
    Validation
    """
    def preprocess_predict_nifti(self, input_files, output_file=None, softmax_ouput_file=None):
        """
        Use this to predict new data
        :param input_files:
        :param output_file:
        :param softmax_ouput_file:
        :return:
        """
        d, s, properties = self.preprocess_patient(input_files)
        pred = self.predict_preprocessed_data_return_softmax(d, self.data_aug_params["mirror"], 1, False, 1,
                                                             self.data_aug_params['mirror_axes'], True, True, 2,
                                                             self.patch_size, True)
        pred = pred.transpose([0] + [i + 1 for i in self.transpose_backward])

        save_segmentation_nifti_from_softmax(pred, output_file, properties, 3, None, None, None, softmax_ouput_file,
                                             None)

    def preprocess_patient(self, input_files):
        """
        Used to predict new unseen data. Not used for the preprocessing of the training/test data
        :param input_files:
        :return:
        """
        from nnunet.preprocessing.preprocessing import GenericPreprocessor, PreprocessorFor2D
        if self.threeD:
            preprocessor = GenericPreprocessor(self.normalization_schemes, self.use_mask_for_norm,
                                               self.transpose_forward, self.intensity_properties)
        else:
            preprocessor = PreprocessorFor2D(self.normalization_schemes, self.use_mask_for_norm,
                                             self.transpose_forward, self.intensity_properties)

        d, s, properties = preprocessor.preprocess_test_case(input_files,
                                                             self.plans['plans_per_stage'][self.stage]['current_spacing'])
        return d, s, properties

    def predict_preprocessed_data_return_softmax(self, data, do_mirroring, num_repeats, use_train_mode, batch_size,
                                                 mirror_axes, tiled, tile_in_z, step, min_size, use_gaussian):
        """
        Don't use this. If you need softmax output, use preprocess_predict_nifti and set softmax_output_file. ### todo:???
        :param data:
        :param do_mirroring:
        :param num_repeats:
        :param use_train_mode:
        :param batch_size:
        :param mirror_axes:
        :param tiled:
        :param tile_in_z:
        :param step:
        :param min_size:
        :param use_gaussian:
        :param use_temporal:
        :return:
        """
        assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        return self.network.predict_3D(data, do_mirroring, num_repeats, use_train_mode, batch_size, mirror_axes,
                                       tiled, tile_in_z, step, min_size, use_gaussian=use_gaussian,
                                       pad_border_mode=self.inference_pad_border_mode,
                                       pad_kwargs=self.inference_pad_kwargs)[2]

    def validate(self, do_mirroring=True, use_train_mode=False, tiled=True, step=2, save_softmax=True,
                 use_gaussian=True, compute_global_dice=True, override=True, validation_folder_name='validation'):
        """
        2018_12_05: I added global accumulation of TP, FP and FN for the validation in here. This is because I believe
        that selecting models is easier when computing the Dice globally instead of independently for each case and
        then averaging over cases. The Lung dataset in particular is very unstable because of the small size of the
        Lung Lesions.
        My theory is that even though the global Dice is different than the acutal target metric it is still a good
        enough substitute that allows us to get a lot more stable results when rerunning the same experiment twice.

        FYI: computer vision community uses the global jaccard for the evaluation of Cityscapes etc, not the per-image
        jaccard averaged over images.

        The reason I am accumulating TP/FP/FN here and not from the nifti files (which are used by our Evaluator) is
        that all predictions made here will have identical voxel spacing whereas voxel spacings in the nifti files
        will be different (which we could compensate for by using the volume per voxel but that would require the
        evaluator to understand spacings which is does not at this point)

        :param do_mirroring:
        :param use_train_mode:
        :param mirror_axes:
        :param tiled:
        :param tile_in_z:
        :param step:
        :param use_nifti:
        :param save_softmax:
        :param use_gaussian:
        :param use_temporal_models:
        :return:
        """
        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        if self.dataset_val is None:
            self.load_dataset()
            self.do_split()

        output_folder = join(self.output_folder, validation_folder_name)  ### output_folder: fold_2/
        maybe_mkdir_p(output_folder)

        if do_mirroring:
            mirror_axes = self.data_aug_params['mirror_axes'] ### 默认是没有的，所以你开也白开。。。。。。。。。
        else:
            mirror_axes = ()

        pred_gt_tuples = []

        export_pool = Pool(4)
        results = []
        global_tp = OrderedDict()
        global_fp = OrderedDict()
        global_fn = OrderedDict()

        for k in self.dataset_val.keys():
            properties = self.dataset[k]['properties']
            fname = properties['list_of_data_files'][0].split("/")[-1][:-12]
            if override or (not isfile(join(output_folder, fname + ".nii.gz"))):
                try:
                    data = np.load(self.dataset[k]['data_file'])['data']
                except Exception as _:
                    data = np.load(self.dataset[k]['data_file'].replace('.npz','.npy'))


                print(k, data.shape)
                data[-1][data[-1] == -1] = 0

                softmax_pred = self.predict_preprocessed_data_return_softmax(data[:-1], do_mirroring, 1, use_train_mode, 1,
                                                                             mirror_axes, tiled, True, step,
                                                                             self.patch_size, use_gaussian=use_gaussian)

                if compute_global_dice:
                    predicted_segmentation = softmax_pred.argmax(0)
                    gt_segmentation = data[-1]
                    labels = properties['classes']
                    labels = [int(i) for i in labels if i > 0]
                    for l in labels:
                        if l not in global_fn.keys():
                            global_fn[l] = 0
                        if l not in global_fp.keys():
                            global_fp[l] = 0
                        if l not in global_tp.keys():
                            global_tp[l] = 0
                        conf = ConfusionMatrix((predicted_segmentation == l).astype(int),
                                               (gt_segmentation == l).astype(int))
                        conf.compute()
                        global_fn[l] += conf.fn
                        global_fp[l] += conf.fp
                        global_tp[l] += conf.tp

                softmax_pred = softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])

                if save_softmax:
                    softmax_fname = join(output_folder, fname + ".npz")
                else:
                    softmax_fname = None

                """There is a problem with python process communication that prevents us from communicating obejcts 
                larger than 2 GB between processes (basically when the length of the pickle string that will be sent is 
                communicated by the multiprocessing.Pipe object then the placeholder (%i I think) does not allow for long 
                enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually 
                patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will 
                then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either 
                filename or np.ndarray and will handle this automatically"""
                if np.prod(softmax_pred.shape) > (2e9 / 4 * 0.9):  # *0.9 just to be save
                    np.save(join(output_folder, fname + ".npy"), softmax_pred)
                    softmax_pred = join(output_folder, fname + ".npy")
                results.append(export_pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                         ((softmax_pred, join(output_folder, fname + ".nii.gz"),
                                                           properties, 3, None, None, None, softmax_fname, None),
                                                          )
                                                         )
                               )

            pred_gt_tuples.append([join(output_folder, fname + ".nii.gz"),
                                   join(self.gt_niftis_folder, fname + ".nii.gz")])

        _ = [i.get() for i in results]
        print("finished prediction, now evaluating...")

        task = self.dataset_directory.split("/")[-1]
        job_name = self.experiment_name
        _ = aggregate_scores(pred_gt_tuples, labels=list(range(self.num_classes)),
                             json_output_file=join(output_folder, "summary.json"),
                             json_name=job_name + " val tiled %s" % (str(tiled)),
                             json_author="Fabian",
                             json_task=task, num_threads=3)
        if compute_global_dice:
            global_dice = OrderedDict()
            all_labels = list(global_fn.keys())
            for l in all_labels:
                global_dice[int(l)] = float(2 * global_tp[l] / (2 * global_tp[l] + global_fn[l] + global_fp[l]))
            write_json(global_dice, join(output_folder, "global_dice.json"))







import os
import shutil

class models_genesis_config:
    model = 'Vnet'
    suffix = 'genesis_chest_ct'
    exp_name = model + '-' + suffix
    
    # data
    variety = 3
    datasets=['luna16', 'ctpa', 'dsb17', 'lits17', 'lndb19', 'kits19', 'rsnastr20']
    luna16_data = '/mnt/dataset/shared/zongwei/LUNA16'
    ctpa_data = '/mnt/dataset/Lung/PE/PE_detection_2016/dicom'
    dsb17_data = '/mnt/dataset/shared/zongwei/dsb2017/dsb2017-lung-data/stage1-mnt'
    lits17_data = '/mnt/dataset/shared/zongwei/LiTS/Tr'
    lndb19_data = '/mnt/dataset/shared/zongwei/LNDb'
    kits19_data = '/mnt/dataset/shared/zongwei/kits19'
    rsnastr20_data = '/mnt/dataset/shared/zguo32/rsna-pe-detection'

    hu_min = -1000.0
    hu_max = 1000.0
    num_subvol_per_patient = 6
    hu_thred = (-150.0 - hu_min) / (hu_max - hu_min)
    lung_max = 0.15
    len_vision = 3
    input_rows = 64
    input_cols = 64 
    input_deps = 32
    crop_rows_min, crop_rows_max = 32, 128
    crop_cols_min, crop_cols_max = 32, 128
    crop_deps_min, crop_deps_max = 16, 64
    nb_class = 1
    sample_png_rate = 0.01
    
    # model pre-training
    verbose = 1
    weights = None
    batch_size = 12
    optimizer = 'sgd'
    workers = 8
    max_queue_size = workers * 8
    save_samples = 'png'
    nb_epoch = 10000
    patience = 50
    lr = 0.1
    steps_per_epoch = 200
    validation_steps = 200

    # image deformation
    nonlinear_rate = 0.8
    paint_rate = 0.9
    outpaint_rate = 0.8
    inpaint_rate = 1.0 - outpaint_rate
    local_rate = 0.0
    flip_rate = 0.4
    
    def __init__(self, args):
        if args.data is not None:
            self.data = args.data
            self.luna16_data = os.path.join(self.data, 'luna16')
            self.ctpa_data = os.path.join(self.data, 'ctpa')
            self.dsb17_data = os.path.join(self.data, 'dsb17')
            self.lits17_data = os.path.join(self.data, 'lits17')
            self.lndb19_data = os.path.join(self.data, 'lndb19')
            self.kits19_data = os.path.join(self.data, 'kits19')
            self.rsnastr20_data = os.path.join(self.data, 'rsnastr20')
        if args.weights is not None and args.weights != 'None':
            self.weights = args.weights

        # logs
        self.model_path = 'pretrained_weights'
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.logs_path = os.path.join(self.model_path, 'Logs')
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)
        self.sample_path = 'pair_samples'
        if not os.path.exists(self.sample_path):
            os.makedirs(self.sample_path)
        shutil.rmtree(os.path.join(self.sample_path, self.exp_name), ignore_errors=True)
        if not os.path.exists(os.path.join(self.sample_path, self.exp_name)):
            os.makedirs(os.path.join(self.sample_path, self.exp_name))
    
    def display(self):
        '''Display Configuration values.'''
        print('\nConfigurations:')
        for a in dir(self):
            if not a.startswith('__') and not callable(getattr(self, a)):
                print('{:30} {}'.format(a, getattr(self, a)))
        print('\n')

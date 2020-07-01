% 20200428 by Dushan N. Wadduwage
% edit this file to initiate parameters for the dabba_mu object

function pram = pram_init()

  pram.is_imgTranslator_on = 1;
  pram.is_imgProcessor_on  = 0;
  
%% Engelward lab tissue data
%   pram.dir_dataroot_0       = '/home/harvard/Dropbox (Harvard University)/WorkingData/20200426_Bevin_Nuc/Nuclei Counter (20x, 40x)/20x Nuclei Counter/20200609_20x_annotation_hand/';
%   pram.dir_imds_gt_0        = [pram.dir_dataroot_0 'Originals'];  
%   pram.dir_imds_exp_0       = [pram.dir_dataroot_0 'Originals'];
%   pram.dir_pxds_gt_hand_0   = [pram.dir_dataroot_0 'Pxd_hand'];  
%   pram.dir_pxds_gt_algo_0   = [pram.dir_dataroot_0 'Pxd_algo'];  
%   pram.ds_imread_extensions = {'.tif','.png'}
%   pram.ds_imread_channedls    = 1;
%   pram.ds_imread_read_method  = 'rescale10k';   % {'bfopen_3D2maxproj-rescale10k','log','mean-histeq','mean-zerocenter','max-zerocenter','rescale10k-zerocenter','rescale10k','rescale1k','max','mean','none'} 
%   pram.exp_type               = 'h2ax_adv';         % {'h2ax_direct','h2ax_adv','h2ax'}

%% Seeber lab cell data
  pram.dir_dataroot_0       = '/home/harvard/Dropbox (Harvard University)/WorkingData/20200210_Andrew_3D_foci/';
  pram.dir_imds_gt_0        = [pram.dir_dataroot_0 '63xZseriesSmall'];  
  pram.dir_imds_exp_0       = [pram.dir_dataroot_0 '20xZseriesSmall'];     %'40xZseriesSmallWF_dapi'
  pram.ds_imread_extensions = {'.stk'};
  pram.max_tr_size          = 5e4;
  pram.ds_imread_varTh      = .5e-3; % varience threshold used to ommit empty images
  pram.ds_imread_augment    = 0;  % {0=no image augmentations ,1=do image augmentation}
  pram.ds_imread_channedls  = 1;
  pram.ds_imread_chText     = '_w3';                                % used to select the images of a specific channel in dir
  pram.ds_imread_read_method= 'bfopen_3D2maxproj-max-zerocenter';   % {'bfopen_3D2maxproj-rescale10k','log','mean-histeq','mean-zerocenter','max-zerocenter','rescale10k-zerocenter','rescale10k','rescale1k','max','mean','none'} 
  pram.exp_type             = 'h2ax';                               % {'h2ax_direct','h2ax_adv','h2ax'}
  
%%  
  pram.dir_dataroot   = './_DATA/Tr_data/';
  pram.dir_imds_gt    = [pram.dir_dataroot 'Originals'];  
  pram.dir_imds_exp   = [pram.dir_dataroot 'Originals']; 
  pram.dir_pxds_gt    = [pram.dir_dataroot 'pxd'];  

  pram.rsf          = 63/20;
  pram.Nx           = 128;
  pram.Nc           = 1;
  pram.N_classes    = 3;                                                % # segmented classes
  pram.classNames   = ["bg","fg","w_bg"];
  pram.pxLblIds       = [0 1 2];

  pram.executionEnvironment   = 'auto';
  pram.gammaMse               = 0.01;
  pram.gammaCyc               = 1;
  pram.gammaId                = 0.01;
  
  pram.numEpochs                  = 10;      
  pram.miniBatchSize              = 48;
  pram.learnRateDropFactor        = 0.1;
  pram.learnRateDropInterval      = 10;
  
  pram.learnRateImgprocessor      = 0.0002;% 0.0002;
  pram.learnRateDiscriminator     = 0.0001;% 0.0001;    
  pram.learnRate_encoder          = 0.0002;% 0.0002;
  pram.learnRate_decoder          = 0.0002;% 0.0002;

  pram.gradientDecayFactor        = 0.5;
  pram.squaredGradientDecayFactor = 0.999;

  pram.trailingAvgDiscriminator   = [];
  pram.trailingAvgSqDiscriminator = [];    
  pram.trailingAvgImgprocessor    = [];
  pram.trailingAvgSqImgprocessor  = [];

  pram.trailingAvg_D_I            = [];
  pram.trailingAvgSq_D_I          = [];
  pram.trailingAvg_D_J            = [];
  pram.trailingAvgSq_D_J          = [];
  pram.trailingAvg_encoder        = [];
  pram.trailingAvgSq_encoder      = [];
  pram.trailingAvg_decoder        = [];
  pram.trailingAvgSq_decoder      = [];

end





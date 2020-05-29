extPath = '/home/harvard/Dropbox (Harvard University)/WorkingData/2019-12-29_DEEP_TFM_Cheng/qd_50_/';
extNameStem = 'qd_50_';

dataPath = '/home/harvard/Dropbox (Harvard University)/WorkingData/2019-12-29_DEEP_TFM_Cheng/beads_0_5__50_/';
dataNameStem = 'beads_50_';

crop_rect_name = 'crop_rect_20191229_50_fgbg64.mat';
crop_rect_name_ex = 'crop_rect_20191229_50_fgbg64.mat';
EmGain = 100;

Nx = 128;
Ny = 128;
Nz = 1;
ny = 1;                         % demag block size, s.t. Y image size is (Nx/nx,Ny,ny)
nx = 1;

sl_ex = [];                     % [um] excitation scattering length (excluded in this model)
sl_em = 62;                     % [um] emission scattering length 
                                    
Nt_mx = 32;
dx = .75;                       % [um] TF image's pixel size 
z = 7*sl_em;

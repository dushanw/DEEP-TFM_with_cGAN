% 2021-04-26 by Duahan N. Wadduwage

clc;clear all;close all
addpath('./_extPatternsets/')

N_sls       = 2;
pram        = f_pram_init();
pram.Nt     = 32;
pram.pattern_typ  = 'dmd_exp_tfm_beads_7sls_20201219';
pram.dataset      = 'beads'; 

[E Y_exp X_refs pram] = f_get_extPettern(pram);
% pram.maxcount     = 5;

%% on-gpu-only
pram.z0_um  = - pram.sl * N_sls;
pram.dz     = pram.dx;                              % [um], depth (z=0 is the surface and -ve is below)
pram.Nz     = 100;

% reset(gpuDevice(1));reset(gpuDevice(2));          % to avoid "Out of memory on device..." errors 
% PSFs        = f_simPSFs3D(pram);
switch N_sls
  case 2
    load('./_PSFs/PSFs17-Apr-2021 16:18:25.mat')    % z0 = -2 sls    
  case 4
    load('./_PSFs/PSFs26-Apr-2021 05:15:18.mat')    % z0 = -4 sls
  case 6
    load('./_PSFs/PSFs20-Apr-2021 05:40:28.mat')    % z0 = -6 sls
end

%N_beads     = round(pram.Nz * 100/64);             % new on 2021-04-16
N_beads     = round(pram.Nz * 100/256);             % new on 2021-04-16

X0          = f_genobj_beads3D_1um_4um(N_beads,pram);

load('./_emhist/emhist_beads_14-May-2021.mat');     % upt to 100 photons
[Yhat Xgt]  = f_fwd3D(X0,E,PSFs,emhist,pram);

DataIn      = single(gather(Yhat));
DataGt      = single(gather(Xgt )); 

dest_dir    = ['~/Documents/tempData/'];
file_name   = sprintf('DataIn_and_Gt_%dsls_%s.mat',N_sls,date);
save([dest_dir file_name],'DataIn','DataGt','pram');

file_name

%% on local
dest_dir    = ['~/Documents/tempData/'];
file_name   = 'DataIn_and_Gt_4sls_14-May-2021.mat';% copy filename from the remote host
system(['scp harvard@10.245.73.7:' dest_dir file_name ' ./xx_temp'])

load(['./xx_temp/' file_name])

% imagesc([mean(Y_exp.beads1,3) mean(DataIn(:,:,:,1),3) mean(DataGt(:,:,:,1),3)]);axis image
imagesc([rescale(mean(Y_exp.beads1,3)) rescale(mean(DataIn(:,:,:,1),3)) rescale(mean(DataGt(:,:,:,1),3))]);axis image

%% temp
dir_name    = './14-May-2021_beads_maxCount-5_Nbeads-39/';

II          = [];
for ii = [2 4 6]
  file_name   = sprintf('DataIn_and_Gt_%dsls_14-May-2021.mat',ii);% copy filename from the remote host
  load([dir_name file_name])

  II  = [II; [rescale(mean(Y_exp.beads1,3)) rescale(mean(DataIn(:,:,:,1),3)) rescale(mean(DataGt(:,:,:,1),3))]];
%   figure;
%   imagesc([mean(Y_exp.beads1,3) mean(DataIn(:,:,:,1),3) mean(DataGt(:,:,:,1),3)]);axis image;colorbar
  
%   figure;
%   imagesc([rescale(mean(Y_exp.beads1,3)) rescale(mean(DataIn(:,:,:,1),3)) rescale(mean(DataGt(:,:,:,1),3))]);axis image
%   title(replace([dir_name file_name],'_','-'))
%   saveas(gca,[dir_name file_name '.tif']);
end
imagesc(II);axis image;axis off
title(replace(dir_name,'_','-'))
saveas(gca,[dir_name 'all.tif']);
close all








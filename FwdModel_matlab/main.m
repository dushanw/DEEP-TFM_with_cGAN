% 20201122 by Dushan N. Wadduwage
% main forward model for cGAN-DEEP-TFM

clc;clear all;close all
addpath('./_extPatternsets/')

pram                  = f_pram_init();
[E Y_exp X_refs pram] = f_get_extPettern(pram);

%% simulate sPSF, exPSF, and emPSF
% PSFs = f_simPSFs(pram);
load('./_PSFs/PSFs27-Dec-2020 04_21_23.mat')    % load('./_PSFs/PSFs27-Dec-2020 04:21:23.mat')

%% simulate training data  
N_beads     = pram.Nz * 500/64;
% X0        = f_genobj_beads3D(pram.Ny,pram.Nx,pram.Nz,N_beads);
X0          = f_genobj_beads3D_1um_4um(N_beads,pram);
pram.Nz     = size(X0,3);
X0          = reshape(X0,[pram.Ny,pram.Nx,1,pram.Nz]);

tic
[Yhat Xgt]  = f_fwd(X0,E,PSFs,pram);
toc

%% FTS convolution <on test>
% % Y_conv = conv2(X0(:,:,1,50),PSFs.exPSF,'same'); % this is the same as what's implemented using FFT below 
% fft_Ny      = size(PSFs.exPSF,1);
% fft_Nx      = size(PSFs.exPSF,2);
% ifft_strt_y = round(fft_Ny/2);
% ifft_strt_x = round(fft_Nx/2);
% ifft_end_y  = ifft_strt_y + pram.Ny - 1;
% ifft_end_x  = ifft_strt_x + pram.Nx - 1;
% 
% fft2_sPSF_x_fft2_emPSF  = (fft2(PSFs.sPSF) .* fft2(PSFs.emPSF));
% 
% Z_conv = conv2(E(:,:,1).*X0(:,:,1,50),PSFs.sPSF,'same');
% Y_conv = conv2(Z_conv,PSFs.emPSF,'same');
% 
% Y_conv = conv2(E(:,:,1).*X0(:,:,1,50)*1e3,PSFs.emConvSPSF,'same');
% 
% 
% % fft2_sPSF_x_fft2_emPSF  = fft2(PSFs.sPSF);
% % Y_conv = conv2(E(:,:,1).*X0(:,:,1,50),PSFs.sPSF,'same');
% 
% Y    = ifft2(fft2(E.*X0,fft_Ny,fft_Nx) .* fft2_sPSF_x_fft2_emPSF);
% Y    = fftshift(fftshift(Y,1),2);
% Y    = Y(ifft_strt_y:ifft_end_y,ifft_strt_x:ifft_end_x,:,:);
% Y    = abs(Y);
% 
% figure;imagesc(imtile(Y(:,:,1,50)));axis image; colorbar
% 
% figure;imagesc([abs(Y(:,:,1,50)) Y_conv]);axis image; colorbar
% figure;imagesc([abs(Y(:,:,1,50)) - Y_conv]);axis image; colorbar
% % note: there's a small spatial shit in fft compared to conv2, see why?
% 
% 
% figure;imagesc([abs(Y(1:end-1,1:end-1,1,50)) - Y_conv(2:end,2:end)]);axis image; colorbar




% 20201122 by Dushan N. Wadduwage
% main forward model for cGAN-DEEP-TFM

clc;clear all;close all
addpath('./_extPatternsets/')

pram    = f_pram_init();
pram.Nt = 32;


[E Y_exp X_refs pram] = f_get_extPettern(pram);

%% simulate sPSF, exPSF, and emPSF
PSFs = f_simPSFs(pram);
% load('./_PSFs/PSFs27-Dec-2020 04_21_23.mat')        % on Macbook
% load('./_PSFs/PSFs04-Jan-2021 02:13:34.mat')        % on GPU

%% simulate training data  
Nmb   = 32;                                          % minibatch size is selected based on f_fwd's run time... 
                                                      % ~500 works ok on GPU.
tic
t = 1;
for j = 1:ceil(pram.Nb/Nmb) 
  j
  N_beads     = pram.Nz * 500/64;
  X0 = [];
  for i=1:round(Nmb/pram.Nz)
    X0_temp   = f_genobj_beads3D_1um_4um(N_beads,pram);  
    X0        = cat(4,X0,reshape(X0_temp,[pram.Ny,pram.Nx,1,size(X0_temp,3)]));
  end
  Nmb_t       = size(X0,4);
  
  [Yhat Xgt]  = f_fwd(X0,E,PSFs,pram);
  DataIn(:,:,:,t:t+Nmb_t-1) = gather(Yhat);
  DataGt(:,:,:,t:t+Nmb_t-1) = gather(Xgt); 
  t = t+Nmb_t;
end
toc
DataIn      = single(DataIn);
DataGt      = single(DataGt);

%% save simulation data
N_sls       = abs(pram.z0_um/pram.sl);
saveDir     = ['./_results/_cnn_synthTrData/' date '/' pram.pattern_typ '/']; 
nameStem    = sprintf('beads_data_%dsls_',N_sls);
mkdir(saveDir)

save([saveDir nameStem datestr(datetime('now')) '_pram_plusplus.mat'],'Y_exp','X_refs','pram','E','PSFs');

DataIn_tr    = DataIn(:,:,:,129:end);
DataGt_tr    = DataGt(:,:,:,129:end);
fileNameStem = [saveDir nameStem '_tr.h5'];
f_writeDataset_hdf5(fileNameStem,DataIn_tr,DataGt_tr);
% test read
% dataIp_rd = h5read(fileNameStem,'/input');
% dataGt_rd = h5read(fileNameStem,'/gt');

DataIn_test  = DataIn(:,:,:,1:128);
DataGt_test  = DataGt(:,:,:,1:128);
fileNameStem = [saveDir nameStem '_test.h5'];
f_writeDataset_hdf5(fileNameStem,DataIn_test,DataGt_test);

DataIn_real_beads1 = Y_exp.beads1;
DataGt_real_beads1 = X_refs.beads1_sf_wf0;
fileNameStem = [saveDir nameStem '_real_beads1.h5'];
f_writeDataset_hdf5(fileNameStem,DataIn_real_beads1,DataGt_real_beads1);

DataIn_real_beads2 = Y_exp.beads2;
DataGt_real_beads2 = X_refs.beads2_sf_wf0;
fileNameStem = [saveDir nameStem '_real_beads2.h5'];
f_writeDataset_hdf5(fileNameStem,DataIn_real_beads2,DataGt_real_beads2);


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

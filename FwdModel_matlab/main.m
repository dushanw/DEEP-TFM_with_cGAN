% 20201122 by Dushan N. Wadduwage
% main forward model for cGAN-DEEP-TFM

% copy from the gpu to RC
% ex: scp -r ./dmd_exp_tfm_beads_7sls_20201219 wadduwage@login.rc.fas.harvard.edu:/n/holyscratch01/wadduwage_lab
% zip: zip -r beads_2021-02-06.zip dmd_exp_tfm_beads_7sls_20201219

clc;clear all;close all
addpath('./_extPatternsets/')

pram    = f_pram_init();
pram.Nt = 32;
[E Y_exp X_refs pram] = f_get_extPettern(pram);

switch pram.sim2dOr3d
  case '2D'
    %% simulate sPSF, exPSF, and emPSF
    PSFs = f_simPSFs(pram);
    % load('./_PSFs/PSFs27-Dec-2020 04_21_23.mat')        % on Macbook
    % load('./_PSFs/PSFs04-Jan-2021 02:13:34.mat')        % on GPU

    %% simulate training data  
    N_beads     = pram.Nz * 1000/64;
    t = 1;
    clear X0
    for j = 1:ceil(pram.Nb/pram.Nz) 
      j
      X0_temp               = f_genobj_beads3D_1um_4um(N_beads,pram);  
      Nmb_t                 = size(X0_temp,3);
      X0(:,:,1,t:t+Nmb_t-1) = gather(X0_temp); 
      t = t+Nmb_t;
    end
    pram.Nb = size(X0,4);

    % minibatch size is selected based on f_fwd's run time. ~48 works ok on GPU.
    Nmb     = 48;                                        

    t = 1;
    clear DataIn DataGt
    DataIn = zeros(pram.Ny,pram.Nx,pram.Nt,floor(pram.Nb/Nmb)*Nmb,'single');
    DataGt = zeros(pram.Ny,pram.Nx,1      ,floor(pram.Nb/Nmb)*Nmb,'single');
    for j = 1:floor(pram.Nb/Nmb)
      tic

      [Yhat Xgt]  = f_fwd(X0(:,:,:,t:t+Nmb-1),E,PSFs,pram);
      DataIn(:,:,:,t:t+Nmb-1) = single(gather(Yhat));
      DataGt(:,:,:,t:t+Nmb-1) = single(gather(Xgt)); 
      t = t+Nmb;

      [j round(toc)]  
    end

%% save simulation data
N_sls       = abs(PSFs.pram.z0_um/PSFs.pram.sl);
saveDir     = ['./_results/_cnn_synthTrData/' date '/' pram.pattern_typ '/']; 
nameStem    = sprintf('beads_data_%dsls_',N_sls);
mkdir(saveDir)

save([saveDir nameStem datestr(datetime('now')) '_pram_plusplus.mat'],'Y_exp','X_refs','pram','E','PSFs');

% test read
% dataIp_rd = h5read(fileNameStem,'/input');
% dataGt_rd = h5read(fileNameStem,'/gt');

DataIn_test  = DataIn(:,:,:,1:128);
DataGt_test  = DataGt(:,:,:,1:128);
fileNameStem = [saveDir nameStem '_test.h5'];
f_writeDataset_hdf5(fileNameStem,DataIn_test,DataGt_test);

DataIn       = DataIn(:,:,:,129:end);
DataGt       = DataGt(:,:,:,129:end);
fileNameStem = [saveDir nameStem '_tr.h5'];
f_writeDataset_hdf5(fileNameStem,DataIn,DataGt);

DataIn_real_beads1 = Y_exp.beads1;
DataGt_real_beads1 = X_refs.beads1_sf_wf0;
fileNameStem = [saveDir nameStem '_real_beads1.h5'];
f_writeDataset_hdf5(fileNameStem,DataIn_real_beads1,DataGt_real_beads1);

DataIn_real_beads2 = Y_exp.beads2;
DataGt_real_beads2 = X_refs.beads2_sf_wf0;
fileNameStem = [saveDir nameStem '_real_beads2.h5'];
f_writeDataset_hdf5(fileNameStem,DataIn_real_beads2,DataGt_real_beads2);
  case '3D'
    %% simulate sPSF, exPSF, and emPSF
    PSFs = f_simPSFs3D(pram);
            
    %% simulate training data  
    N_beads     = round(pram.Nz * 1000/64);
    N_beads     = round(pram.Nz * 100/64);% new on 2021-04-16
    t = 1;
    clear X0
    for j = 1:ceil(pram.Nb/pram.Nz)
      j
      X0_temp               = f_genobj_beads3D_1um_4um(N_beads,pram);
      Nmb_t                 = size(X0_temp,3);
      X0(:,:,1,t:t+Nmb_t-1) = gather(X0_temp); 
      t = t+Nmb_t;
    end
    pram.Nb = size(X0,4);

    % minibatch size is selected based on f_fwd's run time. ~48 works ok on GPU.
    Nmb     = 48;                                        

    t = 1;
    clear DataIn DataGt
    DataIn = zeros(pram.Ny,pram.Nx,pram.Nt,floor(pram.Nb/Nmb)*Nmb,'single');
    DataGt = zeros(pram.Ny,pram.Nx,1      ,floor(pram.Nb/Nmb)*Nmb,'single');
    for j = 1:floor(pram.Nb/Nmb)
      tic

      [Yhat Xgt]  = f_fwd3D(X0(:,:,:,t:t+Nmb-1),E,PSFs,pram);
      DataIn(:,:,:,t:t+Nmb-1) = single(gather(Yhat));
      DataGt(:,:,:,t:t+Nmb-1) = single(gather(Xgt)); 
      t = t+Nmb;

      [j round(toc)]  
    end

end


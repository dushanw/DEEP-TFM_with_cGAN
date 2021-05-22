% 20201122 by Dushan N. Wadduwage
% main forward model for cGAN-DEEP-TFM

% copy from the gpu to RC
% ex: scp -r ./dmd_exp_tfm_beads_7sls_20201219 wadduwage@login.rc.fas.harvard.edu:/n/holyscratch01/wadduwage_lab
% zip: zip -r beads_2021-02-06.zip dmd_exp_tfm_beads_7sls_20201219

clc;clear all;close all
addpath('./_extPatternsets/')
% load('../__manuscript/_figures/_inferno-tables')

pram                  = f_pram_init();
pram.pattern_typ      = 'dmd_exp_tfm_mouse_20201224_200um';   % pram.maxcount = {10.8549 [for 400um],
                                                              %                  25.0622 [for 350um],
                                                              %                  22.9540 [for 300um],
                                                              %                  46.7010 [for 200um],
                                                              %                  56.0317 [for 100um],
                                                              %                  25.0622 [for sf   ],
pram.dataset          = 'mouse_200um';
pram.Nt               = 32;

[E Y_exp X_refs pram] = f_get_extPettern(pram);
pram.maxcount

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
  case '3D'
    switch pram.dataset
      case 'beads'
        pram.dz = pram.dx;
        pram.Nb = 5000;
        %% simulate sPSF, exPSF, and emPSF
        % reset(gpuDevice(1));reset(gpuDevice(2));          % to avoid "Out of memory on device..." errors 
        % pram.z0_um      = -6*pram.sl;                     % [um]      depth (z=0 is the surface and -ve is below)
        % PSFs = f_simPSFs3D(pram);
        % load('./_PSFs/PSFs17-Apr-2021 16:18:25.mat')      % z0 = -2 sls
        % load('./_PSFs/PSFs26-Apr-2021 05:15:18.mat')      % z0 = -4 sls  
          load('./_PSFs/PSFs20-Apr-2021 05:40:28.mat')      % z0 = -6 sls
        
        %% load emhist for camera noise model  
        load('./_emhist/emhist_beads_14-May-2021.mat');  % upt to  15 photons  
          
        %% simulate training data  
        %N_beads     = round(pram.Nz * 1000/64);            % used in 2D case

        pram.Nz     = 100; 
        N_beads     = round(pram.Nz * 100/64);              % new on 2021-04-16

        clear DataIn DataGt
        DataIn = zeros(pram.Ny,pram.Nx,pram.Nt,pram.Nb,'single');
        DataGt = zeros(pram.Ny,pram.Nx,1      ,pram.Nb,'single');          

        X0                        = f_genobj_beads3D_1um_4um(N_beads,pram);
        [Yhat Xgt]                = f_fwd3D(X0,E,PSFs,emhist,pram);
        Nmb                       = size(Xgt,4);

        tic
        t   = 1;
        for j = 1:floor(pram.Nb/Nmb)
          X0                      = f_genobj_beads3D_1um_4um(N_beads,pram);
         [Yhat Xgt]               = f_fwd3D(X0,E,PSFs,emhist,pram);

          Nmb                     = size(Xgt,4);
          DataIn(:,:,:,t:t+Nmb-1) = single(gather(Yhat));
          DataGt(:,:,:,t:t+Nmb-1) = single(gather(Xgt )); 

          fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b')
          fprintf('\nbatch = %0.4d | b = %0.5d | time = %0.4d/%0.4d [mins]',...
                               j,          t,  round(toc/60), round(pram.Nb*toc/(t*60)))                         
          t = t+Nmb;
        end

        DataIn                    = DataIn(:,:,:,1:t-1);
        DataGt                    = DataGt(:,:,:,1:t-1);
      case 'mouse_200um'
        % pram.dz = pram.dx;
        pram.Nb = 5000;
        %% simulate sPSF, exPSF, and emPSF        
        % reset(gpuDevice(1));reset(gpuDevice(2));          % to avoid "Out of memory on device..." errors 
        % PSFs = f_simPSFs3D(pram);
        % load('./_PSFs/PSFs17-Apr-2021 16:18:25.mat')      % z0 = -2 sls (for 256x256 data size)
        % load('./_PSFs/PSFs02-May-2021 22:52:41.mat')      % z0 = -6 sls (for 326x326 data size - i.e. neuron/brain-vasculature data PSFs are 1305×1305×67 of size)
        load('./_PSFs/PSFs04-May-2021 09:52:47.mat')        % z0 = -4 sls (for 326x326 data size - i.e. neuron/brain-vasculature data PSFs are 1305×1305×67 of size)
                        
        %% load emhist for camera noise model  
        load('./_emhist/emhist_29-Apr-2021 02:09:25.mat');  % upt to 100 photons
        
        %% simulate training data
        pram.Nz     = 100;

        clear DataIn DataGt
        DataIn = zeros(pram.Ny,pram.Nx,pram.Nt,pram.Nb,'single');
        DataGt = zeros(pram.Ny,pram.Nx,1      ,pram.Nb,'single');          
          
        % read all cells     
        load('./_datasets/PS_SOM_mice_20190317.mat')        % loads variable named Data
        Nb_per_stack                = 20;                   % rough value of training images generated by scanning one stack, set as 10 in the f_fwd3d
        pram.Npch_perCell           = round(pram.Nb/(length(Data.cell)*Nb_per_stack));
        
        tic
        t   = 1;
        for i=1:length(Data.cell)
          V_ps                      = Data.cell{i};
          X0s                       = f_genobj_neuronPatches(V_ps,pram);
          % figure;imagesc(max(X0s{1},[],3));axis image;colorbar
          
          for j = 1:length(X0s)
           [Yhat Xgt]               = f_fwd3D(X0s{j},E,PSFs,emhist,pram);

            Nmb                     = size(Xgt,4);
            DataIn(:,:,:,t:t+Nmb-1) = single(gather(Yhat));
            DataGt(:,:,:,t:t+Nmb-1) = single(gather(Xgt ));            
            
            fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b')
            fprintf('\nbatch = %0.4d | b = %0.5d | time = %0.4d/%0.4d [mins]',...
                                 j,          t,  round(toc/60), round(pram.Nb*toc/(t*60)))                         
            t = t+Nmb;
          end
        end
        DataIn                    = DataIn(:,:,:,1:t-1);
        DataGt                    = DataGt(:,:,:,1:t-1);
    end
end

%% save simulation data
N_sls       = round(abs(PSFs.pram.z0_um/PSFs.pram.sl));
saveDir     = ['./_results/_cnn_synthTrData/' date '/' pram.pattern_typ '/'];
nameStem    = sprintf('%s_data_%dsls_',pram.dataset,N_sls);
mkdir(saveDir)

save([saveDir nameStem datestr(datetime('now')) '_pram_plusplus.mat'],'Y_exp','X_refs','pram','E','PSFs','-v7.3');

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

%% %% for mouse 200um
DataIn_anml1_r1_200um = Y_exp.anml1_r1_200um; 
DataGt_anml1_r1_200um = X_refs.anml1_r1_200um_avg0;
fileNameStem = [saveDir nameStem '_anml1_r1_200um.h5'];
f_writeDataset_hdf5(fileNameStem,DataIn_anml1_r1_200um,DataGt_anml1_r1_200um);

DataIn_anml1_r2_200um = Y_exp.anml1_r2_200um; 
DataGt_anml1_r2_200um = X_refs.anml1_r2_200um_avg0;
fileNameStem = [saveDir nameStem '_anml1_r2_200um.h5'];
f_writeDataset_hdf5(fileNameStem,DataIn_anml1_r2_200um,DataGt_anml1_r2_200um);

DataIn_anml2_r1_200um = Y_exp.anml2_r1_200um;
DataGt_anml2_r1_200um = X_refs.anml2_r1_200um_avg0;
fileNameStem = [saveDir nameStem '_anml2_r1_200um.h5'];
f_writeDataset_hdf5(fileNameStem,DataIn_anml2_r1_200um,DataGt_anml2_r1_200um);

%% for mouse 300um
DataIn_anml1_r1_300um = Y_exp.anml1_r1_300um; 
DataGt_anml1_r1_300um = X_refs.anml1_r1_300um_avg0;
fileNameStem = [saveDir nameStem '_anml1_r1_300um.h5'];
f_writeDataset_hdf5(fileNameStem,DataIn_anml1_r1_300um,DataGt_anml1_r1_300um);

DataIn_anml1_r2_300um = Y_exp.anml1_r2_300um; 
DataGt_anml1_r2_300um = X_refs.anml1_r2_300um_avg0;
fileNameStem = [saveDir nameStem '_anml1_r2_300um.h5'];
f_writeDataset_hdf5(fileNameStem,DataIn_anml1_r2_300um,DataGt_anml1_r2_300um);

DataIn_anml2_r1_300um = Y_exp.anml2_r1_300um;
DataGt_anml2_r1_300um = X_refs.anml2_r1_300um_avg0;
fileNameStem = [saveDir nameStem '_anml2_r1_300um.h5'];
f_writeDataset_hdf5(fileNameStem,DataIn_anml2_r1_300um,DataGt_anml2_r1_300um);

DataIn_anml2_r2_300um = Y_exp.anml2_r2_300um;
DataGt_anml2_r2_300um = X_refs.anml2_r2_300um_avg0;
fileNameStem = [saveDir nameStem '_anml2_r2_300um.h5'];
f_writeDataset_hdf5(fileNameStem,DataIn_anml2_r2_300um,DataGt_anml2_r2_300um);

%% for beads
DataIn_real_beads1 = Y_exp.beads1;
DataGt_real_beads1 = X_refs.beads1_sf_wf0;
fileNameStem = [saveDir nameStem '_real_beads1.h5'];
f_writeDataset_hdf5(fileNameStem,DataIn_real_beads1,DataGt_real_beads1);

DataIn_real_beads2 = Y_exp.beads2;
DataGt_real_beads2 = X_refs.beads2_sf_wf0;
fileNameStem = [saveDir nameStem '_real_beads2.h5'];
f_writeDataset_hdf5(fileNameStem,DataIn_real_beads2,DataGt_real_beads2);

%% temp section
max_input_photons = max(poissrnd(pram.maxcount,[1 1000]))*2;
N_reps            = pram.cam_emhist_Nreps;
emhist            = f_genEmhist(max_input_photons,N_reps,pram);
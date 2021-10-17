% 20201122 by Dushan N. Wadduwage
% main forward model for cGAN-DEEP-TFM

clc;clear all;close all
addpath('./_extPatternsets/')
% load('../__manuscript/_figures/_inferno-tables')

%% set parameters
pram                  = f_pram_init();
%pram.pattern_typ      = 'dmd_exp_tfm_beads_7sls_20201219';    %the experimental beads
pram.pattern_typ      = 'dmd_exp_tfm_mouse_20201224_100um';  % the experimental BV

                                                                % maxcounts from the calibration processes.
                                                             % pram.maxcount = {10.8549 [for mouse_400um], 8-sls
                                                             %                  25.0622 [for mouse_350um], 7-sls
                                                             %                  22.9540 [for mouse_300um], 6-sls
                                                             %                  46.7010 [for mouse_200um], 4-sls
                                                             %                  56.0317 [for mouse_100um], 2-sls
                                                             %                  25.0622 [for mouse_sf   ], 0-sls
pram.dataset          = 'mouse_bv_100um';     % select the case  
pram.Nt               = 32;     %number og patterns
pram.Nb               = 500;   %Batch dimension
%pram.dx = 1.625;
%pram.Nx = 66;
%pram.Ny = 66;
%% load data
[E Y_exp X_refs pram] = f_get_extPettern(pram);
%pram.maxcount        = 20;    %have to set maxcount for beads data  %for BV can use experimental max count. 
                                                             % overwrite the maxcount from the calibration as the 
                                                             % calibration seems off. We need to comeback and deal with 
                                                             % it if the fixed value doesnt work for all cases <2021-05-24>.

%% simulate sPSF, exPSF, and emPSF        
pram.z0_um          = -2*pram.sl;                          % [um] depth (z=0 is the surface and -ve is below), set for beads
reset(gpuDevice(1));reset(gpuDevice(2));                   % to avoid "Out of memory on device..." errors 
%PSFs = f_simPSFs3D(pram); %can comment and use presaved PSFs                                 % simuated and saved in ./_PSFs/ dir as
                                                             %   PSFs23-May-2021 02:57:32.mat - z0_um:  4.4450   (i.e. surface)
                                                             %   PSFs23-May-2021 14:46:23.mat - z0_um: -94.5550  (i.e. 2-sls, for mouse_100um)
                                                             %   PSFs04-May-2021 09:52:47.mat - z0_um: -194.5550 (i.e. 4-sls, for mouse_200um)
                                                             %   PSFs02-May-2021 22:52:41.mat - z0_um: -294.5550 (i.e. 6-sls, for mouse_300um)

%% simulate for camera noise model (generate emhist)  
% max_input_photons = max(poissrnd(pram.maxcount,[1 1000]))*2;
% N_reps            = pram.cam_emhist_Nreps;
% emhist            = f_genEmhist(max_input_photons,N_reps,pram);
                                                             % simulated and saved in ./_emhist dir as,
                                                             %   emhist_29-Apr-2021 02:09:25.mat (upt to 100 photons)  

%% run simulation
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
        pram.dz = pram.dx; % <try withouht this so dz = 1>
        %% load sPSF, exPSF, and emPSF <try -2sls and -4sls and -6sls if -5sls didnt work>
        % load('./_PSFs/PSFs17-Apr-2021 16:18:25.mat')      % z0 = -2 sls 
        % load('./_PSFs/PSFs26-Apr-2021 05:15:18.mat')      % z0 = -4 sls  
          load('./_PSFs/PSFs25-May-2021 01:34:24.mat')      % z0 = -5 sls
        % load('./_PSFs/PSFs20-Apr-2021 05:40:28.mat')      % z0 = -6 sls
        
        % load emhist (for camera noise model)
        load('./_emhist/emhist_29-Apr-2021 02:09:25.mat');  % upt to 100 photons
    
        %% simulate training data
        pram.Nz     = 100;
        %N_beads    = round(pram.Nz * 1000/64);            % used in 2D case
        N_beads     = round(pram.Nz * 100/64);             % new on 2021-04-16

        clear DataIn DataGt
        DataIn      = zeros(pram.Ny,pram.Nx,pram.Nt,pram.Nb,'single');
        DataGt      = zeros(pram.Ny,pram.Nx,1      ,pram.Nb,'single');          
        
        X0          = f_genobj_beads3D_1um_4um(N_beads,pram);
        %fprintf(size(X0))
        [Yhat Xgt]  = f_fwd3D(X0,E,PSFs,emhist,pram);
        Nmb         = size(Xgt,4);  %

        tic;
        t   = 1;
        for j = 1:floor(pram.Nb/Nmb)
          disp(j)
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
        
      case 'mouse_Synbv_100um'
        % load sPSF, exPSF, and emPSF
        load('./_PSFs/PSFs23-May-2021 14:46:23.mat')        % z0 = -2 sls (for 326x326 data size - i.e. neuron/brain-vasculature data PSFs are 1305×1305×67 of size)             
        % load emhist (for camera noise model)  
        load('./_emhist/emhist_29-Apr-2021 02:09:25.mat');  % upt to 100 photons
        disp(pram.maxcount)       
        clear DataIn DataGt
        DataIn = zeros(pram.Ny,pram.Nx,pram.Nt,pram.Nb,'single');
        DataGt = zeros(pram.Ny,pram.Nx,1      ,pram.Nb,'single');
          
        tic
        t   = 1;
        dataPath_root = './_datasets/SynVesSAP';            % data root
        files = dir([dataPath_root '*/*.mat']);
        %disp(length(SynBloodVes))
        for j=1:length(files)
          fname = fullfile(files(j).folder,files(j).name);
          load(fname);
          for i=1:length(SynBloodVes)
            X0s                        = single(SynBloodVes{i});
            [Yhat Xgt]               = f_fwd3D(X0s,E,PSFs,emhist,pram);
       
            Nmb                     = size(Xgt,4);
            DataIn(:,:,:,t:t+Nmb-1) = single(gather(Yhat));
            DataGt(:,:,:,t:t+Nmb-1) = single(gather(Xgt ));            

            fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b')
            fprintf('b = %0.5d | time = %0.4d/%0.4d [mins]',...
                                           t,  round(toc/60), round(pram.Nb*toc/(t*60)))                         
            t = t+Nmb;
          end
        end 
        DataIn                    = DataIn(:,:,:,1:t-1);
        DataGt                    = DataGt(:,:,:,1:t-1);
        
      case 'mouse_Synbv_300um'
        % load sPSF, exPSF, and emPSF
        load('./_PSFs/PSFs02-May-2021 22:52:41.mat')    % z0= -6 sls (300um)               
        % load emhist (for camera noise model)  
        load('./_emhist/emhist_29-Apr-2021 02:09:25.mat');  % upt to 100 photons
        disp(pram.maxcount)       
        clear DataIn DataGt
        DataIn = zeros(pram.Ny,pram.Nx,pram.Nt,pram.Nb,'single');
        DataGt = zeros(pram.Ny,pram.Nx,1      ,pram.Nb,'single');

        tic
        t   = 1;
        dataPath_root = './_datasets/SynVesSAP';            % data root
        files = dir([dataPath_root '*/*.mat']);
        %disp(length(SynBloodVes))
        for j=1:length(files)
          fname = fullfile(files(j).folder,files(j).name);
          load(fname);
          for i=1:length(SynBloodVes)
            X0s                        = single(SynBloodVes{i});
            [Yhat Xgt]               = f_fwd3D(X0s,E,PSFs,emhist,pram);
       
            Nmb                     = size(Xgt,4);
            DataIn(:,:,:,t:t+Nmb-1) = single(gather(Yhat));
            DataGt(:,:,:,t:t+Nmb-1) = single(gather(Xgt ));            

            fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b')
            fprintf('b = %0.5d | time = %0.4d/%0.4d [mins]',...
                                           t,  round(toc/60), round(pram.Nb*toc/(t*60)))                         
            t = t+Nmb;
          end
        end 
        DataIn                    = DataIn(:,:,:,1:t-1);
        DataGt                    = DataGt(:,:,:,1:t-1);
        %%
      case 'mouse_VesSAPbv_100um'
        % load sPSF, exPSF, and emPSF
        load('./_PSFs/PSFs23-May-2021 14:46:23.mat')        % z0 = -2 sls (for 326x326 data size - i.e. neuron/brain-vasculature data PSFs are 1305×1305×67 of size)
                        
        % load emhist (for camera noise model)  
        load('./_emhist/emhist_29-Apr-2021 02:09:25.mat');  % upt to 100 photons
        
        clear DataIn DataGt
        DataIn = zeros(pram.Ny,pram.Nx,pram.Nt,pram.Nb,'single');
        DataGt = zeros(pram.Ny,pram.Nx,1      ,pram.Nb,'single');
          
        % read all cells
        %load('./_datasets/VesSAP_mouse_3Dvol_27062021.mat')        % loads variable named Data
        load('./_datasets/VesSAP_mouse_3Dvol_08072021.mat')
        tic
        t   = 1;
        disp(length(VesSAPBV))
        for i=1:length(VesSAPBV)
          for j=1:length(VesSAPBV{i})
              X0s                        = single(VesSAPBV{i}{j});

              [Yhat Xgt]               = f_fwd3D(X0s,E,PSFs,emhist,pram);

              Nmb                     = size(Xgt,4);
              DataIn(:,:,:,t:t+Nmb-1) = single(gather(Yhat));
              DataGt(:,:,:,t:t+Nmb-1) = single(gather(Xgt ));            

              fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b')
              fprintf('\nbatch = %0.4d |b = %0.5d | time = %0.4d/%0.4d [mins]',...
                                  j,           t,  round(toc/60), round(pram.Nb*toc/(t*60)))                         
              t = t+Nmb;
          end
        end 
        DataIn                    = DataIn(:,:,:,1:t-1);
        DataGt                    = DataGt(:,:,:,1:t-1);
        
      case 'mouse_neuronal_100um'
        % load sPSF, exPSF, and emPSF
        load('./_PSFs/PSFs23-May-2021 14:46:23.mat')        % z0 = -2 sls (for 326x326 data size - i.e. neuron/brain-vasculature data PSFs are 1305×1305×67 of size)
                        
        % load emhist (for camera noise model)  
        load('./_emhist/emhist_29-Apr-2021 02:09:25.mat');  % upt to 100 photons
        
        clear DataIn DataGt
        DataIn = zeros(pram.Ny,pram.Nx,pram.Nt,pram.Nb,'single');
        DataGt = zeros(pram.Ny,pram.Nx,1      ,pram.Nb,'single');
          
        % read all cells
        load('./_datasets/PS_SOM_mice_20190317.mat')        % loads variable named Data
        pram.Nz                     = 100;                  % sets Nz of the data cube
        Nb_per_stack                = 20;                   % rough value of training images generated by scanning one stack, set as 10 in the f_fwd3d
        pram.Npch_perCell           = round(pram.Nb/(length(Data.cell)*Nb_per_stack));
        
        tic
        t   = 1;
        for i=1:length(Data.cell)
          V_ps0                      = Data.cell{i};
          V_ps = imresize(V_ps0,Data.pram.dx_ps/pram.dx);
          
          X0s                       = f_genobj_neuronPatches(V_ps,pram);
          
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
        
      case 'mouse_neuronal_300um'
        % load sPSF, exPSF, and emPSF
        load('./_PSFs/PSFs02-May-2021 22:52:41.mat')    % z0= -6 sls (300um)
                        
        % load emhist (for camera noise model)  
        load('./_emhist/emhist_29-Apr-2021 02:09:25.mat');  % upt to 100 photons
        
        clear DataIn DataGt
        DataIn = zeros(pram.Ny,pram.Nx,pram.Nt,pram.Nb,'single');
        DataGt = zeros(pram.Ny,pram.Nx,1      ,pram.Nb,'single');
          
        % read all cells
        load('./_datasets/PS_SOM_mice_20190317.mat')        % loads variable named Data
        pram.Nz                     = 100;                  % sets Nz of the data cube
        Nb_per_stack                = 20;                   % rough value of training images generated by scanning one stack, set as 10 in the f_fwd3d
        pram.Npch_perCell           = round(pram.Nb/(length(Data.cell)*Nb_per_stack));
        
        tic
        t   = 1;
        for i=1:length(Data.cell)
          V_ps0                     = Data.cell{i};
          V_ps                      = imresize(V_ps0,pram.Data.dx_ps/pram.dx);
          X0s                       = f_genobj_neuronPatches(V_ps,pram);
          
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
    case 'mouse_bv_100um'
        % load sPSF, exPSF, and emPSF
        load('./_PSFs/PSFs23-May-2021 14:46:23.mat')        % z0 = -2 sls (for 326x326 data size - i.e. neuron/brain-vasculature data PSFs are 1305×1305×67 of size)
                        
        % load emhist (for camera noise model)  
        load('./_emhist/emhist_29-Apr-2021 02:09:25.mat');  % upt to 100 photons
        
        clear DataIn DataGt
        DataIn = zeros(pram.Ny,pram.Nx,pram.Nt,pram.Nb,'single');
        DataGt = zeros(pram.Ny,pram.Nx,1      ,pram.Nb,'single');
          
        % read all cells
        %load('./_datasets/VesSAP_mouse_3Dvol_27062021.mat')        % loads variable named Data
        load('./_datasets/BV_03102021.mat')

        tic
        t   = 1;
        %disp(length(VesSAPBV))
        V_ps0                   = single(Data.cell);
        Vps                     = imresize(V_ps0,Data.pram.dx/pram.dx);
        numlist = [1:round(size(Vps,1)/8):size(Vps,1)-size(E,1)];
        for i = 1:length(numlist)
            for j = 1:length(numlist)
                X0s = Vps(numlist(i):numlist(i)+size(E,1)-1,numlist(j):numlist(j)+size(E,1)-1,:);
                [Yhat Xgt]              = f_fwd3D(X0s,E,PSFs,emhist,pram);

                Nmb                     = size(Xgt,4);
                DataIn(:,:,:,t:t+Nmb-1) = single(gather(Yhat));
                DataGt(:,:,:,t:t+Nmb-1) = single(gather(Xgt ));            

                fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b')
                fprintf('b = %0.5d | time = %0.4d/%0.4d [mins]',...
                                               t,  round(toc/60), round(pram.Nb*toc/(t*60)))                         
                t = t+Nmb;
            end
        end 
        
        DataIn                    = DataIn(:,:,:,1:t-1);
        DataGt                    = DataGt(:,:,:,1:t-1);
    end    
end

%% save simulated data
N_sls         = round(abs(PSFs.pram.z0_um/PSFs.pram.sl));
saveDir       = ['./_results/_cnn_synthTrData/' date '/' pram.pattern_typ '/'];
% saveDir       = ['./_results/_cnn_synthTrData/' date '_maxcountsCalib/' pram.pattern_typ '/'];
nameStem      = sprintf('%s_data_%dsls_%dmc',pram.dataset,N_sls,pram.maxcount);
mkdir(saveDir)
save([saveDir nameStem datestr(datetime('now')) '_pram_plusplus.mat'],'Y_exp','X_refs','pram','E','PSFs','-v7.3');

% test read
% dataIp_rd   = h5read(fileNameStem,'/input');
% dataGt_rd   = h5read(fileNameStem,'/gt');

DataIn_test   = DataIn(:,:,:,1:128);
DataGt_test   = DataGt(:,:,:,1:128);
fileNameStem  = [saveDir nameStem '_test.h5'];
f_writeDataset_hdf5(fileNameStem,DataIn_test,DataGt_test);

DataIn        = DataIn(:,:,:,129:end);
DataGt        = DataGt(:,:,:,129:end);
fileNameStem  = [saveDir nameStem '_tr.h5'];
f_writeDataset_hdf5(fileNameStem,DataIn,DataGt);

fields_Yexp   = fieldnames(Y_exp);
fields_Xrefs  = fieldnames(X_refs);
for i=1:length(fields_Yexp)  
  DataIn_exp  = Y_exp. (fields_Yexp {i}); 
  DataGt_exp  = X_refs.(fields_Xrefs{i});
  fileNameStem= [saveDir nameStem fields_Yexp{i} '_exp.h5'];
  f_writeDataset_hdf5(fileNameStem,DataIn_exp,DataGt_exp);
end



% next to do is to go through and modify for the 4th domention of X0 in puts


function [Yhat Xgt] = f_fwd3D(X0,E,PSFs,emhist,pram)
  % Input dimentionality
  %   X0          - [y,x,z]
  %   E           - [y,x,t=Nt]
  %   PSFs.xxPSF  - [y,x,z]

  % Dimentionality used within the function (here t = patterns,b = instances) 
  %   X0          - [y,x,z  ,t=1 ,b  ]
  %   E           - [y,x,z=1,t=Nt,b=1]
  %   PSFs.xxPSF  - [y,x,z  ,t=1 ,b=1]
  %   Y0          - [y,x,z=1,t=Nt,b  ]
  
  % Output dimentionality
  %   Yhat        - [y,x,t=Nt,b]
  %   Xgt         - [y,x,t=1 ,b]
    
  %% preprocess inputs (for size, resolution, and dimentionality, and gpu-use)
  % Nz_X0   = size(X0,3);
%   exPSF   =single(imresize3(PSFs.exPSF,PSFs.pram.dx/pram.dx,'Antialiasing',true));% same as the below
%   emPSF   = single(imresize3(PSFs.emPSF,PSFs.pram.dx/pram.dx,'Antialiasing',true));
%   sPSF    = single(imresize3(PSFs.sPSF ,PSFs.pram.dx/pram.dx,'Antialiasing',true));  
  rsf_x     = PSFs.pram.dx/pram.dx;
  rsf_y     = rsf_x;
  rsf_z     = PSFs.pram.dx/pram.dz;  
  exPSF     = single(imresize3(PSFs.exPSF,'Scale',[rsf_y,rsf_x,rsf_z],'Method','linear','Antialiasing',true));
  emPSF     = single(imresize3(PSFs.emPSF,'Scale',[rsf_y,rsf_x,rsf_z],'Method','linear','Antialiasing',true));
  sPSF      = single(imresize3(PSFs.sPSF ,'Scale',[rsf_y,rsf_x,rsf_z],'Method','linear','Antialiasing',true));
  
  E         = reshape(E,[size(E,1),size(E,2),1,size(E,3)]);
  E_gt      = single(ones(pram.Ny,pram.Nx,1,1));
  %X0       = reshape(X0,[size(X0,1),size(X0,2),size(X0,3),1,size(X0,4)]);
  if pram.useGPU ==1
    X0      = gpuArray(X0);
    E       = gpuArray(E);
    emPSF   = gpuArray(emPSF);
    exPSF   = gpuArray(exPSF);
    sPSF    = gpuArray(sPSF);
  end
  X0        = padarray(X0,round([(size(exPSF,1)-pram.Ny)/2 (size(exPSF,2)-pram.Nx)/2 0 0 0]),0,'both');
  X0        = X0(1:size(exPSF,1),1:size(exPSF,2),:,:,:);
  
  %% fwd model - equation: Em(x,y,z,t) = emPSF **3d {sPSF **2d [(exPSF **3d E).*X0]}, here **=conv
  Eex_3D    = f_conv3nd(exPSF,E,'same');
  
  vol_Nz    = size(exPSF,3);
  vol_inits = [1:5:size(X0,3)-vol_Nz];
  % vol_inits = round(linspace(1,size(X0,3)-vol_Nz-1,10));
  
  Nb        = length(vol_inits);
  meanX0    = mean(X0(:));
  b_t       = 1;
  for b = 1:Nb
    X0_vol  = X0(:,:,vol_inits(b):vol_inits(b)+vol_Nz-1);
    if mean(X0_vol(:))>meanX0
      vol_inits_valid(b_t) = vol_inits(b);
      b_t   = b_t+1;
    end
  end
  vol_inits = vol_inits_valid;
  Nb        = b_t - 1;
  
  for b = 1:Nb
    % b
    X0_vol  = X0(:,:,vol_inits(b):vol_inits(b)+vol_Nz-1);
    X0_vol  = imrotate(X0_vol,90*rem(b,4));
    X_ex    = Eex_3D .* X0_vol;
    
    for j=1:size(X_ex,4)                              % using the loop in the function is slow for some reason
      X_sctterd(:,:,:,j) = f_conv2nd(sPSF ,X_ex(:,:,:,j),'same');
    end
  
    for j=1:size(X_ex,4)                              % using the loop in the function is slow for some reason
      X_em(:,:,:,j)      = f_conv3nd(emPSF,X_sctterd(:,:,:,j),'same');
    end
    
    %% gt image (as in PS-TPM in the absence of scattering)
    Xgt_3D    = f_conv3nd(exPSF,X0_vol,'same');
  
    %% postprocess (cropping)
    y_range   = round(size(X_em,1)/2 - pram.Nx/2)+1:round(size(X_em,1)/2 + pram.Nx/2);
    x_range   = round(size(X_em,2)/2 - pram.Nx/2)+1:round(size(X_em,2)/2 + pram.Nx/2);

%   Eex_3D    = Eex_3D   (y_range,x_range,:,:,:);
%   X0_vol    = X0_vol   (y_range,x_range,:,:,:);
%   Xgt_3D    = Xgt_3D   (y_range,x_range,:,:,:);
%   X_ex      = X_ex     (y_range,x_range,:,:,:);
%   X_sctterd = X_sctterd(y_range,x_range,:,:,:);
%   X_em      = X_em     (y_range,x_range,:,:,:);

    %% images (Xgt,Y0)
    Xgt       = Xgt_3D(y_range,x_range,ceil(size(X_em,3)/2),:,:);
    Y0        = X_em  (y_range,x_range,ceil(size(X_em,3)/2),:,:);
    
    %% match experimental counts (refer to f_get_extPettern and f_read_data on the original data folder)
    % Note: look up why there's a scaling factor 5? we removed this part
    % Y0        = double(5*pram.maxcount*Y0/max(Y0(:)));
    % Xgt       = 5*Xgt./max(Xgt(:));    
    Y0        = double(pram.maxcount*Y0/max(Y0(:)));
    Xgt       = Xgt./max(Xgt(:));
    
    Y0_all  (:,:,:,:,b) = Y0;
    Xgt_all (:,:,:,:,b) = Xgt;
  end
  
  %% load emccd noise distributions for Yhat    
  % max_input_photons = max(poissrnd(max(Y0(:)),[1 1000]))*2;
  % N_reps            = pram.cam_emhist_Nreps;
  % emhist            = f_genEmhist(max_input_photons,N_reps,pram);
%   try
%     load('./_emhist/emhist_03-Jan-2021 07_55_31.mat');% upt to  15 photons
%   catch
%     load('./_emhist/emhist_03-Jan-2021 07:55:31.mat');% upt to  15 photons
%     load('./_emhist/emhist_29-Apr-2021 02:09:25.mat');% upt to 100 photons
%   end
  [Yhat_all YhatADU]= f_simulateIm_emCCD(Y0_all,emhist,pram);
  
  %% change dims for output
  Xgt       = reshape(Xgt_all ,[pram.Ny pram.Nx 1       Nb]);
  Yhat      = reshape(Yhat_all,[pram.Ny pram.Nx pram.Nt Nb]);
end

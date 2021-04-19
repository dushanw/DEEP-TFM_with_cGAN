% next to do is to go through and modify for the 4th domention of X0 in puts


function [Yhat Xgt] = f_fwd3D(X0,E,PSFs,pram)
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
  
  %% load emccd noise distributions for Yhat  
  try
    load('./_emhist/emhist_03-Jan-2021 07_55_31.mat') 
  catch
    load('./_emhist/emhist_03-Jan-2021 07:55:31.mat')
  end
  
  %% preprocess inputs (for size, resolution, and dimentionality, and gpu-use)
  %Nz_X0     = size(X0,3);
  exPSF     = single(imresize3(PSFs.exPSF,PSFs.pram.dx/pram.dx,'Antialiasing',true));
  emPSF     = single(imresize3(PSFs.emPSF,PSFs.pram.dx/pram.dx,'Antialiasing',true));
  sPSF      = single(imresize3(PSFs.sPSF ,PSFs.pram.dx/pram.dx,'Antialiasing',true));
  E         = reshape(E,[size(E,1),size(E,2),1,size(E,3)]);
  %X0        = reshape(X0,[size(X0,1),size(X0,2),size(X0,3),1,size(X0,4)]);
  if pram.useGPU ==1
    % X0      = gpuArray(X0);
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
  Nb        = length(vol_inits);
  for b = 1:Nb
    b
    X0_vol  = gpuArray(X0(:,:,vol_inits(b):vol_inits(b)+vol_Nz-1));
    X_ex    = Eex_3D .* X0_vol;
    
    for j=1:size(X_ex,4)                              % using the loop in the function is slow for some reason
      X_sctterd(:,:,:,j) = f_conv2nd(sPSF ,X_ex(:,:,:,j),'same');
    end
  
    for j=1:size(X_ex,4)                              % using the loop in the function is slow for some reason
      X_em(:,:,:,j)      = f_conv3nd(emPSF,X_sctterd(:,:,:,j),'same');
    end
    
    %% gt image (as in the absence of scattering)
    Xgt_3d    = f_conv3nd(exPSF,X0,'same');
  
    %% postprocess (cropping)
    y_range   = round(size(X_em,1)/2 - pram.Nx/2)+1:round(size(X_em,1)/2 + pram.Nx/2);
    x_range   = round(size(X_em,2)/2 - pram.Nx/2)+1:round(size(X_em,2)/2 + pram.Nx/2);    

%   Eex_3D    = Eex_3D   (y_range,x_range,:,:,:);
%   X0        = X0       (y_range,x_range,:,:,:);
%   Xgt       = Xgt      (y_range,x_range,:,:,:);
%   X_ex      = X_ex     (y_range,x_range,:,:,:);
%   X_sctterd = X_sctterd(y_range,x_range,:,:,:);
%   X_em      = X_em     (y_range,x_range,:,:,:);

    %% images (Xgt,Y0)
    Xgt       = Xgt_3d(y_range,x_range,ceil(size(X_em,3)/2),:,:);
    Y0        = X_em  (y_range,x_range,ceil(size(X_em,3)/2),:,:);
    
    %% backup code with dim(X0) = [Ny Nx Nz 1 Nb]  
%   X_ex      = Eex_3D .* X0;
%   
%   % X_sctterd = f_conv2nd(sPSF,X_ex,'same');        % this is slower than looping outside the function as below  
%   tic
%   for i=1:size(X_ex,5)
%     for j=1:size(X_ex,4)                              % using the loop in the function is slow for some reason
%       X_sctterd(:,:,:,j,i) = f_conv2nd(sPSF ,X_ex(:,:,:,j,i),'same');
%     end
%   end
%   toc
%   % X_em      = f_conv3nd(emPSF,X_sctterd,'same');  % this is slower than looping outside the function as below
%   for i=1:size(X_ex,5)  
%     for j=1:size(X_ex,4)                              % using the loop in the function is slow for some reason
%       X_em(:,:,:,j,i)      = f_conv3nd(emPSF,X_sctterd(:,:,:,j,i),'same');
%     end
%   end
%   %% gt image (as in the absence of scattering)
%   Xgt       = f_conv3nd(exPSF,X0,'same');
%   
%   %% postprocess (cropping)
%   y_range   = round(size(X_em,1)/2 - pram.Nx/2)+1:round(size(X_em,1)/2 + pram.Nx/2);
%   x_range   = round(size(X_em,2)/2 - pram.Nx/2)+1:round(size(X_em,2)/2 + pram.Nx/2);    
%     
%   Eex_3D    = Eex_3D   (y_range,x_range,:,:,:);
%   X0        = X0       (y_range,x_range,:,:,:);
%   Xgt       = Xgt      (y_range,x_range,:,:,:);
%   X_ex      = X_ex     (y_range,x_range,:,:,:);
%   X_sctterd = X_sctterd(y_range,x_range,:,:,:);
%   X_em      = X_em     (y_range,x_range,:,:,:);
% 
%   %% images (Xgt,Y0)
%   Xgt       = Xgt (:,:,ceil(size(X_em,3)/2),:,:);
%   Y0        = X_em(:,:,ceil(size(X_em,3)/2),:,:);

    %% cont...
    Y0        = double(5*pram.maxcount*Y0/max(Y0(:))); 
    Xgt       = 5*Xgt./max(Xgt(:));
  
    [Yhat YhatADU]      = f_simulateIm_emCCD(Y0,emhist,pram);
    
    Yhat_all(:,:,:,:,b) = Yhat;
    Xgt_all (:,:,:,:,b) = Xgt;
  end
  
  
  % max_input_photons = max(poissrnd(max(Y0(:)),[1 1000]))*2;
  % N_reps            = pram.cam_emhist_Nreps;
  % emhist            = f_genEmhist(max_input_photons,N_reps,pram);
%   try
%     load('./_emhist/emhist_03-Jan-2021 07_55_31.mat') 
%   catch
%     load('./_emhist/emhist_03-Jan-2021 07:55:31.mat')
%   end
%   [Yhat YhatADU]= f_simulateIm_emCCD(Y0_all,emhist,pram);
  
  %% change dims for output
  Xgt       = reshape(Xgt_all ,[pram.Ny pram.Nx 1       Nb]); 
  Yhat      = reshape(Yhat_all,[pram.Ny pram.Nx pram.Nt Nb]);
end





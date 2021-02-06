
function [Yhat Xgt] = f_fwd(X0,E,PSFs,pram)

  emConvSPSF  = imresize(PSFs.emConvSPSF,PSFs.pram.dx/pram.dx,'bilinear');
  emPSF       = imresize(PSFs.emPSF,PSFs.pram.dx/pram.dx,'bilinear');

  if pram.useGPU ==1
    X0          = gpuArray(X0);
    E           = gpuArray(E);
    emConvSPSF  = gpuArray(emConvSPSF);
    emPSF       = gpuArray(emPSF);  
  end
    
  
  Y0  = f_conv2nd(E.*X0,emConvSPSF,'same');    
  Xgt = f_conv2nd(X0,emPSF,'same');    
  
%   for j=1:size(X0,4)
%     for i=1:pram.Nt
%       Y0(:,:,i,j) = conv2(E(:,:,i).*X0(:,:,1,j),emConvSPSF,'same');          
%     end
%     Xgt(:,:,1,j)  =  conv2(X0(:,:,1,j),emPSF,'same');    
%   end

  Y0  = double(5*pram.maxcount*Y0/max(Y0(:))); 
  Xgt = 5*Xgt./max(Xgt(:));

  % max_input_photons = max(poissrnd(max(Y0(:)),[1 1000]))*2;
  % N_reps            = pram.cam_emhist_Nreps;
  % emhist            = f_genEmhist(max_input_photons,N_reps,pram);
  try
    load('./_emhist/emhist_03-Jan-2021 07_55_31.mat') 
  catch
    load('./_emhist/emhist_03-Jan-2021 07:55:31.mat')
  end

  [Yhat YhatADU]    = f_simulateIm_emCCD(Y0,emhist,pram);
end
function [A A_spx A_ps] = f_gen_fwdA(E,PSFs,pram)

  emConvSPSF  = imresize(PSFs.emConvSPSF,PSFs.pram.dx/pram.dx,'bilinear')*(pram.dx/PSFs.pram.dx)^2;
  emPSF       = imresize(PSFs.emPSF     ,PSFs.pram.dx/pram.dx,'bilinear')*(pram.dx/PSFs.pram.dx)^2;
  exPSF       = imresize(PSFs.exPSF     ,PSFs.pram.dx/pram.dx,'bilinear')*(pram.dx/PSFs.pram.dx)^2;
  % sPSF      = imresize(PSFs.sPSF      ,PSFs.pram.dx/pram.dx,'bilinear')*(pram.dx/PSFs.pram.dx)^2;
  
  % assume no photon loss
  emConvSPSF  = emConvSPSF./sum(emConvSPSF(:));
  
  if pram.useGPU ==1
    E           = gpuArray(E);
    emConvSPSF  = gpuArray(emConvSPSF);
    emPSF       = gpuArray(emPSF);  
  end
    
  E           = padarray(E,[1 1],0,'pre');
  for t=1:pram.Nt     
    E(:,:,t)  = conv2(E(:,:,t),exPSF,'same');  
  end
  E           = E(1:end-1,1:end-1,:);

  X_temp = zeros(pram.Ny,pram.Nx);
  for i=1:pram.Ny*pram.Nx 
    fprintf('%d/%d\n',i,pram.Ny*pram.Nx )
    
    X_temp(:)=0;
    X_temp(i)=1;        

    for t=1:pram.Nt
      Y_temp    (:,:,t) = conv2(E(:,:,t).*X_temp,emConvSPSF,'same');
      Y_temp_spx(:,:,t) = conv2(E(:,:,t).*X_temp,emConvSPSF);
    end
    A(:,i)     = Y_temp(:);
    A_spx(:,i) = sum(sum(Y_temp_spx,1),2);

    Y_temp_ps  = conv2(X_temp,exPSF,'same');
    A_ps(:,i)  = Y_temp_ps(:);
  end

end
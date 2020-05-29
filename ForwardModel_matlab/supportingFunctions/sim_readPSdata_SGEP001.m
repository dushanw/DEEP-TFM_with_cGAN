
function [X_gt] = sim_readPSdata_SGEP001(dataPath,dataName,Nx,Ny,Nz,channel,exPSF,dx,dx_gt)

    for i=1:Nz*3
        X_gt(:,:,i) = imread([dataPath dataName],i);
    end
    X_gt = X_gt(:,:,channel:3:end);% select first channel
    X_gt = X_gt-1;X_gt(find(X_gt<0))=0;% substract read noise       
    
    for i=1:Nz
        X_gt(:,:,i) = conv2(X_gt(:,:,i),exPSF,'same'); 
    end
    X_gt = imresize(X_gt,dx_gt/dx);

    X_gt = X_gt(1:Ny,1:Nx,:);        
    X_gt = double(X_gt);
    
    X_gt(:,:,1:Nz) = X_gt(:,:,Nz:-1:1);
end
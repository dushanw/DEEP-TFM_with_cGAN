

function Y = sim_genPatterndImages(X_gt_z,sPSF,H_ext,N_expTimeFactor)
    
    X_gt_z = X_gt_z*N_expTimeFactor;
    Nt = size(H_ext,3);
    for i=1:Nt
        Y(:,:,i) = conv2(H_ext(:,:,i).*X_gt_z,sPSF,'same');                         
    end
    
    % Y = poissrnd(Y);% add noise
end 


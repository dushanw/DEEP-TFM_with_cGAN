% 20200417 by Dushan N. Wadduwage
% Input     = [N N Nc] image 
% Output    = [Nx Nx Nc floor(N/Nx) floor(N/Nx)] image
% ignors last parts of the images of sizes below [Nx Nx]

function Y = img2blocks(X,Nx)

    kr  = 0;
    for i=1:Nx:size(X,1)-Nx
        kr = kr+1;
        kc = 0;
        for j=1:Nx:size(X,2)-Nx
            kc = kc+1;
            Y(:,:,:,kr,kc) = X(i:i+Nx-1,j:j+Nx-1,:);            
        end
    end

end
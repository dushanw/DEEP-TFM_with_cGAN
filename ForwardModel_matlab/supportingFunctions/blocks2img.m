% 20200417 by Dushan N. Wadduwage
% Input     = [N N Nc] image 
% Output    = [Nx Nx Nc floor(N/Nx) floor(N/Nx)] image
% ignors last parts of the images of sizes below [Nx Nx]

function Z = blocks2img(Y)

    Z = [];
    for i=1:size(Y,4)
        Zr = [];
        for j=1:size(Y,5)
            Zr = cat(2,Zr,Y(:,:,:,i,j));
        end        
        Z = cat(1,Z,Zr);
    end


end
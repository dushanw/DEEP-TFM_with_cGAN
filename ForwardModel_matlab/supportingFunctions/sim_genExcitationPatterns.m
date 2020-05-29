

function H_ext = sim_genExcitationPatterns(Nx,Ny,Nt,patternType,exPSF,dx,dx_gt)

    switch patternType
        case 'fullyRandom'
            H_ext = randi([0 1],Ny,Nx,Nt);
        case '10pcntFilledRandom'
            H_ext = randi([0 10],Ny,Nx,Nt);
            H_ext(find(H_ext(:)<9))=0;
            H_ext(find(H_ext(:)>=9))=1;                       
        case '1pcntFilledRandom'
            H_ext = randi([0 100],Ny,Nx,Nt);
            H_ext(find(H_ext(:)<99))=0;
            H_ext(find(H_ext(:)>=99))=1;            
        case 'mmm'
            H_ext = zeros(Ny,Nx,Nt);
            delta_y = Ny/sqrt(Ny*Nx/Nt);
            delta_x = Ny/sqrt(Ny*Nx/Nt);
            t=1;
            for i=1:sqrt(Nt)
                for j=1:sqrt(Nt)                    
                    H_ext(i:delta_y:end,j:delta_x:end,t)=1;
                    t=t+1;
                end
            end            
        case 'hadamardRandom'
            H_bar_rnd = randi([0,1],Nx,Ny);
            H_rnd = H_bar_rnd*2-1;
            H_had = hadamard(Nt);
            for i=1:Nt
                H_temp = repmat(reshape(H_had(i,:),[sqrt(Nt) sqrt(Nt)]),[Ny/sqrt(Nt) Nx/sqrt(Nt)]);

                Ht(:,:,i) = H_temp;% hadamard
                Ht(:,:,i) = H_temp.*H_rnd;% randomized hadamard    
            end
            H_ext = (Ht+1)/2;
    end
    
    H_ext = imresize(H_ext,dx/dx_gt,'nearest');
    for i=1:Nt
        H_ext(:,:,i) = conv2(H_ext(:,:,i),exPSF,'same'); 
    end
    H_ext = double(imresize(H_ext,dx_gt/dx,'nearest'));
end
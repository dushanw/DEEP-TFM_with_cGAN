function [X0 Segments] = gen_synth_20190718(Nx,Ny,Nz,bg1,bg2,s)

    X0 = zeros(Nx,Ny,Nz);
    
    O_bg1 = zeros(Nx,Ny)+bg1;
    
    O_bg2 = mean(insertShape(zeros(Nx,Ny),'FilledCircle',[Ny/2.5 Nx/2.5 10]),3);
    O_bg2 = O_bg2 * bg2/max(O_bg2(:));
    
    O_s   = mean(insertShape(zeros(Nx,Ny),'FilledCircle',[Ny/2 Nx/2 2]),3);
    O_s   = O_s * s/max(O_s(:));

    O = O_s + O_bg1 + O_bg2;
    %imagesc(O)
    
    Segments = zeros(Nx,Ny);
    Segments(find(O_bg2(:)>0)) = 1;
    Segments(find(O_s(:)>0)) = 2;
    
    X0 = repmat(O,1,1,Nz);
end
function X0 = gen_beadsIn2D(Nx,Ny,bg)
    
    X0 = zeros(Nx,Ny)+bg;   
    N_objects = 10;
    for i=1:N_objects    
        X0   = mean(insertShape(X0,'FilledCircle',[randi(Ny) randi(Nx) randi([8 10])],'Opacity',rand),3);        
    end
    
end

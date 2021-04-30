
function X0 = f_genobj_neuronPatches(V_ps,pram)
  
  Ny0         = pram.Ny;
  Nx0         = pram.Nx;
  
  np          = 100;                                % The maximum photon value used for normarlization  
  [y_inds,x_inds] = find(max(V_ps,[],3) > 0.2*np);  % Find object regions

  inds_valid      = (y_inds<size(V_ps,1)-Ny0-1) .* (x_inds<size(V_ps,2)-Nx0-1);         
                                                    % Valid indices
  y_inds          = y_inds(inds_valid>0);
  x_inds          = x_inds(inds_valid>0);

  randIdcs        = randperm(length(x_inds));
  y_inds_rnd      = y_inds(randIdcs);                % Now the patch upper-left corner indexs are randomly arranged in x,y 
  x_inds_rnd      = x_inds(randIdcs);

  for i = 1:pram.Npch_perCell                        % Loop to randomly select NN_x * NN_y small patterns
    ind_y         = y_inds_rnd(i);
    ind_x         = x_inds_rnd(i);
    ind_yr        = ind_y-round(Ny0/2):ind_y-round(Ny0/2)+Ny0-1;
    ind_xr        = ind_x-round(Nx0/2):ind_x-round(Nx0/2)+Nx0-1;

    X0{i}   = V_ps (ind_yr,ind_xr,:);
  end
  % volshow(X0)  
end







% 20181107 by Dushan N. Wadduwage

function Xhat = f_rec_inv_canonPrior(pram,y,A,gamma)

  %% optimization over alpha  
  n     = size(A,2);  
  y     = double(y);
  
  cvx_begin 
      variable x(n)
      minimize(norm(A*x-y,2) + gamma*norm(x,1))

      subject to
%          x >= 0;          
  cvx_end
  Xhat = reshape(x,pram.Ny,pram.Nx);    
end

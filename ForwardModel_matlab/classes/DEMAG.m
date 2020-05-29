% 20190729 by Dushan N. Wadduwage
% demagnifier class

classdef DEMAG < handle
   properties
      A_demag      
      Ny
      Nx
      ny
      nx      
   end
   methods       
       function obj = DEMAG(Ny,Nx,ny,nx)
           obj.Ny = Ny;
           obj.Nx = Nx;
           obj.ny = ny;
           obj.nx = nx;
                      
           X_inds = reshape(1:Ny*Nx,Ny,Nx);
           X_inds = im2col(X_inds,[ny nx],'distinct');
           
           obj.A_demag = sparse(Ny*Nx/(ny*nx),Ny*Nx);
           for i=1:size(obj.A_demag,1)
               obj.A_demag(i,X_inds(:,i))=1;
           end           
       end
       
       function Xdm = demagnify(self,X)
           for t=1:size(X,3)
               Xt = X(:,:,t);
               Xdmt = self.A_demag*Xt(:);
               Xdmt = reshape(Xdmt,[self.Ny/self.ny self.Nx/self.nx]);           
               Xdm(:,:,t) = Xdmt;
           end
       end
       
   end
end









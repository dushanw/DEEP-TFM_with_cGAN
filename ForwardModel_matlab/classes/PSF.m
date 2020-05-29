
classdef PSF < handle
   properties
      psf
      otf
      A_psf
      % A_otf
   end
   methods       
       function obj = PSF(data,Nx,Ny)
           if isempty(data)
               obj.psf = zeros(Ny,Nx);
               obj.psf(Ny/2,Nx/2)=1;
           else               
               obj.psf = data;               
           end
           obj.otf = fftshift(fft2(obj.psf));
           obj.A_psf = genConvMat(obj.psf);           
       end
       
       function self = genConvMat(self)
           obj.A_psf = genConvMat(obj.psf);           
       end
       
       function self = updatePSF(self)
           self.psf = fftshift(ifft2(self.otf));  
           self.A_psf = genConvMat(self.psf);
       end
       
       function self = updateOTF(self)
           self.oft = fftshift(fft2(self.psf));       
       end              
   end
end
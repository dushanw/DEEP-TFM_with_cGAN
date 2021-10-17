% 20210417 by Dushan N. Wadduwage
% This will perform nd convolution comparable to Matlab's convnd, but using
% fftn for enhanced speed
%
% inputs: A : nd array
%         B : nd array
% output  C : nd array

function C = f_convnd(A,B,shape)

  fft_Ny      = size(A,1)+size(B,1)-1;
  fft_Nx      = size(A,2)+size(B,2)-1;
  fft_Nz      = size(A,3)+size(B,3)-1;
  
  C = ifftn(fftn(A,[fft_Ny,fft_Nx,fft_Nz]) .* fftn(B,[fft_Ny,fft_Nx,fft_Nz]));    
  C = abs(C);

  if ~isempty(shape)
    switch shape
      case 'same'
        y_range = round(fft_Ny/2 - size(A,1)/2)+1:round(fft_Ny/2 + size(A,1)/2);
        x_range = round(fft_Nx/2 - size(A,2)/2)+1:round(fft_Nx/2 + size(A,2)/2);
        z_range = round(fft_Nz/2 - size(A,3)/2)+1:round(fft_Nz/2 + size(A,3)/2);
        C = C(y_range,x_range,z_range,:);
    end
  end

end
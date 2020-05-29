% 20181019 by Dushan N. Wadduwage
% data class definition

classdef IMAGESTACK < handle
   properties
      Data
      Nx
      Ny
      ny
      nx
      Nt
      fftFlag
      diagFlag
      columnFlag
      blkcolumnFlag
      bkldiagFlag
      originalFormFlag
   end
   methods    
       % constructor
       function obj = IMAGESTACK(data,dataPath,nameStem,Nx,Ny,Nt,EmGain,crop_rect_name)
           if isempty(data)             
               try load(crop_rect_name);
               catch crop_rect = genCropRect(dataPath,nameStem,crop_rect_name);
               end
               try obj.Data = readimages_singleFile(dataPath,nameStem,crop_rect,Nx,Ny,Nt,EmGain);
               catch obj.Data = readimages(dataPath,nameStem,crop_rect,Nx,Ny,Nt,EmGain);
               end
           else    
               obj.Data = data;
           end
           obj.Nx = size(obj.Data,1);
           obj.Ny = size(obj.Data,2);
           obj.Nt = size(obj.Data,3);
           obj.ny = 1;
           obj.nx = 1;
           obj.fftFlag = 0;
           obj.diagFlag = 0;
           obj.columnFlag = 0;
           obj.bkldiagFlag = 0;
           obj.originalFormFlag = 1;
       end
       
       % re-shape fucntions
       function self = stack(self)
           self.Data = reshape(self.Data,[self.Nx self.Ny self.Nt]);
       end
              
       function self = columnize(self)
           self.Data = self.Data(:);
           self.columnFlag = 1;
           self.originalFormFlag = 0;
       end
       
       function self = diagonalize(self) 
           if self.diagFlag==0
               tempData = reshape(self.Data,[self.Nx*self.Ny self.Nt]);
               A_hight = self.Nx*self.Ny*self.Nt;
               A_width = self.Nx*self.Ny;          

               diagonalPositions = -([1:A_width:A_hight]-1);
               self.Data = spdiags(tempData,diagonalPositions,A_hight,A_width);
               self.diagFlag = 1;
               self.originalFormFlag = 0;
           end
       end
       
       function self = blkDiagonalize(self,ny,nx) 
            if self.diagFlag==0 && self.bkldiagFlag==0
                temp_A = [];
                
                for i=1:self.Nt
                    temp = im2col(self.Data(:,:,i),[ny nx],'distinct');
                    temp = num2cell(temp',2);
                    temp_A = [temp_A; sparse(blkdiag(temp{:}))];                                       
                end
                self.Data = temp_A;
                self.bkldiagFlag=1;
                self.originalFormFlag = 0;
                
                self.ny = ny;
                self.nx = nx;
            end
       end
       
       function self = blkColumnize(self,ny,nx)
           if self.originalFormFlag == 1
               for t=1:self.Nt
                   Data_temp(:,:,t) = im2col(self.Data(:,:,t),[ny nx],'distinct');
               end
               self.Data = Data_temp(:);
               self.originalFormFlag = 0;
               self.blkcolumnFlag = 1;

               self.ny = ny;
               self.nx = nx;
               
           else
               display('Couldn not blkColumnize :-/');
           end
       end
       
       function self = de_blkColumnize(self,ny,nx)
           if self.Nt == 1      
               self.Data = reshape(self.Data,[self.ny*self.nx self.Ny*self.Nx/(self.ny*self.nx)]);
               self.Data = col2im(self.Data,[self.ny self.nx],[self.Ny self.Nx],'distinct');                
               
               self.ny = ny;
               self.nx = nx;
           else 
               display('Couldn not de_blkColumnize :-/');
           end
       end

       
       % FFT and IFFT
       function self = fft2stack(self)           
           self.Data = reshape(self.Data,[self.Nx self.Ny self.Nt]);
           for i=1:self.Nt
               self.Data(:,:,i) = fftshift(fft2(self.Data(:,:,i)));           
           end
           self.fftFlag = 1;
       end
       
       function self = ifft2stack(self)           
           self.Data = reshape(self.Data,[self.Nx self.Ny self.Nt]);
           for i=1:self.Nt
               self.Data(:,:,i) = ifft2(fftshift(self.Data(:,:,i)));           
           end
           fftFlag = 0;
       end
            
       % Numerical on 2D and 3D - overloading
       function imgStackOut = mtimes(imgStack,img)% '*'
           for i=1:imgStack.Nt
               tempData(:,:,i) = imgStack.Data(:,:,i)*img.Data;
           end         
           imgStackOut = IMAGESTACK(tempData);
       end
       
       function imgStackOut = times(imgStack,img)% '.*'
           for i=1:imgStack.Nt
               tempData(:,:,i) = imgStack.Data(:,:,i).*img.Data;
           end           
           imgStackOut = IMAGESTACK(tempData);
       end
       
       % visualization
       function implay(self,I_max_mult)
            implay(I_max_mult*(self.Data-min(self.Data(:)))/max(self.Data(:)));
       end
       
       % Image saving
       function imsave(self,nameStem,I_max_mult)
           self.Data = reshape(self.Data,[self.Nx self.Ny self.Nt]);
           
           I_max = max(self.Data(:))/I_max_mult;
           I_min = 0;%min(self.Data(:));
           for i=1:self.Nt
               Itemp = self.Data(:,:,i);
               Itemp = uint16(2^16*(Itemp-I_min)/I_max);
               imwrite(Itemp,sprintf('%s_%d.tif',nameStem,i));               
           end
       end
              
       function imsave1(self,nameStem,I_max_mult)
           self.Data = reshape(self.Data,[self.Nx self.Ny self.Nt]);
           
           Itemp = self.Data(:,:,1);           
           I_max = max(Itemp(:))/I_max_mult;
           I_min = 0;%min(Itemp(:));
           
           Itemp = uint16(2^16*(Itemp-I_min)/I_max);               
           imwrite(Itemp,sprintf('%s_%d.tif',nameStem,i));                          
       end
       
       function imsave_mip(self,nameStem,I_max_mult,plane,pixelRatio)% maximum_intensity_projection
           self.Data = reshape(self.Data,[self.Nx self.Ny self.Nt]);
           
           switch plane
               case 'xy'
                   Itemp = max(self.Data,[],3);           
               case 'xz'
                   Itemp = squeeze(max(self.Data,[],2));
                   Itemp = imresize(Itemp,[size(Itemp,1) size(Itemp,2)*pixelRatio]);
                   Itemp = Itemp';
           end
           
           I_max = max(Itemp(:))/I_max_mult;
           I_min = 0;%min(Itemp(:));
           
           Itemp = uint16(2^16*(Itemp-I_min)/I_max);               
           imwrite(Itemp,sprintf('%s_%d.tif',nameStem,i));                          
       end

   end
end








% 20181129 by Dushan N. Wadduwage
% data class definition

classdef SIM_EXPERIMENT < handle
   properties     
       % simulation parameters
       sl_em
       sl_ex 
       patternType      % E {'hadamardRandom','fullyRandom'};
       Nt_mx
       N_expTimeFactor  % max counts per pixel (per t)
       dx_gt            % [um] PS image's pixel size
       dx               % [um] TF image's pixel size 
       Nx
       Ny
       Nz
       Nt
       z_range
       
       exPSF
       Ex
       X0
       
       sPSF       
       Y
       z_ind_now        % ind
       z_now
       Xhat
       Xhat_TFM
       Xhat_PSMPM
       Xhat_GT
   end
   methods
       % constructor
       function obj = SIM_EXPERIMENT(parameterFile)
           % simulation parameters 
           run(parameterFile);                                
           
           obj.sl_em = sl_em;
           obj.sl_ex = sl_ex;
           obj.patternType = patternType;    
           obj.Nt_mx = Nt_mx;
           obj.N_expTimeFactor = N_expTimeFactor; 
           obj.dx_gt = dx_gt;
           obj.dx = dx;
           obj.Nx = Nx;
           obj.Ny = Ny;
           obj.Nz = Nz;
           obj.z_range = z_range;
           obj.z_ind_now = 1;% [um]
           obj.z_now = obj.z_range(obj.z_ind_now);
           
           obj.exPSF = PSF(sim_get_modeled_exPSF([],sl_ex,dx_gt));% no z or ls_ex dependence 
           
           obj.Ex = CALIB_IMAGESTACK(sim_genExcitationPatterns(Nx,Ny,Nt_mx,patternType,obj.exPSF.psf,dx,dx_gt)); 
           
           load([dataPath dataName])
           obj.X0 = IMAGESTACK(imresize(X0_Neuron0_PSTPM,[Ny Nx]));
           obj.Xhat = REC_IMAGESTACK(zeros(Ny,Nx));
       end
       
       function image_DEEPTFM(self,new_z_ind,t_exp,Nt)
           self.z_ind_now = new_z_ind;        
           self.z_now = self.z_range(new_z_ind);
           self.N_expTimeFactor = t_exp;
           self.Nt = Nt;
           
           self.sPSF = PSF(sim_get_modeled_sPSF(self.z_now,self.sl_em,self.dx,self.Nx));
           self.Y = MEASURE_IMAGESTACK(sim_genPatterndImages(self.X0.Data(:,:,self.z_ind_now),...
                                                            self.sPSF.psf,...
                                                            self.Ex.Data(:,:,1:self.Nt)/self.Nt,...
                                                            self.N_expTimeFactor));
           self.Y.addPoissonNoise;                                                                                                                           
       end
       
       function self = image_TFM(self)
           self.Xhat_TFM = MEASURE_IMAGESTACK(conv2(self.X0.Data(:,:,self.z_ind_now)*self.N_expTimeFactor,...
                                                    self.sPSF.psf,'same'));
           self.Xhat_TFM.addPoissonNoise;
           %self.Xhat_TFM.Data = self.Xhat_TFM.Data/Nt_DEEPTFM;
       end 
       
       function self = image_PSMPM(self)
           self.Xhat_PSMPM = MEASURE_IMAGESTACK(self.X0.Data(:,:,self.z_ind_now)*self.N_expTimeFactor);
           self.Xhat_PSMPM.addPoissonNoise; 
           %self.Xhat_PSMPM.Data = self.Xhat_PSMPM.Data/Nt_DEEPTFM;
       end 
       
       function self = image_GT(self)
           self.Xhat_GT = MEASURE_IMAGESTACK(self.X0.Data(:,:,self.z_ind_now)*self.N_expTimeFactor);           
       end 
              
       function self = estimate_Xhat_wls(self)
           Y_temp = self.Y;
           Y_temp.Data(find(Y_temp.Data(:)<0))=0;
           
%           Y_temp.preprocess;
           Y_temp.columnize;
           
           Ex_temp = CALIB_IMAGESTACK(self.Ex.Data(:,:,1:self.Nt));
%           Ex_temp.preprocess;
           Ex_temp.diagonalize;
           Ac = repmat({sparse(self.sPSF.A_psf)},self.Nt,1);
           Ac = blkdiag(Ac{:});
           A = Ac*Ex_temp.Data;
           
           %A_waverec2 =getWaveletmatrices(self.Xhat.Ny,self.Xhat.Nx);
           load(sprintf('A_waverec2_%dx%d.mat',[self.Ny self.Nx]))          
           A_wavedec2 = inv(A_waverec2);
           
%           x0 = A\Y_temp.Data;
           
%           self.Xhat.Data = estimate_Xhat_cvx(Y_temp.Data,Ex_temp.Data,A_wavedec2);
           self.Xhat.Data = estimate_Xhat_cvx_wls(Y_temp.Data,A,A_wavedec2);
%           x = estimate_Xhat_cvx_wls(y,A,Psy)
           
           self.Xhat.Data(find(self.Xhat.Data(:)<0)) = 0; 
           self.Xhat.stack; 
           
       end       
   end
end

           
           

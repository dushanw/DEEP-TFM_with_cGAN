% 20181107 by Dushan N. Wadduwage
% 20201223 edited by DNW to improve speed using GPU and fmin

function [Xhat FitInfo] = f_rec_inv_wlPrior_lasso(pram,y,A,wname,lasso_lambda)

  tic
  disp(['Preprocessing inputs | t = ' num2str(toc) '[s]'])
  
  % parameters
  if isempty(wname) 
    wname       = 'db4';
  end
   
  y = A'*y;           % convert to a (Ny*Nx) by (Ny*Nx) system
  A = A'*A;

  c = sum(A,1);
  A = A./max(c);
  c = c./max(c);

  w = (sqrt(1./(y+1)));

  %% optimization over alpha 
  disp(['Generating wavelet matrix | t = ' num2str(toc) '[s]'])
  Psy   = getWaveletmatrices(pram.Ny,pram.Nx,wname);
  
  A_x_Psy = full(A) * Psy;
  
  Psy   = single(full(Psy));
  y     = single(y);

  disp(['Lasso started | t = ' num2str(toc) '[s]'])
  if ~isempty(lasso_lambda)
    [alpha FitInfo] = lasso(A_x_Psy,y,'Options',statset('UseParallel',true),'Lambda',lasso_lambda); 
  else
    [alpha FitInfo] = lasso(A_x_Psy,y,'Options',statset('UseParallel',true));
  end
  disp(['Lasso done! | t = ' num2str(toc) '[s]'])

  x     = Psy*alpha;
  x(x<0)= 0;
  
  Xhat = reshape(x,pram.Ny,pram.Nx,size(x,2)); 
end


function A_waverec2 = getWaveletmatrices(h,w,wname)
  X           = zeros(h,w);
  L           = wmaxlev(size(X),wname);
  [c s]       = wavedec2(X,2,wname);
  A_waverec2  = sparse(h*w,length(c));

  parfor i=1:length(c)
    c1{i}    = zeros(size(c));
    c1{i}(i) = 1;
    A_waverec2(:,i)=sparse(reshape(waverec2(c1{i},s,wname),[h*w,1])); 
  end
  done=1;
end



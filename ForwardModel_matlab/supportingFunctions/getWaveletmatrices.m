% 20180113 by Dushan N. Wadduwage
% wavelet matrices 

function [A_waverec2]=getWaveletmatrices(h,w)

    X = zeros(h,w);
    [c s]=wavedec2(X,2,'haar');  
    A_waverec2 = sparse(h*w,h*w);
    for i=1:h*w
        i
        c(:)=0;
        c(i)=1;
        Y_temp = waverec2(c,s,'haar');
        A_waverec2(:,i)=sparse(reshape(Y_temp,[h*w,1])); 
    end
end




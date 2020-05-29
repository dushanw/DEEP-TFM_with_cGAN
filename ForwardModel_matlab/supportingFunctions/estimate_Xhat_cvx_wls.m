% 20181107 by Dushan N. Wadduwage

function x = estimate_Xhat_cvx_wls(y,A,Psy)

    y(find(y(:)<0))=0;% no negative measurements 
    y = y./max(y);
    
    y = A'*y;
    A = A'*A;
    

    c = sum(A,1);
    A = A./max(c);
    c = c./max(c);

    n = size(A,2);
    gamma = 0;%8e-6;% Nt,32
    
%    w = diag(sqrt(1./(y+1)));
    w = (sqrt(1./(y+1)));
    
    L1sum = norm(y(:),1);
    L2sum = sqrt(L1sum); % Target of least square estimation based on Poisson statistics
    eps = L2sum*1.5;

    cvx_begin 
        variable x(n)
        minimize(norm(w.*(A*x-y),2) + gamma*norm(Psy*x,1))        
        subject to
            x >= 0;
            %norm( A * x - y, 2 )<=eps;
    cvx_end

    x(find(isnan(x)))=0;
    x(find(x(:)<0))=0;
end
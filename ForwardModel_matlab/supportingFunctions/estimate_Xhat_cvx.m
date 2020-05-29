% 20181107 by Dushan N. Wadduwage

function x = estimate_Xhat_cvx(y,A,Psy)

    n = size(A,2);
    % gamma = 0.00015;% Nt,16
    % gamma = 0.00001;% Nt,32
    gamma = 0;% Nt,32
    
    cvx_begin 
        variable x(n)
        %minimize(norm( A * x - y, 2 ) + gamma*norm(Psy*x,1))
        minimize(norm( A * x - y, 2 ))
        subject to
            x >= 0;
    cvx_end

    x(find(isnan(x)))=0;

end
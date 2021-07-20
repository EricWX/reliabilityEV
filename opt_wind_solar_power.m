function [beta_wind, alpha_wind,  r_solar_M_H, pred_wind] = ...
    opt_wind_solar_power(quant,lambda, Wind, Wind_date, Solar, window)

Wind = Wind/window;
Solar = Solar/window;

Month_wind = month(Wind_date);
Hour_wind = hour(Wind_date);
n_r_wind = length(Hour_wind);
Months = zeros(n_r_wind,12);
Hours = zeros(n_r_wind,24);
for i = 1:12
    Months(:,i) = (Month_wind == i);
end
for i = 1:24
    Hours(:,i) = (Hour_wind == i-1);
end


A_wind = [Months, Hours];
A_solar_M = Months;
A_solar_H = Hours;
n_wind = 12 + 24;

n_quant = length(quant);

diff_matrix_beta = sparse(diff(diag(ones(n_quant,1))));
diff_matrix_alpha = sparse(diff(diff(diag(ones(n_quant-1,1)))));
diff_beta_tail = diff(diag(ones(2,1)));
diff_q = diff(diag(ones(n_quant,1)));
A_wind = sparse(A_wind);

A_Month_Hour = zeros(n_r_wind, 24*12);
for i = 1:24
    for j = 1:12
        A_Month_Hour(:, 24*(j-1)+i) = (Hour_wind == i-1) .* (Month_wind == j);
    end
end

lambda1 = lambda(1);
lambda2 = lambda(2);
lambda3 = lambda(3);
lambda4 = lambda(4);
lambda5 = lambda(5);
% lambda1 = 100;
% lambda2 = 50;
% lambda3 = 100;
% lambda4 = 100;

tic;
cvx_begin quiet
    variables beta_wind(n_quant,n_wind) alpha_wind(n_quant) r_solar_M_H(288)
    ob1_pre1 = A_wind * beta_wind' + ones(n_r_wind,1)*alpha_wind' +  ...
        A_Month_Hour * r_solar_M_H .* Solar * ones(1,n_quant);
    ob1_pre2 = Wind * ones(1,n_quant) - ob1_pre1;
    ob1 = 0.5 * sum(sum(abs(ob1_pre2))) + ones(1,n_r_wind) * ob1_pre2 * (quant - 0.5)';
    ob2_pre = diff_matrix_beta * beta_wind;
    ob2 = sum(sum(ob2_pre .* ob2_pre));
    ob3_pre = diff_matrix_alpha * alpha_wind(2:(end));
    ob3 = sum(ob3_pre .* ob3_pre);
    
    r_solar_matrix = reshape(r_solar_M_H,[24,12]);
%     ob4_pre = r_solar_matrix - 2*circshift(r_solar_matrix,1)+circshift(r_solar_matrix,1);
    ob4_pre = r_solar_matrix - circshift(r_solar_matrix,1);
    ob4 = sum(sum( ob4_pre .* ob4_pre));
    %%ob7 is the regulization of r_solar_month
    r_solar_matrix_t = r_solar_matrix';
    ob5_pre = r_solar_matrix_t - circshift(r_solar_matrix_t,1);
    ob5 = sum(sum( ob5_pre .* ob5_pre));
    
    ob6 = sum(r_solar_M_H.*r_solar_M_H);
   pred_wind = A_wind * beta_wind' + ones(n_r_wind,1)*alpha_wind' + ...
        A_Month_Hour * r_solar_M_H .* Solar * ones(1,n_quant);
    minimize( ob1 + lambda1 * ob2 + lambda2 * ob3 + lambda3 * ob4 + ...
        lambda4 * ob5 + lambda5  * ob6);
    subject to 
       pred_wind(:,1) >= 0
       diff_q * pred_wind' >= 0
       
       diff(diag(ones(2,1))) * beta_wind(1:2,:) == 0
        diff(diag(ones(2,1))) * beta_wind((end-1):end,:) == 0
        r_solar_matrix(1:6,:) == 0
        r_solar_matrix(21:24,:) == 0
        
%        diff_beta_tail * beta_wind((end-1):end,:) == 0
%        r_solar_matrix(1,:) == 0
%        r_solar_matrix(24,:) == 0
%         r_solar_matrix(1:6,:) == 0
%         r_solar_matrix(21:24,:) == 0
 cvx_end
 
 beta_wind = beta_wind * window;
 alpha_wind = alpha_wind * window;
 pred_wind = pred_wind * window;
 
toc;
% pred_wind = A_wind * beta_wind' + ones(n_r_wind,1)*alpha_wind' + ...
%     A_solar_M * r_solar_M .* Solar * ones(1,n_quant) +...
%     A_solar_H * r_solar_H .* Solar * ones(1,n_quant);

% function [beta_wind, alpha_wind,  r_solar_M_H, pred_wind] = opt_wind_solar(quant,lambda, Wind, Wind_date...
%             , Solar)
% 
% Month_wind = month(Wind_date);
% Hour_wind = hour(Wind_date);
% n_r_wind = length(Hour_wind);
% Months = zeros(n_r_wind,12);
% Hours = zeros(n_r_wind,24);
% for i = 1:12
%     Months(:,i) = (Month_wind == i);
% end
% for i = 1:24
%     Hours(:,i) = (Hour_wind == i-1);
% end
% 
% 
% A_wind = [Months, Hours];
% A_solar_M = Months;
% A_solar_H = Hours;
% n_wind = 12 + 24;
% 
% n_quant = length(quant);
% 
% diff_matrix_beta = sparse(diff(diag(ones(n_quant,1))));
% diff_matrix_alpha = sparse(diff(diff(diag(ones(n_quant-2,1)))));
% diff_q = diff(diff(diag(ones(n_quant,1))));
% A_wind = sparse(A_wind);
% 
% A_Month_Hour = zeros(n_r_wind, 24*12);
% for i = 1:24
%     for j = 1:12
%         A_Month_Hour(:, 24*(j-1)+i) = (Hour_wind == i-1) .* (Month_wind == j);
%     end
% end
% 
% lambda1 = lambda(1);
% lambda2 = lambda(2);
% lambda3 = lambda(3);
% lambda4 = lambda(4);
% 
% % lambda1 = 100;
% % lambda2 = 50;
% % lambda3 = 100;
% % lambda4 = 100;
% 
% tic;
% cvx_begin quiet
%     variables beta_wind(n_quant,n_wind) alpha_wind(n_quant) r_solar_M_H(288)
%     ob1_pre1 = A_wind * beta_wind' + ones(n_r_wind,1)*alpha_wind' +  ...
%         A_Month_Hour * r_solar_M_H .* Solar * ones(1,n_quant);
%     ob1_pre2 = Wind * ones(1,n_quant) - ob1_pre1;
%     ob1 = 0.5 * sum(sum(abs(ob1_pre2))) + ones(1,n_r_wind) * ob1_pre2 * (quant - 0.5)';
%     ob2_pre = diff_matrix_beta * beta_wind;
%     ob2 = sum(sum(ob2_pre .* ob2_pre));
%     ob3_pre = diff_matrix_alpha * alpha_wind(2:(end-1));
%     ob3 = sum(ob3_pre .* ob3_pre);
%     
%     r_solar_matrix = reshape(r_solar_M_H,[24,12]);
%     ob4_pre = r_solar_matrix - circshift(r_solar_matrix,1);
%     ob4 = sum(sum( ob4_pre .* ob4_pre));
%     %%ob7 is the regulization of r_solar_month
%     r_solar_matrix_t = r_solar_matrix';
%     ob5_pre = r_solar_matrix_t - circshift(r_solar_matrix_t,1);
%     ob5 = sum(sum( ob5_pre .* ob5_pre));
%    pred_wind = A_wind * beta_wind' + ones(n_r_wind,1)*alpha_wind' + ...
%         A_Month_Hour * r_solar_M_H .* Solar * ones(1,n_quant);
%     minimize( ob1 + lambda1 * ob2 + lambda2 * ob3 + lambda3 * ob4 + lambda4 * ob5);
%     subject to 
%        pred_wind(:,1) >= 0
%        diff_q * pred_wind' >= 0
%         r_solar_matrix(1:6,:) == 0
%         r_solar_matrix(21:24,:) == 0
%  cvx_end
% toc;


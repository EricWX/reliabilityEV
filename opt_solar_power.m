function [beta_solar,pred_solar] =opt_solar_power(quant,Solar, Date,lambda, scale)

Solar = Solar/scale;

Month = month(Date);
Hour = hour(Date);
n_r = length(Solar);
Months = zeros(n_r,12);
Hours = zeros(n_r,24);
% for i = 1:12
%     Months(:,i) = (Month == i);
% end
% 
% for i = 1:24
%     Hours(:,i) = (Hour == i-1);
% end
for i = 1:12
    for j = 1:24
        Month_Hour(:,(i-1)*24 + j) = (Month == i).*(Hour == j-1);
    end
end
    


A = Month_Hour;
n_solar = 12 * 24;

n_quant = length(quant);

diff_matrix_beta = sparse(diff(diff(diag(ones(n_quant,1)))));
diff_beta_tail = diff(diag(ones(2,1)));
diff_matrix_alpha = sparse(diff(diff(diag(ones(n_quant-1,1)))));
diff_matrix_q = sparse(diff(diag(ones(n_quant,1))));


A = sparse(A);

lambda1 = lambda(1);
lambda2 = lambda(2);



tic;
cvx_solver SDPT3

cvx_begin 
    variables beta_solar(n_quant,n_solar)
    ob1_pre = (Solar * ones(1,n_quant) - (A * beta_solar'))  ;
    ob1 = 0.5 * sum(sum(abs(ob1_pre))) + ones(1,n_r) * ob1_pre * (quant - 0.5)';
%     ob2_pre = diff_matrix_beta * beta_solar;
%     ob2 = sum(sum(ob2_pre .* ob2_pre));
%     ob3_pre = diff_matrix_alpha * alpha_solar(2:(end));
%     ob3 = sum(ob3_pre .* ob3_pre);
    minimize( ob1  ) % + lambda1 * ob2 +lambda2 * ob3
    pred_solar = (A * beta_solar');
    subject to 
        pred_solar(:,1) >= 0          
%         diff_beta_tail * beta_solar(1:2,:) == 0          
%         diff_beta_tail * beta_solar((end-1):end,:) == 0
        diff_matrix_q * pred_solar' >= 0
        
cvx_end
% cvx_begin 
%     variables beta_solar(n_quant,n_solar) alpha_solar(n_quant) 
%     ob1_pre = (Solar * ones(1,n_quant) - (A * beta_solar' + ones(n_r,1)*alpha_solar'))  ;
%     ob1 = 0.5 * sum(sum(abs(ob1_pre))) + ones(1,n_r) * ob1_pre * (quant - 0.5)';
%     ob2_pre = diff_matrix_beta * beta_solar;
%     ob2 = sum(sum(ob2_pre .* ob2_pre));
%     ob3_pre = diff_matrix_alpha * alpha_solar(2:(end));
%     ob3 = sum(ob3_pre .* ob3_pre);
%     minimize( ob1 + lambda1 * ob2 ) %+lambda2 * ob3
%     pred_solar = (A * beta_solar' + ones(n_r,1)*alpha_solar');
%     subject to 
%         pred_solar(:,1) >= 0          
%         diff_beta_tail * beta_solar(1:2,:) == 0          
%         diff_beta_tail * beta_solar((end-1):end,:) == 0
%         diff_matrix_q * pred_solar' >= 0
%         
%  cvx_end
toc;
% pred_solar = (A * beta_solar' + ones(n_r,1)*alpha_solar').* (day_night_id * ones(1,n_quant));


beta_solar = beta_solar * scale;
% alpha_solar = alpha_solar * scale;
pred_solar = pred_solar * scale;
cvx_solver SeDuMi


end


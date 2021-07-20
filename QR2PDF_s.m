function [list_wind, dis_q_wind] = QR2PDF_s(quant, pred_data, scale_factor)

scale_factor = scale_factor ;
pred_data = pred_data/scale_factor;
real_predict_data_add100 = pred_data;
quant_wind_add100 = quant;
% 
% if abs(real_predict_data_add100(end)) < 100
%     d = 1;             % Sampling interval for the PDF model 
% else
%     d = 5;
% end    

d = 1;

rho = 10^12/d^3;        % regularization parameter 
rho = 0;
[x_wind,q_wind] = qCDF2PDF(real_predict_data_add100, quant_wind_add100,d,rho);
x_wind  = x_wind * scale_factor;

ind = (q_wind < 0);
q_wind(ind) = [];
x_wind(ind) = [];
% x_wind = x_wind * (1 - r_wind)* fudge_factor;
% q_wind = q_wind / (fudge_factor *(1 - r_wind));
max_rz = floor(max(x_wind)) ;
min_rz = ceil(min(x_wind));
list_wind = min_rz:max_rz;

% plot(x_wind, q_wind)
dis_q_wind = interp1(x_wind,q_wind,list_wind);
dis_q_wind = dis_q_wind/sum(dis_q_wind);

                       
                       
                       
                       
% Month_real = kron(speye(12), ones(24*7*2,1));
% Hour_real = kron(ones(12*7,1),kron(speye(24),ones(2,1)));
% time_real_wind = [Month_real,Hour_real];
% n_real_wind = 24 * 12 * 7 * 2;
% pred_real_wind = time_real_wind * beta_wind' + ones(n_real_wind,1) * alpha_wind';
% %wind
% real_predict_wind_add100 = [zeros(n_real_wind,1), pred_real_wind,ones(n_real_wind,1) * max_wind];
% sample_value = [];
% prob_sample = [];
% sample_step = 1;
% d = 1;             % Sampling interval for the PDF model 
% rho = 10^9/d^3;        % regularization parameter 
% % u =[]; %PDF arguments
% % g_wind = []; %PDF values
% %%% get the phf for wind
% quant_wind_add100 = [0,quant_wind,1];
% for i = 1 : n_real_wind
%     % get the cdf through qQR2CDF() function
%     %[sample_value{i},prob_sample{i}] = qQR2CDF(quant_wind,real_predict_wind_add100(i,:),sample_step); 
%     % get the pdf through qCDF2PDF() function 
%     [u_wind,g_wind] = qCDF2PDF(real_predict_wind_add100(i,:),quant_wind_add100,d,rho);
%     ind = (g_wind < 10^(-9));
%     g_wind(ind) = [];
%     u_wind(ind) = [];
%     
%     u_wind = u_wind * (1 - Month_real(i,:)*r_wind)* fudge_factor;
%     g_wind = g_wind / (fudge_factor *(1 - Month_real(i,:)*r_wind));
%     max_rz = floor(max(u_wind)) ;
%     min_rz = ceil(min(u_wind));
%     list_wind{i} = min_rz:max_rz;
%     dis_q_wind{i} = interp1(u_wind,g_wind,list_wind{i});
%     dis_q_wind{i} = dis_q_wind{i}/sum(dis_q_wind{i});
%     display(i);
% end
end

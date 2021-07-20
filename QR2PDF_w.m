function [list_wind, dis_q_wind] = QR2PDF_w(quant, pred_data, scale_factor)

pred_data = pred_data/scale_factor;



real_predict_data_add100 = pred_data; % before[pred_data(1)-1, pred_data]
quant_wind_add100 = quant;
if abs(real_predict_data_add100(end)) < 100
    d = 1;             % Sampling interval for the PDF model 
else
    d = 5;           %before 2
end                 % Sampling interval for the PDF model 
rho = 10^9/d^3;        % regularization parameter before, 10^9/d^3
[x_wind,q_wind] = qCDF2PDF(real_predict_data_add100, quant_wind_add100,d,rho);
x_wind  = x_wind * scale_factor;

ind = (q_wind < 0);
q_wind(ind) = [];
x_wind(ind) = [];

max_rz = floor(max(x_wind)) ;
min_rz = ceil(min(x_wind));
list_wind = min_rz:max_rz;
dis_q_wind = interp1(x_wind,q_wind,list_wind);
dis_q_wind = dis_q_wind/sum(dis_q_wind);
                       
                       
                       
                       
end

function [q_demand_comb,sample_value_comb] = tailSample_new(quant_demand, pred_data,...
                paramEsts_right, paramEsts_left)
% input: 
%quant_demand : the array of quantile of demand
%beta_demand, alpha_demand, rho_demand: estimated parameter in QR model
%paramEsts_right,paramEsts_left: estimated QR tail model, pareto
%                                   distributed
%time_d_real: matrix that implies the date, month and hour

kHat_left      = paramEsts_left(1);   % Tail index parameter
sigmaHat_left  = paramEsts_left(2);

kHat_right      = paramEsts_right(1);   % Tail index parameter
sigmaHat_right  = paramEsts_right(2);

sample_right = (quant_demand(end)+0.0001) : 0.0005 : 0.9999;
sample_left = .0001 : 0.0005 :(quant_demand(1) - 0.0001);


% sample of left and right tail through q 
sample_value_right = double(pred_data(end) + ...
             gpinv((sample_right-quant_demand(end))/(1 - quant_demand(end)), kHat_right,sigmaHat_right));
sample_value_left =double( pred_data(1) - ...
             gpinv((quant_demand(1) - sample_left)/quant_demand(1), kHat_left,sigmaHat_left));
q_demand_comb = [sample_left,quant_demand,sample_right];
sample_value_comb = [sample_value_left, pred_data, sample_value_right];


end
function [paramEsts_left,paramEsts_right,Error_left_pos,Error_right_pos] = demand_tail_est(tail_n,quant, ...
    pred_demand_clear, Demand_clear)

% q_left =   quant(tail_n);
% q_right = quant(n_quant + 1 - tail_n);

n_quant = length(quant);
pred_left = pred_demand_clear(:,tail_n);
pred_right = pred_demand_clear(:,n_quant + 1 - tail_n);

Error_left = pred_left - Demand_clear;
Error_right = Demand_clear - pred_right;
Error_left = sort(Error_left,'descend');
Error_right = sort(Error_right,'descend');

Error_left_pos = Error_left(Error_left > 0 );
Error_right_pos = Error_right(Error_right > 0);

paramEsts_left = gpfit(Error_left_pos);
% kHat_left      = paramEsts_left(1);   % Tail index parameter
% sigmaHat_left  = paramEsts_left(2);

paramEsts_right = gpfit(Error_right_pos);
% kHat_right      = paramEsts_right(1);   % Tail index parameter
% sigmaHat_right  = paramEsts_right(2);
end
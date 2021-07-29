function [beta_demand, alpha_demand, r_wind_M_H, r_solar_M_H, pred_demand] = ...
    opt_demand_solar_power(quant, Demand, Date, Holidays, Wind, Solar,lambda, window)


Demand = Demand/window;
Wind = Wind/window;
Solar = Solar/window;

Month = month(Date);
Weekday = weekday(Date);
Hour = hour(Date);

n_r = length(Date);
Months = zeros(n_r,12);
Weekdays = zeros(n_r,7);
Hours = zeros(n_r,24);
Month_Hour = zeros(n_r, 24*12);
% Holidays = datetime(['01-Jan-2016';'18-Jan-2016';...  
%     '15-Feb-2016'; '30-May-2016';...
%     '04-Jul-2016'; '05-Sep-2016'; '10-Oct-2016';...
%     '11-Nov-2016';'24-Nov-2016';'26-Dec-2016']);
Holidayis =ismember(Date,Holidays);             % create holiday array of the year


for i = 1:12
    Months(:,i) = (Month == i);
end
for i = 1:7
    Weekdays(:,i) = (Weekday == i);
end
for i = 1:24
        Hours(:,i) = (Hour == i - 1);    
end
for i = 1:24
    for j = 1:12
        Month_Hour(:, 24*(j-1)+i) = (Hour == i-1) .* (Month == j);
    end
end
        
Holiday = [Holidayis,1 - Holidayis];




n_quant = length(quant);
A1 = [Months, Weekdays, Hours, Holiday]; % A1 is the model matrix
n = 12 + 7 + 24 + 2;    % 44 regressor in total, 12 months, 7 weekdays, 24 hours, and one holiday

% A2_month = Months;                    % A2 is the matrix to month, in order to indentify different
% A2_hour = Hours;                                % \gamma  for wind     


diff_matrix_beta = diff(diag(ones(n_quant,1)));         % first differentiate matrix, in order
                                                        % to have penalty on beta 
diff_matrix_alpha = diff(diag(ones(n_quant-2,1)),2);  % sencond differentiate matrix, in order
                                                        % to have penalty on alpha 

diff_matrix_q = diff(diag(ones(n_quant,1)));                                                         
                                                        % transfer_matrix = kron(diag(ones(12,1)),ones(1,24));
diff_mon = diff(diag(ones(12,1)),2); % 2nd 
diff_hour = diff(diag(ones(24,1)),2);
% diff_hour_night = diff_hour;
% diff_hour_night(7:24,:) = 0;
% diff_hour_day = diff_hour;
% diff_hour_day(1:6,:) = 0;
                                                        
A1 = sparse(A1);
A2_month = sparse(Months);
A2_hour = sparse(Hours);
A3_month_hour = sparse(Month_Hour);


lambda1 = lambda(1);     % penalty on beta 
lambda2 = lambda(2);      % penalty on alpha
lambda3 = lambda(3);    % penalty on gamma, these three could be larger and need to be adjusted
lambda4 = lambda(4);
lambda5 = lambda(5);
lambda6 = lambda(6);
lambda7 = lambda(7);


%% solve 
% cvx_solver SeDuMi
tic;
cvx_begin
    variables beta_demand(n_quant,n) alpha_demand(n_quant) 
    variables r_wind_M_H(288)  r_solar_M_H(288)
    % ob1 is the object on quantile model
    ob1_pre1 = A1 * beta_demand' + ones(n_r,1)*alpha_demand' ...
        + A3_month_hour * r_wind_M_H .* Wind * ones(1,n_quant)...
        + A3_month_hour * r_solar_M_H .* Solar * ones(1,n_quant);

%         + A2_hour * r_wind_H .* Wind * ones(1,n_quant)  ...
        %+ A2_hour * r_solar_H .* Solar * ones(1,n_quant);
    ob1_pre2 = Demand * ones(1,n_quant) - ob1_pre1;
    ob1 = 0.5 * sum(sum(abs(ob1_pre2))) + ones(1,n_r) * ob1_pre2 * (quant - 0.5)';
    % ob2 is the regulization of beta
    ob2_pre = diff_matrix_beta * beta_demand;
    ob2 = sum(sum(ob2_pre .* ob2_pre));
    %ob3 is the regulization of alpha
    ob3_pre = diff_matrix_alpha * alpha_demand(2:(end-1));
    ob3 = sum(ob3_pre .* ob3_pre);
%     %ob4 is the regulization of r_wind_M
%     ob4_pre = r_wind_M - circshift(r_wind_M,1);
%     ob4 = sum( ob4_pre .* ob4_pre);
%     %%ob5 is the regulization of r_wind_H    
%     ob5_pre = r_wind_H - circshift(r_wind_H,1);
%     ob5 = sum( ob5_pre .* ob5_pre);
    % combine ob4 and ob5
    r_wind_matrix = reshape(r_wind_M_H,[24,12]);
    ob4_pre = r_wind_matrix - circshift(r_wind_matrix,1);
    ob4 = sum(sum( ob4_pre .* ob4_pre));
    %%ob7 is the regulization of r_solar_month
    r_wind_matrix_t = r_wind_matrix';
    ob5_pre = r_wind_matrix_t - circshift(r_wind_matrix_t,1);
    ob5 = sum(sum( ob5_pre .* ob5_pre));
    %%ob6 is the regulization of r_solar_hour
    r_solar_matrix = reshape(r_solar_M_H,[24,12]);
    ob6_pre = r_solar_matrix- circshift(r_solar_matrix,1) ;
    ob6 = sum(sum( ob6_pre .* ob6_pre));
    %%ob7 regu of r_solar_month
    r_solar_matrix_t = r_solar_matrix';
    ob7_pre = r_solar_matrix_t - circshift(r_solar_matrix_t,1);
    ob7 = sum(sum( ob7_pre .* ob7_pre));
    
    %%ob8 regulation norm of r_solar
    ob8 = sum(r_solar_M_H.*r_solar_M_H);
    
    pred_demand = A1 * beta_demand' + ones(n_r,1)*alpha_demand' + ...
    A3_month_hour * r_wind_M_H .* Wind * ones(1,n_quant) + ...
    A3_month_hour * r_solar_M_H .* Solar * ones(1,n_quant);
%     A2_hour * r_wind_H .* Wind * ones(1,n_quant) + ...
    
%     %ob6 is the regulization of r_solar_M
%     ob6_pre =  r_solar_M - circshift(r_solar_M,1);
%     ob6 = sum(ob6_pre .* ob6_pre);
%     %ob7 is the regulization of r_solar_H
%     ob7_pre =  r_solar_H - circshift(r_solar_H,1);
%     ob7 = sum(ob7_pre .* ob7_pre);
    
    minimize( ob1 + lambda1 * ob2 + lambda2 * ob3 + lambda3 *ob4 ...
           + lambda4 * ob5 + lambda5 * ob6 + lambda6 * ob7 + lambda7*ob8);
    % for q = 0.05 0.1 0.15 0.2 and q = 0.8 0.85 0.9 0.95, I assume beta
    % are constant at each side
    subject to
        diff(diag(ones(2,1))) * beta_demand(1:2,:) == 0
        diff(diag(ones(2,1))) * beta_demand((end-1):end,:) == 0
        r_solar_matrix(1:6,:) == 0
        r_solar_matrix(21:24,:) == 0
        pred_demand(:,1) >= 0 
        diff_matrix_q * pred_demand' >= 0  
%         sum(r_wind_M_H)==0
cvx_end
toc;

%     A2_hour * r_solar_H .* Solar * ones(1,n_quant)...
%     + A2_month * r_solar_M .* Solar * ones(1,n_quant);
% beta_demand; 
% alpha_demand; 
% r_wind;
beta_demand = beta_demand * window;
alpha_demand = alpha_demand * window;
pred_demand = pred_demand * window;


end

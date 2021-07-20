%validation 


load solar13_17
load LoadCA
load solar18
load load2016_r
load load2017_r
load load2018_r


load2018_r = load2018_r';

load_train = [load2016_r, load2017_r]'/max(load2018_r) * 1000;

load_test = load2018_r/max(load2018_r) * 1000;
solar13_17 = solar13_17 * 1000;
solar2018 = solar2018 * 1000;

n_load_train = length(load_train);
solar16_17 = solar13_17(end-n_load_train+1:end);

%2018 time
t18s = datetime(2018,1,1,0,0,0);
t18e = datetime(2018,12,31,23,0,0);

%13_17 time
t13s = datetime(2013, 1, 1, 0, 0, 0);
t17e = datetime(2017,12,31,23,0,0);

%16_17 time
t16s = datetime(2016, 1, 1, 0, 0, 0);


date13_17 = (t13s : hours(1) : t17e)';
date16_17 = (t16s : hours(1) : t17e)';
date18 = (t18s : hours(1) : t18e)';

n_data18 = length(date18); 

% Holidays_18 = datetime(['01-Jan-2018';'15-Jan-2018';...  % the holiday of 2017
%     '19-Feb-2018';'30-Mar-2018'; '28-May-2018';...
%     '04-Jul-2018'; '03-Sep-2018'; ...
%     '22-Nov-2018';'25-Dec-2018']);

Holidays_18 = holidays(t18s, t18e);
Holidays_16_17 = holidays(t16s, t17e);


quant = 0.05:.1:.95;
n_quant = length(quant);

[A_solar_train, month_hour_matrix_train_solar] = date_matrix(date13_17,'solar', 0);
[A_demand_train, month_hour_matrix_train] = date_matrix(date16_17,'demand', Holidays_18);


[A_solar_test, month_hour_matrix_test_solar] = date_matrix(date18,'solar', 0);
[A_demand_test, month_hour_matrix_test] = date_matrix(date18,'demand', Holidays_18);

%% Indicating Matrix for simplified  data
Month_real = kron(speye(12), ones(24*7*2,1));
Hour_real = kron(ones(12*7,1),kron(speye(24),ones(2,1)));
Weekday_real = kron(ones(12,1), kron(speye(7),ones(24*2,1)));
Holiday_real = kron(ones(12*7*24,1),speye(2));

time_real_solar_power = kron(speye(12),kron(ones(7,1),kron(speye(24),ones(2,1))));
time_real_variance = kron(speye(12),kron(speye(7),kron(speye(24),ones(2,1))));

time_real_w_s = [Month_real,Hour_real];
time_real_demand = [Month_real,Weekday_real,Hour_real,Holiday_real];


day_night_indicator13_17 = dayNight(diff(solar13_17), date13_17(2:end));
day_night_indicator18 = dayNight(diff(solar2018), date18(2:end));
% 
% lambda = [10^(-1), 10^(0)]; % HyperParameter
% [beta_solar_ramp,alpha_solar_ramp, pred_solar_ramp] =opt_solar_ramp(quant,diff(solar2018), date18(2:end),lambda);

 
j = 1
for i = 1:7
    show = sprintf('i : %d, j : %d ', i,j);
    disp(show)
    lambda = [10^(5), 10^(2)];  % -1,-1

    r_solar13_17 = diff(solar13_17);
    r_solar18 = diff(solar2018);

    [beta_solar_ramp,alpha_solar_ramp, pred_solar_ramp] =opt_solar_ramp(quant, r_solar13_17 , date13_17(2:end),lambda);        
    pred_solar_train = (month_hour_matrix_train_solar * beta_solar_ramp' + ones(length(date13_17),1)*alpha_solar_ramp');
%          [train_stats,bin_summary_train] = test_error(quant, r_solar13_17 , pred_solar_train(2:end,:));


%         [beta_solar_ramp,alpha_solar_ramp, pred_solar_ramp] =opt_solar_ramp(quant, r_solar18, date18(2:end),lambda);

%         [beta_solar,alpha_solar, pred_solar] =opt_solar_power(quant_s_w,Solar12_16, date12_16,lambda);  
%         [beta_solar, alpha_solar,pred_solar] =opt_solar(quant_solar, day_night_indicator,Solar_train, date15,lambda);
    %train error
    pred_solar_test = (month_hour_matrix_test_solar * beta_solar_ramp' + ones(length(date18),1)*alpha_solar_ramp');
     ind_train = find(day_night_indicator13_17>0);
     ind_test = find(day_night_indicator18>0);

     pred_solar_test = pred_solar_test(2:end, :);
     pred_solar_train = pred_solar_train(2:end, :);

     pred_solar_train_chosen = pred_solar_train(ind_train,:);
     pred_solar_test_chosen = pred_solar_test(ind_test,:);

     Solar_train_chosen = r_solar13_17(ind_train);
     Solar_test_chosen = r_solar18(ind_test);

     [train_stats,bin_summary_train] = test_error(quant, Solar_train_chosen , pred_solar_train_chosen)
     [test_stats,bin_summary_test] = test_error(quant, Solar_test_chosen , pred_solar_test_chosen)

%         %test error
%         pred_solar_test = (month_hour_matrix_test_solar * beta_solar_ramp' + ones(length(date17),1)*alpha_solar_ramp');
% %          ind = find(day_night_indicator17>0);
% %          pred_solar_test_chosen = pred_solar_test(ind,:);
% %          Solar_test_chosen = Solar17(ind,:);
%         [test_stats,bin_summary_test] = test_error(quant, Solar_test_chosen , pred_solar_test)
    test_stats_train_list(i) = train_stats;
    test_stats_test_list(i) = test_stats;

end



% validate load
test_stats_train_list_demand = 0;
test_stats_test_list_demand  = 0;
for i = 1:8
    show = sprintf('i : %d, j : %d ', i,j);
    disp(show)
    lambda = [10^(1),10^(0),10^(3), 10^(3), 10^(1)];
    
    r_demand_train = diff(load_train);
    r_demand_test = diff(load_test);
    r_solar_train = diff(solar16_17);
    r_solar_test = diff(solar2018);
    
    [beta_demand_ramp, alpha_demand_ramp, r_demand_solar_M_H_ramp, pred_demand_ramp] = ...
    opt_demand_solar_ramp(quant, r_demand_train, date16_17(2:end),...
    Holidays_16_17(2:end),  r_solar_train, lambda);
    

    figure
    x = 1 : 24;
    y = 1 : 12;
    F = (reshape(r_demand_solar_M_H_ramp,[24,12]))';
    % F= r_demand_solar_M_H_ramp';
    surf(x,y,F)
    xlim([1,24])
    ylim([1,12])
    zlim([-.8,.6])
    xlabel('Hour')
    ylabel('Month')
    zlabel('Impact Factor')
    zlabel('\Gamma_{LS}')
    
    title('Coefficients \gamma(Month,Hour)')
    
    pred_demand_train = A_demand_train * beta_demand_ramp' + ones(length(date16_17),1)*alpha_demand_ramp' + ...
        month_hour_matrix_train * r_demand_solar_M_H_ramp  .* [0; r_solar_train] * ones(1,n_quant);
    
    pred_demand_test = A_demand_test * beta_demand_ramp' + ones(length(date18),1)*alpha_demand_ramp' + ...
        month_hour_matrix_test * r_demand_solar_M_H_ramp  .* [0; r_solar_test] * ones(1,n_quant);
    
    pred_demand_train = pred_demand_train(2:end, :);
    pred_demand_test = pred_demand_test(2:end, :);
    %train error
    [train_stats,bin_summary_train] = test_error(quant, r_demand_train , pred_demand_train)
    [test_stats,bin_summary_test] = test_error(quant, r_demand_test , pred_demand_test)
    
    test_stats_train_list_demand(i) = train_stats;
    test_stats_test_list_demand(i) = test_stats;

end


















Holidays_15 = datetime(['01-Jan-2015';'19-Jan-2015';...  % the holiday of 2015
    '16-Feb-2015'; '25-May-2015';...
    '03-Jul-2015'; '07-Sep-2015'; '12-Oct-2015';...
    '11-Nov-2015';'26-Nov-2015';'25-Dec-2015']);
Holidays_16 = datetime(['01-Jan-2016';'18-Jan-2016';...  % the holiday of 2016
    '15-Feb-2016'; '30-May-2016';...
    '04-Jul-2016'; '05-Sep-2016'; '10-Oct-2016';...
    '11-Nov-2016';'24-Nov-2016';'26-Dec-2016']);
Holidays_17 = datetime(['02-Jan-2017';'16-Jan-2017';...  % the holiday of 2017
    '20-Feb-2017'; '29-May-2017';...
   '04-Jul-2017'; '04-Sep-2017'; '09-Oct-2017';...
    '10-Nov-2017';'23-Nov-2017';'25-Dec-2017']);


% datetime(a,'Format','d-MMM-y')

t12s = datetime(2012,1,1,0,0,0);
t12e = datetime(2012,12,31,23,0,0);
%2015
t15s = datetime(2015,1,1,0,0,0);
t15e = datetime(2015,12,31,23,0,0);
%2016
t16s = datetime(2016,1,1,0,0,0);
t16e = datetime(2016,12,31,23,0,0);
%2017
t17s = datetime(2017,1,1,0,0,0);
t17e = datetime(2017,12,31,23,0,0);
%date
date12_16 = (t12s : hours(1) : t16e)';
date15_16 = (t15s : hours(1) : t16e)';
date17 = (t17s : hours(1) : t17e)';

n_data12_16 = length(date12_16); % data length of 2012 - 2016
n_data15_16 = length(date15_16); % data length of 2015 - 2016
n_data17 = length(date17);       % data length of 2017



[A_solar_train, month_hour_matrix_train_solar] = date_matrix(date12_16,'solar', 0);
[A_solar_test, month_hour_matrix_test_solar] = date_matrix(date17,'solar', 0);

[A_demand_train, month_hour_matrix_train] = date_matrix(date15_16,'demand', Holidays_15_16);
[A_demand_test, month_hour_matrix_test] = date_matrix(date17,'demand', Holidays_17);

test_stats_train_list = zeros(3,3);
test_stats_test_list = zeros(3,3);
day_night_indicator12_16 = dayNight(Solar12_16, date12_16);
day_night_indicator17 = dayNight(Solar17, date17);
n_quant = length(quant);


parfor i = 2:3
        show = sprintf('i : %d, j : %d ', i,j);
        disp(show)
        lambda = [10^(-1), 10^(0)];  % -1,-1
        [beta_solar,alpha_solar, pred_solar] =opt_solar_power(quant_s_w,Solar12_16, date12_16,lambda);  
%         [beta_solar, alpha_solar,pred_solar] =opt_solar(quant_solar, day_night_indicator,Solar_train, date15,lambda);
        %train error
        pred_solar_train = (month_hour_matrix_train_solar * beta_solar' + ones(length(date12_16),1)*alpha_solar');
         ind = find(day_night_indicator12_16>0);
         pred_solar_train_chosen = pred_solar_train(ind,:);
         Solar_train_chosen = Solar12_16(ind);
        [train_stats,bin_summary_train] = test_error(quant, Solar_train_chosen , pred_solar_train_chosen)
        
        %test error
        pred_solar_test = (month_hour_matrix_test_solar * beta_solar' + ones(length(date17),1)*alpha_solar');
         ind = find(day_night_indicator17>0);
         pred_solar_test_chosen = pred_solar_test(ind,:);
         Solar_test_chosen = Solar17(ind,:);
        [test_stats,bin_summary_test] = test_error(quant, Solar_test_chosen , pred_solar_test_chosen)
        test_stats_train_list(i,j) = train_stats;
        test_stats_test_list(i,j) = test_stats;

end


%% Wind Validation
[A_wind_train, month_hour_matrix_train] = date_matrix(date15_16,'wind', 0);
[A_wind_test, month_hour_matrix_test] = date_matrix(date17,'wind', 0);

wind_train = zeros(1);
wind_test = zeros(1);
for i = 1:6
    lambda_wind = [10^(0),10^(1),10^(4),10^(4),1000];
    [beta_wind, alpha_wind,  r_solar_M_H, pred_wind] = ...
        opt_wind_solar_power(quant_s_w,lambda_wind, Wind15_16, date15_16, Solar15_16);
    r_wind_solar_matrix = reshape(r_solar_M_H,[24,12]);

    figure
    x = 1 : 24;
    y = 1 : 12;
    F = (reshape(r_solar_M_H,[24,12]))';
    surf(x,y,F)
    xlim([1,24])
    ylim([1,12])
    xlabel('Hour')
    ylabel('Month')
    zlabel('Total impact factor')

    pred_wind_train = A_wind_train * beta_wind' + ones(n_data15_16,1)*alpha_wind' + ...
        month_hour_matrix_train * r_solar_M_H  .* Solar15_16 * ones(1,n_quant+1);
    pred_wind_test = A_wind_test * beta_wind' + ones(n_data17,1)*alpha_wind' + ...
         month_hour_matrix_test * r_solar_M_H .* Solar17 * ones(1,n_quant+1);
    [train_stats,bin_summary_train] = test_error(quant, Wind15_16 , pred_wind_train)
    [test_stats,bin_summary_test] = test_error(quant, Wind2017 , pred_wind_test)

    wind_train(i) = train_stats;
    wind_test(i) = test_stats;
end




[A_demand_train, month_hour_matrix_train] = date_matrix(date15_16,'demand', Holidays_15_16);
[A_demand_test, month_hour_matrix_test] = date_matrix(date17,'demand', Holidays_17);

demand_train = zeros(1);
demand_test = zeros(1);
a = [0.5,5,10,50];
for i = 1:5
    show = sprintf('i : %d', i);
    disp(show)
    lambda_demand_power = [10^(3),10^(0), 10^(6),10^(6),10^(7),10^(1),0];
%     [beta_demand, alpha_demand, r_wind_Month_Hour, r_solar_Month_Hour, pred_demand] = opt_demand_solar(...
%         quant_demand, Demand_train, date15, Holidays_15, Wind_train, Solar_train,lambda);
    [beta_demand, alpha_demand, r_demand_wind_M_H, r_demand_solar_M_H, pred_demand] = ...
        opt_demand_solar_power(quant, Demand15_16, date15_16,...i
        Holidays_15_16, Wind15_16, Solar15_16,lambda_demand_power);%lambda 7 parameters

    r_demand_wind_matrix  = reshape(r_demand_wind_M_H,[24,12]);
    r_demand_solar_matrix  = reshape(r_demand_solar_M_H,[24,12]);


%     figure
%     x = 1 : 24;
%     y = 1 : 12;
%     F = r_demand_wind_matrix';
%     surf(x,y,F)
%     xlim([1,24])
%     ylim([1,12])
%     xlabel('Hour')
%     ylabel('Month')
%     zlabel('Total impact factor')
%     title('Wind Impact on Demand COEFFICIENTS \gamma(Month,Hour)')

    figure
    x = 1 : 24;
    y = 1 : 12;
    F = (reshape(r_demand_solar_M_H,[24,12]))';
    surf(x,y,F)
    xlim([1,24])
    ylim([1,12])
    xlabel('Hour')
    ylabel('Month')
    zlabel('Impact Factor \beta_{LS}')
    title('COEFFICIENTS \beta(Month,Hour)')
    
    pred_demand_train = A_demand_train * beta_demand' + ones(n_data15_16,1)*alpha_demand' + ...
    month_hour_matrix_train * r_demand_wind_M_H .* Wind15_16 * ones(1,n_quant) + ...
    month_hour_matrix_train * r_demand_solar_M_H .* Solar15_16 * ones(1,n_quant);
    %
    [test_stats15,bin_summary15] =test_error(quant, Demand15_16, pred_demand_train)
    demand_train(i) = test_stats15;

    pred_demand_test = A_demand_test * beta_demand' + ones(n_data17,1)*alpha_demand' + ...
    month_hour_matrix_test * r_demand_wind_M_H .* Wind2017 * ones(1,n_quant) + ...
    month_hour_matrix_test * r_demand_solar_M_H .* Solar17 * ones(1,n_quant);
        [test_stats16,bin_summary16] = test_error(quant, Demand2017, pred_demand_test)
        disp(test_stats16)
        demand_test(i) = test_stats16;
end





lambda = [10^(0), 10^(-1)];  
quant = 0.05:.1:.95;
[beta_solar,alpha_solar, pred_solar] =opt_solar_power(quant,WA_Solar_11_15, date11_15,lambda);




%% QQ and PP plot 
subplot(1,3,1)

 [paramEsts_left,paramEsts_right,Error_left_pos,Error_right_pos] 
Error_left_pos_sort = sort(Error_left_pos)
 
Em_summer_quan = gpcdf(Error_left_pos_sort,paramEsts_left(1),paramEsts_left(2));
plot(Em_summer_quan)

[paramEsts_left,paramEsts_right,Error_left_pos,Error_right_pos] = ...
    demand_tail_est(tail_n,quant,pred_demand, conv_hour(window, Demand15_16));
%% QQ plot
kHat_left = paramEsts_left(1);
sigmaHat_left = paramEsts_left(2);
kHat_right = paramEsts_right(1);
sigmaHat_right = paramEsts_right(2);
q_left = quant(1);
q_right = quant(end);

set(0,'defaultAxesFontName','Times New Roman');
set(0,'defaultAxesFontSize',2.7*k_scaling);

set(0,'defaultTextFontName','Times New Roman');
set(0,'defaultTextFontSize',2.7*k_scaling);

figure
subplot(1,3,1)
% left qq model
n_left = length(Error_left_pos);
theoretical_q = (0:(n_left-1))/n_left;
theoretical_quantile = gpinv(theoretical_q,kHat_left,sigmaHat_left);
Error_left_pos_sort = sort(Error_left_pos,'descend');
theoretical_quantile_sort = sort(theoretical_quantile,'descend');


scatter(-theoretical_quantile_sort/1000,-Error_left_pos_sort/1000,'+')
hold on
plot(0:-0.001:-.1,0:-0.001:-.1,'-r','LineWidth',2)
 xlim([-.1,0])
 ylim([-.1,0])
% xlabel('GW')
% ylabel('GW')
title('LEFT TAIL QQ (q=0.05)');

% plot(pred_demand(1,:)) 
% sample_right = .95 : 0.0001 : 0.999;
% sample_value_right = pred_demand(1,end) + gpinv((sample_right-q_right)/0.05, kHat_right,sigmaHat_right);
% sample_left = .0001 : 0.0001 : 0.05;
% sample_value_left = pred_demand(1,1) - gpinv((q_left - sample_left)/0.05, kHat_left,sigmaHat_left);
% plot([sample_left,quant,sample_right],[sample_value_left, pred_demand(1,:), sample_value_right]) 
% 
box on
axis square
subplot(1,3,2)
%PP plot
n_r = size(pred_demand,1);
p_demand_model = zeros(n_r,1);
for i = 1 : n_r
    if Demand15_16(i) < pred_demand(i,1)
        p_demand_model(i) = q_left - gpcdf((pred_demand(i,1) - Demand15_16(i)),kHat_left,sigmaHat_left)*q_left;
    elseif Demand15_16(i) > pred_demand(i,end)
        p_demand_model(i) = q_right + gpcdf((Demand15_16(i) - pred_demand(i,end)),kHat_right,sigmaHat_right)*(1-q_right);
    else 
        p_demand_model(i) = interp1(pred_demand(i,:),quant,Demand15_16(i));
    end
end
p_demand_model_sort = sort(p_demand_model);
empirical_q = (1:n_r)/n_r;
plot(empirical_q,p_demand_model_sort,'+b')
xlim([0,1])
ylim([0,1])
title('DEMAND QR MODEL PP')
hold on
plot(0:0.01:1,0:0.01:1,'-r','LineWidth',2)
box on
axis square
subplot(1,3,3)
% right qq model
n_right = length(Error_right_pos);
theoretical_q = (0:(n_right-1))/n_right;
theoretical_quantile_right = gpinv(theoretical_q,kHat_right,sigmaHat_right);
Error_right_pos_sort = sort(Error_right_pos);
theoretical_quantile_sort_right = sort(theoretical_quantile_right);
scatter(theoretical_quantile_sort_right/1000,Error_right_pos_sort/1000,'+')
hold on
plot(0:0.001:.3,0:.001:.3,'-r','LineWidth',2)
xlim([0,.3])
ylim([0,.3])
% xlabel('GW')
% ylabel('GW')
title('RIGHT TAIL QQ (q=0.95)')
box on
axis square

x_width =18.2386; % NOT TESTED, MIGHT REQUIRE TWEAKING
y_height = 4.4667; % NOT TESTED, MIGHT REQUIRE TWEAKING
FigHandle = gcf;
set(FigHandle, 'PaperUnits', 'centimeters');
set(FigHandle, 'PaperPosition', [0 0 x_width y_height]);
%%% set figure to have no margins
FigHandle.PaperPositionMode = 'auto';
fig_pos = FigHandle.PaperPosition;
FigHandle.PaperSize = [fig_pos(3) fig_pos(4)];
%i= 2052
print(gcf, '-bestfit','-dpdf', 'plots/QQ_PP.pdf');


figure; plot(median_xl_list)
hold on 
scatter(1:length(comb_train), -Demand15_16* Demand_sc_train)

%% L - W - S validation 
comb_train =-( Demand15_16 * Demand_sc_train - Wind15_16 * Wind_sc_train - Solar15_16 * Solar_sc_train);
comb_test =-(Demand2017 * Demand_sc_train  - Wind2017 * Wind_sc_train - Solar17 *1000 * Solar_sc_train);
train_bins = zeros(11,1);
median_x_list = [0,0,0];
median_xl_list= [0,0,0];
median_xw_list= [0,0,0];
median_xs_list= [0,0,0];
for i = 1:length(comb_train)
    Hour_temp = hour(date15_16(i)) + 1;
    Month_temp = month(date15_16(i));
    
    r_lw = r_demand_wind_matrix(Hour_temp, Month_temp);
    r_ls = r_demand_solar_matrix(Hour_temp, Month_temp);
    r_ws = r_wind_solar_matrix(Hour_temp, Month_temp);
    
    demand_tilde =   A_demand_train(i,:) * beta_demand' + alpha_demand';
    demand_factor = Demand_sc_train ; 
        
    wind_tilde = A_wind_train(i,:) * beta_wind' +  alpha_wind';
    wind_factor = Wind_sc_train - Demand_sc_train * r_lw;
    wind_scale = wind_tilde * wind_factor;
%         Wind_nameplate = wind_factor * 1000 * window;

    solar_tilde = (month_hour_matrix_train_solar2(i,:) * beta_solar' +  alpha_solar')*1000;
    solar_factor = Wind_sc_train * r_ws + Solar_sc_train - Demand_sc_train * r_lw * r_ws ...
        - Demand_sc_train * r_ls;
    solar_scale = solar_tilde * solar_factor;
%         Solar_nameplate =  solar_factor*1000 * window;

    [q_demand_comb,demand_sample] = tailSample_new(quant, demand_tilde,...
        paramEsts_right, paramEsts_left);
    demand_sample_scale = demand_sample * demand_factor;
    [list_demand,dis_q_demand] = QR2PDF_demand(q_demand_comb,demand_sample_scale, ...
        1);
    list_demand = - list_demand;
    min_demand = min(list_demand);
    max_demand = max(list_demand);
    num_demand = 1 : (window) : length(list_demand);
    list_demand_center = list_demand - min_demand;
    list_demand_center = list_demand_center(num_demand);
    dis_q_demand = dis_q_demand(num_demand) * window;
    dis_q_demand = fliplr(dis_q_demand);


    if abs(wind_scale(1) - wind_scale(end)) <= 24
        mid = round((wind_scale(1) + wind_scale(end))/2);
        min_wind = mid;
        list_wind_center = 0;
        dis_q_wind = 1;
    else
        [list_wind, dis_q_wind] = QR2PDF_w(quant_s_w, wind_scale);
        min_wind = min(list_wind);
        max_wind = max(list_wind);
        list_wind_center = list_wind - min_wind;
        num_wind = 1 : (window) : length(list_wind);
        list_wind_center = list_wind_center(num_wind);        
        dis_q_wind = dis_q_wind(num_wind) * window;
    end
    %Solar QR to PDF

    if abs(solar_scale(end) - solar_scale(1)) <= 24
        mid = round((solar_scale(end) + solar_scale(1))/2);
        min_Solar = mid;
        list_solar_center = 0;
        dis_q_solar = 1;
    else
        [list_solar, dis_q_solar] = QR2PDF_s(quant_s_w, solar_scale);
        min_Solar = min(list_solar);
        max_Solar = max(list_solar);

        list_solar_center = list_solar - min_Solar;
        num_solar = 1 : (window) : length(list_solar);
        list_solar_center = list_solar_center(num_solar);        
        dis_q_solar = dis_q_solar(num_solar) * window ;
    end

    q_G_L_W_S = conv(conv(dis_q_demand, dis_q_wind),dis_q_solar);
    q_G_L_W_S = q_G_L_W_S/sum(q_G_L_W_S);
    x_G_L_W_S = 1 : (window ) : ((length(q_G_L_W_S) - 1) * (window) + 1);
    
    %% W+S-L
    [~,median_q1] = min(abs(cumsum(q_G_L_W_S) - 0.1));
    median_x1 =  x_G_L_W_S(median_q1);
    median_x_list(i,1) = median_x1 + min_wind + min_demand + min_Solar ;
    
    [~,median_q2] = min(abs(cumsum(q_G_L_W_S) - 0.5));
    median_x2 =  x_G_L_W_S(median_q2);
    median_x_list(i,2) = median_x2 + min_wind + min_demand + min_Solar;
    
    [~,median_q3] = min(abs(cumsum(q_G_L_W_S) - 0.9));
    median_x3 =  x_G_L_W_S(median_q3);
    median_x_list(i,3) = median_x3 + min_wind + min_demand + min_Solar;
    
    %% -L
    [~,median_q1] = min(abs(cumsum(dis_q_demand) - 0.1));
    median_x1 =  x_G_L_W_S(median_q1);
    median_xl_list(i,1) = median_x1 + min_demand;
    
    [~,median_q2] = min(abs(cumsum(dis_q_demand) - 0.5));
    median_x2 =  x_G_L_W_S(median_q2);
    median_xl_list(i,2) = median_x2 + min_demand;
    
    [~,median_q3] = min(abs(cumsum(dis_q_demand) - 0.9));
    median_x3 =  x_G_L_W_S(median_q3);
    median_xl_list(i,3) = median_x3 + min_demand;
    
    %% W
    [~,median_q1] = min(abs(cumsum(dis_q_wind) - 0.1));
    median_x1 =  x_G_L_W_S(median_q1);
    median_xw_list(i,1) = median_x1 + min_wind;
    
    [~,median_q2] = min(abs(cumsum(dis_q_wind) - 0.5));
    median_x2 =  x_G_L_W_S(median_q2);
    median_xw_list(i,2) = median_x2 + min_wind;
    
    [~,median_q3] = min(abs(cumsum(dis_q_wind) - 0.9));
    median_x3 =  x_G_L_W_S(median_q3);
    median_xw_list(i,3) = median_x3 + min_wind;
    
    %% S
    [~,median_q1] = min(abs(cumsum(dis_q_solar) - 0.1));
    median_x1 =  x_G_L_W_S(median_q1);
    median_xs_list(i,1) = median_x1 + min_Solar;
    
    [~,median_q2] = min(abs(cumsum(dis_q_solar) - 0.5));
    median_x2 =  x_G_L_W_S(median_q2);
    median_xs_list(i,2) = median_x2 + min_Solar;
    
    [~,median_q3] = min(abs(cumsum(dis_q_solar) - 0.9));
    median_x3 =  x_G_L_W_S(median_q3);
    median_xs_list(i,3) = median_x3 + min_Solar;
        
    x_G_L_W_S = x_G_L_W_S  + min_wind + min_demand + min_Solar;
    
    [~, x_id] = min(abs(x_G_L_W_S - comb_train(i)));
    q_id = sum(q_G_L_W_S(1:x_id));
    bin_id  = floor((q_id - 0.05) *10) + 2;
    train_bins(bin_id) = train_bins(bin_id) + 1;
    
    if ~mod(i,100)
            display(i);
    end
end
pearson_test(train_bins, quant)


median_x_list = [0,0,0];
median_xl_list= [0,0,0];
median_xw_list= [0,0,0];
median_xs_list= [0,0,0];
test_bins = zeros(11,1);
for i = 1:length(comb_test)
    Hour_temp = hour(date17(i)) + 1;
    Month_temp = month(date17(i));
    
    r_lw = r_demand_wind_matrix(Hour_temp, Month_temp);
    r_ls = r_demand_solar_matrix(Hour_temp, Month_temp);
    r_ws = r_wind_solar_matrix(Hour_temp, Month_temp);
    
    demand_tilde =   A_demand_test(i,:) * beta_demand' + alpha_demand';
    demand_factor = Demand_sc_train ; 
        
    wind_tilde = A_wind_test(i,:) * beta_wind' +  alpha_wind';
    wind_factor = Wind_sc_train - Demand_sc_train * r_lw;
    wind_scale = wind_tilde * wind_factor;
%         Wind_nameplate = wind_factor * 1000 * window;

    solar_tilde = (month_hour_matrix_test_solar(i,:) * beta_solar' +  alpha_solar')*1000;
    solar_factor = Wind_sc_train * r_ws + Solar_sc_train - Demand_sc_train * r_lw * r_ws ...
        - Demand_sc_train * r_ls;
    solar_scale = solar_tilde * solar_factor;
%         Solar_nameplate =  solar_factor*1000 * window;

    [q_demand_comb,demand_sample] = tailSample_new(quant, demand_tilde,...
        paramEsts_right, paramEsts_left);
    demand_sample_scale = demand_sample * demand_factor;
    [list_demand,dis_q_demand] = QR2PDF_demand(q_demand_comb,demand_sample_scale, ...
        1);
    list_demand = - list_demand;
    min_demand = min(list_demand);
    max_demand = max(list_demand);
    num_demand = 1 : (window) : length(list_demand);
    list_demand_center = list_demand - min_demand;
    list_demand_center = list_demand_center(num_demand);
    dis_q_demand = dis_q_demand(num_demand) * window;
    dis_q_demand = fliplr(dis_q_demand);


    if abs(wind_scale(1) - wind_scale(end)) <= 24
        mid = round((wind_scale(1) + wind_scale(end))/2);
        min_wind = mid;
        list_wind_center = 0;
        dis_q_wind = 1;
    else
        [list_wind, dis_q_wind] = QR2PDF_w(quant_s_w, wind_scale);
        min_wind = min(list_wind);
        max_wind = max(list_wind);
        list_wind_center = list_wind - min_wind;
        num_wind = 1 : (window) : length(list_wind);
        list_wind_center = list_wind_center(num_wind);        
        dis_q_wind = dis_q_wind(num_wind) * window;
    end
    %Solar QR to PDF

    if abs(solar_scale(end) - solar_scale(1)) <= 24
        mid = round((solar_scale(end) + solar_scale(1))/2);
        min_Solar = mid;
        list_solar_center = 0;
        dis_q_solar = 1;
    else
        [list_solar, dis_q_solar] = QR2PDF_s(quant_s_w, solar_scale);
        min_Solar = min(list_solar);
        max_Solar = max(list_solar);

        list_solar_center = list_solar - min_Solar;
        num_solar = 1 : (window) : length(list_solar);
        list_solar_center = list_solar_center(num_solar);        
        dis_q_solar = dis_q_solar(num_solar) * window ;
    end

    q_G_L_W_S = conv(conv(dis_q_demand, dis_q_wind),dis_q_solar);
    q_G_L_W_S = q_G_L_W_S/sum(q_G_L_W_S);
    x_G_L_W_S = 1 : (window ) : ((length(q_G_L_W_S) - 1) * (window) + 1);
    
    x_G_L_W_S = x_G_L_W_S  + min_wind + min_demand + min_Solar;
    
    [~, x_id] = min(abs(x_G_L_W_S - comb_test(i)));
    q_id = sum(q_G_L_W_S(1:x_id));
    bin_id  = floor((q_id - 0.05) *10) + 2;
    test_bins(bin_id) = test_bins(bin_id) + 1;
    
    if ~mod(i,100)
            display(i);
    end
end


pearson_test(test_bins, quant)



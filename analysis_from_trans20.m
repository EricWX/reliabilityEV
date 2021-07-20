clear all
figure
x= datasample(load2019_r,100)/1000;
cdfplot(x)
h = cdfplot(x)
[h,stats] = cdfplot(x)
%% load data solar, wind, demand
% load load2016_r
load load2017_r
load load2018_r
load load2019_r

% load_train = [load2017_r, load2018_r]';
load_train = [load2017_r, load2018_r]'/max(load2019_r) * 1000 /.93;
load_test = load2019_r'/max(load2019_r) * 1000;

load solar_train
load solar_test

solar_train_sub = solar_train(end-17520+1:end) * 1000;
solar_train = solar_train * 1000;
solar_test = solar_test * 1000;

load wind_train
load wind_test

wind_train = wind_train * 1000;
wind_test = wind_test * 1000;

%% time

%2019 time
t19s = datetime(2019,1,1,0,0,0);
t19e = datetime(2019,12,31,23,0,0);

%15_18 time
t15s = datetime(2015, 1, 1, 0, 0, 0);
t18e = datetime(2018,12,31,23,0,0);

%17_18 time
t17s = datetime(2017, 1, 1, 0, 0, 0);

date15_18 = (t15s : hours(1) : t18e)';
date17_18 = (t17s : hours(1) : t18e)';
date19 = (t19s : hours(1) : t19e)';


n_data19 = length(date19); 

Holidays_17_18 = holidays(t17s, t18e);
Holidays_19 = holidays(t19s, t19e);

quant = 0.05:.1:.95;


Month_real = kron(speye(12), ones(24*7*2,1));
Hour_real = kron(ones(12*7,1),kron(speye(24),ones(2,1)));
Weekday_real = kron(ones(12,1), kron(speye(7),ones(24*2,1)));
Holiday_real = kron(ones(12*7*24,1),speye(2));

time_real_solar_power = kron(speye(12),kron(ones(7,1),kron(speye(24),ones(2,1))));
time_real_variance = kron(speye(12),kron(speye(7),kron(speye(24),ones(2,1))));

time_real_w_s = [Month_real,Hour_real];
time_real_demand = [Month_real,Weekday_real,Hour_real,Holiday_real];
p_holiday = length(Holidays_19)/365/(24*7*12);
p_not_holiday = (1-length(Holidays_19)/365)/(24*7*12);
n_state = 24 * 12 * 7 * 2;
weight_q_list = zeros(n_state,1);
time_real_r = zeros(n_state,12*24);
weight_q_list_month = zeros(n_state,12);


for i = 1 : n_state
    month_temp = find(Month_real(i,:) == 1);
    
    time_real_r(i,:) = kron(Month_real(i,:), Hour_real(i,:));
    temp_time_real_demand = time_real_demand(i,:);
    if temp_time_real_demand(end) == 0 
        weight_q = p_holiday * 365 * 24;
    else 
        weight_q = p_not_holiday * 365 * 24;
    end
    weight_q_list(i) = weight_q;
    weight_q_list_month(i, month_temp) = weight_q;
end

%% Scenario Input

% %90/10 load 
% Load9010_2015 = prctile(Demand2015,90)/sum(Demand2015);
% Load9010_2016 = prctile(Demand2016,90)/sum(Demand2016);
% Load9010_2017 = prctile(Demand2017,90)/sum(Demand2017);
% 
% ratio9010_annual = mean([Load9010_2015,Load9010_2016,Load9010_2017]);
% sum(Demand201)
% %wind namaple
% ratiowind_2015 = max(Wind2015)/sum(Wind2015);
% ratiowind_2016 = max(Wind2016)/sum(Wind2016);
% ratiowind_2017 = max(Wind2017)/sum(Wind2017);
% 
% rationameplate_annual = mean([ratiowind_2015,ratiowind_2016,ratiowind_2017]);

 
%% Solar Optimization
lambda = [10^(-1), 10^(0)]; % HyperParameter
quant_s_w = [0, quant,1];
[beta_solar,alpha_solar, pred_solar] =opt_solar_power(quant_s_w, solar_train, date15_18,lambda);

% %% Wind Optimization 
% lambda = [10^(1),10^(1),10^(1),10^(3),10^(2)]; % HyperParameter
% lambda = [10^(0),10^(1),10^(2),10^(4),10^(-3)];
lambda = [10^(0),10^(1),10^(4),10^(4),1000];
 [beta_wind, alpha_wind,  r_solar_M_H, pred_wind] = ...
    opt_wind_solar_power(quant_s_w,lambda, wind_train, date17_18, solar_train_sub);
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
zlabel('Impact Factor')

%% Demand Optimization
% 
% lambda_demand_power = [10^(3),10^(-2), 10^(4),10^(1),10^(7)*7,10^(3),10^(1)];
% lambda_demand_power = [10^(3),10^(-2), 10^(4),10^(1),10^(7)*7,10^(2),10^(-2)];
lambda_demand_power = [10^(3),10^(0), 10^(6),10^(6),10^(7),10^(1),0];
[beta_demand, alpha_demand, r_demand_wind_M_H, r_demand_solar_M_H, pred_demand] = ...
    opt_demand_solar_power(quant, load_train, date17_18,...
    Holidays_17_18, wind_train, solar_train_sub, lambda_demand_power);%lambda 7 parameters
r_demand_wind_matrix  = reshape(r_demand_wind_M_H,[24,12]);
r_demand_solar_matrix  = reshape(r_demand_solar_M_H,[24,12]);


figure
x = 1 : 24;
y = 1 : 12;
F = r_demand_wind_matrix';
surf(x,y,F)
xlim([1,24])
ylim([1,12])
xlabel('Hour')
ylabel('Month')
zlabel('Impact Factor')
% title('Wind Impact on Demand COEFFICIENTS \gamma(Month,Hour)')
    
figure
x = 1 : 24;
y = 1 : 12;
F = (reshape(r_demand_solar_M_H,[24,12]))';
surf(x,y,F)
xlim([1,24])
ylim([1,12])
xlabel('Hour')
ylabel('Month')
zlabel('Impact Factor')
% title('Solar Impact on Demand COEFFICIENTS \beta(Month,Hour)')
 % Demand tail
% tail_n = 1;
% [Var_paramEsts_left,Var_paramEsts_right,Var_Error_left_pos,Var_Error_right_pos] = ...
%     demand_tail_est(tail_n,quant,kron(pred_variance_15,ones(12,1)), variance_demand_15');
% p_power_g_0 = 0;


tail_n = 1;
[paramEsts_left,paramEsts_right,Error_left_pos,Error_right_pos] = ...
    demand_tail_est(tail_n,quant,pred_demand, load_train);


%% Generation - Outage
load generationunits
op = (generationunits.Status == 'OP');
p1 = (generationunits.Primary1 ~= 'NA');
p2 = (generationunits.Primary1 ~= 'WND');
p3 = (generationunits.Primary1 ~= 'WH');
p4 = (generationunits.Primary1 ~= 'SUN');

%coal
p5 = (generationunits.Primary1 ~= 'BIT');
p6 = (generationunits.Primary1 ~= 'COL');
p7 = (generationunits.Primary1 ~= 'LIG');
p8 = (generationunits.Primary1 ~= 'PC');
p9 = (generationunits.Primary1 ~= 'SC');
p10 = (generationunits.Primary1 ~= 'SUB');
p11 = (generationunits.Primary1 ~= 'WC');
%OIL
p12 = (generationunits.Primary1 ~= 'DFO');
p13 = (generationunits.Primary1 ~= 'JF');
p14 = (generationunits.Primary1 ~= 'KER');
p15 = (generationunits.Primary1 ~= 'OIL');
p16 = (generationunits.Primary1 ~= 'RFO');
p17 = (generationunits.Primary1 ~= 'WO');

%Nature Gas
p18 = (generationunits.Primary1 ~= 'BFG');
p19 = (generationunits.Primary1 ~= 'GAS');
p20 = (generationunits.Primary1 ~= 'NG');
p21 = (generationunits.Primary1 ~= 'OG');
p22 = (generationunits.Primary1 ~= 'PG');


c1 = (generationunits.Capacity  > 20);
n1 = (generationunits.netMWh  > 10000); % Filter the generation units


Eford = 6.97/100;
id1_ge = op & p1 & p2 & p3 & p4 & c1 & n1;
id2_coal = p5 & p6 & p7 & p8 & p9 & p10 & p11;
id3_oil = p12 & p13 & p14 & p15 & p16 & p17;
id4_ng = p18 & p19 & p20 & p21 & p22;
Capacity = generationunits.Capacity(id1_ge & id2_coal & id3_oil & id4_ng);
Capacity_ng = sort(generationunits.Capacity(~p20),'descend');
Capacity_ng = Capacity_ng(1:125);
Capacity = [Capacity; Capacity_ng];
sum(Capacity)
% sum(generationunits.Capacity(op & p1 & p2 & p3 & p4 & c1 & n1))
Probability = ones(size(Capacity))*(Eford);

window = 1;
[x_MW,dist_conv] = Cap_Outage(Probability, Capacity,window);
% %Distribution
% min_x_MW = min(x_MW);
% max_x_MW = max(x_MW);
% x_MW_center = x_MW - min_x_MW;
% num_x_MW = 1 : (window * 500) : length(x_MW);
% x_MW_center_used = x_MW_center(num_x_MW);
% dist_conv_select = zeros(length(x_MW_center_used),1);
% loc_value = find(dist_conv>0);
% for i = 1:length(loc_value)
%    [~,loc_temp] = min(abs(num_x_MW - loc_value(i)));
%    dist_conv_select(loc_temp) = dist_conv(loc_value(i));
% end

%% Indicating Matrix for real data
[A_wind_train, month_hour_matrix_train] = date_matrix(date17_18,'wind', 0);
[A_wind_test, month_hour_matrix_test] = date_matrix(date19,'wind', 0);

[A_solar_train, month_hour_matrix_train_solar] = date_matrix(date15_18,'solar', 0);
[A_solar_train2, month_hour_matrix_train_solar2] = date_matrix(date17_18,'solar', 0);
[A_solar_test, month_hour_matrix_test_solar] = date_matrix(date19,'solar', 0);

[A_demand_train, month_hour_matrix_train] = date_matrix(date17_18,'demand', Holidays_17_18);
[A_demand_test, month_hour_matrix_test] = date_matrix(date19,'demand', Holidays_19);

%% Indicating Matrix for simplified  data
Month_real = kron(speye(12), ones(24*7*2,1));
Hour_real = kron(ones(12*7,1),kron(speye(24),ones(2,1)));
Weekday_real = kron(ones(12,1), kron(speye(7),ones(24*2,1)));
Holiday_real = kron(ones(12*7*24,1),speye(2));

time_real_solar_power = kron(speye(12),kron(ones(7,1),kron(speye(24),ones(2,1))));
time_real_variance = kron(speye(12),kron(speye(7),kron(speye(24),ones(2,1))));

time_real_w_s = [Month_real,Hour_real];
time_real_demand = [Month_real,Weekday_real,Hour_real,Holiday_real];
p_holiday = length(Holidays_17_18)/365/(24*7*12);
p_not_holiday = (1-length(Holidays_17_18)/365)/(24*7*12);
n_state = 24 * 12 * 7 * 2;
weight_q_list = zeros(n_state,1);
time_real_r = zeros(n_state,12*24);

for i = 1 : n_state
    time_real_r(i,:) = kron(Month_real(i,:), Hour_real(i,:));
    temp_time_real_demand = time_real_demand(i,:);
    if temp_time_real_demand(end) == 0 
        weight_q = p_holiday * 365 * 24;
    else 
        weight_q = p_not_holiday * 365 * 24;
    end
    weight_q_list(i) = weight_q;
end




%% Energy find lower and upper boundary
beta_demand_list =cell(1,24) ;
alpha_demand_list = cell(1,24);
beta_wind_list = cell(1,24);
alpha_wind_list = cell(1,24);
beta_solar_list = cell(1,24);
alpha_solar_list = cell(1,24);
r_lw_list = cell(1,24);
r_ls_list = cell(1,24);
r_ws_list = cell(1,24);
pred_demand_list = cell(1,24);
x_MW_list = cell(1,24);
dist_conv_list = cell(1,24);



beta_wind_list3 = cell(1,24);
alpha_wind_list3 = cell(1,24);
r_ws_list3 = cell(1,24);


for window = 1:24
    
%     conv_demand = conv_hour(window, load_train);
% %     wind
     conv_wind = conv_hour(window, wind_train);
     conv_solar_sub = conv_hour(window, solar_train_sub);
%     conv_solar = conv_hour(window, solar_train);
    
    % QR demand 
    Date_window = date17_18(window:end);
    Date15_18_window = date15_18(window:end);
    
    
    % energy opt for demand
%     lambda_demand_power = [10^(3),10^(-2), 10^(4),10^(1),10^(7)*7,10^(3),10^(1)];
%     lambda_demand_power = [10^(3),10^(0), 10^(6),10^(6),10^(7),10^(1),0];
%     [beta_demand, alpha_demand, r_demand_wind_M_H, r_demand_solar_M_H, pred_demand] = ...
%         opt_demand_solar_power(quant, conv_demand, Date_window,...
%         Holidays_17_18, conv_wind, conv_solar_sub,lambda_demand_power, window);%lambda 7 parameters
%     r_demand_wind_matrix  = reshape(r_demand_wind_M_H,[24,12]);
%     r_demand_solar_matrix  = reshape(r_demand_solar_M_H,[24,12]);
%     energy opt for wind    
%     lambda = [10^(0),10^(1),10^(2),10^(4),10^(-3)];
%     lambda_wind = [10^(0),10^(1),10^(4),10^(4),1000];

     lambda_wind = [10^(-1),10^(-1),10^(6),10^(6),10^(1)];
     scale = ceil(max(conv_wind)/1000);
     [beta_wind, alpha_wind,  r_solar_M_H, pred_wind] = ...
         opt_wind_solar_power(quant_s_w,lambda_wind, conv_wind, ...
         Date_window, conv_solar_sub, scale);
     r_wind_solar_matrix = reshape(r_solar_M_H,[24,12]);
     
%     energy opt for solar
%     lambda = [10^(-1), 10^(0)];
%     scale = ceil(max(conv_solar)/1000);
%     [beta_solar, pred_solar] =opt_solar_power([0,quant_s_w],conv_solar, Date15_18_window,lambda, scale);
%     
%     beta_demand_lzist{window} = beta_demand ;
% %     alpha_demand_list{window} = alpha_demand;
     beta_wind_list3{window} = beta_wind;
     alpha_wind_list3{window} = alpha_wind ;
%     beta_solar_list3{window} = beta_solar;
%     alpha_solar_list3{window} = alpha_solar;
%     r_lw_list{window} = r_demand_wind_matrix;
%     r_ls_list{window} = r_demand_solar_matrix;
    r_ws_list3{window} = r_wind_solar_matrix;
%     pred_demand_list{window} = pred_demand;
    display("window finished:" + window)
end



for window = 1:24
    maxSolar_list = maxSolar(conv_hour(window, solar_train), date15_18(window:end));
    beta_solar_list{window}(end,:) = maxSolar_list;
end

for window = 1:24
  
   
    conv_solar = conv_hour(window, solar_train);
    
    Date15_18_window = date15_18(window:end);
    
 
%     energy opt for solar
    lambda = [10^(-1), 10^(0)];
    [beta_solar,alpha_solar, pred_solar] =opt_solar_power(quant_s_w,conv_solar, Date15_18_window,lambda, 20*window);
% 
    display("window finished:" + window)
end

%% Scenario Input
% new_set = 1;
% if new_set == 0
%     load beta_demand_list
%     load alpha_demand_list
%     load beta_wind_list
%     load alpha_wind_list
%     load beta_solar_list
%     load alpha_solar_list
%     load r_lw_list
%     load r_ls_list
%     load r_ws_list
%     load pred_demand_list
% else
    load beta_demand_list_2
    load alpha_demand_list_2
    load beta_wind_list
    load alpha_wind_list
    load beta_solar_list
%     load alpha_solar_list
    load r_lw_list_2
    load r_ls_list_2
    load r_ws_list
    load pred_demand_list_2
    
    load alpha_wind_list3
    load beta_wind_list3
    load r_ws_list3
    
    
    beta_demand_list = beta_demand_list_2;
    alpha_demand_list = alpha_demand_list_2;
%     beta_wind_list = beta_wind_list_2;
%     alpha_wind_list = alpha_wind_list_2;
    r_lw_list = r_lw_list_2;
    r_ls_list = r_ls_list_2;
%     r_ws_list = r_ws_list_2;
    pred_demand_list = pred_demand_list_2;
    
    r_ws_list = r_ws_list3;
    alpha_wind_list = alpha_wind_list3;
    beta_wind_list = beta_wind_list3;
    
% end
%% EV Profile
% real estimate
EV_profile = [450,250,200,150,120,130,...
          200,300,280,260,250,240,...
          260,320, 390, 550, 680,880,...
          1000,950, 880, 820,730, 600];

% profile set
      
EV_power_profile = EV_profile/sum(EV_profile) * 38000;
EV_Energy_profile = (EV_power_profile + circshift(EV_power_profile,1))/2




%% Scenario 0

Demand_sc_train = 46.487; %current demand
Solar_sc_train =  14.116  ; %  3/1/2021
Wind_sc_train = 6.890; % 3/1/2021
Fixed_transfer = 10000;

pump_storage_power = 4500; %4500MW/4500MWh
pump_storage_energy = 4500; %4500MW/4500MWh
pump_period = 1; %1h

battery_storage_power = 500;% 1500MW/6000MWh 
battery_storage_energy = 2000;% 1500MW/6000MWh 
battery_period = 4; %4h

EV_daily = 8.9857*1000; % GWh 100% 30% - 38GWh


Battery_op = [-1/6,-1/6,-1/6,-1/6,-1/6,-1/6,...
              0,0,0,0,0,0,...
              0,0,0,0,0.00,0.2,...
              0.3,0.25,0.2,0.05,0,0];
Pump_op= Battery_op;

EV_op_perfect = -[450,250,200,150,120,130,...
          200,300,280,260,250,240,...
          260,320, 390, 550, 680,880,...
          1000,950, 880, 820,730, 600];
EV_op_perfect = - EV_op_perfect/sum(EV_op_perfect);



load generationunits
op = (generationunits.Status == 'OP');
p1 = (generationunits.Primary1 ~= 'NA');
p2 = (generationunits.Primary1 ~= 'WND');
p3 = (generationunits.Primary1 ~= 'WH');
p4 = (generationunits.Primary1 ~= 'SUN');

%coal
p5 = (generationunits.Primary1 ~= 'BIT');
p6 = (generationunits.Primary1 ~= 'COL');
p7 = (generationunits.Primary1 ~= 'LIG');
p8 = (generationunits.Primary1 ~= 'PC');
p9 = (generationunits.Primary1 ~= 'SC');
p10 = (generationunits.Primary1 ~= 'SUB');
p11 = (generationunits.Primary1 ~= 'WC');
%OIL
p12 = (generationunits.Primary1 ~= 'DFO');
p13 = (generationunits.Primary1 ~= 'JF');
p14 = (generationunits.Primary1 ~= 'KER');
p15 = (generationunits.Primary1 ~= 'OIL');
p16 = (generationunits.Primary1 ~= 'RFO');
p17 = (generationunits.Primary1 ~= 'WO');

%Nature Gas
p18 = (generationunits.Primary1 ~= 'BFG');
p19 = (generationunits.Primary1 ~= 'GAS');
p20 = (generationunits.Primary1 ~= 'NG');
p21 = (generationunits.Primary1 ~= 'OG');
p22 = (generationunits.Primary1 ~= 'PG');


c1 = (generationunits.Capacity  > 20);
n1 = (generationunits.netMWh  > 10000); % Filter the generation units


Eford = 6.97/100;
id1_ge = op & p1 & p2 & p3 & p4 & c1 & n1;
id2_coal = p5 & p6 & p7 & p8 & p9 & p10 & p11;
id3_oil = p12 & p13 & p14 & p15 & p16 & p17;
id4_ng = p18 & p19 & p20 & p21 & p22;
Capacity = generationunits.Capacity(id1_ge & id2_coal & id3_oil & id4_ng);
Capacity_ng = sort(generationunits.Capacity(~p20),'descend');
Capacity_ng = Capacity_ng(1:107);
Capacity = [Capacity; Capacity_ng];
sum(Capacity)
% sum(generationunits.Capacity(op & p1 & p2 & p3 & p4 & c1 & n1))
Probability = ones(size(Capacity))*(Eford);


%% Scenario 1

Demand_sc_train = 46.487; %current demand
Solar_sc_train =  14.116 ; %  3/1/2021
Wind_sc_train = 6.890; % 3/1/2021
Fixed_transfer = 10000;

pump_storage_power = 4500; %4500MW/4500MWh
pump_storage_energy = 4500; %4500MW/4500MWh
pump_period = 1; %1h

battery_storage_power = 1500;% 1500MW/6000MWh 
battery_storage_energy = 6000;% 1500MW/6000MWh 
battery_period = 4; %4h

EV_daily = 113.7432*1000; % GWh 100% 30% - 38GWh


Battery_op = [-1/6,-1/6,-1/6,-1/6,-1/6,-1/6,...
              0,0,0,0,0,0,...
              0,0,0,0,0,0.2,...
              0.3,0.3,0.2,0,0,0];
Pump_op= Battery_op;

EV_op_perfect = [-1/40,-1/20,-1/20,-1/20,-1/20,-1/20,...
              -1/20,-1/20,-1/20 ,-3/40,-3/40,-1/10,...
              -3/40,-3/40,-1/20,-1/20,0,0, ...
              0,0,0,0,-1/40,-1/20];


Capacity = generationunits.Capacity(id1_ge & id2_coal & id3_oil & id4_ng);
Capacity_ng = sort(generationunits.Capacity(~p20),'descend');
Capacity_ng = Capacity_ng(1:125);
Capacity = [Capacity; Capacity_ng];
sum(Capacity)
% sum(generationunits.Capacity(op & p1 & p2 & p3 & p4 & c1 & n1))
Probability = ones(size(Capacity))*(Eford);
          

%% Scenario 2

Demand_sc_train = 46.487; %current demand
Solar_sc_train =  25.9 ; %  3/1/2021
Wind_sc_train = 6.890 + 10; % 3/1/2021
Fixed_transfer = 10000;

pump_storage_power = 4500; %4500MW/4500MWh
pump_storage_energy = 4500; %4500MW/4500MWh
pump_period = 1; %1h

battery_storage_power = 1500;% 1500MW/6000MWh 
battery_storage_energy = 6000;% 1500MW/6000MWh 
battery_period = 4; %4h

EV_daily = 113.7432*1000/3; % GWh 100% 30% - 38GWh


Battery_op =  [-1/6,-1/6,-1/6,-1/6,-1/6,-1/6,...
              0,0,0,0,0,0,...
              0,0,0,0,0.00,0.1,...
              0.25,0.25,0.25,0.15,0,0];
Pump_op= Battery_op;

EV_op_perfect = [-1/160,-1/40,-1/40,-1/40,-1/40,-1/40,...
              -1/20,-5/80,-7/80 ,-4/40,-4/40,-1/10,...
              -4/40,-4/40,-3/40,-4/80,-1/40,0, ...
              0,0,0,0,0,-1/160];


Capacity = generationunits.Capacity(id1_ge & id2_coal & id3_oil & id4_ng);
Capacity_ng = sort(generationunits.Capacity(~p20),'descend');
Capacity_ng = [Capacity_ng(1:90);Capacity_ng(110)] ;
Capacity = [Capacity; Capacity_ng];
sum(Capacity)
% sum(generationunits.Capacity(op & p1 & p2 & p3 & p4 & c1 & n1))
Probability = ones(size(Capacity))*(Eford);



%% Scenario C

Demand_sc_train = 46.487; %current demand
Solar_sc_train =  58 ; %  3/1/2021
Wind_sc_train = 36; % 3/1/2021
Fixed_transfer = 20000;

pump_storage_power = 6000; %4500MW/4500MWh
pump_storage_energy = 6000; %4500MW/4500MWh
pump_period = 1; %1h

battery_storage_power = 4500;% 1500MW/6000MWh 
battery_storage_energy = 18000;% 1500MW/6000MWh 
battery_period = 4; %4h

EV_daily = 113.7432*1000; % GWh 100% 30% - 38GWh


Battery_op = [0,0,0,0,0,-1/48,...
              -3/48,-5/48,-5/48,-1/12,-1/12,-5/48,...
              -5/48,-5/48,-5/48,-1/12,-1/24,0.1,...
              0.2,0.2,0.25,0.15,0.05,0.05];
Pump_op= Battery_op;

EV_op_perfect = [-1/80,-1/80,-1/80,-1/40,-1/40,-1/80,...
              -3/80,-5/80,-7/80 ,-4/40,-4/40,-1/10,...
              -4/40,-4/40,-3/40,-5/80,-4/80,-1/80, ...
              0,0,0,0,0,-1/80];


load generationunits
op = (generationunits.Status == 'OP');
p1 = (generationunits.Primary1 ~= 'NA');
p2 = (generationunits.Primary1 ~= 'WND');
p3 = (generationunits.Primary1 ~= 'WH');
p4 = (generationunits.Primary1 ~= 'SUN');

%coal
p5 = (generationunits.Primary1 ~= 'BIT');
p6 = (generationunits.Primary1 ~= 'COL');
p7 = (generationunits.Primary1 ~= 'LIG');
p8 = (generationunits.Primary1 ~= 'PC');
p9 = (generationunits.Primary1 ~= 'SC');
p10 = (generationunits.Primary1 ~= 'SUB');
p11 = (generationunits.Primary1 ~= 'WC');
%OIL
p12 = (generationunits.Primary1 ~= 'DFO');
p13 = (generationunits.Primary1 ~= 'JF');
p14 = (generationunits.Primary1 ~= 'KER');
p15 = (generationunits.Primary1 ~= 'OIL');
p16 = (generationunits.Primary1 ~= 'RFO');
p17 = (generationunits.Primary1 ~= 'WO');

%Nature Gas
p18 = (generationunits.Primary1 ~= 'BFG');
p19 = (generationunits.Primary1 ~= 'GAS');
p20 = (generationunits.Primary1 ~= 'NG');
p21 = (generationunits.Primary1 ~= 'OG');
p22 = (generationunits.Primary1 ~= 'PG');


c1 = (generationunits.Capacity  > 20);
n1 = (generationunits.netMWh  > 10000); % Filter the generation units


Eford = 6.97/100;
id1_ge = op & p1 & p2 & p3 & p4 & c1 & n1;
id2_coal = p5 & p6 & p7 & p8 & p9 & p10 & p11;
id3_oil = p12 & p13 & p14 & p15 & p16 & p17;
id4_ng = p18 & p19 & p20 & p21 & p22;
Capacity = generationunits.Capacity(id1_ge & id2_coal & id3_oil & id4_ng) * 1.6;
% Capacity_ng = sort(generationunits.Capacity(~p20),'descend');
% Capacity_ng = Capacity_ng(1:125);
% Capacity = [Capacity; Capacity_ng];
sum(Capacity)
% sum(generationunits.Capacity(op & p1 & p2 & p3 & p4 & c1 & n1))
Probability = ones(size(Capacity))*(Eford);




%% Scenario D


Demand_sc_train = 46.487; %current demand
Solar_sc_train =  58 ; %  3/1/2021
Wind_sc_train = 48.5; % 3/1/2021
Fixed_transfer = 19000;

pump_storage_power = 6000; %4500MW/4500MWh
pump_storage_energy = 6000; %4500MW/4500MWh
pump_period = 1; %1h

battery_storage_power = 4500;% 1500MW/6000MWh 
battery_storage_energy = 18000;% 1500MW/6000MWh 
battery_period = 4; %4h

EV_daily = 113.7432*1000; % GWh 100% 30% - 38GWh


Battery_op = [0,-1/24,-1/24,-1/24,-1/24,-1/12,...
              -1/12,-1/12,-1/12,0,0,-1/12,...
              -1/12,-1/12,-1/12,-1/12,-1/12,0.1,...
              0.2,0.2,0.25,0.15,0.05,0.05];
          
% Battery_op = [0,-1/24,-1/24,-1/24,-1/24,-1/12,...
%               -1/24,-1/24,-1/12,-1/12,-1/12,-1/12,...
%               -1/12,-1/12,-1/12,-1/12,-1/48,0.1,...
%               0.2,0.2,0.25,0.15,0.05 + 1/48,0.05];
% Pump_op= Battery_op;

          
Pump_op= Battery_op;

EV_op_perfect = [-1/80,-1/80,-1/80,-1/40,-1/40,-1/80,...
              -3/80,-5/80,-7/80 ,-4/40,-4/40,-1/10,...
              -4/40,-4/40,-3/40,-5/80,-4/80,-1/80, ...
              0,0,0,0,0,-1/80];


load generationunits
op = (generationunits.Status == 'OP');
p1 = (generationunits.Primary1 ~= 'NA');
p2 = (generationunits.Primary1 ~= 'WND');
p3 = (generationunits.Primary1 ~= 'WH');
p4 = (generationunits.Primary1 ~= 'SUN');

%coal
p5 = (generationunits.Primary1 ~= 'BIT');
p6 = (generationunits.Primary1 ~= 'COL');
p7 = (generationunits.Primary1 ~= 'LIG');
p8 = (generationunits.Primary1 ~= 'PC');
p9 = (generationunits.Primary1 ~= 'SC');
p10 = (generationunits.Primary1 ~= 'SUB');
p11 = (generationunits.Primary1 ~= 'WC');
%OIL
p12 = (generationunits.Primary1 ~= 'DFO');
p13 = (generationunits.Primary1 ~= 'JF');
p14 = (generationunits.Primary1 ~= 'KER');
p15 = (generationunits.Primary1 ~= 'OIL');
p16 = (generationunits.Primary1 ~= 'RFO');
p17 = (generationunits.Primary1 ~= 'WO');

%Nature Gas
p18 = (generationunits.Primary1 ~= 'BFG');
p19 = (generationunits.Primary1 ~= 'GAS');
p20 = (generationunits.Primary1 ~= 'NG');
p21 = (generationunits.Primary1 ~= 'OG');
p22 = (generationunits.Primary1 ~= 'PG');


c1 = (generationunits.Capacity  > 20);
n1 = (generationunits.netMWh  > 10000); % Filter the generation units


Eford = 6.97/100;
id1_ge = op & p1 & p2 & p3 & p4 & c1 & n1;
id2_coal = p5 & p6 & p7 & p8 & p9 & p10 & p11;
id3_oil = p12 & p13 & p14 & p15 & p16 & p17;
id4_ng = p18 & p19 & p20 & p21 & p22;
Capacity = generationunits.Capacity(id1_ge & id2_coal & id3_oil & id4_ng) * 1.6;
% Capacity_ng = sort(generationunits.Capacity(~p20),'descend');
% Capacity_ng = Capacity_ng(1:125);
% Capacity = [Capacity; Capacity_ng];
sum(Capacity)
% sum(generationunits.Capacity(op & p1 & p2 & p3 & p4 & c1 & n1))
Probability = ones(size(Capacity))*(Eford);



% 
% input_capacity = 32000;
% 
% power_capacity = generation_capacity + input_capacity;
% % ramp_capacity_up = 6000
% Import Profile


f_factor = 1.4;
Demand_sc_train = 125*10^6 * Demand_scale_factor * f_factor; %current demand
% Wind_sc_train = 1.9136; %5.625 MW
Wind_sc_train = 2.8*10^6 * Wind_scale_factor;
Solar_sc_train = 0.5698 ; % 2.1MW
generation_scale = 1;
Battery_energy_capacity = 0;
Battery_power_capacity = 0;
Pump_energy_capacity = 1.5*10^3*8;
Pump_power_capacity = 1.5*10^3;
Fixed_transfer = 1.7*10^3; %GW




LOLH_list = zeros(24,1);
p_power_g_upper = zeros(n_state,1);
p_power_g_0_list={};

for window = 1
    beta_demand = beta_demand_list{window} ;
    alpha_demand = alpha_demand_list{window};
    beta_wind = beta_wind_list{window};
    alpha_wind = alpha_wind_list{window};
    beta_solar = beta_solar_list{window};
%     alpha_solar = alpha_solar_list{window};
    r_demand_wind_matrix = r_lw_list{window};
    r_demand_solar_matrix = r_ls_list{window};
    r_wind_solar_matrix = r_ws_list{window};
    pred_demand = pred_demand_list{window};
    
    sample_size = window*5;

    tail_n = 1;
    [paramEsts_left,paramEsts_right,Error_left_pos,Error_right_pos] = ...
        demand_tail_est(tail_n,quant,pred_demand, conv_hour(window, load_train));
%     G_capacity = 5.5 * 10^3 * generation_scale; % kw

%     gas_turbines = ones(7,1) * G_capacity;
%     Outage_prob = ones(7,1) * Eford/100;
% 
%     [x_MW,dist_conv] = Cap_Outage(Outage_prob, gas_turbines,window);

    [x_MW,dist_conv] = Cap_Outage(Probability, Capacity, window);

    min_x_MW = min(x_MW);
    max_x_MW = max(x_MW);
    x_MW_center = x_MW - min_x_MW;
    num_x_MW = 1 : (sample_size) : length(x_MW);
    x_MW_center_used = x_MW_center(num_x_MW);
    dist_conv_select = zeros(length(x_MW_center_used),1);
    loc_value = find(dist_conv>0);
    for i = 1:length(loc_value)
       [~,loc_temp] = min(abs(num_x_MW - loc_value(i)));
       dist_conv_select(loc_temp) = dist_conv(loc_value(i));
    end
    dist_conv_select = dist_conv_select/sum(dist_conv_select);

    p_power_g_0 = zeros(n_state,1);

    parfor i = 1:n_state
%         bat = battery_assign_day;
%         pump = pump_assign_day;
        r_lw = Hour_real(i,:) * r_demand_wind_matrix * Month_real(i,:)';
        r_ls = Hour_real(i,:) * r_demand_solar_matrix * Month_real(i,:)';
        r_ws = Hour_real(i,:) * r_wind_solar_matrix * Month_real(i,:)';


        demand_tilde =   time_real_demand(i,:) * beta_demand' + alpha_demand';
        demand_factor = Demand_sc_train ;


        wind_tilde = time_real_w_s(i,:) * beta_wind' +  alpha_wind';
        wind_factor = Wind_sc_train - Demand_sc_train * r_lw;
        wind_scale = wind_tilde * wind_factor;
%         Wind_nameplate = wind_factor * 1000 * window;

        solar_tilde = (time_real_solar_power(i,:) * beta_solar');
        solar_factor = Wind_sc_train * r_ws + Solar_sc_train - Demand_sc_train * r_lw * r_ws ...
            - Demand_sc_train * r_ls;
        solar_scale = solar_tilde * solar_factor;
%         Solar_nameplate =  solar_factor*1000 * window;

        [q_demand_comb,demand_sample] = tailSample_new(quant, demand_tilde,...
            paramEsts_right, paramEsts_left);
        demand_sample_scale = demand_sample * demand_factor;
        scale = ceil(max(demand_sample_scale)/(2000*demand_factor));
        [list_demand,dis_q_demand] = QR2PDF_demand(q_demand_comb,demand_sample_scale, ...
            scale);
%         figure;plot(list_demand, cumsum(dis_q_demand));
%         hold on
%         plot(demand_sample_scale,q_demand_comb)
%         
        list_demand = - list_demand;
        min_demand = min(list_demand);
        max_demand = max(list_demand);
        num_demand = 1 : (sample_size) : length(list_demand);
        list_demand_center = list_demand - min_demand;
        list_demand_center = list_demand_center(num_demand);
        dis_q_demand = dis_q_demand(num_demand) * sample_size;
        dis_q_demand = fliplr(dis_q_demand);
% [~,a] = min(abs(cumsum(dis_q_demand) - 0.05))
% num_demand(a) + min_demand


        if abs(wind_scale(1) - wind_scale(end)) <= max([2*sample_size, 24])
            mid = round((wind_scale(1) + wind_scale(end))/2);
            min_wind = mid;
            list_wind_center = 0;
            dis_q_wind = 1;
        else
            scale = ceil(max(wind_scale)/(500*wind_factor));
            [list_wind, dis_q_wind] = QR2PDF_w(quant_s_w, wind_scale, scale);
            plot(list_wind, dis_q_wind)
            min_wind = min(list_wind);
            max_wind = max(list_wind);
            list_wind_center = list_wind - min_wind;
            num_wind = 1 : (sample_size) : length(list_wind);
            list_wind_center = list_wind_center(num_wind);        
            dis_q_wind = dis_q_wind(num_wind) * sample_size;
        end
        %Solar QR to PDF

        if abs(solar_scale(end) - solar_scale(1)) <= max([2*sample_size, 24]) 
            mid = round((solar_scale(end) + solar_scale(1))/2);
            min_Solar = mid;
            list_solar_center = 0;
            dis_q_solar = 1;
        else
            scale = ceil(max(solar_scale)/(1000*solar_factor));
            [list_solar, dis_q_solar] = QR2PDF_s([quant_s_w], solar_scale, scale);
            
            min_Solar = min(list_solar);
            max_Solar = max(list_solar);

            list_solar_center = list_solar - min_Solar;
            num_solar = 1 : (sample_size) : length(list_solar);
            list_solar_center = list_solar_center(sample_size);        
            dis_q_solar = dis_q_solar(num_solar) * sample_size ;
        end

        q_G_L_W_S = conv(conv(conv(dis_q_demand, dis_q_wind),dis_q_solar),...
            dist_conv_select);
        q_G_L_W_S = q_G_L_W_S/sum(q_G_L_W_S);
        x_G_L_W_S = 1 : (sample_size ) : ((length(q_G_L_W_S) - 1) * (sample_size) + 1);

        if window <= 22
            EV_changing_engery = 0;
        else
            EV_changing_engery = (window - 22)/(24-22) * EV_daily;
        end 
         pump_enegy = pump_storage_energy;
        if window <= 4
            battery_energy = battery_storage_power * window;
           
        elseif window <= 20
            battery_energy = battery_storage_energy;
        else
            pump_enegy = pump_storage_energy * (24-window)/4;
            battery_energy = battery_storage_energy * (24-window)/4;
        end

%         if window <= 7
%             pump_energy = Pump_power_capacity * window;
%         else
%             pump_energy = Pump_energy_capacity;
%         end 
        x_G_L_W_S = x_G_L_W_S + min_x_MW + min_wind + min_demand + min_Solar +...
                     + pump_storage_energy + battery_energy - ...
                 EV_changing_engery + Fixed_transfer * window;
        [~,b] = min(abs(x_G_L_W_S));
        p_power_g_0(i) = sum(q_G_L_W_S(1:b));  

        % uppper bound
        if window == 1
            day_id = find(time_real_demand(i,:) == 1);
            month_temp = day_id(1);
            hour_temp = day_id(3) - 19;


            q_G_L_W_S_upper = conv(conv(conv(dis_q_demand, dis_q_wind),dis_q_solar),...
                dist_conv_select);
            q_G_L_W_S_upper = q_G_L_W_S_upper/sum(q_G_L_W_S_upper);
            x_G_L_W_S = 1 : (sample_size ) : ((length(q_G_L_W_S_upper) - 1) * (sample_size ) + 1);

            hour_id = find(Hour_real(i,:)>0);

            Battery_power = Battery_op(hour_temp) * battery_storage_energy;
            Pump_power = Pump_op(hour_temp) * pump_storage_energy;
            EV_power = EV_op_perfect(hour_temp) * EV_daily;

%             Battery_power = battery_assign(month_temp, hour_temp);
%             Pump_power = pump_assign(month_temp, hour_temp);
%             Battery_power = battery_assign( hour_temp);
%             Pump_power = pump_assign(hour_temp);
            x_G_L_W_S = x_G_L_W_S + min_x_MW + min_wind + min_demand + min_Solar +...
                         + Battery_power + Pump_power + EV_power + ...
                         Fixed_transfer;
            [~,b] = min(abs(x_G_L_W_S));
            p_power_g_upper(i) = sum(q_G_L_W_S_upper(1:b));  
        end
        if ~mod(i,400)
            display(i);
        end
    end
    sum(p_power_g_upper.*weight_q_list)
    
    LOLH_list(window) = sum(p_power_g_0.*weight_q_list);   
    p_power_g_0_list{window} = p_power_g_0;
    display("window finished:" + window + ", Lower Bound: " + LOLH_list(window))
end

LOLH_list
LOLH_upper = sum(p_power_g_upper.*weight_q_list)


figure
plot(LOLH_list)
grid
listOfProb_power = zeros(n_data17,12);
for n = 1
    p_power_g_0 = p_power_g_0_list{n};
    for i = 1 : length(date19)
        temp = A_demand_test(i,:);
        [~,a]=ismember(temp,time_real_demand,'rows');
        listOfProb_power(i,n) = p_power_g_0(a);
        listOfProb_power_upper(i) = p_power_g_upper(a);
    end
end

%results:
% s1
% upper 2.2445
% LOLH_list =
% 
%     0.3401
%     0.9086
%     1.0711
%     1.0336
%     0.9706
%     0.8083
%     0.6328
%     0.4660
%     0.3460
%     0.2367
%     0.1681
%     0.1150
%     0.0659
%     0.0409
%     0.0268
%     0.0139
%     0.0078
%     0.0051
%     0.0048
%     0.0047
%     0.0052
%     0.0060
%     0.0078
%     0.0124



% 
% LOLH_list =
%     0.3095
%     0.8323
%     0.9858
%     0.9552
%     0.9801
%     0.8731
%          0
%          0
%          0
%          0
%          0
%          0
%          0
%          0
%          0
%          0
%          0
%          0
%          0
%          0
%          0
%          0
%     0.1593
%     0.8987





% 
s1 = [
    0.3721
    1.0092
    1.2201
    1.2168
    1.1884
    1.0386
    0.8564
    0.6676
    0.5264
    0.3846
    0.2953
    0.2181
    0.1409
    0.0980
    0.0745
    0.0492
    0.0351
    0.0226
    0.0188
    0.0155
    0.0176
    0.0207
    0.0430
    0.1393]

% % 
% s2 =[
%     0.0805
%     0.2745
%     0.3304
%     0.3134
%     0.3126
%     0.2686
%     0.2183
%     0.1571
%     0.1103
%     0.0561
%     0.0416
%     0.0243
%     0.0070
%     0.0056
%     0.0036
%     0.0022
%     0.0016
%     0.0012
%     0.0012
%     0.0012
%     0.0016
%     0.0021
%     0.0220
%     0.3149]

% s3
% 
% LOLH_list =
% 
%     0.0979
%     0.3164
%     0.3758
%     0.3545
%     0.3504
%     0.2982
%     0.2407
%     0.1728
%     0.1219
%     0.0643
%     0.0481
%     0.0291
%     0.0091
%     0.0069
%     0.0043
%     0.0025
%     0.0019
%     0.0015
%     0.0014
%     0.0014
%     0.0019
%     0.0024
%     0.0201
%     0.1948

% 
% s4
% 
% LOLH_list =
% 
%     0.0725
%     0.2497
%     0.3007
%     0.2848
%     0.2767
%     0.2345
%     0.1977
%     0.1418
%     0.0992
%     0.0437
%     0.0371
%     0.0219
%     0.0057
%     0.0057
%     0.0038
%     0.0023
%     0.0017
%     0.0012
%     0.0012
%     0.0011
%     0.0015
%     0.0020
%     0.0219
%     0.3000






figure
plot(0:23, s2)
xlim([0,23])
xlabel('Length of Window (Hour)')
ylabel('Hours')




figure
EV_op_predict = [450,250,200,150,120,130,...
          200,300,280,260,250,240,...
          260,320, 390, 550, 680,880,...
          1000,950, 880, 820,730, 600];
EV_op_predict = EV_op_predict/sum(EV_op_predict);

EV_op_perfect = -[-1/80,-1/40,-1/40,-3/80,-1/40,-1/40,...
              -1/20,-3/40,-3/40 ,-4/40,-4/40,-1/10,...
              -4/40,-4/40,-3/40,-2/40,-1/40,0, ...
              0,0,0,0,0,0]
          
figure
plot(0:23, EV_op_perfect*113.7432*1000)

xlim([0,23])
xlabel('Hour of a Day')
ylabel('MW')













% %% Battery opertion
Battery_op = [-1/6,-1/6,-1/6,-1/6,-1/6,-1/6,0,0,0,0,0,0.0,0,0,0.05,...
    0.11,0.18,0.3,0.2,0.13,0.03,0,0,0];
% Pump_op = zeros(1,24);
% Pump_op(13:19) = 1/7;
% Pump_op(2:5) = -1/6;
Pump_op= [-1/6,-1/6,-1/6,-1/6,-1/6,-1/6*2/3,...
          -1/6*1/3,0,0,0,0,0,...
          1/7*1/3,1/7*2/3,1/7,1/7,1/7,1/7,...
          1/7,1/7*2/3,1/7*1/3,0,0,0]
% Pump_op([15,22]) = 1/14;
%% Scenario 1
load Probability
load Outage
f_factor = 1.4;
Demand_sc_train = 125*10^6 * Demand_scale_factor * f_factor; %current demand
% Wind_sc_train = 1.9136; %5.625 MW
Wind_sc_train = 2.8*10^6 * Wind_scale_factor;
Solar_sc_train = 0.5698 ; % 2.1MW
generation_scale = 1;
Battery_energy_capacity = 0;
Battery_power_capacity = 0;
Pump_energy_capacity = 1.5*10^3*8;
Pump_power_capacity = 1.5*10^3;
Fixed_transfer = 1.7*10^3; %GW


r_bu = Battery_power_capacity ;
r_pu = Pump_power_capacity ;
r_fu = Fixed_transfer;
r_gu = sum(Outage)*.095;

r_bd = Battery_power_capacity ;
r_pd = Pump_power_capacity;
r_fd = Fixed_transfer;
r_gd = sum(Outage)*.18;

ramp_up_r =  r_fu + r_gu;

ramp_up_limit  = r_bu + r_pu + r_fu + r_gu ;
ramp_down_limit  = r_bd + r_pd + r_fd + r_gd;



%% Scenario 7
% new 3    
% - the Central renewables scenario: 
% 1.7-2GW  fixed + approx. 10% wind + approx. 10% solar +  Battery 0.5GW-5 hour;
% take baseline and retire a couple of biggest dispatchable generators to get LOLH
%of about 2 
load Probability
load Outage
generation_sort = sort(Outage);
generation_prop = generation_sort/sum(generation_sort);
plot(generation_sort,cumsum(generation_prop)) %% Thresh 290

% Retire two biggest ones
Probability = Probability(Outage <705);
Outage = Outage(Outage<705);
sum(Outage)


f_factor = 1.4;
Demand_sc_train = 125*10^6 * Demand_scale_factor * f_factor; %current demand
% Wind_sc_train =    1.9136; %5.625 MW
Wind_sc_train = 19*10^6 * Wind_scale_factor;
Solar_sc_train = 12.5698 ; % 2.1MW
 
generation_scale = 1;
Battery_power_capacity = 1500;
Battery_energy_capacity = Battery_power_capacity*5;
Pump_energy_capacity = 1.5*10^3*8;
Pump_power_capacity = 1.5*10^3;
Fixed_transfer = 1.7*10^3; %GW

r_bu = Battery_power_capacity ;
r_pu = Pump_power_capacity ;
r_fu = Fixed_transfer;
r_gu = sum(Outage)*.095;

r_bd = Battery_power_capacity ;
r_pd = Pump_power_capacity;
r_fd = Fixed_transfer;
r_gd = sum(Outage)*.18;

ramp_up_r =  r_fu + r_gu;

ramp_up_limit  = r_bu + r_pu + r_fu + r_gu ;
ramp_down_limit  = r_bd + r_pd + r_fd + r_gd;

%% Scenario 8
% new 4    
% - the Central renewables scenario: 
% 1.7-2GW  fixed + approx. 10% wind + approx. 10% solar +  Battery 0.5GW-5 hour;
% take baseline and retire a couple of biggest dispatchable generators to get LOLH
%of about 2 
load Probability
load Outage
generation_sort = sort(Outage);
generation_prop = generation_sort/sum(generation_sort);
plot(generation_sort,cumsum(generation_prop)) %% Thresh 290

% Retire two biggest ones
Probability = Probability(Outage <705);
Outage = Outage(Outage<705);
sum(Outage)


f_factor = 1.4;
Demand_sc_train = 125*10^6 * Demand_scale_factor * f_factor; %current demand
% Wind_sc_train =    1.9136; %5.625 MW
Wind_sc_train = 29*10^6 * Wind_scale_factor;
Solar_sc_train = 12.5698 ; % 2.1MW
 
generation_scale = 1;
Battery_power_capacity = 1500;
Battery_energy_capacity = Battery_power_capacity*5;
Pump_energy_capacity = 1.5*10^3*8;
Pump_power_capacity = 1.5*10^3;
Fixed_transfer = 0.7*10^3; %GW


r_bu = Battery_power_capacity ;
r_pu = Pump_power_capacity ;
r_fu = Fixed_transfer;
r_gu = sum(Outage)*.095;

r_bd = Battery_power_capacity ;
r_pd = Pump_power_capacity;
r_fd = Fixed_transfer;
r_gd = sum(Outage)*.18;

ramp_up_r =  r_fu + r_gu;

ramp_up_limit  = r_bu + r_pu + r_fu + r_gu ;
ramp_down_limit  = r_bd + r_pd + r_fd + r_gd;



%% Scenario 9
% new 5    
% - the Central renewables scenario: 
% 1.7-2GW  fixed + approx. 10% wind + approx. 10% solar +  Battery 0.5GW-5 hour;
% take baseline and retire a couple of biggest dispatchable generators to get LOLH
%of about 2 
load Probability
load Outage
generation_sort = sort(Outage);
generation_prop = generation_sort/sum(generation_sort);
plot(generation_sort,cumsum(generation_prop)) %% Thresh 290

% Retire two biggest ones
Probability = Probability(Outage <705);
Outage = Outage(Outage<705);
sum(Outage)


f_factor = 1.4;
Demand_sc_train = 125*10^6 * Demand_scale_factor * f_factor; %current demand
% Wind_sc_train =    1.9136; %5.625 MW
Wind_sc_train = 19*10^6 * Wind_scale_factor;
Solar_sc_train = 18.5698 ; % 2.1MW
 
generation_scale = 1;
Battery_power_capacity = 1500;
Battery_energy_capacity = Battery_power_capacity*5;
Pump_energy_capacity = 1.5*10^3*8;
Pump_power_capacity = 1.5*10^3;
Fixed_transfer = 0.7*10^3; %GW

r_bu = Battery_power_capacity ;
r_pu = Pump_power_capacity ;
r_fu = Fixed_transfer;
r_gu = sum(Outage)*.095;

r_bd = Battery_power_capacity ;
r_pd = Pump_power_capacity;
r_fd = Fixed_transfer;
r_gd = sum(Outage)*.18;

ramp_up_r =  r_fu + r_gu;

ramp_up_limit  = r_bu + r_pu + r_fu + r_gu ;
ramp_down_limit  = r_bd + r_pd + r_fd + r_gd;


%% Scenario 10
% new 6    
% - the Central renewables scenario: 
% 1.7-2GW  fixed + approx. 10% wind + approx. 10% solar +  Battery 0.5GW-5 hour;
% take baseline and retire a couple of biggest dispatchable generators to get LOLH
%of about 2 
load Probability
load Outage
generation_sort = sort(Outage);
generation_prop = generation_sort/sum(generation_sort);
plot(generation_sort,cumsum(generation_prop)) %% Thresh 290

% Retire two biggest ones
Probability = Probability(Outage <705);
Outage = Outage(Outage<705);
sum(Outage)


f_factor = 1.4;
Demand_sc_train = 125*10^6 * Demand_scale_factor * f_factor; %current demand
% Wind_sc_train =    1.9136; %5.625 MW
Wind_sc_train = 19*10^6 * Wind_scale_factor;
Solar_sc_train = 12.5698 ; % 2.1MW
 
generation_scale = 1;
Battery_power_capacity = 2700;
Battery_energy_capacity = Battery_power_capacity*5;
Pump_energy_capacity = 1.5*10^3*8;
Pump_power_capacity = 1.5*10^3;
Fixed_transfer = 0.7*10^3; %GW


r_bu = Battery_power_capacity ;
r_pu = Pump_power_capacity ;
r_fu = Fixed_transfer;
r_gu = sum(Outage)*.095;

r_bd = Battery_power_capacity ;
r_pd = Pump_power_capacity;
r_fd = Fixed_transfer;
r_gd = sum(Outage)*.14;

ramp_up_r =  r_fu + r_gu;

ramp_up_limit  = r_bu + r_pu + r_fu + r_gu ;
ramp_down_limit  = r_bd + r_pd + r_fd + r_gd;




% f_factor = 1.4;
% Demand_sc_train = 125*10^6 * Demand_scale_factor * f_factor; %current demand
% % Wind_sc_train = 1.9136; %5.625 MW
% Wind_sc_train = 6*10^6 * Wind_scale_factor;
% Solar_sc_train = 10; % 2.1MW
% generation_scale = 1;
% Battery_energy_capacity = 2*10^3*4;
% Battery_power_capacity = 2*10^3;
% Pump_energy_capacity = 1.7*10^3*7;
% Pump_power_capacity = 1.7*10^3;
% Fixed_transfer = 0.7*10^3; %GW

%%
window = 1;
[x_MW,dist_conv] = Cap_Outage(Probability, Outage*generation_scale,window);
%Distribution
min_x_MW = min(x_MW);
max_x_MW = max(x_MW);
x_MW_center = x_MW - min_x_MW;
num_x_MW = 1 : (window * 500) : length(x_MW);
x_MW_center_used = x_MW_center(num_x_MW);
dist_conv_select = zeros(length(x_MW_center_used),1);
loc_value = find(dist_conv>0);
for i = 1:length(loc_value)
   [~,loc_temp] = min(abs(num_x_MW - loc_value(i)));
   dist_conv_select(loc_temp) = dist_conv(loc_value(i));
end


tic;

beta_demand = beta_demand_list{1} ;
alpha_demand = alpha_demand_list{1};
beta_wind = beta_wind_list{1};
alpha_wind = alpha_wind_list{1};
beta_solar = beta_solar_list{1};
alpha_solar = alpha_solar_list{1};
r_demand_wind_matrix = r_lw_list{1};
r_demand_solar_matrix = r_ls_list{1};
r_wind_solar_matrix = r_ws_list{1};
pred_demand = pred_demand_list{1};
[x_MW,dist_conv] = Cap_Outage(Probability, Outage*generation_scale,1);
min_x_MW = min(x_MW);
max_x_MW = max(x_MW);
x_MW_center = x_MW - min_x_MW;
num_x_MW = 1 : (window * 500) : length(x_MW);
x_MW_center_used = x_MW_center(num_x_MW);
dist_conv_select = zeros(length(x_MW_center_used),1);
loc_value = find(dist_conv>0);

tail_n = 1;
[paramEsts_left,paramEsts_right,Error_left_pos,Error_right_pos] = ...
    demand_tail_est(tail_n,quant,pred_demand, Demand15_16);

for i = 1:length(loc_value)
   [~,loc_temp] = min(abs(num_x_MW - loc_value(i)));
   dist_conv_select(loc_temp) = dist_conv(loc_value(i));
end
parfor i = 1:n_state
    r_lw = Hour_real(i,:) * r_demand_wind_matrix * Month_real(i,:)';
    r_ls = Hour_real(i,:) * r_demand_solar_matrix * Month_real(i,:)';
    r_ws = Hour_real(i,:) * r_wind_solar_matrix * Month_real(i,:)';
    
%     
%     Demand_deviation_tilde = time_real_variance(i,:) * r_M_W_H';
%     variance_factor = Demand_sc_train;
%    
    demand_tilde =   time_real_demand(i,:) * beta_demand' + alpha_demand';
    
    demand_factor = Demand_sc_train;
    
    
    wind_tilde = time_real_w_s(i,:) * beta_wind' +  alpha_wind';
    wind_factor = Wind_sc_train - Demand_sc_train * r_lw;
    wind_scale = wind_tilde * wind_factor;
    tic
    Wind_nameplate = wind_factor*1000;


    solar_tilde = (time_real_solar_power(i,:) * beta_solar' +  alpha_solar')*1000;
    solar_factor = Wind_sc_train * r_ws + Solar_sc_train - Demand_sc_train * r_lw * r_ws ...
        - Demand_sc_train * r_ls;
    solar_scale = solar_tilde * solar_factor;
    Solar_nameplate =  solar_factor*1000;

%             (Hour_real(i,:) * r_demand_wind_matrix * Month_real') * Wind * ones(1,n_quant) + ...
%             A3_month_hour * r_solar_M_H .* Solar * ones(1,n_quant)
%             

%             temp_time_real_wind = time_real_w_s(i,:) * Wind_factor(i)  ;
%             temp_time_real_demand = time_real_demand(i,:);
%             temp_r_wind = Month_real(i,:) * r_wind_use ;

    [q_demand_comb,demand_sample] = tailSample_new(quant, demand_tilde,...
        paramEsts_right, paramEsts_left);
    demand_sample_scale = demand_sample * demand_factor;
    [list_demand,dis_q_demand] = QR2PDF_demand(q_demand_comb,demand_sample_scale, ...
        1);
    list_demand = - list_demand;
    min_demand = min(list_demand);
    max_demand = max(list_demand);
    num_demand = 1 : (window * 500) : length(list_demand);
    list_demand_center = list_demand - min_demand;
    list_demand_center = list_demand_center(num_demand);
    dis_q_demand = dis_q_demand(num_demand) * window * 500;

    %Intra variance to PDF
%     [q_variance_comb,variance_sample] = tailSample_new(quant, ...
%         Demand_deviation_tilde,Var_paramEsts_right, Var_paramEsts_left);
%     variance_sample_scale = variance_sample * variance_factor;
%     [list_variance,dis_q_variance] = QR2PDF_demand(q_variance_comb,variance_sample_scale, ...
%         1);
%     list_variance = - list_variance;
%     min_variance = min(list_variance);
%     max_variance = max(list_variance);
%     num_variance = 1 : (window * 10) : length(list_variance);
%     list_variance_center = list_variance - min_variance;
%     list_variance_center = list_variance_center(num_variance);
%     dis_q_variance = dis_q_variance(num_variance) * window * 10;


    %wind QR to PDF
    if abs(wind_scale(1) - wind_scale(end)) <= 24
        mid = round((wind_scale(1) + wind_scale(end))/2);
        min_wind = mid;
        list_wind_center = 0;
        dis_q_wind = 1;
    else
        [list_wind, dis_q_wind] = QR2PDF_w(quant, wind_scale,...
            Wind_nameplate);
        min_wind = min(list_wind);
        max_wind = max(list_wind);
        list_wind_center = list_wind - min_wind;
        num_wind = 1 : (window * 500) : length(list_wind);
        list_wind_center = list_wind_center(num_wind);        
        dis_q_wind = dis_q_wind(num_wind) * window * 500;
    end
    %Solar QR to PDF
    
    if abs(solar_scale(end) - solar_scale(1)) <= 24
        mid = round((solar_scale(end) + solar_scale(1))/2);
        min_Solar = mid;
        list_solar_center = 0;
        dis_q_solar = 1;
    else
        [list_solar, dis_q_solar] = QR2PDF_s(quant, solar_scale);
        min_Solar = min(list_solar);
        max_Solar = max(list_solar);

        list_solar_center = list_solar - min_Solar;
        num_solar = 1 : (window * 500) : length(list_solar);
        list_solar_center = list_solar_center(num_solar);        
        dis_q_solar = dis_q_solar(num_solar) * window * 500;
    end
    %Intra hour variance to PDF

    q_G_L_W_S = conv(conv(conv(dis_q_demand, dis_q_wind),dis_q_solar),...
        dist_conv_select);
    q_G_L_W_S = q_G_L_W_S/sum(q_G_L_W_S);
    x_G_L_W_S = 1 : (window * 500) : ((length(q_G_L_W_S) - 1) * (window * 500) + 1);


    x_G_L_W_S = x_G_L_W_S + min_x_MW + min_wind + min_demand + min_Solar +...
                 + Pump_power_capacity +Fixed_transfer + Battery_power_capacity;
    [~,b] = min(abs(x_G_L_W_S));
    p_power_g_0(i) = sum(q_G_L_W_S(1:b));  
    if ~mod(i,100)
        
        display(i);
    end
    
end
LOLH = sum(p_power_g_0.*weight_q_list')
toc



LOLH = sum(p_power_g_0.*weight_q_list')



listOfProb_power = zeros(length(date17),1);
for i = 1 : length(date17)
    temp = A_demand_test(i,:);
    [~,a]=ismember(temp,time_real_demand,'rows');
    listOfProb_power(i) = p_power_g_0(a);
end
sum(listOfProb_power)

plot(listOfProb_power)
xlabel('Hour of Year 2015')
ylabel('Probability')
grid
title('LOLP Of 2015')
%% ENergy
%% Energy find lower and upper boundary
beta_demand_list =cell(1,12) ;
alpha_demand_list = cell(1,12);
beta_wind_list = cell(1,12);
alpha_wind_list = cell(1,12);
beta_solar_list = cell(1,12);
alpha_solar_list = cell(1,12);
r_lw_list = cell(1,12);
r_ls_list = cell(1,12);
r_ws_list = cell(1,12);
pred_demand_list = cell(1,12);
x_MW_list = cell(1,12);
dist_conv_list = cell(1,12);
for window = [1:3,5:12]
    
    conv_demand = conv_hour(window, Demand15_16);
    % wind
    conv_wind = conv_hour(window, Wind15_16);
    conv_solar = conv_hour(window, Solar15_16);
    conv_solar12_16 = conv_hour(window, Solar12_16);
    
    % QR demand 
    Date_window = date15_16(window:end);
    Date11_15_window = date12_16(window:end);
    
    
    % energy opt for demand
%     lambda_demand_power = [10^(3),10^(-2), 10^(4),10^(1),10^(7)*7,10^(3),10^(1)];
%     lambda_demand_power = [10^(3),10^(0), 10^(6),10^(6),10^(7),10^(1),0];
%     [beta_demand, alpha_demand, r_demand_wind_M_H, r_demand_solar_M_H, pred_demand] = ...
%         opt_demand_solar_power(quant, conv_demand, Date_window,...
%         Holidays_15_16, conv_wind, conv_solar,lambda_demand_power);%lambda 7 parameters
%     r_demand_wind_matrix  = reshape(r_demand_wind_M_H,[24,12]);
%     r_demand_solar_matrix  = reshape(r_demand_solar_M_H,[24,12]);
%     energy opt for wind    
    lambda = [10^(0),10^(1),10^(2),10^(4),10^(-3)];
    lambda_wind = [10^(0),10^(1),10^(4),10^(4),1000];
    [beta_wind, alpha_wind,  r_solar_M_H, pred_wind] = ...
        opt_wind_solar_power(quant_s_w,lambda_wind, conv_wind, Date_window, conv_solar);
    r_wind_solar_matrix = reshape(r_solar_M_H,[24,12]);
%     energy opt for solar
    lambda = [10^(-1), 10^(0)];
    [beta_solar,alpha_solar, pred_solar] =opt_solar_power(quant_s_w,conv_solar12_16, Date11_15_window,lambda);
%     
%     beta_demand_list{window} = beta_demand ;
%     alpha_demand_list{window} = alpha_demand;
    beta_wind_list{window} = beta_wind;
    alpha_wind_list{window} = alpha_wind ;
    beta_solar_list{window} = beta_solar;
    alpha_solar_list{window} = alpha_solar;
%     r_lw_list{window} = r_demand_wind_matrix;
%     r_ls_list{window} = r_demand_solar_matrix;
    r_ws_list{window} = r_wind_solar_matrix;
%     pred_demand_list{window} = pred_demand;

end




LOLH_list = zeros(12,1);
p_power_g_upper = zeros(n_state,1);
p_power_g_0_list={};
for window = 1
    beta_demand = beta_demand_list{window} ;
    alpha_demand = alpha_demand_list{window};
    beta_wind = beta_wind_list{window};
    alpha_wind = alpha_wind_list{window};
    beta_solar = beta_solar_list{window};
    alpha_solar = alpha_solar_list{window};
    r_demand_wind_matrix = r_lw_list{window};
    r_demand_solar_matrix = r_ls_list{window};
    r_wind_solar_matrix = r_ws_list{window};
    pred_demand = pred_demand_list{window};
    

    tail_n = 1;
    [paramEsts_left,paramEsts_right,Error_left_pos,Error_right_pos] = ...
        demand_tail_est(tail_n,quant,pred_demand, conv_hour(window, Demand15_16));
%     G_capacity = 5.5 * 10^3 * generation_scale; % kw

%     gas_turbines = ones(7,1) * G_capacity;
%     Outage_prob = ones(7,1) * Eford/100;
% 
%     [x_MW,dist_conv] = Cap_Outage(Outage_prob, gas_turbines,window);
    [x_MW,dist_conv] = Cap_Outage(Probability, Outage*generation_scale,window);

    min_x_MW = min(x_MW);
    max_x_MW = max(x_MW);
    x_MW_center = x_MW - min_x_MW;
    num_x_MW = 1 : (window) : length(x_MW);
    x_MW_center_used = x_MW_center(num_x_MW);
    dist_conv_select = zeros(length(x_MW_center_used),1);
    loc_value = find(dist_conv>0);
    for i = 1:length(loc_value)
       [~,loc_temp] = min(abs(num_x_MW - loc_value(i)));
       dist_conv_select(loc_temp) = dist_conv(loc_value(i));
    end
    dist_conv_select = dist_conv_select/sum(dist_conv_select);
    
    p_power_g_0 = zeros(n_state,1);
    
    parfor i = 1:n_state
%         bat = battery_assign_day;
%         pump = pump_assign_day;
        r_lw = Hour_real(i,:) * r_demand_wind_matrix * Month_real(i,:)';
        r_ls = Hour_real(i,:) * r_demand_solar_matrix * Month_real(i,:)';
        r_ws = Hour_real(i,:) * r_wind_solar_matrix * Month_real(i,:)';
        

        demand_tilde =   time_real_demand(i,:) * beta_demand' + alpha_demand';
        demand_factor = Demand_sc_train ;


        wind_tilde = time_real_w_s(i,:) * beta_wind' +  alpha_wind';
        wind_factor = Wind_sc_train - Demand_sc_train * r_lw;
        wind_scale = wind_tilde * wind_factor;
%         Wind_nameplate = wind_factor * 1000 * window;

        solar_tilde = (time_real_solar_power(i,:) * beta_solar' +  alpha_solar')*1000;
        solar_factor = Wind_sc_train * r_ws + Solar_sc_train - Demand_sc_train * r_lw * r_ws ...
            - Demand_sc_train * r_ls;
        solar_scale = solar_tilde * solar_factor;
%         Solar_nameplate =  solar_factor*1000 * window;
        
        [q_demand_comb,demand_sample] = tailSample_new(quant, demand_tilde,...
            paramEsts_right, paramEsts_left);
        demand_sample_scale = demand_sample * demand_factor;
        [list_demand,dis_q_demand] = QR2PDF_demand(q_demand_comb,demand_sample_scale, ...
            1);
%         figure;plot(list_demand, cumsum(dis_q_demand));
%         hold on
%         plot(demand_sample_scale,q_demand_comb)
%         
        list_demand = - list_demand;
        min_demand = min(list_demand);
        max_demand = max(list_demand);
        num_demand = 1 : (window) : length(list_demand);
        list_demand_center = list_demand - min_demand;
        list_demand_center = list_demand_center(num_demand);
        dis_q_demand = dis_q_demand(num_demand) * window;
        dis_q_demand = fliplr(dis_q_demand);
% [~,a] = min(abs(cumsum(dis_q_demand) - 0.05))
% num_demand(a) + min_demand

        
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
        
        q_G_L_W_S = conv(conv(conv(dis_q_demand, dis_q_wind),dis_q_solar),...
            dist_conv_select);
        q_G_L_W_S = q_G_L_W_S/sum(q_G_L_W_S);
        x_G_L_W_S = 1 : (window ) : ((length(q_G_L_W_S) - 1) * (window) + 1);

        if window <= 5
            Battery_energy = Battery_power_capacity * window;
        else
            Battery_energy = Battery_energy_capacity;
        end 
        if window <= 7
            pump_energy = Pump_power_capacity * window;
        else
            pump_energy = Pump_energy_capacity;
        end 
        x_G_L_W_S = x_G_L_W_S + min_x_MW + min_wind + min_demand + min_Solar +...
                     + Battery_energy + pump_energy + Fixed_transfer*window;
        [~,b] = min(abs(x_G_L_W_S));
        p_power_g_0(i) = sum(q_G_L_W_S(1:b));  
        
        % uppper bound
        if window == 1
            day_id = find(time_real_demand(i,:) == 1);
            month_temp = day_id(1);
            hour_temp = day_id(3) - 19;
            

            q_G_L_W_S_upper = conv(conv(conv(dis_q_demand, dis_q_wind),dis_q_solar),...
                dist_conv_select);
            q_G_L_W_S_upper = q_G_L_W_S_upper/sum(q_G_L_W_S_upper);
            x_G_L_W_S = 1 : (window ) : ((length(q_G_L_W_S_upper) - 1) * (window ) + 1);
            
%             hour_id = find(Hour_real(i,:)>0);
%              Battery_power = Battery_op(hour_temp) * Battery_energy_capacity;
%              Pump_power = Pump_op(hour_temp) * Pump_energy_capacity;
             Battery_power = battery_assign(month_temp, hour_temp);
             Pump_power = pump_assign(month_temp, hour_temp);
%              Battery_power = battery_assign( hour_temp);
%              Pump_power = pump_assign(hour_temp);
            x_G_L_W_S = x_G_L_W_S + min_x_MW + min_wind + min_demand + min_Solar +...
                         + Battery_power + Pump_power + Fixed_transfer;
            [~,b] = min(abs(x_G_L_W_S));
            p_power_g_upper(i) = sum(q_G_L_W_S_upper(1:b));  
        end
        if ~mod(i,100)
            display(i);
        end
    end
    LOLH_list(window) = sum(p_power_g_0.*weight_q_list);   
    p_power_g_0_list{window} = p_power_g_0;
end

LOLH_list
LOLH_upper = sum(p_power_g_upper.*weight_q_list)


listOfProb_power = zeros(n_data17,12);
for n = 1
    p_power_g_0 = p_power_g_0_list{n};
    for i = 1 : n_data17
        temp = A_demand_test(i,:);
        [~,a]=ismember(temp,time_real_demand,'rows');
        listOfProb_power(i,n) = p_power_g_0(a);
        listOfProb_power_upper(i) = p_power_g_upper(a);
    end
end
figure 
plot(listOfProb_power_upper)


plot(LOLH_list)
grid
xlabel('Energy Window Length (Hours)')
ylabel('LOLH (Hours)')
title('LOLH')
LOLH_upper = sum(p_power_g_upper.*weight_q_list);

 


%% Choose potential battery and pump storage operation
q_thresh = 0.004;
q_thresh = [0.001,0.001,0.002,...
            0.0001, 0.0001, 0.002, ...
            0.004,0.006,0.002,...
            0.0001, 0.0001, 0.001]
window = 1
beta_demand = beta_demand_list{window} ;
alpha_demand = alpha_demand_list{window};
beta_wind = beta_wind_list{window};
alpha_wind = alpha_wind_list{window};
beta_solar = beta_solar_list{window};
alpha_solar = alpha_solar_list{window};
r_demand_wind_matrix = r_lw_list{window};
r_demand_solar_matrix = r_ls_list{window};
r_wind_solar_matrix = r_ws_list{window};
pred_demand = pred_demand_list{window};


tail_n = 1;
[paramEsts_left,paramEsts_right,Error_left_pos,Error_right_pos] = ...
    demand_tail_est(tail_n,quant,pred_demand, conv_hour(window, Demand15_16));
%     G_capacity = 5.5 * 10^3 * generation_scale; % kw

%     gas_turbines = ones(7,1) * G_capacity;
%     Outage_prob = ones(7,1) * Eford/100;
% 
%     [x_MW,dist_conv] = Cap_Outage(Outage_prob, gas_turbines,window);
[x_MW,dist_conv] = Cap_Outage(Probability, Outage*generation_scale,window);

min_x_MW = min(x_MW);
max_x_MW = max(x_MW);
x_MW_center = x_MW - min_x_MW;
num_x_MW = 1 : (window) : length(x_MW);
x_MW_center_used = x_MW_center(num_x_MW);
dist_conv_select = zeros(length(x_MW_center_used),1);
loc_value = find(dist_conv>0);
for i = 1:length(loc_value)
   [~,loc_temp] = min(abs(num_x_MW - loc_value(i)));
   dist_conv_select(loc_temp) = dist_conv(loc_value(i));
end
dist_conv_select = dist_conv_select/sum(dist_conv_select);

x_boundary_list = zeros(n_state,1);

parfor i = 1:n_state
    r_lw = Hour_real(i,:) * r_demand_wind_matrix * Month_real(i,:)';
    r_ls = Hour_real(i,:) * r_demand_solar_matrix * Month_real(i,:)';
    r_ws = Hour_real(i,:) * r_wind_solar_matrix * Month_real(i,:)';


    demand_tilde =   time_real_demand(i,:) * beta_demand' + alpha_demand';
    demand_factor = Demand_sc_train ;


    wind_tilde = time_real_w_s(i,:) * beta_wind' +  alpha_wind';
    wind_factor = Wind_sc_train - Demand_sc_train * r_lw;
    wind_scale = wind_tilde * wind_factor;
%         Wind_nameplate = wind_factor * 1000 * window;


    solar_tilde = (time_real_solar_power(i,:) * beta_solar' +  alpha_solar')*1000;
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
    num_demand = 1 : (window ) : length(list_demand);
    list_demand_center = list_demand - min_demand;
    list_demand_center = list_demand_center(num_demand);
    dis_q_demand = dis_q_demand(num_demand) * window;



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
        dis_q_solar = dis_q_solar(num_solar) * window;
    end

    q_G_L_W_S = conv(conv(conv(dis_q_demand, dis_q_wind),dis_q_solar),...
        dist_conv_select);
    q_G_L_W_S = q_G_L_W_S/sum(q_G_L_W_S);
    x_G_L_W_S = 1 : (window) : ((length(q_G_L_W_S) - 1) * (window) + 1);

    if window <= 4
        Battery_energy = Battery_power_capacity * window;
    else
        Battery_energy = Battery_energy_capacity;
    end 
    if window <= 7
        pump_energy = Pump_power_capacity * window;
    else
        pump_energy = Pump_energy_capacity;
    end 
    x_G_L_W_S = x_G_L_W_S + min_x_MW + min_wind + min_demand + min_Solar +...
                 + Battery_energy + pump_energy + Fixed_transfer*window;
    [~,b] = min(abs(x_G_L_W_S));
    p_power_g_0(i) = sum(q_G_L_W_S(1:b));  

    % uppper bound
    month_temp = find(Month_real(i,:)>0);

    q_G_L_W_S_upper = conv(conv(conv(dis_q_demand, dis_q_wind),dis_q_solar),...
        dist_conv_select);
    q_G_L_W_S_upper = q_G_L_W_S_upper/sum(q_G_L_W_S_upper);
    x_G_L_W_S = 1 : (window ) : ((length(q_G_L_W_S_upper) - 1) * (window) + 1);

%         hour_temp = find(Hour_real(i,:)>0);
%         Battery_power = Battery_op(hour_temp) * Battery_energy_capacity;
%         Pump_power = Pump_op(hour_temp) * Pump_energy_capacity;

    x_G_L_W_S = x_G_L_W_S + min_x_MW + min_wind + min_demand + min_Solar +...
                     Fixed_transfer;
   acumprob = cumsum(q_G_L_W_S);
    [~, a] = min(abs(acumprob - q_thresh(month_temp)));
    x_boundary = x_G_L_W_S(a);
    x_boundary_list(i) = x_boundary;   
    
    if ~mod(i,100)
        display(i);
    end
end

listOfBound = zeros(n_data17,1);
for i = 1 : n_data17
    temp = A_demand_test(i,:);
    [~,a]=ismember(temp,time_real_demand,'rows');
    listOfBound(i) = x_boundary_list(a);
end

%change the battery profile month by month
month_id = 1;
weekday_id = 3;
hour_id = 1;
boundary_month = zeros(12, 24);
for i = 1:length(date17)
    day_id = find(A_demand_test(i,:) == 1);
    month_temp = day_id(1);
    weekday_temp = day_id(2) - 12;
    if month_temp == month_id
        if weekday_temp == weekday_id
            boundary_month(month_temp, hour_id) = listOfBound(i);
            hour_id = hour_id + 1;
            if hour_id > 24
                hour_id= 1;
                month_id = month_id + 1;
            end
        end
    end
end
figure


battery_assign = zeros(12,24);
pump_assign = zeros(12,24);
for i = 1:12
    Peak_day_quantile = boundary_month(i,:);    
    cvx_begin
        variables batteries_supply(24) pump_supply(24)
%         minimize(sum(batteries_supply.*batteries_supply)) % minimize( sum (|battery_discharge|)
        minimize(sum(norm(batteries_supply,1)+ norm(pump_supply,1)))
%          dual variables y{24}
        subject to 
            sum(batteries_supply) == 0
            sum(pump_supply) == 0
%             sum(batteries_supply([1:7,22:24])) == -Battery_energy_capacity
%             sum(pump_supply([1:7,22:24])) == -Pump_energy_capacity
%             sum(batteries_supply(8:21)) = Battery_energy_capacity
%             sum(pump_supply(8:21)) <= Pump_energy_capacity
%             batteries_supply([1:7,23:24]) >= - Battery_energy_capacity/5
%             pump_supply([1:7,23:24]) >= - Pump_energy_capacity/6
%             batteries_supply(8:22) <=  Battery_power_capacity * 2
%             pump_supply(8:22) <=  Pump_power_capacity *1.2
%             batteries_supply(8:11) ==  0
%             pump_supply(8:11) ==  0
            %charge                            
            for j = 1:24
                Peak_day_quantile(j) + batteries_supply(j) + pump_supply(j) >= 0 
            end
            pump_supply <= Pump_power_capacity
            sum(abs(batteries_supply)) <= 2 * Battery_energy_capacity
            sum(abs(pump_supply)) <= 2 * Pump_energy_capacity
%             batteries_supply - circshift(batteries_supply,1) <= 500
%             pump_supply - circshift(pump_supply,1) <= 300
% %             tril(ones(18)) * batteries_supply(7:end) <= 0             % lower boundary
% %             tril(ones(18)) * batteries_supply(7:end) + Total_storage  >= 0% upper boundary
%             cumsum(batteries_supply(8:22)) <= 0
%             cumsum(batteries_supply(7:end)) + Total_storage >= 0
%             for i = 1:18
%                 Peak_day_quantile(6 + i,i) - sum(batteries_supply(7:i+6)) >= 0 : y{i+6}
%             end
%             for i = 1:12
%                 for j = 1:19-i
%                     sum(batteries_supply(j:j+i-1)) + Peak_day_quantile(j+i-1,i) >= 0  % satisfy the quantile limit
%                 end
%             end
    cvx_end        
    battery_assign(i,:) = batteries_supply';
    pump_assign(i,:) = pump_supply';
    
end



%4369-4526
list_peak_week = 4369:4536;
Peak_x_boundary = listOfBound(list_peak_week);
% Total_storage = Battery_energy_capacity + Pump_energy_capacity;
battery_assign = zeros(7,24);
pump_assign = zeros(7,24);
for wd = 3
    Peak_day_quantile = Peak_x_boundary((wd-1)*24 + 1 : wd*24);    
    cvx_begin
        variables batteries_supply(24) pump_supply(24)
%         minimize(sum(batteries_supply.*batteries_supply)) % minimize( sum (|battery_discharge|)
        minimize(sum(norm(batteries_supply,1)+ norm(pump_supply,1)))
%          dual variables y{24}
        subject to 
            sum(batteries_supply) == 0
            sum(pump_supply) == 0
%             sum(batteries_supply([1:7,22:24])) == -Battery_energy_capacity
%             sum(pump_supply([1:7,22:24])) == -Pump_energy_capacity
%             sum(batteries_supply(8:21)) = Battery_energy_capacity
%             sum(pump_supply(8:21)) <= Pump_energy_capacity
%             batteries_supply([1:7,23:24]) >= - Battery_energy_capacity/5
%             pump_supply([1:7,23:24]) >= - Pump_energy_capacity/6
%             batteries_supply(8:22) <=  Battery_power_capacity * 2
%             pump_supply(8:22) <=  Pump_power_capacity *1.2
%             batteries_supply(8:11) ==  0
%             pump_supply(8:11) ==  0
            %charge                            
            for i = 1:24
                Peak_day_quantile(i) + batteries_supply(i) + pump_supply(i) >= 0 
            end
%             batteries_supply - circshift(batteries_supply,1) <= 500
%             pump_supply - circshift(pump_supply,1) <= 300
% %             tril(ones(18)) * batteries_supply(7:end) <= 0             % lower boundary
% %             tril(ones(18)) * batteries_supply(7:end) + Total_storage  >= 0% upper boundary
%             cumsum(batteries_supply(8:22)) <= 0
%             cumsum(batteries_supply(7:end)) + Total_storage >= 0
%             for i = 1:18
%                 Peak_day_quantile(6 + i,i) - sum(batteries_supply(7:i+6)) >= 0 : y{i+6}
%             end
%             for i = 1:12
%                 for j = 1:19-i
%                     sum(batteries_supply(j:j+i-1)) + Peak_day_quantile(j+i-1,i) >= 0  % satisfy the quantile limit
%                 end
%             end
    cvx_end        
    battery_assign(wd,:) = batteries_supply';
    pump_assign(wd,:) = pump_supply';
end
battery_assign_day = battery_assign(3,:);
pump_assign_day = pump_assign(3,:);


%% ramp rate

% Demand and wind

%% solar ramp
lambda = [10^(-1), 10^(0)]; % HyperParameter
[beta_solar_ramp,alpha_solar_ramp, pred_solar_ramp] =opt_solar_ramp(quant,diff(Solar12_16), date12_16(2:end),lambda);

%% wind ramp
lambda = [10^(0),10^(1),10^(4),10^(4),1000];
 [beta_wind_ramp, alpha_wind_ramp,  r_solar_M_H_ramp, pred_wind_ramp] = ...
    opt_wind_solar_ramp(quant,lambda, diff(Wind15_16), date15_16(2:end), diff(Solar15_16));
r_wind_solar_ramp_matrix = reshape(r_solar_M_H_ramp,[24,12]);

%% demand ramp
lambda = [10^(3),10^(-2), 10^(4),10^(1),10^(7)*7,10^(2),10^(-2)];
lambda = [10^(3),10^(0), 10^(6),10^(6),10^(7),10^(1),0];
[beta_demand_ramp, alpha_demand_ramp, r_demand_wind_M_H_ramp, r_demand_solar_M_H_ramp, pred_demand_ramp] = ...
    opt_demand_solar_ramp(quant, diff(Demand15_16), date15_16(2:end),...
    Holidays_15_16(2:end), diff(Wind15_16), diff(Solar15_16),lambda);%lambda 7 parameters
r_demand_wind_ramp_matrix  = reshape(r_demand_wind_M_H_ramp,[24,12]);
r_demand_solar_ramp_matrix  = reshape(r_demand_solar_M_H_ramp,[24,12]);



%% Tail of D, W, S

tail_n = 1;
[paramEsts_left_D_ramp,paramEsts_right_D_ramp,Error_left_pos_D_ramp,Error_right_pos_D_ramp] = ...
    demand_tail_est(tail_n,quant,pred_demand_ramp, diff(Demand15_16));


[paramEsts_left_W_ramp,paramEsts_right_W_ramp,Error_left_pos_W_ramp,Error_right_pos_W_ramp] = ...
    demand_tail_est(tail_n,quant,pred_wind_ramp, diff(Wind15_16));


[paramEsts_left_S_ramp,paramEsts_right_S_ramp,Error_left_pos_S_ramp,Error_right_pos_S_ramp] = ...
    demand_tail_est(tail_n,quant,pred_solar_ramp, diff(Solar12_16));

[x_MW,dist_conv] = Cap_Outage(Probability, Outage*generation_scale,1);
x_MW = x_MW - sum(Outage);
plot(x_MW,dist_conv)
dist_ramp_outage = conv(dist_conv, fliplr(dist_conv')');
x_MW_neg = [x_MW(1:end-1)];
x_MW_ramp = [x_MW_neg,0,-fliplr(x_MW_neg)];
plot(x_MW_ramp,dist_ramp_outage)
min_X_MW_ramp = min(x_MW_ramp);

for i = 1:n_state
    
    

    r_lw = Hour_real(i,:) * r_demand_wind_ramp_matrix * Month_real(i,:)';
    r_ls = Hour_real(i,:) * r_demand_solar_ramp_matrix * Month_real(i,:)';
    r_ws = Hour_real(i,:) * r_wind_solar_ramp_matrix * Month_real(i,:)';
    
%     
%     Demand_deviation_tilde = time_real_variance(i,:) * r_M_W_H';
%     variance_factor = Demand_sc_train;
%    
    demand_tilde =   time_real_demand(i,:) * beta_demand_ramp' + alpha_demand_ramp';   
    demand_factor = Demand_sc_train;
    
    
    wind_tilde = time_real_w_s(i,:) * beta_wind_ramp' +  alpha_wind_ramp';
    wind_factor = Wind_sc_train - Demand_sc_train * r_lw;
    wind_scale = wind_tilde * wind_factor;
    
%     Wind_nameplate = wind_factor*1000;


    solar_tilde = (time_real_solar_power(i,:) * beta_solar_ramp' +  alpha_solar_ramp')*1000;
    solar_factor = Wind_sc_train * r_ws + Solar_sc_train - Demand_sc_train * r_lw * r_ws ...
        - Demand_sc_train * r_ls;
    solar_scale = solar_tilde * solar_factor;
%     Solar_nameplate =  solar_factor*1000;

%             (Hour_real(i,:) * r_demand_wind_matrix * Month_real') * Wind * ones(1,n_quant) + ...
%             A3_month_hour * r_solar_M_H .* Solar * ones(1,n_quant)
%             

%             temp_time_real_wind = time_real_w_s(i,:) * Wind_factor(i)  ;
%             temp_time_real_demand = time_real_demand(i,:);
%             temp_r_wind = Month_real(i,:) * r_wind_use ;
    
% Demand QR to PDF
    [q_demand_comb,demand_sample] = tailSample_new(quant, demand_tilde,...
        paramEsts_right_D_ramp, paramEsts_left_D_ramp);
    demand_sample_scale = demand_sample * demand_factor;
    [list_demand,dis_q_demand] = QR2PDF_demand(q_demand_comb,demand_sample_scale, ...
        1);
    list_demand = - list_demand;
    min_demand = min(list_demand);
    max_demand = max(list_demand);
    num_demand = 1  : length(list_demand);
    list_demand_center = list_demand - min_demand;
    list_demand_center = list_demand_center(num_demand);
    dis_q_demand = dis_q_demand(num_demand);
%     dis_q_demand = fliplr(dis_q_demand);
% Wind QR to PDF
    min_wind = 0;
    if abs(wind_scale(1) - wind_scale(end)) <= 24
        mid = round((wind_scale(1) + wind_scale(end))/2);
        min_wind = mid;
        list_wind_center = 0;
        dis_q_wind = 1;
    else
         [q_wind_comb,wind_sample] = tailSample_new(quant, wind_tilde,...
            paramEsts_right_W_ramp, paramEsts_left_W_ramp);
        wind_sample_scale = wind_sample * wind_factor;
        if wind_factor < 0
            wind_sample_scale = fliplr(wind_sample_scale);
            q_wind_comb = 1- fliplr(q_wind_comb);
        end
        [list_wind,dis_q_wind] = QR2PDF_demand(q_wind_comb,wind_sample_scale, ...
            1);
        list_wind = - list_wind;
        min_wind = min(list_wind);
        max_wind = max(list_wind);
        num_wind = 1  : length(list_wind);
        list_wind_center = list_wind - min_wind;
        list_wind_center = list_wind_center(num_wind);
        dis_q_wind = dis_q_wind(num_wind) ;
    end
% Solar QR to PDF
    min_solar = 0;
    if abs(solar_scale(end) - solar_scale(1)) <= 24
        mid = round((solar_scale(end) + solar_scale(1))/2);
        min_Solar = mid;
        list_solar_center = 0;
        dis_q_solar = 1;
    else
        [q_solar_comb,solar_sample] = tailSample_new(quant, solar_tilde,...
            paramEsts_right_S_ramp, paramEsts_left_S_ramp);
        solar_sample_scale = solar_sample * solar_factor;
        if solar_factor < 0
           solar_sample_scale = fliplr(solar_sample_scale);
            q_solar_comb =1- fliplr(q_solar_comb);
        end
        [list_solar,dis_q_solar] = QR2PDF_demand(q_solar_comb,solar_sample_scale, ...
           1);

        list_solar = - list_solar;
        min_solar = min(list_solar);
        max_solar = max(list_solar);
        num_solar = 1  : length(list_solar);
        list_solar_center = list_solar - min_solar;
        list_solar_center = list_solar_center(num_solar);
        dis_q_solar = dis_q_solar(num_solar) ;
    end
%     %wind QR to PDF
%     if abs(wind_scale(1) - wind_scale(end)) <= 24
%         mid = round((wind_scale(1) + wind_scale(end))/2);
%         min_wind = mid;
%         list_wind_center = 0;
%         dis_q_wind = 1;
%     else
%         [list_wind, dis_q_wind] = QR2PDF_w(quant, wind_scale,...
%             Wind_nameplate);
%         min_wind = min(list_wind);
%         max_wind = max(list_wind);
%         list_wind_center = list_wind - min_wind;
%         num_wind = 1 : (window * 500) : length(list_wind);
%         list_wind_center = list_wind_center(num_wind);        
%         dis_q_wind = dis_q_wind(num_wind) * window * 500;
%     end
%     %Solar QR to PDF
%     
%     if abs(solar_scale(end) - solar_scale(1)) <= 24
%         mid = round((solar_scale(end) + solar_scale(1))/2);
%         min_Solar = mid;
%         list_solar_center = 0;
%         dis_q_solar = 1;
%     else
%         [list_solar, dis_q_solar] = QR2PDF_s(quant, solar_scale);
%         min_Solar = min(list_solar);
%         max_Solar = max(list_solar);
% 
%         list_solar_center = list_solar - min_Solar;
%         num_solar = 1 : (window * 500) : length(list_solar);
%         list_solar_center = list_solar_center(num_solar);        
%         dis_q_solar = dis_q_solar(num_solar) * window * 500;
%     end
    %Intra hour variance to PDF
    day_id = find(time_real_demand(i,:) == 1);
    month_temp = day_id(1);
    hour_temp = day_id(3) - 19;
%     if hour_temp == 1
%         battery_ramp = battery_assign_day(1) - battery_assign_day(end);
%         pump_ramp = pump_assign_day(1) - pump_assign_day(end);
%     else
%         battery_ramp = battery_assign_day(hour_temp) - battery_assign_day(hour_temp - 1);
%         pump_ramp = pump_assign_day(hour_temp) - pump_assign_day(hour_temp - 1);
%     end
% 
%      if hour_temp == 1
%         battery_ramp =0;
%         pump_ramp = 0;
%     else
%         battery_ramp = battery_assign(month_temp, hour_temp) - battery_assign(month_temp, hour_temp - 1);
%         pump_ramp = pump_assign(month_temp, hour_temp) - pump_assign(month_temp, hour_temp - 1);
%     end
    
    q_G_L_W_S = conv(conv(conv(dis_q_demand, dis_q_wind),dis_q_solar),dist_ramp_outage);
    q_G_L_W_S = q_G_L_W_S/sum(q_G_L_W_S);
    x_G_L_W_S = 1 : length(q_G_L_W_S);
    
    x_G_L_W_S = x_G_L_W_S  + min_wind + min_demand + min_Solar + min_X_MW_ramp;
%     [~,b] = min(abs(x_G_L_W_S - ramp_up_limit));
%     [~,c] = min(abs(x_G_L_W_S - ramp_down_limit));
    [~,d] = min(abs(x_G_L_W_S - ramp_up_r));
    
%     p_up_g_0(i) =  sum(q_G_L_W_S(b:end));  
%     p_down_g_0(i) = sum(q_G_L_W_S(c:end));  
    p_up_real(i) =  sum(q_G_L_W_S(d:end));
    
    if ~mod(i,100)
         
        display(i);
    end
    
end
% LOLH_up = sum(p_up_g_0.*weight_q_list')
LOLH_up_real = sum(p_up_real.*weight_q_list')

LOLH_down = sum(p_down_g_0.*weight_q_list')
LOLH = LOLH_up +LOLH_down
%tail for W S D

listOfProb_ramp_up = zeros(n_data17,1);
listOfProb_ramp_down = zeros(n_data17,1);
listOfProb_ramp_up_real = zeros(n_data17,1);

% p_power_g_0 = p_power_g_0_list{n};
for i = 1 : n_data17
    temp = A_demand_test(i,:);
    [~,a]=ismember(temp,time_real_demand,'rows');
%     listOfProb_ramp_up(i) = p_up_g_0(a);
%     listOfProb_ramp_down(i) = p_down_g_0(a);
    listOfProb_ramp_up_real(i) = p_up_real(a);
end

figure 
% plot(listOfProb_ramp_up_real(8065:8065+7*24-1))
plot(listOfProb_ramp_up_real(7393:7393+7*24-1))
hold on
plot(listOfProb_ramp_down)

 

tail_n = 1;
[paramEsts_left_d_ramp,paramEsts_right_d_ramp,Error_left_pos_d_ramp,Error_right_pos_d_ramp] = demand_tail_est(tail_n,quant, ...
            demand_ramp_pred, diff(diag(ones(length(Demand2015),1))) * Demand2015);

[paramEsts_left_w_ramp,paramEsts_right_w_ramp,Error_left_pos_w_ramp,Error_right_pos_w_ramp] = demand_tail_est(tail_n,quant, ...
            wind_ramp_pred, diff(diag(ones(length(Wind_r),1))) * Wind_r);
[paramEsts_left_s_ramp,paramEsts_right_s_ramp,Error_left_pos_s_ramp,Error_right_pos_s_ramp] = demand_tail_est(tail_n,quant, ...
            pred_solar_ramp, diff(diag(ones(length(Solar_r),1))) * Solar_r);

        
        
p_ramp_g_0 = zeros(n_state,1);
ramp_capacity = 4200;
for i = 1 : n_state
    demand_ramp_tem =   time_real_w_s(i,:) * beta_demand_ramp' + alpha_demand_ramp';
    wind_ramp_tem =   time_real_w_s(i,:) * beta_wind_ramp' + alpha_wind_ramp';
    solar_ramp_tem =   time_real_w_s(i,:) * beta_solar_ramp' + alpha_solar_ramp';

%             (Hour_real(i,:) * r_demand_wind_matrix * Month_real') * Wind * ones(1,n_quant) + ...
%             A3_month_hour * r_solar_M_H .* Solar * ones(1,n_quant)
%             

%             temp_time_real_wind = time_real_w_s(i,:) * Wind_factor(i)  ;
%             temp_time_real_demand = time_real_demand(i,:);
%             temp_r_wind = Month_real(i,:) * r_wind_use ;
    %sample
    [q_demand_ramp,demand_sample_ramp] = tailSample_new(quant_ramp, demand_ramp_tem,...
        paramEsts_left_d_ramp, paramEsts_right_d_ramp);
    [q_wind_ramp,wind_sample_ramp] = tailSample_new(quant_ramp, wind_ramp_tem,...
        paramEsts_left_w_ramp, paramEsts_right_w_ramp);
    [q_solar_ramp,solar_sample_ramp] = tailSample_new(quant_ramp, demand_ramp_tem,...
        paramEsts_left_s_ramp, paramEsts_right_s_ramp);
    %pdf cdf
    [list_demand_ramp,dis_q_demand_ramp] = QR2PDF_demand(q_demand_ramp,demand_sample_ramp,1);
    [list_wind_ram,dis_q_wind_ramp] = QR2PDF_demand(q_wind_ramp,wind_sample_ramp,1);
    [list_solar_ram,dis_q_solar_ramp] = QR2PDF_demand(q_solar_ramp,solar_sample_ramp,1);
    
    %list 
    %demand
    list_demand_ramp = - list_demand_ramp;
    min_demand = min(list_demand_ramp);
    max_demand = max(list_demand_ramp);
    num_demand = 1 : 8 : length(list_demand_ramp);
    list_demand_center = list_demand_ramp - min_demand;
    list_demand_center = list_demand_center(num_demand);
    dis_q_demand_ramp = dis_q_demand_ramp(num_demand) * 8;
    dis_q_demand_ramp = fliplr(dis_q_demand_ramp);
    
    
    %wind
    min_wind = min(list_wind_ram);
    max_wind = max(list_wind_ram);
    num_wind = 1 : 8 : length(list_wind_ram);
    list_wind_center = list_wind_ram - min_wind;
    list_wind_center = list_wind_center(num_wind);
    dis_q_wind_ramp = dis_q_wind_ramp(num_wind) * 8;

    %solar
    day_night_id = Hour_real(i,:) * day_night_indicator * Month_real(i,:)';
    if day_night_id == 0                
        list_solar_center = 0;
        dis_q_solar_ramp = 1;
    else 

        min_solar = min(list_solar_ram);
        max_solar = max(list_solar_ram);
        num_solar = 1 : 8 : length(list_solar_ram);
        list_solar_center = list_solar_ram - min_solar;
        list_solar_center = list_solar_center(num_solar);
        dis_q_solar_ramp = dis_q_solar_ramp(num_solar) *  8;
 
    end

    
    
%     min_solar = min(list_solar_ram);
%     max_solar = max(list_solar_ram);
%     num_solar = 1 : (window * 8) : length(list_solar_ram);
%     list_solar_center = list_solar_ram - min_solar;
%     list_solar_center = list_solar_center(num_solar);
%     dis_q_solar_ramp = dis_q_solar_ramp(num_solar) * window * 8;
%     
%     

    q_G_L_W_S = conv(conv(dis_q_demand_ramp, dis_q_wind_ramp),dis_q_solar_ramp);
    q_G_L_W_S = q_G_L_W_S/sum(q_G_L_W_S);
    x_G_L_W_S = 1 : 8 : ((length(q_G_L_W_S) - 1) * 8 + 1);


    x_G_L_W_S_down = x_G_L_W_S + min_wind + min_demand + min_solar + ramp_capacity;
    x_G_L_W_S_up = x_G_L_W_S + min_wind + min_demand + min_solar - ramp_capacity;

%     cumQ = cumsum(q_G_L_W);
%     q_G_L_W = q_G_L_W/sum(q_G_L_W);

%     if(ismember(0,x_G_L_W))
%         a = find(x_G_L_W == 0);
%         p_power_g_0(i) = sum(q_G_L_W(1:a));  
%     end
    [~,b1] = min(abs(x_G_L_W_S_down));
    [~,b2] = min(abs(x_G_L_W_S_up));
    p_ramp_g_0(i) = sum(q_G_L_W_S(1:b1))  ;  
%     display(i);
%      tempProb = zeros((30 + 100)*10^3 + 1 ,1);
%      tempProb(ismember(cumX, x_G_L_W)) = q_G_L_W;
%      cumProb = cumProb + tempProb * ((rem(i,2) == 1) * p_holiday + (rem(i,2) ~= 1) * p_not_holiday);
%      hold on;
%      display(i);

end


LOLH_ramp = sum(p_ramp_g_0 .*  weight_q_list);
%1.8494
n = length(date15)
listOfProb_power = zeros(n,1);
for i = 1 : (n)
    temp = A15(i,:);
    [~,a]=ismember(temp,time_real_w_s,'rows');
    listOfProb_power(i) = p_ramp_g_0(a);
end

 
for n = 1:1000
    l(n) = - sum(ones(n,1)/n .* log(ones(n,1)/n))
end
plot(l)



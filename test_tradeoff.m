%% Scenarios
Demand_sc_train = 46.487; %current demand
Solar_sc_train =  20; % 2.1MW
generation_capacity = 1035 + 4395 + 238 + 52 + 43 + 121 + 176;
input_capacity = 28500;

Fixed_transfer = 20000;
ramp_capacity_up = (10+76.93+4.33+3.4+2.9+3.15) * 47  ;
ramp_capacity_down = (10+76.93+4.33+3.4+2.9+3.15) * 40 + 1500 ;
power_capacity = generation_capacity + input_capacity;
%% Curtailment for each month
 

LOLH_curtail_ramp_up = zeros(10, 12);
LOLH_curtail_ramp_down = zeros(10, 12);
LOLH_curtail_power = zeros(10, 12);

for k = 1:10
curtailment = [1, 1, 1, ...
    1, 1, 1, ...
    1, 1, 1,...
    1,1,1] * .1 * k;

solar_bound = zeros(n_state, 1);

p_ramp_down = 0;
p_ramp_up = 0;
n_state = 4032;
%% Risk for ramp
for i = 1:n_state
    
    day_id = find(time_real_demand(i,:) == 1);
    month_temp = day_id(1);
    hour_temp = day_id(3) - 19;
    curtail_q = curtailment(month_temp);
    
    r_ls = Hour_real(i,:) * r_demand_solar_ramp_matrix * Month_real(i,:)';

    demand_tilde =   time_real_demand(i,:) * beta_demand_ramp' + alpha_demand_ramp';   
    demand_factor = Demand_sc_train;
    demand_tilde_scale = demand_tilde * demand_factor;

    solar_tilde = (time_real_solar_power(i,:) * beta_solar_ramp' +  alpha_solar_ramp');
    solar_factor = Solar_sc_train  - Demand_sc_train * r_ls;
    solar_scale = solar_tilde * solar_factor * curtail_q;

    
    
    
% Demand QR to PDF
    [q_demand_comb,demand_sample] = tailSample_new(quant, demand_tilde_scale,...
        paramEsts_right_D_ramp, paramEsts_left_D_ramp);
    [list_demand,dis_q_demand] = QR2PDF_demand(q_demand_comb,demand_sample, ...
        1);
    min_demand = min(list_demand);
    max_demand = max(list_demand);
    num_demand = 1  : length(list_demand);
    list_demand_center = list_demand - min_demand;
    list_demand_center = list_demand_center(num_demand);
   
% Solar QR to PDF
    min_solar = 0;
    if abs(solar_scale(end) - solar_scale(1)) <= 24
        mid = round((solar_scale(end) + solar_scale(1))/2);
        min_solar = mid;
        list_solar_center = 0;
        dis_q_solar = 1;
        solar_bound(i) = 0;
    else
        [q_solar_comb,solar_sample] = tailSample_new(quant, solar_scale,...
            paramEsts_right_S_ramp, paramEsts_left_S_ramp);
%         solar_sample_scale = solar_sample * solar_factor;
        if solar_factor < 0
           solar_sample = fliplr(solar_sample);
            q_solar_comb =1- fliplr(q_solar_comb);
        end
        [list_solar,dis_q_solar] = QR2PDF_demand(q_solar_comb,solar_sample, ...
           1);

        list_solar = - list_solar;
        min_solar = min(list_solar);
        max_solar = max(list_solar);
        num_solar = 1  : length(list_solar);
        list_solar_center = list_solar - min_solar;
        list_solar_center = list_solar_center(num_solar);
        dis_q_solar = dis_q_solar(num_solar) ;
        dis_q_solar = fliplr(dis_q_solar);       
    end  
    
    q_G_L_W_S = conv(dis_q_demand, dis_q_solar);
    q_G_L_W_S = q_G_L_W_S/sum(q_G_L_W_S);
    x_G_L_W_S = 1 : length(q_G_L_W_S);
    
    x_G_L_W_S = x_G_L_W_S  + min_demand + min_solar;

    x_G_L_W_S_down = x_G_L_W_S + min_demand + min_solar + ramp_capacity_down;
    x_G_L_W_S_up = x_G_L_W_S  + min_demand + min_solar - ramp_capacity_up;

%     cumQ = cumsum(q_G_L_W);
%     q_G_L_W = q_G_L_W/sum(q_G_L_W);

%     if(ismember(0,x_G_L_W))
%         a = find(x_G_L_W == 0);
%         p_power_g_0(i) = sum(q_G_L_W(1:a));  
%     end
    [~,b1] = min(abs(x_G_L_W_S_down));
    [~,b2] = min(abs(x_G_L_W_S_up));
    p_ramp_down(i) = sum(q_G_L_W_S(1:b1))  ;  
%     [~,d] = min(abs(x_G_L_W_S - ramp_up_r));
    p_ramp_up(i) =  sum(q_G_L_W_S(b2:end));
    
    if ~mod(i,100)
         
        display(i);
    end
    
end


%% Risk for power
p_power = 0;
n_state = 4032;
for i = 1:n_state
    
    day_id = find(time_real_demand(i,:) == 1);
    month_temp = day_id(1);
    hour_temp = day_id(3) - 19;
    curtail_q = curtailment(month_temp);
    
    r_ls = Hour_real(i,:) * r_demand_solar_matrix * Month_real(i,:)';

    demand_tilde =   time_real_demand(i,:) * beta_demand' + alpha_demand';   
    demand_factor = Demand_sc_train;
    demand_sample_scale = demand_tilde * demand_factor;

    solar_tilde = (time_real_solar_power(i,:) * beta_solar' +  alpha_solar');
    solar_factor = Solar_sc_train  - Demand_sc_train * r_ls;
    solar_scale = solar_tilde * solar_factor * curtail_q;

    
    
    
% Demand QR to PDF
    [q_demand_comb,demand_sample] = tailSample_new(quant, demand_sample_scale,...
        paramEsts_right, paramEsts_left);
    
    [list_demand,dis_q_demand] = QR2PDF_demand(q_demand_comb,demand_sample, ...
        1);
    list_demand = - list_demand;
    min_demand = min(list_demand);
    max_demand = max(list_demand);
    num_demand = 1  : length(list_demand);
    list_demand_center = list_demand - min_demand;
    list_demand_center = list_demand_center(num_demand);
    dis_q_demand = fliplr(dis_q_demand);
    
% Solar QR to PDF
    min_solar = 0;
    if abs(solar_scale(end) - solar_scale(1)) <= 24
        mid = round((solar_scale(end) + solar_scale(1))/2);
        min_solar = mid;
        list_solar_center = 0;
        dis_q_solar = 1;
        solar_bound(i) = 0;
    else

        [list_solar, dis_q_solar] = QR2PDF_s(quant_s_w, solar_scale);
        min_solar = min(list_solar);
        max_Solar = max(list_solar);

        list_solar_center = list_solar - min_solar;
        num_solar = 1 : length(list_solar);
        list_solar_center = list_solar_center(num_solar);        
        dis_q_solar = dis_q_solar(num_solar);

    end  
    
    q_G_L_W_S = conv(dis_q_demand, dis_q_solar);
    q_G_L_W_S = q_G_L_W_S/sum(q_G_L_W_S);
    x_G_L_W_S = 1 : length(q_G_L_W_S);
    
    if hour_temp > 18 && hour_temp < 23
        p_c = power_capacity + 8000;
    elseif ismember(hour_temp, [16, 17, 18,23, 24])
        p_c = power_capacity + 6000;
    else
        p_c = power_capacity;
    end
    x_G_L_W_S = x_G_L_W_S + min_demand + min_solar + p_c;
    
    [~,b] = min(abs(x_G_L_W_S));
    p_power(i) = sum(q_G_L_W_S(1:b))  ;  
%     [~,d] = min(abs(x_G_L_W_S - ramp_up_r));
    
    if ~mod(i,100)
         
        display(i);
    end
end

LOLH_curtail_ramp_up(k, :) = p_ramp_up * weight_q_list_month; 
LOLH_curtail_ramp_down(k, :) = p_ramp_down * weight_q_list_month;
LOLH_curtail_power(k, :) = p_power * weight_q_list_month;
end

LOLH_curtail_ramp_up
LOLH_curtail_ramp_down
LOLH_curtail_power

size(weight_q_list)

LOLH_ramp_d = sum(p_ramp_down' .*  weight_q_list)
LOLH_ramp_u = sum(p_ramp_up' .*  weight_q_list)
LOLH_power = sum(p_power' .*  weight_q_list)


LOLH_ramp_d_m = sum(p_ramp_down' .*  weight_q_list_month)
LOLH_ramp_u_m = sum(p_ramp_up' .*  weight_q_list_month)
LOLH_power_m = sum(p_power' .*  weight_q_list_month)

figure
plot(0.1:0.1:1, fliplr(LOLH_curtail_power(:, 7)))
hold on
plot(0.1:0.1:1, fliplr(LOLH_curtail_ramp_up(:, 7)))
xlabel('Percentage of Curtail')
plot(0.1:0.01:1, 2*ones(length(0.1:0.01:1),1),'b')
legend('Power LOLH', 'Ramp up LOLH', 'LOLH = 2')
grid
title('July Power LOLH and Ramp up LOLH Trade-off Analysis')


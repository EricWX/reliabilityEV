lambda = [10^(-1), 10^(0)]; % HyperParameter
quant_s_w = [quant,1];
[beta_solar,alpha_solar, pred_solar] =opt_solar_power(quant_s_w,solar2018, date18,lambda);

%% Wind Optimization 


lambda_demand_power =  [10^(3),10^(0), 10^(7),10^(1),0];
[beta_demand, alpha_demand, r_demand_solar_M_H, pred_demand] = ...
    opt_demand_solar_power(quant, load2018_r, date18,...
    Holidays_18, solar2018,lambda_demand_power);%lambda 7 parameters
r_demand_solar_matrix  = reshape(r_demand_solar_M_H,[24,12]);

tail_n = 1;
[paramEsts_left,paramEsts_right,Error_left_pos,Error_right_pos] = ...
    demand_tail_est(tail_n,quant,pred_demand/1000, load2018_r/1000);

curtailment = [.7, .7, .7, ...
        .5, .5, .5, ...
        .7, .7, .5,...
        .8,.8,.8];
certain_quantile = 0.95;

solar_bound = zeros(n_state, 1);
for i = 1:n_state
    day_id = find(time_real_demand(i,:) == 1);
    month_temp = day_id(1);
    hour_temp = day_id(3) - 19;
    curtail_q = curtailment(month_temp);
    r_ls = Hour_real(i,:) * r_demand_solar_matrix * Month_real(i,:)';
    
%     
%     Demand_deviation_tilde = time_real_variance(i,:) * r_M_W_H';
%     variance_factor = Demand_sc_train;
%    
    demand_tilde =   (time_real_demand(i,:) * beta_demand' + alpha_demand') /1000;
    
    demand_factor = Demand_sc_train;
    
    
    solar_tilde = (time_real_solar_power(i,:) * beta_solar' +  alpha_solar');
    solar_factor =  Solar_sc_train - Demand_sc_train * r_ls;
    solar_scale = solar_tilde * solar_factor;
    Solar_nameplate =  solar_factor;

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
    num_demand = 1  : length(list_demand);
    list_demand_center = list_demand - min_demand;
    list_demand_center = list_demand_center(num_demand);
    dis_q_demand = dis_q_demand(num_demand);
    dis_q_demand = fliplr(dis_q_demand);
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


  
    %Solar QR to PDF
    
    if abs(solar_scale(end) - solar_scale(1)) <= 80
        mid = round((solar_scale(end) + solar_scale(1))/2);
        min_Solar = mid;
        list_solar_center = 0;
        dis_q_solar = 1;
    else
        [list_solar, dis_q_solar] = QR2PDF_s(quant_s_w, solar_scale);
        min_Solar = min(list_solar);
        max_Solar = max(list_solar);

        list_solar_center = list_solar - min_Solar;
        num_solar = 1 : length(list_solar);
        list_solar_center = list_solar_center(num_solar);        
        dis_q_solar = dis_q_solar(num_solar);
        
        acc_dis_q_solar = cumsum(dis_q_solar);
        [~, c_position] = min(abs(acc_dis_q_solar - (1 - certain_quantile)));
        solar_bound(i) = c_position + min_Solar;
        list_solar_center = list_solar_center(c_position:end);
        dis_q_solar = dis_q_solar(c_position:end);
        dis_q_solar(1) = 1 - curtail_q;
        dis_q_solar = dis_q_solar/sum(dis_q_solar);
    end
    %Intra hour variance to PDF

    q_G_L_W_S = conv(dis_q_demand,dis_q_solar);
    q_G_L_W_S = q_G_L_W_S/sum(q_G_L_W_S);
    x_G_L_W_S = 1 : length(q_G_L_W_S);


    x_G_L_W_S = x_G_L_W_S + min_demand + min_Solar +...
                 + generation_capacity  +Fixed_transfer ;
    [~,b] = min(abs(x_G_L_W_S));
    p_power_g_0(i) = sum(q_G_L_W_S(1:b));  
    if ~mod(i,100)
        
        display(i);
    end
end


list_solar_bound = zeros(length(date18), 1);
listOfProb_power = zeros(length(date18), 1);
for i = 1 : length(date18)
    temp = A_demand_test(i,:);
    [~,a] = ismember(temp,time_real_demand,'rows');
    listOfProb_power(i) = p_power_g_0(a);
    
    list_solar_bound(i) = solar_bound(a);
end

curtail_solar = zeros(length(date18),1);
for i = 1:length(date18)
    if Solar_sc_train * solar2018(i) > list_solar_bound(i)
        curtail_solar(i) = list_solar_bound(i);
    else
        curtail_solar(i) = solar2018(i) * Solar_sc_train;
    end
end
figure
plot(solar2018*Solar_sc_train)
hold on
plot(curtail_solar)

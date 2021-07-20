function[list_demand,dis_q_demand] = QR2PDF_demand(q_demand_comb,sample_value_comb, ...
    scale_factor)

sample_value_comb = sample_value_comb/scale_factor;

sample_step_demand = 5;             % Sampling interval for the PDF model
rho_demand = 10^12/sample_step_demand^3;
[x_demand, q_demand] = qCDF2PDF(sample_value_comb,q_demand_comb,sample_step_demand,rho_demand);
% ind = (q_demand< 10^(-7));
% q_demand(ind) = [];
% x_demand(ind) = [];
x_demand = x_demand * scale_factor;
q_demand = fliplr(q_demand)/ scale_factor;

    pdf_demand_clean = q_demand;
    ind_clean_demand = find(pdf_demand_clean < 0);
    pdf_demand_clean(ind_clean_demand) = [];
    x_demand(ind_clean_demand) = [];

max_n_demand = floor(max(x_demand)) ;
min_n_demand = ceil(min(x_demand));
list_demand = min_n_demand :max_n_demand;
dis_q_demand = interp1(x_demand,pdf_demand_clean,list_demand);
dis_q_demand = dis_q_demand/sum(dis_q_demand);


end

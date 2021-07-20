function [test_stats,bin_summary] =test_error(quant, real_data, pred_data)

n_q = length(quant);
n_data = length(real_data);
prob_list = zeros(n_q+1,1);
prob_list(1) = quant(1);
for i = 2:n_q
    prob_list(i) = quant(i) - quant(i-1);
end
prob_list(n_q+1) = 1 - quant(end);



% Ind = (real_data > 0);
% for i = 1 : n_q
%     Ind = Ind .* (pred_data(:,i) > 0);
% end

point_bin = ((real_data - pred_data(:,1)) <= 0);
for i = 1 : (n_q - 1)
    test_vector = ((real_data - pred_data(:,i)) > 0 & (real_data - pred_data(:,i+1)) <= 0);
    point_bin = point_bin +  test_vector * (i+1);
end
point_bin = point_bin +  ((real_data - pred_data(:,n_q)) > 0) * (n_q+1);
bin_summary = zeros(n_q+1,1);

for i = 1: (n_q+1)
    bin_summary(i) = sum(point_bin == i);
end
% n_over0 = sum(bin_summary(2:end));
% test_stats = (bin_summary(2:end) - n_over0 * prob_list(2:end)).^2./(prob_list(2:end)*n_over0);

n_over0 = sum(bin_summary);
test_stats = sum((bin_summary - n_data * prob_list).^2./n_data);


dimention = n_q;

end
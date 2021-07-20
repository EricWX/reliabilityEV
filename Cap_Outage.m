function [x_MW,dist_conv24] = Cap_Outage(Probability, Outage,n_hour)


Probability_clear = Probability;
Outage_clear = Outage;
ind_0 = find(Outage == 0);
Probability_clear(ind_0) = [];
Outage_clear(ind_0) = [];
Prob_h = 1 - Probability_clear; % get the probability that is not outage.
%%%
n_Outage = length(Outage_clear);   % get the number of generators
p_round = zeros(n_Outage,4); % create a matrix to store the information of each generation
                             % column 1 is the floor of the capacity
                             % column 2 is the probability of the floor of
                             % the capacity
                             % column 3 is the probability of the ceil of
                             % the capacity
                             % colum 4 is the probability of fail with zero
                             % capacity  
iflO = (floor(Outage_clear)==Outage_clear);         % logical array 
cO = ceil(Outage_clear);                      % 
cO(iflO) = Outage_clear(iflO)+1;              % fix ceiling for integer Outage 
p_round = [cO, Prob_h.*(cO-Outage_clear), Prob_h.*(Outage_clear-floor(Outage_clear)), 1-Prob_h];
for i = 1:n_Outage
    if Outage_clear(i) < 1
        p_round(i,2) = p_round(i,4) + p_round(i,2);
        p_round(i,4) = 0;
    end
end
%

tic;
grid_n = ceil(sum(Outage_clear)) + 20;
% initial value
% Create an array that, the first element is 
% the probability when c = 0, the
% second element is the probability
% when c = 1. .... 

%For each generation, there
%are at most three elements with
%nonzero probability.

dist_conv = zeros(p_round(1,1)+1,1);   
                                       
dist_conv(1) = p_round(1,4);
dist_conv(p_round(1,1)) = p_round(1,2);
dist_conv(p_round(1,1)+1) = p_round(1,3);
temp_dist = zeros(grid_n,1);
temp_dist = double(temp_dist);
dist_conv = double(dist_conv);
% convolution
% Using loop to convolution
for i = 2:n_Outage
    new_power = zeros((p_round(i,1)+1),1);
    new_power(1) = p_round(i,4);
    new_power(p_round(i,1)) = p_round(i,2);
    new_power(p_round(i,1) + 1) = p_round(i,3);
    dist_conv = conv(dist_conv,new_power);
end
%
% plot the convolution
dist_conv24 = dist_conv;


for i = 2:n_hour
    dist_conv24 = conv(dist_conv24,dist_conv);
end

n_conv = length(dist_conv24); % the final lenght of the convolution
x_MW = 0:(n_conv-1); % argument in MW 
ind = (dist_conv24 < 10^(-8));
dist_conv24(ind) = [];
x_MW(ind)=[];

% plot the generation capacity pdf 
% Transition from MW to GW, x-axis/1000, p * 1000
%x_GW = x_MW/1000; 

end
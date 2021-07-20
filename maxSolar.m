function max_solar_list = maxSolar(Solar, Date)

[A_solar_test, month_hour_matrix] = date_matrix(Date,'solar', 0);

max_solar_list = zeros(1, 288);
for i = 1:length(Date)
    id = find(month_hour_matrix(i,:) == 1);
    max_solar_list(id) = max([max_solar_list(id), Solar(i)]);
end

% 
% Hour = hour(Date);
% Month = month(Date);
% 
% n_r = length(Solar);
% Months = zeros(n_r,12);
% Hours = zeros(n_r,24);
% day_night_indicator = zeros(24,12);
% 
% for i = 1:12
%     Months(:,i) = (Month == i);
% end
% for i = 1:24
%     Hours(:,i) = (Hour == i-1);
% end
% 
% 
% for i = 1:n_r
%     if Solar(i) == 0
%         n = find(Months(i,:) == 1);
%         m = find(Hours(i,:) == 1);
%         day_night_indicator(m,n) = day_night_indicator(m,n) + 1;        
%     end    
% end
% day_night_matrix = (day_night_indicator == 0);
% 
% for i = 1:n_r
%     day_night_list(i) = Hours(i,:) * day_night_matrix * Months(i,:)';
% end
% day_night_list = day_night_list';
end


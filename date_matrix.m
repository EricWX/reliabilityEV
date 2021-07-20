function [date_matrix, month_hour] = date_matrix(date, variable, Holidays)
n_r = length(date);
Month = month(date);
Hour = hour(date);
Months = zeros(n_r,12);
Hours = zeros(n_r,24);
month_hour = zeros(n_r, 24*12);

for i = 1:12
    Months(:,i) = (Month == i);
end
Months = sparse(Months);
for i = 1:24
    Hours(:,i) = (Hour == i - 1);    
end
Hours = sparse(Hours);
for i = 1:12
    for j = 1:24
        month_hour(:,(i-1)*24 + j) = (Month == i).*(Hour == j-1);
    end
end
month_hour = sparse(month_hour);

if isequal(variable, 'wind') || isequal(variable, 'solar')
    
    date_matrix = [Months,Hours];
    date_matrix = sparse(date_matrix);
    
else 
    
    Weekday = weekday(date);
    
    Weekdays = zeros(n_r,7);
    Holidayis =ismember(date,Holidays);             % create holiday array of the year


    
    for i = 1:7
        Weekdays(:,i) = (Weekday == i);
    end


    Holiday = [Holidayis,1 - Holidayis];

    date_matrix = sparse([Months, Weekdays, Hours, Holiday]);
    
end
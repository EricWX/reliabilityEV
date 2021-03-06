%%  deal with raw data

load_temp = load1(zone == 'CA ISO');
hour_temp = hour(hour1(zone == 'CA ISO'));
date_temp = Date(zone == 'CA ISO');
hour_temp = hour(date_temp);

load2018_r = [load_temp(1)];
idx = 1;
for i =2:length(load_temp)
    gap = hour_temp(i) - hour_temp(i-1);
    if gap == -23
        idx = idx + 1;
        load2018_r(idx) = load_temp(i);
    
    elseif gap >= 1
        for j = 1:gap
            idx = idx + 1;
            load2018_r(idx) = load_temp(i-1) + j * (load_temp(i) - load_temp(i-1))/gap;
        end
    end
end
%2018 outlier -- 5965

load2018_r(5965) = (load2018_r(5964) + load2018_r(5966))/2; 




% For 2017 data
load1 = load1(1:89261);
Hour = Hour(1:89261);
Date = Date(1:89261);
zone = zone(1:89261);

load_temp = load1(zone == 'CA ISO');
hour_temp = hour(Hour(zone == 'CA ISO'));
date_temp = Date(zone == 'CA ISO');


load2017_r = [load_temp(1)];
idx = 1;
for i =2:length(load_temp)
    gap = hour_temp(i) - hour_temp(i-1);
    if gap < 0
        gap = 24 + gap;
    elseif date_temp(i) - date_temp(i-1) > 0
        gap = gap + 24*(date_temp(i) - date_temp(i-1) - 1);
    end
    for j = 1:gap
        idx = idx + 1;
        load2017_r(idx) = load_temp(i-1) + j * (load_temp(i) - load_temp(i-1))/gap;
    end
end



% For 2016 data


load_temp = load2(zone1 == 'CA ISO');
datetime_temp = Date1(zone1 == 'CA ISO');
datetime_temp = datetime(string(datetime_temp));
hour_temp = hour(datetime_temp);
% date_temp = date(datetime_temp);


load2016_r = [load_temp(1)];
idx = 1;
for i =2:length(load_temp)
    gap = hour_temp(i) - hour_temp(i-1);
    if gap < 0
        gap = 24 + gap;
%     elseif date_temp(i) - date_temp(i-1) > 0
%         gap = gap + 24*(date_temp(i) - date_temp(i-1) - 1);
    end
    for j = 1:gap
        idx = idx + 1;
        load2016_r(idx) = load_temp(i-1) + j * (load_temp(i) - load_temp(i-1))/gap;
    end
end
mean(load2018_r)/mean(load2017_r)




% For 2019 data


load_temp = load1(zone == 'CA ISO');
datetime_temp = Date(zone == 'CA ISO');
datetime_temp = datetime(string(datetime_temp));
hour_temp = hour(datetime_temp);
% date_temp = date(datetime_temp);


load2019_r = [load_temp(1)];
idx = 1;
for i =2:length(load_temp)
    gap = hour_temp(i) - hour_temp(i-1);
    if gap < 0
        gap = 24 + gap;
%     elseif date_temp(i) - date_temp(i-1) > 0
%         gap = gap + 24*(date_temp(i) - date_temp(i-1) - 1);
    end
    for j = 1:gap
        idx = idx + 1;
        load2019_r(idx) = load_temp(i-1) + j * (load_temp(i) - load_temp(i-1))/gap;
    end
end



mean(load2018_r)/mean(load2017_r)






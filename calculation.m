% calculate total electricity usage of a week

% number total cars
n_car_ca = 15 *10^6;
% fraction of EV
EV_ratio = 0.079;
% miles per veicle per year 
miles_per_year = 11071;
% ave kWh/100miles
ave_kwh_100miles = 25;

annual_power_usage =  n_car_ca * 1 * miles_per_year/100 * ave_kwh_100miles;
% annual_power_usage = 332400* miles_per_year/100 * ave_kwh_100miles;
dayly_power_usage = annual_power_usage/365 / 10^6; %GW

% weekday and weekend might be different; If it is find the ratio  
% 37.9144KWh daily need to prepare to charge at least





function conv24 = conv_hour(numberOfHour, power)

conv24 = zeros((length(power) - numberOfHour),1);
for i = 1 : (length(power) - numberOfHour + 1)
    conv24(i) = sum(power(i:(i + numberOfHour - 1)));
end
end
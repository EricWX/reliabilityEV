generationunits( && generationunits.Primary1 ! ,:)

op = (generationunits.Status == 'OP');
p1 = (generationunits.Primary1 ~= 'NA');
p2 = (generationunits.Primary1 ~= 'WND');
p3 = (generationunits.Primary1 ~= 'WH');
p4 = (generationunits.Primary1 ~= 'SUN');
c1 = (generationunits.Capacity  > 20);
n1 = (generationunits.netMWh  > 10000);

sum(generationunits.Capacity(op & p1 & p2 & p3 & p4 & c1 & n1))
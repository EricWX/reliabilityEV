import = [6,6,6,6,6,6,...
          5,4,3,2,1,1,...
          1,1,1,2,3,4,...
          5,6,7,7,7,7];
plot(0:23, import)
ylim([0,7])
grid
xlim([0,23])
ylabel('GW')
xlabel('Hour')

title('Import Profile (temp)')

 %% IEEE Standard Figure Configuration - Version 2.0

% run this code before the plot command

%%
% According to the standard of IEEE Transactions and Journals: 

% Times New Roman is the suggested font in labels. 

% For a singlepart figure, labels should be in 8 to 10 points,
% whereas for a multipart figure, labels should be in 8 points.

% Width: column width: 8.8 cm; page width: 18.1 cm.

%% width & hight of the figure
k_scaling = 4;          % scaling factor of the figure
% (You need to plot a figure which has a width of (8.8 * k_scaling)
% in MATLAB, so that when you paste it into your paper, the width will be
% scalled down to 8.8 cm  which can guarantee a preferred clearness.
k_width_hight = 2;      % width:hight ratio of the figure

width = 8.8 * k_scaling;
hight = width / k_width_hight;

%% figure margins
top = 2*.7;  % normalized top margin
bottom = 3*5/5;	% normalized bottom margin
left = 2*.6;	% normalized left margin
right = 0.5/3;  % normalized right margin

%% set default figure configurations
set(0,'defaultFigureUnits','centimeters');
set(0,'defaultFigurePosition',[0 0 17 17/2]);

% % For correlaiton 
% k_scaling = 4;          % scaling factor of the figure
% % (You need to plot a figure which has a width of (8.8 * k_scaling)
% % in MATLAB, so that when you paste it into your paper, the width will be
% % scalled down to 8.8 cm  which can guarantee a preferred clearness.
% k_width_hight = 2;      % width:hight ratio of the figure
% 
% width = 8.8 * k_scaling;
% hight = width / k_width_hight;
% 
% %% figure margins
% top = 2/9;  % normalized top margin
% bottom = 3/2*1.35;	% normalized bottom margin
% left = 3*1.5;	% normalized left margin
% right = 0.5/2;  % normalized right margin


%% For PP plots
% k_scaling = 5;          % scaling factor of the figure
% % (You need to plot a figure which has a width of (8.8 * k_scaling)
% % in MATLAB, so that when you paste it into your paper, the width will be
% % scalled down to 8.8 cm  which can guarantee a preferred clearness.
% k_width_hight = 2;      % width:hight ratio of the figure
% 
% width = 8.8 * k_scaling;
% hight = width / k_width_hight;
% 
% %% figure margins
% top = 2/3*2;  % normalized top margin
% bottom = 3/2;	% normalized bottom margin
% left =  - 3/28;	% normalized left margin
% right = 0;  % normalized right margin
% 
% %% set default figure configurations
% set(0,'defaultFigureUnits','centimeters');
% set(0,'defaultFigurePosition',[0 0 17 17]);
% 
% 
% 
% %% set default figure configurations
% set(0,'defaultFigureUnits','centimeters');
% set(0,'defaultFigurePosition',[0 0 17 17*7/10]);
% 
% %%
% set(0,'defaultLineLineWidth',.7*k_scaling);
% set(0,'defaultAxesLineWidth',0.25*k_scaling);
% 
% set(0,'defaultAxesGridLineStyle',':');
% set(0,'defaultAxesYGrid','on');
% set(0,'defaultAxesXGrid','on');
% 
% set(0,'defaultAxesFontName','Times New Roman');
% set(0,'defaultAxesFontSize',5*k_scaling);
% 
% set(0,'defaultTextFontName','Times New Roman');
% set(0,'defaultTextFontSize',5*k_scaling);
% 
% set(0,'defaultLegendFontName','Times New Roman');
% set(0,'defaultLegendFontSize',3*k_scaling);



% normal
%% set default figure configurations
set(0,'defaultFigureUnits','centimeters');
set(0,'defaultFigurePosition',[0 0 17 17/2]);


% 
% %% set default figure configurations
% set(0,'defaultFigureUnits','centimeters');
% set(0,'defaultFigurePosition',[0 0 17 17*7/10]);

%
set(0,'defaultLineLineWidth',.5*k_scaling);
set(0,'defaultAxesLineWidth',0.25*k_scaling);

set(0,'defaultAxesGridLineStyle',':');
set(0,'defaultAxesYGrid','on');
set(0,'defaultAxesXGrid','on');

set(0,'defaultAxesFontName','Times New Roman');
set(0,'defaultAxesFontSize',4*k_scaling);

set(0,'defaultTextFontName','Times New Roman');
set(0,'defaultTextFontSize',4*k_scaling);

set(0,'defaultLegendFontName','Times New Roman');
set(0,'defaultLegendFontSize',2*k_scaling);


set(0,'defaultAxesUnits','normalized');
set(0,'defaultAxesPosition',[left/width bottom/hight (width-left-right)/width  (hight-bottom-top)/hight]);

% set(0,'defaultAxesColorOrder',[0 0 0]);
set(0,'defaultAxesTickDir','out');

set(0,'defaultFigurePaperPositionMode','auto');

% you can change the Legend Location to whatever as you wish
set(0,'defaultLegendLocation','southeast');
set(0,'defaultLegendBox','on');
set(0,'defaultLegendOrientation','vertical');

% LOLH_curtail_power
%     10.4619
%     3.8568
%     2.1445
%     1.8601
%     1.6584
%     1.3401
%     1.0449
%     0.8804
%     0.8080
%     0.7735
%     0.7513
% 
%     LOLH_curtail_ramp_up
%  0.0343
%     0.0140
%     0.0125
%     0.0125
%     0.0125
%     0.0125
%     0.0136
%     0.1183
%     1.1509
%     4.6259
%    11.6258

% s2 
% LOLH_curtail_power
% 0.2253
%     0.2452
%     0.2579
%     0.2640
%     0.2728
%     0.3070
%     0.5464
%     2.0961
%     6.4763
%    13.9954
%    29.5564
% LOLH_curtail_ramp_up
%  84.1714
%    71.4624
%    54.3257
%    33.7154
%    14.4416
%     3.0417
%     0.1720
%     0.0126
%     0.0125
%     0.0125
%     0.0343

%s3
% LOLH_curtail_power
% 0.2704
%     0.2992
%     0.3374
%     0.3890
%     0.4464
%     0.4971
%     0.5605
%     0.8184
%     4.1122
%    19.4972
%    55.4290
%  LOLH_curtail_ramp_up  
%  141.3575
%   125.0527
%   107.1333
%    92.4284
%    75.9765
%    50.7620
%    19.0744
%     1.6648
%     0.0135
%     0.0125
%     0.0343

% new s2
% power
% 0.1722
%     0.1939
%     0.2215
%     0.2468
%     0.2605
%     0.2687
%     0.3004
%     0.6683
%     3.9481
%    12.2954
%    29.5564
% 
% ramp up
% 74.1536
%    56.5105
%    32.7659
%    11.6379
%     1.8687
%     0.0408
%     0.0125
%     0.0125
%     0.0125
%     0.0125
%     0.0343
  



figure
%s1 
% plot(0:0.1:1, LOLH_curtail_power, '-')
% hold on
% plot(0:0.1:1, LOLH_curtail_ramp_up, '--')
% plot(0:0.01:1, 2*ones(length(0:0.01:1),1), '.')

%s2
% plot(0.4:0.1:.9, LOLH_curtail_power(5:end-1), '-')
% hold on
% plot(0.4:0.1:.9, LOLH_curtail_ramp_up(5:end-1), '--')
% 
% plot(0.4:0.01:.9, 2*ones(length(0.4:0.01:.9),1), '.')

%s3
figure
plot(0.2:0.1:.9, LOLH_curtail_power(3:end-1), '-')
hold on
plot(0.2:0.1:.9, LOLH_curtail_ramp_up(3:end-1), '--')
plot(0.2:0.01:.9, 2*ones(length(0.2:0.01:.9),1), '.')

figure
semilogy(0.2:0.1:.9, LOLH_curtail_power(3:end-1), '-')
hold on
semilogy(0.2:0.1:.9, LOLH_curtail_ramp_up(3:end-1), '--')
semilogy(0.2:0.01:.9, 2*ones(length(0.2:0.01:.9),1), '.')
xlim([.2, .9])

yticks([0.1 1 10])
yticklabels({'10^{-1}','10^{0}','10^{1}'})

xlabel('Percentage of Curtailment')
legend('Power LOLH', 'Ramp up LOLH', 'LOLH = 2')
title('LOLH and LORH Trade-off Analysis, July (Scenario 2)')


x_width =18.2386; % NOT TESTED, MIGHT REQUIRE TWEAKING
y_height = 15.4667; % NOT TESTED, MIGHT REQUIRE TWEAKING
FigHandle = gcf;
set(FigHandle, 'PaperUnits', 'centimeters');
set(FigHandle, 'PaperPosition', [0 0 x_width y_height]);
%%% set figure to have no margins
FigHandle.PaperPositionMode = 'auto';
fig_pos = FigHandle.PaperPosition;
FigHandle.PaperSize = [fig_pos(3) fig_pos(4)];
print(gcf, '-bestfit','-dpdf', 'plots\ramp_trade_off_s2_log3.pdf');


% import profile
k_scaling = 4;          % scaling factor of the figure
% (You need to plot a figure which has a width of (8.8 * k_scaling)
% in MATLAB, so that when you paste it into your paper, the width will be
% scalled down to 8.8 cm  which can guarantee a preferred clearness.
k_width_hight = 2;      % width:hight ratio of the figure

width = 8.8 * k_scaling;
hight = width / k_width_hight;

%% figure margins
top = 2*.7;  % normalized top margin
bottom = 3*5/5;	% normalized bottom margin
left = 2 * 1.2;	% normalized left margin
right = 0.5/3;  % normalized right margin

%% set default figure configurations
set(0,'defaultFigureUnits','centimeters');
set(0,'defaultFigurePosition',[0 0 17 17/2]);

% % For correlaiton 
% k_scaling = 4;          % scaling factor of the figure
% % (You need to plot a figure which has a width of (8.8 * k_scaling)
% % in MATLAB, so that when you paste it into your paper, the width will be
% % scalled down to 8.8 cm  which can guarantee a preferred clearness.
% k_width_hight = 2;      % width:hight ratio of the figure
% 
% width = 8.8 * k_scaling;
% hight = width / k_width_hight;
% 
% %% figure margins
% top = 2/9;  % normalized top margin
% bottom = 3/2*1.35;	% normalized bottom margin
% left = 3*1.5;	% normalized left margin
% right = 0.5/2;  % normalized right margin


%% For PP plots
% k_scaling = 5;          % scaling factor of the figure
% % (You need to plot a figure which has a width of (8.8 * k_scaling)
% % in MATLAB, so that when you paste it into your paper, the width will be
% % scalled down to 8.8 cm  which can guarantee a preferred clearness.
% k_width_hight = 2;      % width:hight ratio of the figure
% 
% width = 8.8 * k_scaling;
% hight = width / k_width_hight;
% 
% %% figure margins
% top = 2/3*2;  % normalized top margin
% bottom = 3/2;	% normalized bottom margin
% left =  - 3/28;	% normalized left margin
% right = 0;  % normalized right margin
% 
% %% set default figure configurations
% set(0,'defaultFigureUnits','centimeters');
% set(0,'defaultFigurePosition',[0 0 17 17]);
% 
% 
% 
% %% set default figure configurations
% set(0,'defaultFigureUnits','centimeters');
% set(0,'defaultFigurePosition',[0 0 17 17*7/10]);
% 
% %%
% set(0,'defaultLineLineWidth',.7*k_scaling);
% set(0,'defaultAxesLineWidth',0.25*k_scaling);
% 
% set(0,'defaultAxesGridLineStyle',':');
% set(0,'defaultAxesYGrid','on');
% set(0,'defaultAxesXGrid','on');
% 
% set(0,'defaultAxesFontName','Times New Roman');
% set(0,'defaultAxesFontSize',5*k_scaling);
% 
% set(0,'defaultTextFontName','Times New Roman');
% set(0,'defaultTextFontSize',5*k_scaling);
% 
% set(0,'defaultLegendFontName','Times New Roman');
% set(0,'defaultLegendFontSize',3*k_scaling);



% normal
%% set default figure configurations
set(0,'defaultFigureUnits','centimeters');
set(0,'defaultFigurePosition',[0 0 17 17/2]);


% 
% %% set default figure configurations
% set(0,'defaultFigureUnits','centimeters');
% set(0,'defaultFigurePosition',[0 0 17 17*7/10]);

%
set(0,'defaultLineLineWidth',.5*k_scaling);
set(0,'defaultAxesLineWidth',0.25*k_scaling);

set(0,'defaultAxesGridLineStyle',':');
set(0,'defaultAxesYGrid','on');
set(0,'defaultAxesXGrid','on');

set(0,'defaultAxesFontName','Times New Roman');
set(0,'defaultAxesFontSize',4*k_scaling);

set(0,'defaultTextFontName','Times New Roman');
set(0,'defaultTextFontSize',4*k_scaling);

set(0,'defaultLegendFontName','Times New Roman');
set(0,'defaultLegendFontSize',2*k_scaling);


set(0,'defaultAxesUnits','normalized');
set(0,'defaultAxesPosition',[left/width bottom/hight (width-left-right)/width  (hight-bottom-top)/hight]);

% set(0,'defaultAxesColorOrder',[0 0 0]);
set(0,'defaultAxesTickDir','out');

set(0,'defaultFigurePaperPositionMode','auto');

% you can change the Legend Location to whatever as you wish
set(0,'defaultLegendLocation','southeast');
set(0,'defaultLegendBox','on');
set(0,'defaultLegendOrientation','vertical');



import1 = [6,6,6,6,6,6,...
          5,4,3,2,1,1,...
          1,1,1,2,3,4,...
          5,6,7,7,7,7];
import2 = [8,8,8,8,8,7,...
          6,5,4,3,2,1,...
          1,1,1,2,4,6,...
          8,9,9,9,9,9];
import3 = [10,10,10,10,10,8,...
          6,4,3,2,1,1,...
          1,1,1,3,5,7,...
          9,11,11,11,11,11];     
figure
plot(0:23, import2)
% hold on
% plot(0:23, import2, ':')
% plot(0:23, import3, '--')
import = [8,8,8,8,8,7,...
          6,5,4,3,2,1,...
          1,1,1,2,4,6,...
          8,9,9,9,9,9];
ylim([0,13])
xlim([0,23])
ylabel('GW')
xlabel('Hour')
% legend('Scenario 1', 'Scenario 2', 'Scenario 3')

title('Import Profile')

x_width =18.2386; % NOT TESTED, MIGHT REQUIRE TWEAKING
y_height = 15.4667; % NOT TESTED, MIGHT REQUIRE TWEAKING
FigHandle = gcf;
set(FigHandle, 'PaperUnits', 'centimeters');
set(FigHandle, 'PaperPosition', [0 0 x_width y_height]);
%%% set figure to have no margins
FigHandle.PaperPositionMode = 'auto';
fig_pos = FigHandle.PaperPosition;
FigHandle.PaperSize = [fig_pos(3) fig_pos(4)];
print(gcf, '-bestfit','-dpdf', 'plots\import_3s_new.pdf');



%% Diff Load
k_scaling = 4;          % scaling factor of the figure
% (You need to plot a figure which has a width of (8.8 * k_scaling)
% in MATLAB, so that when you paste it into your paper, the width will be
% scalled down to 8.8 cm  which can guarantee a preferred clearness.
k_width_hight = 2;      % width:hight ratio of the figure

width = 8.8 * k_scaling;
hight = width / k_width_hight;

%% figure margins
top = 2*.7;  % normalized top margin
bottom = 3*5/5;	% normalized bottom margin
left = 2 * 1.2;	% normalized left margin
right = 0.5/3;  % normalized right margin

%% set default figure configurations
set(0,'defaultFigureUnits','centimeters');
set(0,'defaultFigurePosition',[0 0 17 17/2]);

% % For correlaiton 
% k_scaling = 4;          % scaling factor of the figure
% % (You need to plot a figure which has a width of (8.8 * k_scaling)
% % in MATLAB, so that when you paste it into your paper, the width will be
% % scalled down to 8.8 cm  which can guarantee a preferred clearness.
% k_width_hight = 2;      % width:hight ratio of the figure
% 
% width = 8.8 * k_scaling;
% hight = width / k_width_hight;
% 
% %% figure margins
% top = 2/9;  % normalized top margin
% bottom = 3/2*1.35;	% normalized bottom margin
% left = 3*1.5;	% normalized left margin
% right = 0.5/2;  % normalized right margin


%% For PP plots
% k_scaling = 5;          % scaling factor of the figure
% % (You need to plot a figure which has a width of (8.8 * k_scaling)
% % in MATLAB, so that when you paste it into your paper, the width will be
% % scalled down to 8.8 cm  which can guarantee a preferred clearness.
% k_width_hight = 2;      % width:hight ratio of the figure
% 
% width = 8.8 * k_scaling;
% hight = width / k_width_hight;
% 
% %% figure margins
% top = 2/3*2;  % normalized top margin
% bottom = 3/2;	% normalized bottom margin
% left =  - 3/28;	% normalized left margin
% right = 0;  % normalized right margin
% 
% %% set default figure configurations
% set(0,'defaultFigureUnits','centimeters');
% set(0,'defaultFigurePosition',[0 0 17 17]);
% 
% 
% 
% %% set default figure configurations
% set(0,'defaultFigureUnits','centimeters');
% set(0,'defaultFigurePosition',[0 0 17 17*7/10]);
% 
% %%
% set(0,'defaultLineLineWidth',.7*k_scaling);
% set(0,'defaultAxesLineWidth',0.25*k_scaling);
% 
% set(0,'defaultAxesGridLineStyle',':');
% set(0,'defaultAxesYGrid','on');
% set(0,'defaultAxesXGrid','on');
% 
% set(0,'defaultAxesFontName','Times New Roman');
% set(0,'defaultAxesFontSize',5*k_scaling);
% 
% set(0,'defaultTextFontName','Times New Roman');
% set(0,'defaultTextFontSize',5*k_scaling);
% 
% set(0,'defaultLegendFontName','Times New Roman');
% set(0,'defaultLegendFontSize',3*k_scaling);



% normal
%% set default figure configurations
set(0,'defaultFigureUnits','centimeters');
set(0,'defaultFigurePosition',[0 0 17 17/2]);


% 
% %% set default figure configurations
% set(0,'defaultFigureUnits','centimeters');
% set(0,'defaultFigurePosition',[0 0 17 17*7/10]);

%
set(0,'defaultLineLineWidth',.5*k_scaling);
set(0,'defaultAxesLineWidth',0.25*k_scaling);

set(0,'defaultAxesGridLineStyle',':');
set(0,'defaultAxesYGrid','on');
set(0,'defaultAxesXGrid','on');

set(0,'defaultAxesFontName','Times New Roman');
set(0,'defaultAxesFontSize',4*k_scaling);

set(0,'defaultTextFontName','Times New Roman');
set(0,'defaultTextFontSize',4*k_scaling);

set(0,'defaultLegendFontName','Times New Roman');
set(0,'defaultLegendFontSize',2*k_scaling);


set(0,'defaultAxesUnits','normalized');
set(0,'defaultAxesPosition',[left/width bottom/hight (width-left-right)/width  (hight-bottom-top)/hight]);

% set(0,'defaultAxesColorOrder',[0 0 0]);
set(0,'defaultAxesTickDir','out');

set(0,'defaultFigurePaperPositionMode','auto');

% you can change the Legend Location to whatever as you wish
set(0,'defaultLegendLocation','southeast');
set(0,'defaultLegendBox','on');
set(0,'defaultLegendOrientation','vertical');

figure 
ax_weekday1 = 12 : 24 : 156;
ax_weekday2 = 24 : 24 : 156;
Weekday_xlable = ['SUN';'MON';'TUE';'WED';'THU';'FRI';'SAT'];
ax(1)=newplot;
grid;
set(gcf,'nextplot','add');
set(ax(1),'Xlim',[0,167],'XTick',ax_weekday1,'XTicklabel',Weekday_xlable,'YTickLabel',[]);
ax(2)=axes('position',get(ax(1),'position'),'Visible', 'off');

% pred_demand_ramp_plot = pred_demand_ramp;
% pred_demand_ramp_plot(:,5)= (pred_demand_ramp(:,5) + pred_demand_ramp(:,6))/2;

pred_demand_test = A_demand_test * beta_demand_ramp' + ones(length(date18),1)*alpha_demand_ramp' + ...
        month_hour_matrix_test * r_demand_solar_M_H_ramp  .* [0; r_solar_test] * ones(1,n_quant);

pred_demand_ramp_plot = pred_demand_test;
pred_demand_ramp_plot(:,5)= (pred_demand_test(:,5) + pred_demand_test(:,6))/2;

plot(0:167, pred_demand_ramp_plot(4345:4512,[1,3,5,8,10]) * max(load2018_r)/1000/1000)
hold on
d_l = diff(load_test);
scatter(0:167, d_l(4344:4511) * max(load2018_r)/1000/1000)
set(ax(2),'Xlim',[0,167],'XTick',ax_weekday2,'XTicklabel','');
ylim([-8,4])
% xlabel('Hours of Day')
ylabel('GW')
legend( 'q = 0.05', 'q = 0.25','q = 0.5', 'q = 0.75', 'q = 0.95', 'Real Data') 
title('Ramp of Load in July')  

x_width =18.2386; % NOT TESTED, MIGHT REQUIRE TWEAKING
y_height = 15.4667; % NOT TESTED, MIGHT REQUIRE TWEAKING
FigHandle = gcf;
set(FigHandle, 'PaperUnits', 'centimeters');
set(FigHandle, 'PaperPosition', [0 0 x_width y_height]);
%%% set figure to have no margins
FigHandle.PaperPositionMode = 'auto';
fig_pos = FigHandle.PaperPosition;
FigHandle.PaperSize = [fig_pos(3) fig_pos(4)];
print(gcf, '-bestfit','-dpdf', 'plots\dif_load_new2.pdf');c


%% Coefficient Factor
figure
x = 1 : 24;
y = 1 : 12;
F = (reshape(r_demand_solar_M_H_ramp,[24,12]))';
% F= r_demand_solar_M_H_ramp';
surf(x,y,F)
xlim([1,24])
ylim([1,12])
zlim([-0.12,0.21])
xlabel('Hour')
ylabel('Month')
zlabel('Impact Factor')
zlabel('\Gamma_{WS}')

x_width =18.2386; % NOT TESTED, MIGHT REQUIRE TWEAKING
y_height = 15.4667; % NOT TESTED, MIGHT REQUIRE TWEAKING
FigHandle = gcf;
set(FigHandle, 'PaperUnits', 'centimeters');
set(FigHandle, 'PaperPosition', [0 0 x_width y_height]);
%%% set figure to have no margins
FigHandle.PaperPositionMode = 'auto';
fig_pos = FigHandle.PaperPosition;
FigHandle.PaperSize = [fig_pos(3) fig_pos(4)];
print(gcf, '-bestfit','-dpdf', 'plots\demand_solar.pdf');



%% for coefficient factor

% width & hight of the figure
k_scaling = 4;          % scaling factor of the figure
% (You need to plot a figure which has a width of (8.8 * k_scaling)
% in MATLAB, so that when you paste it into your paper, the width will be
% scalled down to 8.8 cm  which can guarantee a preferred clearness.
k_width_hight = 2;      % width:hight ratio of the figure

width = 8.8 * k_scaling;
hight = width / k_width_hight;

%% figure margins
top = 1.4;  % normalized top margin
bottom = 3*5/7;	% normalized bottom margin
left = 5;	% normalized left margin
right = 0.5/10;  % normalized right margin

%% set default figure configurations
set(0,'defaultFigureUnits','centimeters');
set(0,'defaultFigurePosition',[0 0 17 10]);


set(0,'defaultLineLineWidth',.5*k_scaling);
set(0,'defaultAxesLineWidth',0.25*k_scaling);

set(0,'defaultAxesGridLineStyle',':');
set(0,'defaultAxesYGrid','on');
set(0,'defaultAxesXGrid','on');

set(0,'defaultAxesFontName','Times New Roman');
set(0,'defaultAxesFontSize',4*k_scaling);

set(0,'defaultTextFontName','Times New Roman');
set(0,'defaultTextFontSize',4*k_scaling);

set(0,'defaultLegendFontName','Times New Roman');
set(0,'defaultLegendFontSize',2*k_scaling);


set(0,'defaultAxesUnits','normalized');
set(0,'defaultAxesPosition',[left/width bottom/hight (width-left-right)/width  (hight-bottom-top)/hight]);

% set(0,'defaultAxesColorOrder',[0 0 0]);
set(0,'defaultAxesTickDir','out');

set(0,'defaultFigurePaperPositionMode','auto');

% you can change the Legend Location to whatever as you wish
set(0,'defaultLegendLocation','southeast');
set(0,'defaultLegendBox','on');
set(0,'defaultLegendOrientation','vertical');


figure
x = 1 : 24;
y = 1 : 12;
F = (reshape(r_demand_solar_M_H_ramp,[24,12]))';
% F= r_demand_solar_M_H_ramp';
surf(x,y,F)
xlim([1,24])
ylim([1,12])
zlim([-.8,.6])
xlabel('Hour')
ylabel('Month')
zlabel('Impact Factor')
zlabel('\Gamma_{LS}')


title('Coefficients \gamma(Month,Hour)')

x_width =18.2386; % NOT TESTED, MIGHT REQUIRE TWEAKING
y_height = 15.4667; % NOT TESTED, MIGHT REQUIRE TWEAKING
FigHandle = gcf;
set(FigHandle, 'PaperUnits', 'centimeters');
set(FigHandle, 'PaperPosition', [0 0 x_width y_height]);
%%% set figure to have no margins
FigHandle.PaperPositionMode = 'auto';
fig_pos = FigHandle.PaperPosition;
FigHandle.PaperSize = [fig_pos(3) fig_pos(4)];
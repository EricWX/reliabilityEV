
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

figure
charging = [repelem(0,17), 5, 10 , 15 , 20, 25, 30, 35, 40];
xlim([0, 24]);
xlabel('Window Length (hour)')
ylabel('GW')
plot(0:24, charging)



















% profile of EV and storage
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
left = 4 * 1.2;	% normalized left margin
right = 0.5/3;  % normalized right margin

%% set default figure configurations
set(0,'defaultFigureUnits','centimeters');
set(0,'defaultFigurePosition',[0 0 17 17/2]);



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



EV_energy_requirement = [zeros(1, 23),0.5,1];
Battery_requirement = [0, -0.25, -0.5,-0.75, -ones(1, 17), -0.75, -0.5, -0.25,0];
Pump_requirement = [0, -1, -ones(1, 19), -0.75, -0.5, -0.25,0];

 
figure
plot(0:24, EV_energy_requirement)
hold on
plot(0:24, Battery_requirement, '--')
plot(0:24, Pump_requirement, '--')

ylim([-1,1])
xlim([0,24])
ylabel('Percentage')
xlabel('Hour of A Day')
legend('EV(E_n)', 'Battery(B_n)', 'Pump(P_n)')

title(upper('Required Charging Energy'))

x_width =18.2386; % NOT TESTED, MIGHT REQUIRE TWEAKING
y_height = 15.4667; % NOT TESTED, MIGHT REQUIRE TWEAKING
FigHandle = gcf;
set(FigHandle, 'PaperUnits', 'centimeters');
set(FigHandle, 'PaperPosition', [0 0 x_width y_height]);
%%% set figure to have no margins
FigHandle.PaperPositionMode = 'auto';
fig_pos = FigHandle.PaperPosition;
FigHandle.PaperSize = [fig_pos(3) fig_pos(4)];
print(gcf, '-bestfit','-dpdf', 'plots\energy_profile2.pdf');



EV_op_perfect = - [-1/80,-1/40,-1/40,-1/40,-1/40,-1/40,...
              -1/20,-5/80,-7/80 ,-4/40,-4/40,-1/10,...
              -4/40,-4/40,-3/40,-4/80,-1/40,0, ...
              0,0,0,0,0,-1/80];
          
EV_op_predict = [450,250,200,150,120,130,...
          200,300,280,260,250,240,...
          260,320, 390, 550, 680,880,...
          1000,950, 880, 820,730, 600];
        
EV_op_perfect = [-1/80,-1/80,-1/80,-1/40,-1/40,-1/80,...
              -3/80,-5/80,-7/80 ,-4/40,-4/40,-1/10,...
              -4/40,-4/40,-3/40,-5/80,-4/80,-1/80, ...
              0,0,0,0,0,-1/80]
EV_op_predict = EV_op_predict/sum(EV_op_predict);



Battery_op = [-1/6,-1/6,-1/6,-1/6,-1/6,-1/6,...
              0,0,0,0,0,0,...
              0,0,0,0,0.00,0.2,...
              0.3,0.25,0.2,0.05,0,0];
          
Battery_op = [0,-1/24,-1/24,-1/24,-1/24,-1/12,...
              -1/12,-1/12,-1/12,0,0,-1/12,...
              -1/12,-1/12,-1/12,-1/12,-1/12,0.1,...
              0.2,0.2,0.25,0.15,0.05,0.05];
          
figure
plot(0:23, EV_op_predict)
hold on
plot(0:23, EV_op_perfect)




stairs(0:24, [Battery_op, Battery_op(end)], 'LineWidth',2)
hold on
stairs(0:23, EV_op_predict)

ylim([0,.1])
xlim([0,23])
ylabel('Percentage')
xlabel('Hour of a Day')
legend('Scenario A', 'Scenario C')

title(upper('EV Charging Profile'))

x_width =18.2386; % NOT TESTED, MIGHT REQUIRE TWEAKING
y_height = 15.4667; % NOT TESTED, MIGHT REQUIRE TWEAKING
FigHandle = gcf;
set(FigHandle, 'PaperUnits', 'centimeters');
set(FigHandle, 'PaperPosition', [0 0 x_width y_height]);
%%% set figure to have no margins
FigHandle.PaperPositionMode = 'auto';
fig_pos = FigHandle.PaperPosition;
FigHandle.PaperSize = [fig_pos(3) fig_pos(4)];
print(gcf, '-bestfit','-dpdf', 'plots\EV_charging_profile2.pdf');



% width & hight of the figure
k_scaling = 4;          % scaling factor of the figure
% (You need to plot a figure which has a width of (8.8 * k_scaling)
% in MATLAB, so that when you paste it into your paper, the width will be
% scalled down to 8.8 cm  which can guarantee a preferred clearness.
k_width_hight = 2;      % width:hight ratio of the figure

width = 8.8 * k_scaling;
hight = width / k_width_hight;

%% figure margins
top = 0.8;  % normalized top margin
bottom = 3*5/7;	% normalized bottom margin
left = 5.6;	% normalized left margin
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
F = (r_demand_wind_matrix)';
% F= r_demand_solar_M_H_ramp';
surf(x,y,F)
xlim([1,24])
ylim([1,12])
% zlim([-.8,.6])
xlabel('Hour')
ylabel('Month')
zlabel('Impact Factor')
zlabel('\Gamma_{LW}')

x_width =18.2386; % NOT TESTED, MIGHT REQUIRE TWEAKING
y_height = 15.4667; % NOT TESTED, MIGHT REQUIRE TWEAKING
FigHandle = gcf;
set(FigHandle, 'PaperUnits', 'centimeters');
set(FigHandle, 'PaperPosition', [0 0 x_width y_height]);
%%% set figure to have no margins
FigHandle.PaperPositionMode = 'auto';
fig_pos = FigHandle.PaperPosition;
FigHandle.PaperSize = [fig_pos(3) fig_pos(4)];
print(gcf, '-bestfit','-dpdf', 'plots\gamma_lw.pdf');



figure
x = 1 : 24;
y = 1 : 12;
F = (r_demand_solar_matrix)';
% F= r_demand_solar_M_H_ramp';
surf(x,y,F)
xlim([1,24])
ylim([1,12])
% zlim([-.8,.6])
xlabel('Hour')
ylabel('Month')
zlabel('Impact Factor')
zlabel('\Gamma_{LS}')


x_width =18.2386; % NOT TESTED, MIGHT REQUIRE TWEAKING
y_height = 15.4667; % NOT TESTED, MIGHT REQUIRE TWEAKING
FigHandle = gcf;
set(FigHandle, 'PaperUnits', 'centimeters');
set(FigHandle, 'PaperPosition', [0 0 x_width y_height]);
%%% set figure to have no margins
FigHandle.PaperPositionMode = 'auto';
fig_pos = FigHandle.PaperPosition;
FigHandle.PaperSize = [fig_pos(3) fig_pos(4)];
print(gcf, '-bestfit','-dpdf', 'plots\gamma_ls.pdf');



figure
x = 1 : 24;
y = 1 : 12;
F = (r_wind_solar_matrix)';
% F= r_demand_solar_M_H_ramp';
surf(x,y,F)
xlim([1,24])
ylim([1,12])
% zlim([-.8,.6])
xlabel('Hour')
ylabel('Month')
zlabel('Impact Factor')
zlabel('\Gamma_{WS}')




% title('Coefficients \gamma(Month,Hour)')

x_width =18.2386; % NOT TESTED, MIGHT REQUIRE TWEAKING
y_height = 15.4667; % NOT TESTED, MIGHT REQUIRE TWEAKING
FigHandle = gcf;
set(FigHandle, 'PaperUnits', 'centimeters');
set(FigHandle, 'PaperPosition', [0 0 x_width y_height]);
%%% set figure to have no margins
FigHandle.PaperPositionMode = 'auto';
fig_pos = FigHandle.PaperPosition;
FigHandle.PaperSize = [fig_pos(3) fig_pos(4)];
print(gcf, '-bestfit','-dpdf', 'plots\gamma_ws.pdf');


LOLH_s0 = [0.2393
0.6907
0.8402
0.8330
0.8057
0.6936
0.5602
0.4262
0.3262
0.2279
0.1639
0.1131
0.0633
0.0383
0.0262
0.0147
0.0093
0.0054
0.0047
0.0042
0.0050
0.0060
0.0139
0.0789];
figure
plot(1:24,LOLH_s0)
xlim([1,24])
xlabel('Length of Window(Hour)')
ylabel('LOLH')

LOLH_s1 = [0.2351
0.6588
0.7832
0.7553
0.7719
0.6823
0.5802
0.4493
0.3494
0.2200
0.1860
0.1340
0.0702
0.0559
0.0395
0.0234
0.0146
0.0089
0.0077
0.0071
0.0081
0.0094
0.0216
0.0918]
figure
plot(1:24,LOLH_s1)
xlim([1,24])
xlabel('Length of Window(Hour)')
ylabel('LOLH')


s1_EV= -[-1/160,-1/40,-1/40,-1/40,-1/40,-1/40,...
              -1/20,-5/80,-7/80 ,-4/40,-4/40,-1/10,...
              -4/40,-4/40,-3/40,-4/80,-1/40,0, ...
              0,0,0,0,0,-1/160];
figure
plot(0:23, s1_EV)
xlim([0,23])
xlabel('Hour of a Day')
ylabel('Percentage')

EV_op_perfect = -[450,250,200,150,120,130,...
          200,300,280,260,250,240,...
          260,320, 390, 550, 680,880,...
          1000,950, 880, 820,730, 600];
s0_EV =  EV_op_perfect/sum(EV_op_perfect);
figure
plot(0:23, s0_EV)
xlim([0,23])
xlabel('Hour of a Day')
ylabel('Percentage')

LOLH_s2 = [0.0030
0.0397
0.0618
0.0612
0.0727
0.0711
0.0798
0.0614
0.0459
0.0158
0.0209
0.0138
0.0038
0.0041
0.0026
0.0016
0.0012
0.0010
0.0010
0.0011
0.0015
0.0020
0.0139
0.0546];
figure
plot(1:24,LOLH_s2)
xlim([1,24])
xlabel('Length of Window(Hour)')
ylabel('LOLH')

s2_EV = -[-1/80,-1/80,-1/80,-1/40,-1/40,-1/80,...
              -3/80,-5/80,-7/80 ,-4/40,-4/40,-1/10,...
              -4/40,-4/40,-3/40,-5/80,-4/80,-1/80, ...
              0,0,0,0,0,-1/80];
figure
plot(0:23, s2_EV)
xlim([0,23])
xlabel('Hour of a Day')
ylabel('Percentage')

Battery_op = [-1/6,-1/6,-1/6,-1/6,-1/6,-1/6,...
              0,0,0,0,0,0,...
              0,0,0,0,0.00,0.2,...
              0.3,0.25,0.2,0.05,0,0];





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

figure
charging = [repelem(0,17), 5, 10 , 15 , 20, 25, 30, 35, 40];
xlim([0, 24]);
xlabel('Window Length (hour)')
ylabel('GW')
plot(0:24, charging)



















% profile of EV and storage
k_scaling = 4;          % scaling factor of the figure
% (You need to plot a figure which has a width of (8.8 * k_scaling)
% in MATLAB, so that when you paste it into your paper, the width will be
% scalled down to 8.8 cm  which can guarantee a preferred clearness.
k_width_hight = 2;      % width:hight ratio of the figure

width = 8.8 * k_scaling;
hight = width / k_width_hight;

%% figure margins
top = 2*.7;  % normalized top margin
bottom = 3*3/5;	% normalized bottom margin
left = 4 * 1;	% normalized left margin
right = 0.5/3;  % normalized right margin

%% set default figure configurations
set(0,'defaultFigureUnits','centimeters');
set(0,'defaultFigurePosition',[0 0 17 17/2]);



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
plot(0:167,listOfProb_power_upper(4321:4321+7*24-1))
set(ax(2),'Xlim',[0,168],'XTick',ax_weekday2,'XTicklabel','');
% xlabel('Weekday')
ylabel('Probability')
title('Peak Week LOLP')


x_width =18.2386; % NOT TESTED, MIGHT REQUIRE TWEAKING
y_height = 15.4667; % NOT TESTED, MIGHT REQUIRE TWEAKING
FigHandle = gcf;
set(FigHandle, 'PaperUnits', 'centimeters');
set(FigHandle, 'PaperPosition', [0 0 x_width y_height]);
%%% set figure to have no margins
FigHandle.PaperPositionMode = 'auto';
fig_pos = FigHandle.PaperPosition;
FigHandle.PaperSize = [fig_pos(3) fig_pos(4)];
print(gcf, '-bestfit','-dpdf', 'plots\peak_week_LOLP_original.pdf');
print(gcf, '-bestfit','-dpdf', 'plots\peak_week_LOLP_updated.pdf');




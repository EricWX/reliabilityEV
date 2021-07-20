function [u,g] = qCDF2PDF(x,q,d,rho) 
% qCDF2PDF - compute PDF from quantile model of the CDF 
% [u,g] = qCDF2PDF(x,q,d,rho) 
% INPUTS
%     x - quantile levels
%     q - quantiles 
%     d - sampling interval for the PDF model 
%     rho - regularization parameter  
% OUTPUTS
%     u - PDF arguments
%     g - PDF values  

% Author: Dimitry Gorinevsky 
% Ver. July 7, 2017
     
u = (min(x):d:max(x))';  % Argument samples for the PDF model 
N = length(u);        % problem size  
y = interp1(x,q,u);   % interpolated samples of the CDF at u 
%%% Solve optimal estimation problem 
% f = arg min ||y-f||^2 + rho*||D^3*f||^2 
% Solution: f = (I+rho*(D^3)'*D^3)\y 
I = speye(N,N); 
D3 = diff(diff(diff(I)));  % D^3 matrix 
f = (I+rho*(D3'*D3))\y;   % estimated CDF 
gg = diff(f)/d; % first difference of the CDF (crude estimate of PDF) 
g = ([gg;0]+[0;gg])/2;   	% zero-phase estimate of the PDF 
end

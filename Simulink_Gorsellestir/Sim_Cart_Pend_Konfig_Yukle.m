% Reset
clc
clear all

% Konfigurasyon_Yukle
% excel_datasi        = importdata("Cart_Pend_Sablon.xlsx");
% Zaman_sn            = excel_datasi.data(:,1);
% Cart_Pos_m          = excel_datasi.data(:,2);
% Cart_Vel_m_sn       = excel_datasi.data(:,3);
% Theta_rad           = excel_datasi.data(:,4);
% Theta_dot_rad_sn    = excel_datasi.data(:,5);
% 
% Cart_Pos_m_ts          = timeseries(Cart_Pos_m		 , Zaman_sn);
% Cart_Vel_m_sn_ts       = timeseries(Cart_Vel_m_sn	 , Zaman_sn);
% Theta_rad_ts           = timeseries(Theta_rad		 , Zaman_sn);
% Theta_dot_rad_sn_ts    = timeseries(Theta_dot_rad_sn , Zaman_sn);


M = .5;
m = 0.2;
b = 0.1;
I = 0.006;
g = 9.8;
l = 0.3;

p = I*(M+m)+M*m*l^2; %denominator for the A and B matrices

A = [0      1              0           0;
     0 -(I+m*l^2)*b/p  (m^2*g*l^2)/p   0;
     0      0              0            1;
     0 -(m*l*b)/p       m*g*l*(M+m)/p  0];
B = [     0;
     (I+m*l^2)/p;
          0;
        m*l/p];
C = [1 0 0 0;
     0 0 1 0];
D = [0;
     0];

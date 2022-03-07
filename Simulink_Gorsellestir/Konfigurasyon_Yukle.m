% Reset
clc
clear all

% Konfigurasyon_Yukle

excel_datasi = importdata("Sablon.xlsx");
Zaman_sn  = excel_datasi.data(:,1);
Theta_der = excel_datasi.data(:,2);

Theta_der_ts = timeseries(Theta_der,Zaman_sn);

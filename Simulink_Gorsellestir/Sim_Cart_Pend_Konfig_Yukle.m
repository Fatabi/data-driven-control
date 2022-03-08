% Reset
clc
clear all

% Konfigurasyon_Yukle

excel_datasi        = importdata("Cart_Pend_Sablon.xlsx");
Zaman_sn            = excel_datasi.data(:,1);
Cart_Pos_m          = excel_datasi.data(:,2);
Cart_Vel_m_sn       = excel_datasi.data(:,3);
Theta_der           = excel_datasi.data(:,4);
Theta_dot_der_sn    = excel_datasi.data(:,5);

Cart_Pos_m_ts          = timeseries(Cart_Pos_m		 , Zaman_sn);
Cart_Vel_m_sn_ts       = timeseries(Cart_Vel_m_sn	 , Zaman_sn);
Theta_der_ts           = timeseries(Theta_der		 , Zaman_sn);
Theta_dot_der_sn_ts    = timeseries(Theta_dot_der_sn , Zaman_sn);
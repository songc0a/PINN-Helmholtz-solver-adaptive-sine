function [zb, ff, nt, T_zb]=subwavelet_ricker(dt,f_main,gama)

% nt=256;
% nt=512;
nt=4096;
% nt=2048;
% nt=8192;
for ii=1:nt
    t(ii)=dt*(ii-1);
end

T_zb=1*(1-2*(pi*f_main.*(t-1.5/f_main)).^2).*exp(-(pi*f_main.*(t-1.5/f_main)).^2).*exp(-gama.*(t));

[zb, ff]=fftrl(T_zb,t);
% zb=zb';
ff=ff';

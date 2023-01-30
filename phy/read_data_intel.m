T = readtable('waterfall.xlsx');
s = size(T);
snr = -10:0.1:30;
figure;
for N_MCS = 1:s(1)
    prob = table2array(T(N_MCS, :));
    writematrix([snr; prob],['MCS', num2str(N_MCS), '.csv']);
    plot(snr, prob, 'LineWidth', 1.5);
    hold on;
end


set(gca, 'YScale', 'log');
ylim([10e-11, 1]);
grid on;
xlabel('SNR');
ylabel('BLER');
legend({'MCS 0'; 'MCS 1'; 'MCS 2'; 'MCS 3'; 'MCS 4'; 'MCS 5'; 'MCS 6'; ...
        'MCS 7'; 'MCS 8'; 'MCS 9'; 'MCS 10'; 'MCS 11'; 'MCS 12'; ...
        'MCS 13'; 'MCS 14'; 'MCS 15'; 'MCS 16'; 'MCS 17'; 'MCS 18'; ...
        'MCS 19'; 'MCS 20'; 'MCS 21'; 'MCS 22'; 'MCS 23'; 'MCS 24'; ...
        'MCS 25'; 'MCS 26'; 'MCS 27'});

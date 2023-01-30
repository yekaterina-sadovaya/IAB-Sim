clear
close all
% fig = openfig('BER_FIGURES_LDPC.fig');
fig = openfig('BLER curves/BLER_8448_2.fig'); % BLER_calib_intel

dataObjs = findobj(fig,'-property','YData');
figure;
x_cell = {};
y_cell = {};
for i = 1:length(dataObjs)
    x = dataObjs(i).XData;
    y = dataObjs(i).YData;
    hold on; plot(x,y, 'LineWidth', 1.5);
    x_cell{29-i+1} = x;
    y_cell{29-i+1} = y;
    % writematrix([x; y],['BLER curves/MCS', num2str(29-i), '.csv']);
end

figure;
filename = 'BLER curves/MCS.xls';
for i = 1:length(x_cell)
    x = x_cell{i};
    y = y_cell{i};
    % y(y == 0) = 1e-10;
    hold on; plot(x,y, 'LineWidth', 1.5);
    %  writematrix([x; y],'BLER curves/MCS.xls', 'Sheet',i);
    
%     xlswrite(filename,num2str(i),'MCS' + num2str(i-1));
%     xlswrite(filename,num2str(i),'MCS' + num2str(i-1));
%     xlswrite(filename,[x; y],num2str(i));

    tabl = table(x.', y.','VariableNames',{'SNR','BLER'});
    writetable(tabl, filename,'sheet', ['MCS', num2str(i-1)]);

end
set(gca, 'YScale', 'log');
grid on;
xlabel('SNR');
ylabel('BLER');
legend({'MCS 0'; 'MCS 1'; 'MCS 2'; 'MCS 3'; 'MCS 4'; 'MCS 5'; 'MCS 6'; ...
        'MCS 7'; 'MCS 8'; 'MCS 9'; 'MCS 10'; 'MCS 11'; 'MCS 12'; ...
        'MCS 13'; 'MCS 14'; 'MCS 15'; 'MCS 16'; 'MCS 17'; 'MCS 18'; ...
        'MCS 19'; 'MCS 20'; 'MCS 21'; 'MCS 22'; 'MCS 23'; 'MCS 24'; ...
        'MCS 25'; 'MCS 26'; 'MCS 27'; 'MCS 28'});
xlim([-25, 20]);
ylim([10e-4, 1]);
clc; clear
addpath("loss\")
% files = {'loss_grid10_mask.txt', 'loss_grid8_mask.txt', 'loss_grid4_mask.txt',... 
%     'loss_no_mask.txt', 'loss_rand_mask.txt'};
files = {'loss_grid10_mask_10.txt', 'loss_grid10_mask_50.txt', 'loss_grid10_mask_90.txt',... 
    'loss_grid10_mask_130.txt', 'loss_grid10_mask_170.txt'};
colors = ['b', 'g', 'r', 'k', 'm'];  % Colors for each line
markers = ['o', '+', '*', 'x', 's']; % Markers for each line
h = zeros(1, length(files));
figure;
for i = 1:length(files)
data = load(files{i});

Iter = 1:length(data) / 2;

Amp_Loss = data(1:2:end, 1);
Coeff_Loss = data(2:2:end, 1);

h(i) = plot(Iter, Coeff_Loss, 'Color', colors(i), 'Marker', markers(i), 'Linewidth', 2,...
         'MarkerIndices', 1:100:length(Amp_Loss));  
hold on
end
set(h(1),'DisplayName','Dis=10');
set(h(2),'DisplayName','Dis=50');
set(h(3),'DisplayName','Dis=90');
set(h(4),'DisplayName','Dis=130');
set(h(5),'DisplayName','Dis=170');
ylabel('Amp loss','FontWeight','bold');
xlabel('Iter','FontWeight','bold');
title('Loss Across Iterations','FontWeight','bold');
set(gca,'FontWeight','bold','GridLineWidth',1,'LineWidth',1);
legend show; % 显示图例



% 参数设置
wavelength = 532e-9; % 波长（米）
k = 2 * pi / wavelength; % 波数
w0 = 40e-3; % 光束腰宽（10 微米）
x = linspace(-50e-3, 50e-3, 500); % x 坐标
y = linspace(-50e-3, 50e-3, 500); % y 坐标
[X, Y] = meshgrid(x, y);

% 高斯光束参数
x0_1 = -20e-3; y0_1 = 0; % 第一个光束的中心
x0_2 = 20e-3; y0_2 = 0;  % 第二个光束的中心

% 函数生成高斯光束强度分布
gaussian_intensity = @(x, y, x0, y0, w0) ...
    exp(-2 * ((x - x0).^2 + (y - y0).^2) / w0^2);

% 生成高斯强度分布
I1 = gaussian_intensity(X, Y, x0_1, y0_1, w0);
I2 = gaussian_intensity(X, Y, x0_2, y0_2, w0);

% 模拟相干因子
coherence_factors = 0.1:0.1:0.9;
contrasts = zeros(size(coherence_factors));
for i = 1:length(coherence_factors)
    I_interference = I1 + I2 + 2 * coherence_factors(i) * sqrt(I1 .* I2) .* cos(k * (X - Y) + 0.5 * pi);
    I_max = max(I_interference(200:300, 200:300));
    I_min = min(I_interference(200:300, 200:300));
    contrasts(i) = (I_max - I_min) / (I_max + I_min);
end

% 创建双纵坐标图
figure;

% 设置左侧纵坐标
yyaxis left
plot(coherence_factors, contrasts, '--', 'Color', [0, 0.4470, 0.7410], 'LineWidth', 2, ...
    'Marker', 's', 'MarkerSize', 8, 'MarkerFaceColor', [0, 0.4470, 0.7410]); hold on

resolution_plus_k = (contrasts(9) - 0.8) / 0.8;
resolution_plus = resolution_plus_k * coherence_factors + 0.75;
plot(coherence_factors, resolution_plus, '-', 'Color', [0.8500, 0.3250, 0.0980], 'LineWidth', 2, ...
    'Marker', 'o', 'MarkerSize', 8, 'MarkerFaceColor', [0.8500, 0.3250, 0.0980]); hold on

ylabel('Resolution');
ylim([0 1]);
yticks(0:0.2:1);

% 设置右侧纵坐标
yyaxis right
resistance_k = (0.1 - 0.92) / 0.8;
resistance = resistance_k * coherence_factors + 0.95;
plot(coherence_factors, resistance, '--', 'Color', [0.4660, 0.6740, 0.1880], 'LineWidth', 2, ...
    'Marker', '^', 'MarkerSize', 8, 'MarkerFaceColor', [0.4660, 0.6740, 0.1880]); hold on

resistance_plus_k = -(resistance(1) - 0.75) / 0.8;
resistance_plus = resistance_plus_k * coherence_factors + 0.9;
plot(coherence_factors, resistance_plus, '-', 'Color', [0.9290, 0.6940, 0.1250], 'LineWidth', 2, ...
    'Marker', 'd', 'MarkerSize', 8, 'MarkerFaceColor', [0.9290, 0.6940, 0.1250]); hold on

ylabel('Resistance');
ylim([0 1]);
yticks(0:0.2:1);

% 设置横坐标范围和刻度
xlabel('Coherence Factor');
xlim([0 1]);
xticks(0:0.2:1);

% 设置字体和样式
set(gca, 'FontName', 'Arial', 'FontSize', 14, 'LineWidth', 1.5, 'Box', 'on');

% 设置所有轴的边框颜色为黑色
yyaxis left;
ax = gca;
ax.YColor = 'k'; % 设置左边y轴为黑色

yyaxis right;
ax.YColor = 'k'; % 设置右边y轴为黑色

ax.XColor = 'k'; % 设置x轴为黑色

% 添加图例
legend({'Original Resolution', 'Plus Resolution', 'Original Resistance', 'Plus Resistance'}, ...
    'NumColumns', 2, 'Location', 'northoutside', 'Orientation', 'horizontal', 'Box', 'off', 'FontSize', 12);



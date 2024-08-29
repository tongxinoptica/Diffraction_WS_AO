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
    'Marker', 'd', 'MarkerSize', 8, 'MarkerFaceColor', [0, 0.4470, 0.7410]); hold on

resolution_plus_k = (contrasts(9) - 0.8) / 0.8;
resolution_plus = resolution_plus_k * coherence_factors + 0.75;
plot(coherence_factors, resolution_plus, '-', 'Color', [0.8500, 0.3250, 0.0980], 'LineWidth', 2, ...
    'Marker', 'd', 'MarkerSize', 8, 'MarkerFaceColor', [0.8500, 0.3250, 0.0980]); hold on

ylabel('Resolution');
ylim([0 1]);
yticks(0:1:1);

% 设置右侧纵坐标
yyaxis right
resistance_k = (0.1 - 0.92) / 0.8;
resistance = resistance_k * coherence_factors + 0.95;
plot(coherence_factors, resistance, '--', 'Color', [0.4660, 0.6740, 0.1880], 'LineWidth', 2, ...
    'Marker', 'o', 'MarkerSize', 8, 'MarkerFaceColor', [0.4660, 0.6740, 0.1880]); hold on

resistance_plus_k = -(resistance(1) - 0.75) / 0.8;
resistance_plus = resistance_plus_k * coherence_factors + 0.9;
plot(coherence_factors, resistance_plus, '-', 'Color', [0.9290, 0.6940, 0.1250], 'LineWidth', 2, ...
    'Marker', 'o', 'MarkerSize', 8, 'MarkerFaceColor', [0.9290, 0.6940, 0.1250]); hold on

ylabel('Resistance');
ylim([0 1]);
yticks(0:1:1);

% 设置横坐标范围和刻度
xlabel('Coherence Factor');
xlim([0 1]);
xticks(0:1:1);

% 设置字体和样式
set(gca, 'FontName', 'Arial', 'FontSize', 16, 'LineWidth', 1.5, 'Box', 'on');

% 设置所有轴的边框颜色为黑色
yyaxis left;
ax = gca;
ax.YColor = 'k'; % 设置左边y轴为黑色

yyaxis right;
ax.YColor = 'k'; % 设置右边y轴为黑色

ax.XColor = 'k'; % 设置x轴为黑色

% 添加图例
legend({'Original Resolution', 'Resolution Up', 'Original Resistance', 'Resistance Up'}, ...
    'NumColumns', 2, 'Location', 'northoutside', 'Orientation', 'horizontal', 'Box', 'off', 'FontSize', 12);

%%
% 参数设置
% 参数设置
grid_size = 10; % 三维网格大小
L0 = 1; % 外尺度
l0 = 0.01; % 内尺度
C_n_squared = 1e-14; % 结构常数
dx = 1; % 空间分辨率

% 生成空间频率坐标
x = linspace(-grid_size/2, grid_size/2 - 1, grid_size) * dx;
[X, Y, Z] = meshgrid(x, x, x);
r = sqrt(X.^2 + Y.^2 + Z.^2);

% 生成随机相位扰动
phi = C_n_squared * exp(-r.^2 / (2 * L0^2)) .* exp(-2 * pi^2 * l0^2 * r.^2);

% 添加随机噪声
noise = randn(grid_size, grid_size, grid_size);
phi = phi .* noise;

% 绘制三维大气湍流模型
figure('Color', 'none'); % 设置背景为透明

% 可视化湍流
h = patch(isosurface(X, Y, Z, phi, 0));
set(h, 'FaceColor', [0.4, 0.6, 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.5); % 优雅的颜色
hold on;

% 获取边框坐标
x_limits = [min(x), max(x)];
y_limits = [min(x), max(x)];
z_limits = [min(x), max(x)];

% 定义边框线
lines = [
    x_limits(1), y_limits(1), z_limits(1), x_limits(2), y_limits(1), z_limits(1);
    x_limits(2), y_limits(1), z_limits(1), x_limits(2), y_limits(2), z_limits(1);
    x_limits(2), y_limits(2), z_limits(1), x_limits(1), y_limits(2), z_limits(1);
    x_limits(1), y_limits(2), z_limits(1), x_limits(1), y_limits(1), z_limits(1);
    x_limits(1), y_limits(1), z_limits(2), x_limits(2), y_limits(1), z_limits(2);
    x_limits(2), y_limits(1), z_limits(2), x_limits(2), y_limits(2), z_limits(2);
    x_limits(2), y_limits(2), z_limits(2), x_limits(1), y_limits(2), z_limits(2);
    x_limits(1), y_limits(2), z_limits(2), x_limits(1), y_limits(1), z_limits(2);
    x_limits(1), y_limits(1), z_limits(1), x_limits(1), y_limits(1), z_limits(2);
    x_limits(2), y_limits(1), z_limits(1), x_limits(2), y_limits(1), z_limits(2);
    x_limits(2), y_limits(2), z_limits(1), x_limits(2), y_limits(2), z_limits(2);
    x_limits(1), y_limits(2), z_limits(1), x_limits(1), y_limits(2), z_limits(2);
];

% 绘制边框
% for i = 1:size(lines, 1)
%     plot3([lines(i, 1), lines(i, 4)], [lines(i, 2), lines(i, 5)], [lines(i, 3), lines(i, 6)], ...
%         'k-', 'LineWidth', 2);
% end

axis equal;
set(gca, 'FontName', 'Arial', 'FontSize', 16, 'LineWidth', 1.5);
view(3);
axis off; % 隐藏坐标轴刻度和标签

% 设置照明和材质
camlight;
lighting gouraud;

% 保存为无背景的图片
exportgraphics(gca, '3D_Atmospheric_Turbulence_Model_With_Edges.png', 'BackgroundColor', 'none');

%%
% Define the range for u
u = linspace(-10, 10, 50); % Adjust the range and number of points as needed

% Define the coefficients a
a_values = [0.5, 1, 2];
radiu = [25, 50, 100];

% Create a figure
figure;

% Define elegant colors and markers
colors = [0.2, 0.4, 0.8;   % Soft Blue
          0.8, 0.4, 0.2;   % Warm Orange
          0.4, 0.8, 0.4];  % Soft Green
markers = {'o', 's', 'd'};


% Loop over each value of a
for i = 1:length(a_values)
    a = a_values(i);
    
    % Calculate the adjusted variable a * u
    adjusted_u = a * u;
    
    % Calculate the first-order Bessel function J1(a * u)
    J1_adjusted_u = besselj(1, adjusted_u);
    
    % Handle the case when a * u = 0 to avoid division by zero
    J1_adjusted_u(adjusted_u == 0) = 1; % Use a small number to avoid division by zero
    
    % Calculate the function |2*J1(a * u)| / (a * u)
    f = abs(2 * J1_adjusted_u./ adjusted_u) ;
    
    % Handle the special case for adjusted_u = 0
    % f(adjusted_u == 0) = 0; % The value at a * u = 0 is 0
    
    % Plot the function with specified color and marker
    plot(u, f, 'LineWidth', 2, 'Color', colors(i, :), 'Marker', markers{i},'MarkerSize', 8);
    hold on;
end

% Add labels, title, and legend
xlabel('u');
ylabel('MCF');
set(gca, 'FontSize', 18);
legend(arrayfun(@(x) sprintf('r_p = %.1fmm', x), radiu, 'UniformOutput', false), 'Box', 'off', 'FontSize', 16);

% Set the x and y axis properties
xticks([]); % Remove x-axis ticks
yticks(0:1); % Set y-axis ticks from 0 to 1 with step size of 0.5

% Set the font size for axes

set(gca, 'LineWidth', 1); % Set border width to 2
% set(gca, 'Box', 'off')
% Set the color for the grid and background
grid off;
hold off;




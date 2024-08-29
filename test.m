%设计不同平面显示不同图像的全息图
%%
clc;
clear
l = 1;
width = 1920;  
height = 1080; 
x = linspace(-1, 1, width);
y = linspace(-1, 1, height);
[X, Y] = meshgrid(x, y);
[theta, r] = cart2pol(X, Y);
phase = angle(exp(1i * l * theta));

% 保存相位图像
filename = 'vortex_phase_l1.png';
imwrite(uint8((phase + pi) * 255 / (2 * pi)), filename);

% 显示相位图像
figure;
imagesc(x, y, phase);
axis xy; axis equal; axis tight;
title('Phase of Vortex Beam (l=1)');
xlabel('x');
ylabel('y');
colormap('hsv');
colorbar;

%%
clc;
clear
% 参数设置
lambda = 632.8e-9; % 光波波长（米）
k = 2 * pi / lambda; % 波数
SLM_size = [1024, 1024]; % SLM尺寸
pixel_size = 8e-6; % SLM像素大小（米）
z1 = 0.05; % 图像a的成像距离（米）
z2 = 0.07; % 图像b的成像距离（米）
num_iterations = 500; % 迭代次数

% 生成SLM网格
[x, y] = meshgrid(linspace(-SLM_size(2)/2, SLM_size(2)/2, SLM_size(2)), ...
                  linspace(-SLM_size(1)/2, SLM_size(1)/2, SLM_size(1)));
x = x * pixel_size;
y = y * pixel_size;

% 读取图像a（数字1）和图像b（数字2）的二值图像
img_a = im2double(imread('C:\Users\84056\Desktop\phase test\1.png')); % 确保图像大小与SLM一致
img_b = im2double(imread('C:\Users\84056\Desktop\phase test\3.png')); % 确保图像大小与SLM一致

% 将图像大小调整为SLM的大小
img_a = imresize(img_a, SLM_size);
img_b = imresize(img_b, SLM_size);

% 初始化相位
initial_phase = rand(SLM_size) * 2 * pi;
SLM_field = exp(1j * initial_phase);

% GS算法迭代
for iter = 1:num_iterations
    % 前向传播到z1
    E1 = prop(SLM_field, z1, lambda, pixel_size);
    % 应用目标约束
    E1 = img_a .* exp(1j * angle(E1));
    % 逆传播回SLM平面
    SLM_field_a = prop(E1, -z1, lambda, pixel_size);
    
    % 前向传播到z2
    E2 = prop(SLM_field, z2, lambda, pixel_size);
    % 应用目标约束
    E2 = img_b .* exp(1j * angle(E2));
    % 逆传播回SLM平面
    SLM_field_b = prop(E2, -z2, lambda, pixel_size);
    
    % 组合两个SLM平面的光场信息
    SLM_field = (SLM_field_a + SLM_field_b) / 2;
    % 保持振幅不变
    SLM_field = exp(1j * angle(SLM_field));
end

% 提取最终相位生成全息图
H = angle(SLM_field);

% 显示全息图
figure;
imagesc(H);
colormap(gray);
colorbar;
title('3D Hologram Phase Map');

% 保存全息图
imwrite(mat2gray(H), '3D_hologram5-7.png');

% 重建
z_reconstruct = [z1, z2];
figure;

for idx = 1:length(z_reconstruct)
    z = z_reconstruct(idx);
    E_reconstruct = prop(exp(1j * H), z, lambda, pixel_size);
    
    % 显示重建图像
    subplot(1, 2, idx);
    imagesc(abs(E_reconstruct));
    colormap(gray);
    colorbar;
    title(['Reconstructed Image at ', num2str(z*100), ' cm']);
end

% 辅助函数：菲涅尔传播
function U_out = prop(U_in, z, lambda, pixel_size)
    [M, N] = size(U_in);
    k = 2 * pi / lambda;
    fx = (-N/2 : N/2-1) / (N * pixel_size);
    fy = (-M/2 : M/2-1) / (M * pixel_size);
    [FX, FY] = meshgrid(fx, fy);
    H = exp(1j * k * z * sqrt(1 - (lambda * FX).^2 - (lambda * FY).^2));
    H = fftshift(H);
    U_out = ifft2(fft2(U_in) .* H);
end

function U_out = ASM(U_in, z, lambda, pixel_size)
    [M, N] = size(U_in);
    k = 2 * pi / lambda;
    fx = (-N/2 : N/2-1) / (N * pixel_size);
    fy = (-M/2 : M/2-1) / (M * pixel_size);
    [FX, FY] = meshgrid(fx, fy);
    H = exp(1j * k * z * sqrt(1 - (lambda * FX).^2 - (lambda * FY).^2));
    H = fftshift(H);
    U_out = ifft2(fft2(U_in) .* H);
end





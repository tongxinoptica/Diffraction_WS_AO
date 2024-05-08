clc; clear
%% Draw loss line
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
legend show; 

%%
clc; clear
ref = imread("test_img\exp\8.5\0046.bmp");
ref = im2double(rgb2gray(ref));
ref = imrotate(ref, 0.18);
ref = ref(150:1255,96:1198);
% ref = (ref - mean(ref(:))) / std(ref(:));
% ref = (ref - min(min(ref))) / (max(max(ref)) - min(min(ref)));
figure(1)
imshow(ref)
imwrite(ref,"test_img\exp\ref.bmp", "bmp")
abe = imread("test_img\exp\z3\0041.bmp");
abe = im2double(rgb2gray(abe));
abe = imrotate(abe, 0.2);
abe = abe(150:1255,96:1198);
figure(2)
imshow(abe)
imwrite(abe,"test_img\exp\abe3.bmp", "bmp")
src = imread("test_img\exp\0.00\0033.bmp");
src = im2double(rgb2gray(src));
src = imrotate(src, 0.25);
src = src(100:1210,80:1190);
figure(3)
imshow(src)
imwrite(src,"test_img\exp\src.bmp", "bmp")

%% Deconvolve
clc;clear
grid10 = imread('test_img\gray_grid10.bmp');
grid10 = im2double(grid10);
grid10 = grid10 / max(max(grid10));
imshow(grid10)
N = 1000;
dx = 8e-6;
L = N*dx;
z = 0.1;
lamda = 532e-9;
k = 2*pi/lamda;
[x, y] = meshgrid(linspace(-L/2, L/2, N), linspace(-L/2, L/2, N));
[theta, r] = cart2pol(x, y);
l = 1;
vortex_phase = exp(1i * l * theta);
ss = exp(0.5*1i*k*(x.^2+y.^2)/z);
A = grid10;
B = A.*vortex_phase;
fA = fftshift(fft2(A));
fB = fftshift(fft2(B));
C = ifft2(ifftshift(conv2(fA, fB, 'same')));
res = C ./A./A;
imagesc(angle(res));
%%





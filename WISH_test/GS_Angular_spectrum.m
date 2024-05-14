%%
clc; clear
phase = zeros(1080,1920,8);
for i = 1:8
randomMatrix = rand(108, 192);
phase_p = imresize(randomMatrix, [1080 1920], 'bicubic');
phase(:,:,i) = phase_p;
end

format long;
tic;
dic = 'gray_grid10.bmp';
img=imread(dic);
img = im2double(img);
% Ii = rgb2gray(Ii);
img = imresize(img, [1080,1920]);
pitch = 0.008; % Pixel lenth, unit: mm
[M,N] = size(img);
W = M*pitch; % Lenth
L = N*pitch; % Width
% W = 8.64;  
% L = 15.36;     
ni=1.4935;
lambda=532e-6;          
h=zeros(N+1);
k=2*pi/lambda;
z1 = 30; % 3cm 
dh=100; % Propagate distance
pitch = 0.008;
x1=-L/2:pitch:L/2-pitch;
y1=-W/2:pitch:W/2-pitch;
[x1,y1]=meshgrid(x1,y1);
r0=L/2;
%物场振幅分布,平行光波入射，初始相位取为0，振幅1
%M=lambda*h2/dx1;
%dx2=lambda*h2/L;
dfx=1/L;
dfy=1/W;
fx=-1/2/pitch:dfx:1/2/pitch-dfx;
fy=-1/2/pitch:dfy:1/2/pitch-dfy;
[fx,fy]=meshgrid(fx,fy);
u1 = img;
A1=fftshift(fft2(u1))*pitch.^2;
A2=A1.*exp(1j*k*z1.*sqrt(1-(lambda*fx).^2-(lambda*fy).^2));
u2=ifft2(ifftshift(A2))*dfx*dfy;  % SLM complex light field
% absu2 = abs(u2)./max(max(abs(u2)));
% subplot(1,2,1)
% imshow(absu2)
% holo = angle(u2);
% holo = holo+pi;
% holo=mod(holo,2*pi)/(2*pi);
% subplot(1,2,2)
% imshow(holo)
%% Generate sensor intensity. Simulate captured images in exp
init_u = 0;
sensor = zeros(1080,1920,8);
for i = 1:8
slm_com = u2.*exp(1j*phase(:,:,i));
A1=fftshift(fft2(slm_com))*pitch.^2;
A2=A1.*exp(1j*k*z1.*sqrt(1-(lambda*fx).^2-(lambda*fy).^2));
u2=ifft2(ifftshift(A2))*dfx*dfy;
absu2 = abs(u2)./max(max(abs(u2)));
sensor(:,:,i) = absu2;
u1=ifft2(ifftshift(absu2))*dfx*dfy; % Input plane
init_u = init_u + u1;
end
init_u = init_u / 8;
%%
t_max=200;
e1=zeros(1,t_max);
for t=1:t_max
    mid = 0;
    for i=1:8
        init_u = init_u.*exp(1j*phase(:,:,i)); % slm plane
        A1=fftshift(fft2(init_u))*pitch.^2;
        A2=A1.*exp(1j*k*z1.*sqrt(1-(lambda*fx).^2-(lambda*fy).^2));
        u2=ifft2(ifftshift(A2))*dfx*dfy;  % sensor complex light field
        % absu2 = abs(u2)./max(max(abs(u2))); % Output intensity
        % c2=(sensor(:,:,i)-absu2)*rand(1,1)+sensor(:,:,i);% Random
        u2b=(1./1).*sensor(:,:,i).*exp(1j*angle(u2));
        A2b=fftshift(fft2(u2b))*pitch^2;
        A1b=A2b./exp(1j*k*z1*sqrt(1-(lambda*fx).^2-(lambda*fy).^2)); % Back prop
        u1b=ifft2(ifftshift(A1b))*dfx*dfy; % slm complex light field
        mid = mid + u1b;
    end
    init_u = mid / 8;
    t
end
toc;
rec_in = abs(init_u)./max(max(abs(init_u)));
rec_pha = angle(init_u) + pi;
rec_pha=mod(rec_pha,2*pi)/(2*pi);
figure;
imshow(rec_pha)
title(rec_pha, 'hologram')
figure;
imshow(rec_in)
% imwrite(holo, 'C:\Users\84056\Desktop\phase test\3_0.105_0.075.png')

%% Reconstruction
clc; clear
% phase = im2double(imread('C:\Users\84056\Desktop\phase test\3_0.105_0.075.png'));
phase = im2double(imread('C:\Users\84056\Desktop\neural-3d-holography-main\results\green\slm_0.18_0.16.png'));
% holo = rgb2gray(holo);
pitch = 8e-6; % Pixel lenth, unit: mm
[M,N] = size(phase);
W = M*pitch; % Lenth
L = N*pitch; % Width
lambda=532e-9;          
k=2*pi/lambda;
x1=-L/2:pitch:L/2-pitch;
y1=-W/2:pitch:W/2-pitch;
[x1,y1]=meshgrid(x1,y1);
dfx=1/L;
dfy=1/W;
fx1 = linspace(-1/2/pitch, 1/2/pitch-dfx, M);
fy1 = linspace(-1/2/pitch, 1/2/pitch-dfy, N);
fx=-1/2/pitch:dfx:1/2/pitch-dfx;
fy=-1/2/pitch:dfy:1/2/pitch-dfy;
[fx,fy]=meshgrid(fx,fy);
z = 0.1;
% for z=100:105;
figure;
u1 = exp(1i*2*pi*phase);
A1=fftshift(fft2(u1));
t = k*z*sqrt(1-(lambda*fx).^2-(lambda*fy).^2);
H = exp(1j*t);
% C = readcell('H.csv', 'Delimiter',',');
% complexMatrix = cellfun(@str2num, C);
A2=A1.*H;
u2=ifft2(ifftshift(A2));  % Output complex light field
absu2 = abs(u2)./max(max(abs(u2))); % Output intensity
imshow(absu2);
% holo = angle(u2);
% holo = holo+pi;
% holo=mod(holo,2*pi)/(2*pi);
% imshow(holo)
% end
% u2_real = u2_real./max(max(u2_real));
% PSNR = psnr(u2_real,absu2);

%% Turbulence
figure;
z1 = 60;
z2 = 90;
res = zeros(900,900);
for i=1:50
tur_dic = strcat('F:\Data\turbulence\air_s\a=3.67_cn=5^-15\',num2str(i),'.jpg');
tur = imread(tur_dic);
tur = im2double(tur);
tur = imcrop(tur,[0 0 900 900]);
phase_tur = 2*pi*tur;
u1 = exp(1i*2*pi*holo);
A1=fftshift(fft2(u1))*pitch.^2.*exp(1j*k*z1*sqrt(1-(lambda*fx).^2-(lambda*fy).^2));
u2 = ifft2(ifftshift(A1))*dfx*dfy; 
absu2 = abs(u2);
phase_u2 = angle(u2);
new_phase = mod((phase_u2 + phase_tur), 2*pi);
u3 = absu2.*exp(1i*new_phase);
A3=fftshift(fft2(u3))*pitch.^2.*exp(1j*k*z2*sqrt(1-(lambda*fx).^2-(lambda*fy).^2));
u4 = ifft2(ifftshift(A3))*dfx*dfy; 
absu4 = abs(u4)./max(max(abs(u4)));
res = res + absu4;
end
res = res/50;
res = res./max(max(res));
imshow(res)


%% 
clear;
clc;     
close all;
format long;
tic;
img_dic = 'C:\Users\84056\Desktop\neural-3d-holography-main\data\test\rgb\1.jpg';
img = imread(img_dic);
img = im2double(img);
img = rgb2gray(img);
img = imresize(img, [1080,1920]);
depth_dic = 'C:\Users\84056\Desktop\neural-3d-holography-main\data\test\depth\1.png';
depth = imread(depth_dic);
depth = im2double(depth);
depth = imresize(depth, [1080,1920])*255;
dh=150; % Propagate distance
depth1 = depth;depth2 = depth;depth3 = depth;depth4 = depth;depth5 = depth;
depth1(depth1<=50) = 150; 
depth1(depth1~=150) = 0; mask1 = depth1; mask1(mask1==150) = 1;
real1 = img.*mask1;
depth2(depth2>50&depth<=100) = 151; 
depth2(depth2~=151) = 0; mask2 = depth2; mask2(mask2==151) = 1;
real2 = img.*mask2;
depth3(depth3>100&depth<=150) = 152; 
depth3(depth3~=152) = 0; mask3 = depth3; mask3(mask3==152) = 1;
real3 = img.*mask3;
depth4(depth4>150&depth<=200) = 153; 
depth4(depth4~=153) = 0; mask4 = depth4; mask4(mask4==153) = 1;
real4 = img.*mask4;
depth5(depth5>200&depth<=255) = 154; 
depth5(depth5~=154) = 0; mask5 = depth5; mask5(mask5==154) = 1; % Front
real5 = img.*mask5;

pitch = 0.008; % Pixel lenth, unit: mm
[M,N] = size(img);
W = M*pitch; % Lenth
L = N*pitch; % Width
% W = 8.64;  
% L = 15.36;     
lambda=532e-6;          
k=2*pi/lambda;
x1=-L/2:pitch:L/2-pitch;
y1=-W/2:pitch:W/2-pitch;
[x1,y1]=meshgrid(x1,y1);
r0=L/2;
%M=lambda*h2/dx1;
%dx2=lambda*h2/L;
dfx=1/L;
dfy=1/W;
fx=-1/2/pitch:dfx:1/2/pitch-dfx;
fy=-1/2/pitch:dfy:1/2/pitch-dfy;
[fx,fy]=meshgrid(fx,fy);

% Et=ones(M,N);
% u1_gaus=sqrt(Et);
% u2_real=img;


%% GS_ASM
ins = ones(M,N);
u1 = ins; % Input plane
u2_real=real1;% Real field
z = 150;
t_max=100;

e1=zeros(1,t_max);
for t=1:t_max

    A1=fftshift(fft2(u1));
    A2=A1.*exp(1j*k*z.*sqrt(1-(lambda*fx).^2-(lambda*fy).^2));
    u2=ifft2(ifftshift(A2));  % Output complex light field
    absu2 = abs(u2)./max(max(abs(u2))); % Output intensity
    c2=(u2_real-absu2)*rand(1,1)+u2_real;% Random
    u2b=(1./1).*c2.*exp(1j*angle(u2));

    A2b=fftshift(fft2(u2b));
    A1b=A2b./exp(1j*k*z*sqrt(1-(lambda*fx).^2-(lambda*fy).^2));
    u1b=ifft2(ifftshift(A1b)); % Input complex light field

    u1m=abs(u1b);
    u1=abs(ins).*exp(1j*angle(u1b));
    e1(t)=sqrt(sum(sum((u2.*conj(u2)-img).^2))/sum(sum(img.^2)));%error 
    SSIM = corrcoef(u2_real,absu2);
    if SSIM(1,2)>=0.999
        break
    end
     
end
toc;

figure;
holo = angle(u1);
holo = holo+pi;
holo=mod(holo,2*pi)/(2*pi);
imshow(holo)
title(holo, 'hologram')

%% Reconstruction
figure;
z = 150;
u1 = exp(1i*2*pi*phase);
A1=fftshift(fft2(u1));
A2=A1.*exp(1j*k*z*sqrt(1-(lambda*fx).^2-(lambda*fy).^2));
u2=ifft2(ifftshift(A2));  % Output complex light field
absu2 = abs(u2)./max(max(abs(u2))); % Output intensity
imshow(absu2);
u2_real = u2_real./max(max(u2_real));
PSNR = psnr(u2_real,absu2);

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


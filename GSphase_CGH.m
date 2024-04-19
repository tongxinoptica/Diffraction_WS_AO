clc
clear
tic
% pics = dir('D:\HR_data\process_spe\DIV_process\*.jpg');
% pic_num = length(pics);
% error = (0);
% for i=1:pic_num
%抽样
% image0 = imread('D:\HR_data\DIV_dpi72\gray0810.png');
image0 = imread("C:\Users\84056\Desktop\neural-3d-holography-main\data\test\rgb\1.jpg");
% image0 = imresize(image0,[1024,1024]);
%转为灰度图
image0 = rgb2gray(image0);

image0 = im2double(image0);
image0 = imresize(image0, [1080,1920]);
[M,N] = size(image0);
% lambda = 532e-9;    %波长632nm
% k = 2*pi/lambda;    %波数
% x = linspace(-0.00768,0.00768,N);
% y = linspace(-0.00432,0.00432,M);
% [X,Y] = meshgrid(x,y);
%球面波
% Z = -0.2;
% s = k*Z*(0.5*X.^2/Z^2+0.5*Y.^2/Z^2);
% s = s-min(min(s));
% s = s/max(max(s));
% imshow(s)
%输出原图
% figure;
% subplot(2,2,1);
% imshow(image0);
% title('原图')

%加入随机位相信息
image1 = image0;
phase = 2i*pi*rand(M,N);
image1 = image1.*exp(phase);

%输出随机位相图
% subplot(2,2,2);
% imshow(image1);
% title('随机位相图');


%第一次逆傅里叶变换
image2 = ifft2(ifftshift(image1));

%迭代
result = [];
result2 = [];

for t=1:1:500
    %迭代判据
    imgangle = angle(image2);         %取位相
    image = exp(1i*imgangle);
    image = fftshift(fft2(image));    %还原
    imgabs = abs(image)/max(max(abs(image)));
    sim = corrcoef(image0,imgabs);    %取相关系数
    result= [result;sim(2,1)];
    MES=sum(sum((imgabs-image0).^2))/(M*N);     
    PSNR=20*log10(1/sqrt(MES));
    result2 = [result2;PSNR];
    if sim(1,2) >= 0.999
        %imgangle = angle(image2);        
        %还原
%         image4 = exp(1i*imgangle);
%         image4 = fftshift(fft2(image4));                          
%         imgabs = abs(image4)/max(max(abs(image4)));
        break
    else
        imgangle = angle(image2);
        image2 = exp(1i*imgangle);

        %正傅里叶变换
        image3 = fftshift(fft2(image2));

        %赋反馈振幅
        imgabs = abs(image3)/max(max(abs(image3)));
        imgangle = angle(image3);
        image3 = exp(1i*imgangle);
        %image3 = image3.*(image0+rand(1,1)*(sqrt(mean2((image0-imgabs).^2))));
        image3 = image3.*(image0+rand(1,1)*(image0-imgabs));

        %逆傅里叶变换
        image2 = ifft2(ifftshift(image3));
        % 存储ssim psnr结果

        
        
    end
end



%取位相
imgangle = angle(image2);  

%还原
image4 = exp(1i*imgangle);
image4 = fftshift(fft2(image4));                          
imgabs = abs(image4)/max(max(abs(image4)));
% subplot(2,2,4);
imshow(imgabs);
figure;
%输出位相全息图
%imgangle=imgangle+pi+s*2*pi;
imgangle = imgangle+pi;
imgangle=mod(imgangle,2*pi)/(2*pi);
imshow(imgangle)
% title('还原图')
imwrite(imgangle,'test.png')
% 
% str1=sprintf('进度:%d/%d, PSNR=%d, SSIM=%d',i,pic_num,PSNR,sim(1,2));
disp(['运行时间',num2str(toc)])
% disp(str1)
% end
% error_txt = fopen('D:\HR_data\process_spe\DIV_error.txt', 'w');
% fprintf(error_txt, '%d\n', error);
% fclose(error_txt);
% %输出位相全息图
% subplot(2,2,3);
% imshow(imgangle);
% title('位相全息图')




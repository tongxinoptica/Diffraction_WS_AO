clc
clear
tic
% pics = dir('D:\HR_data\process_spe\DIV_process\*.jpg');
% pic_num = length(pics);
% error = (0);
% for i=1:pic_num
%����
% image0 = imread('D:\HR_data\DIV_dpi72\gray0810.png');
image0 = imread("C:\Users\84056\Desktop\neural-3d-holography-main\data\test\rgb\1.jpg");
% image0 = imresize(image0,[1024,1024]);
%תΪ�Ҷ�ͼ
image0 = rgb2gray(image0);

image0 = im2double(image0);
image0 = imresize(image0, [1080,1920]);
[M,N] = size(image0);
% lambda = 532e-9;    %����632nm
% k = 2*pi/lambda;    %����
% x = linspace(-0.00768,0.00768,N);
% y = linspace(-0.00432,0.00432,M);
% [X,Y] = meshgrid(x,y);
%���沨
% Z = -0.2;
% s = k*Z*(0.5*X.^2/Z^2+0.5*Y.^2/Z^2);
% s = s-min(min(s));
% s = s/max(max(s));
% imshow(s)
%���ԭͼ
% figure;
% subplot(2,2,1);
% imshow(image0);
% title('ԭͼ')

%�������λ����Ϣ
image1 = image0;
phase = 2i*pi*rand(M,N);
image1 = image1.*exp(phase);

%������λ��ͼ
% subplot(2,2,2);
% imshow(image1);
% title('���λ��ͼ');


%��һ���渵��Ҷ�任
image2 = ifft2(ifftshift(image1));

%����
result = [];
result2 = [];

for t=1:1:500
    %�����о�
    imgangle = angle(image2);         %ȡλ��
    image = exp(1i*imgangle);
    image = fftshift(fft2(image));    %��ԭ
    imgabs = abs(image)/max(max(abs(image)));
    sim = corrcoef(image0,imgabs);    %ȡ���ϵ��
    result= [result;sim(2,1)];
    MES=sum(sum((imgabs-image0).^2))/(M*N);     
    PSNR=20*log10(1/sqrt(MES));
    result2 = [result2;PSNR];
    if sim(1,2) >= 0.999
        %imgangle = angle(image2);        
        %��ԭ
%         image4 = exp(1i*imgangle);
%         image4 = fftshift(fft2(image4));                          
%         imgabs = abs(image4)/max(max(abs(image4)));
        break
    else
        imgangle = angle(image2);
        image2 = exp(1i*imgangle);

        %������Ҷ�任
        image3 = fftshift(fft2(image2));

        %���������
        imgabs = abs(image3)/max(max(abs(image3)));
        imgangle = angle(image3);
        image3 = exp(1i*imgangle);
        %image3 = image3.*(image0+rand(1,1)*(sqrt(mean2((image0-imgabs).^2))));
        image3 = image3.*(image0+rand(1,1)*(image0-imgabs));

        %�渵��Ҷ�任
        image2 = ifft2(ifftshift(image3));
        % �洢ssim psnr���

        
        
    end
end



%ȡλ��
imgangle = angle(image2);  

%��ԭ
image4 = exp(1i*imgangle);
image4 = fftshift(fft2(image4));                          
imgabs = abs(image4)/max(max(abs(image4)));
% subplot(2,2,4);
imshow(imgabs);
figure;
%���λ��ȫϢͼ
%imgangle=imgangle+pi+s*2*pi;
imgangle = imgangle+pi;
imgangle=mod(imgangle,2*pi)/(2*pi);
imshow(imgangle)
% title('��ԭͼ')
imwrite(imgangle,'test.png')
% 
% str1=sprintf('����:%d/%d, PSNR=%d, SSIM=%d',i,pic_num,PSNR,sim(1,2));
disp(['����ʱ��',num2str(toc)])
% disp(str1)
% end
% error_txt = fopen('D:\HR_data\process_spe\DIV_error.txt', 'w');
% fprintf(error_txt, '%d\n', error);
% fclose(error_txt);
% %���λ��ȫϢͼ
% subplot(2,2,3);
% imshow(imgangle);
% title('λ��ȫϢͼ')




%%
clc; clear
folder_path = 'asm/ori'; 
image_files = dir(fullfile(folder_path, '*.tif'));
for i = 1:length(image_files)
    % 读取图像
    img = imread(fullfile(folder_path, image_files(i).name));
    img = rgb2gray(img);
    img = im2double(img);
    img = imrotate(img, -1.5, 'bilinear', 'crop'); % 顺时针旋转用负角度
    img = padarray(img, [10,0],0,'pre');
    pre_img = img(1:1800, 465:2264);
    % 显示原始图像和旋转后的图像（可选）
%     figure;
%     subplot(1, 2, 1);
%     imshow(img);
%     title(['Original Image ', num2str(i)]);
%     
%     subplot(1, 2, 2);
%     imshow(rotated_img);
%     title(['Rotated Image ', num2str(i)]);
    
    % 保存旋转后的图像，生成新文件名
    [~, name, ext] = fileparts(image_files(i).name);
    new_filename = fullfile('asm/process', [name, ext]);
    imwrite(pre_img, new_filename);
end
%%
clc;clear
img = imread('asm/process/cabli500_rotated.tif');
img = im2double(img);
img = padarray(img, [10,0],0,'pre');
a = img(1:1800, 465:2264);
imshow(a)
    
    
    
    
    
    
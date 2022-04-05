close all
clear
clc
% reading images
img = imread('pool.jpg');
flag = imread('flag.png');
[m1,n1] = size(img(:,:,1));
[m2,n2] = size(flag(:,:,1));
% 4 points of pool image
p1 = [544 525;749 485;854 506;653 553];
% corners of flag
p2 = [1 1;n2 1;n2 m2;1 m2];
% finding transformation
tform = fitgeotrans(p2,p1,'projective');
% transforming image
img_w = imwarp(flag,tform);
[mt,nt] = size(img_w(:,:,1));
[a,b] = find(img_w(:,:,1)>0);
flag_point = [b(1),a(1)];
pool_point = round(p1(1,:));
% calculating shift for showing in
% correct position
shift = pool_point - flag_point;
x = shift(2);
y = shift(1);
temp = zeros(m1,n1,3);
temp = uint8(temp);
% moving transformed flag to
% correct position
temp(3+x:2+mt+x,3+y:2+nt+y,:) = img_w;
% creating a mask for replacing
% transformed flag to original image
mask = rgb2gray(temp)>0;
temp2 = img.*uint8(~mask);
% showing the results
imshow(flag)
title('Flag')
figure
imshow(img)
title('Location of flag')
hold on
plot(p1(:,1),p1(:,2),'o','markersize',7,'markerfacecolor','black')
figure
imshow(temp+temp2)
title('Result')
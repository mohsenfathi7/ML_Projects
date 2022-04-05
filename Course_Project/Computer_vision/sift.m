% close all
clear
clc
close all
% number of valid matches
n = 20;
% reading image
img0 = imread('monalisa_piece1.jpg');
img2 = imread('monalisa.jpg');
% perfroming SIFT
[des1, loc1] = sift(img0);
[des2, loc2] = sift(img2);
% matching the SIFT descriptor accross two images
[ind1, ind2] = SIFTmatch(des1, des2, n);
% drawing the matched features
match1 = loc1(ind1, [2 1]);
match2 = loc2(ind2, [2 1]);
% estimating geometric tranformation 
tform = estimateGeometricTransform(match1,match2,'projective');
% transforming image
img_w = imwarp(img0,tform);
mask = rgb2gray(img_w)>0;
% removing black area around picture
[x,y] = find(mask==1);
img_n = img_w(min(x):max(x),min(y):max(y),:);
[a,b,~] = size(img_n);
[des3, loc3] = sift(img_n);
[ind11, ind22] = SIFTmatch(des3, des2, n);
match11 = loc3(ind11, [2 1]);
match22 = loc2(ind22, [2 1]);
tform2 = estimateGeometricTransform(match11,match22,'projective');
% corners of image
bx = [1,b,b,1,1];
by = [1,1,a,a,1];
% corners of reactangle shown in the image
[tx,ty] = transformPointsForward(tform2,bx,by);
% showing the results
imshowpair(img2,img0,'montage')
hold on
plot(tx,ty,'linewidth',1.5,'color','yellow')
title('Template location')
hold off
figure
showMatchedFeatures(img2,img0,match2,match1,'montage');
title('Matched points')
%% piece 1 with 15 degrees rotation and 20 offset
close all
% number of valid matches
n = 20;
% reading image
img0 = imread('monalisa_piece1.jpg');
img0 = img0 + 20;
img1 = imrotate(img0,15);
img2 = imread('monalisa.jpg');
% perfroming SIFT
[des1, loc1] = sift(img1);
[des2, loc2] = sift(img2);
% matching the SIFT descriptor accross two images
[ind1, ind2] = SIFTmatch(des1, des2, n);
% drawing the matched features
match1 = loc1(ind1, [2 1]);
match2 = loc2(ind2, [2 1]);
% estimating geometric tranformation 
tform = estimateGeometricTransform(match1,match2,'projective');
% transforming image
img_w = imwarp(img1,tform);
mask = rgb2gray(img_w)>0;
% removing black area around picture
[x,y] = find(mask==1);
img_n = img_w(min(x):max(x),min(y):max(y),:);
[a,b,~] = size(img_n);
[des3, loc3] = sift(img_n);
[ind11, ind22] = SIFTmatch(des3, des2, n);
match11 = loc3(ind11, [2 1]);
match22 = loc2(ind22, [2 1]);
tform2 = estimateGeometricTransform(match11,match22,'projective');
% corners of image
bx = [1,b,b,1,1];
by = [1,1,a,a,1];
% corners of reactangle shown in the image
[tx,ty] = transformPointsForward(tform2,bx,by);
% showing the results
imshowpair(img2,img1,'montage')
hold on
plot(tx,ty,'linewidth',1.5,'color','yellow')
title('Template location (15 degrees rotation)')
hold off
figure
showMatchedFeatures(img2,img1,match2,match1,'montage');
title('Matched points (15 degrees rotation)')
%% piece 1 with 20 degrees rotation and 40 offset
close all
% number of valid matches
n = 20;
% reading image
img0 = imread('monalisa_piece1.jpg');
img0 = img0 + 40;
img1 = imrotate(img0,20);
img2 = imread('monalisa.jpg');
% perfroming SIFT
[des1, loc1] = sift(img1);
[des2, loc2] = sift(img2);
% matching the SIFT descriptor accross two images
[ind1, ind2] = SIFTmatch(des1, des2, n);
% drawing the matched features
match1 = loc1(ind1, [2 1]);
match2 = loc2(ind2, [2 1]);
% estimating geometric tranformation 
tform = estimateGeometricTransform(match1,match2,'projective');
% transforming image
img_w = imwarp(img1,tform);
mask = rgb2gray(img_w)>0;
% removing black area around picture
[x,y] = find(mask==1);
img_n = img_w(min(x):max(x),min(y):max(y),:);
[a,b,~] = size(img_n);
[des3, loc3] = sift(img_n);
[ind11, ind22] = SIFTmatch(des3, des2, n);
match11 = loc3(ind11, [2 1]);
match22 = loc2(ind22, [2 1]);
tform2 = estimateGeometricTransform(match11,match22,'projective');
% corners of image
bx = [1,b,b,1,1];
by = [1,1,a,a,1];
% corners of reactangle shown in the image
[tx,ty] = transformPointsForward(tform2,bx,by);
% showing the results
imshowpair(img2,img1,'montage')
hold on
plot(tx,ty,'linewidth',1.5,'color','yellow')
title('Template location (20 degrees rotation)')
hold off
figure
showMatchedFeatures(img2,img1,match2,match1,'montage');
title('Matched points (20 degrees rotation)')

close all
clc
clear
% reading image
cd 'C:\Users\User\Desktop\HW3'
img0 = imread('im_0.png');
gray0 = rgb2gray(img0);
img1 = imread('im_1.png');
gray1 = rgb2gray(img1);
%% part a
clc
% harris key point
key0 = detectHarrisFeatures(gray0);
% selecting 1000 stongest key point
key0 = key0.selectStrongest(1000);
% harris key point
key1 = detectHarrisFeatures(gray1);
% selecting 1000 stongest key point
key1 = key1.selectStrongest(1000);
% extracting features
[f0, vkey0] = extractFeatures(gray0, key0);
[f1, vkey1] = extractFeatures(gray1, key1);
% matching key points
[ind,dis] = matchFeatures(f0,f1);
mp0 = vkey0(ind(:,1));
mp1 = vkey1(ind(:,2));
% finding matched key points with almost same y
err = abs(mp0.Location(:,2)-mp1.Location(:,2));
loc = err<.7;
mp0 = mp0(loc);
mp1 = mp1(loc);
% showing results
figure; ax = axes;
showMatchedFeatures(img0,img1,mp0,mp1,'montage','Parent',ax);
title(ax, 'Final point matches');
legend(ax, 'Matched points 0','Matched points 1');
%% b
clc
close all
% x and y of matched points
loc0 = mp0.Location;
x_0 = loc0(:,1);
y_0 = loc0(:,2);
loc1 = mp1.Location;
x_1 = loc1(:,1);
y_1 = loc1(:,2);
L = numel(x_0);
A = [];
% creating A matrix
for i = 1:L
    x0 = x_0(i);
    x1 = x_1(i);
    y0 = y_0(i);
    y1 = y_1(i);
    v2 = [x0*x1,x0*y1,x0,y0*x1,y0*y1,y0,x1,y1,1];
    A = cat(1,A,v2);
end
% solving problem with 8-points algorithm
for k= 1:1000
    inlier = 0;
    % selecting 8 random points
    p = randperm(L,8);
    A2 = A(p,:);
    % solving problem with 8-points algorithm
    [~,~,V] = svd(A2);
    f = V(:,end);
    F = reshape(f,[3,3])';
    [U,S,V] = svd(F);
    S(3,3)= 0;
    F = U*S*V';
    F = F';
    % finding distance
    for n=1:L
       d = [mp1(n,:).Location,1]*F*[mp0(n,:).Location';1];
       d = d^2;
       if d<.005
           inlier = inlier+1;
       end
    end
    % checking inliers ratio
    if (inlier/L) > .99
        break
    end
end
% showing epipolar lines
imshow(img1); 
title('Epipolar without Function'); hold on;
plot(mp1.Location(p,1),mp1.Location(p,2),'ro')
epiLines = epipolarLine(F,mp0.Location(p,:));
points = lineToBorderPoints(epiLines,size(img0));
line(points(:,[1,3])',points(:,[2,4])');
%% c
clc
% loading the best estimated F
load('best.mat');
tic
% camera matrix of image 1
p0 = diag(ones(3,1));
p0 = cat(2,p0,zeros(3,1));
% calculating null space of A
[~,~,V] = svd(F');
e = V(:,end);
% creating skew-symmetrix matrix
skew_e = [0,-e(3),e(2);e(3),0,-e(1);-e(2),e(1),0];
% calculating camera matrixof image 2
p1 = skew_e * F;
p1 = cat(2,p1,e);
loc0 = mp0.Location;
x_0 = loc0(:,1);
y_0 = loc0(:,2);
loc1 = mp1.Location;
x_1 = loc1(:,1);
y_1 = loc1(:,2);
% changing k in from 1 to 23 to see the results
k=23;
% creating A matrix
A = [y_0(k)*p0(3,:)-p0(2,:)
     p0(1,:)-x_0(k)*p0(3,:)
     y_1(k)*p1(3,:)-p1(2,:)
     p1(1,:)-x_1(k)*p1(3,:)];
% calculating null space of A
[~,~,V] = svd(A);
t = V(:,end)';
% setting the 4th element to 1 in all points
t = abs(t/t(4));
x_y_z = t(1:3);
% showing results
imshow(img0)
hold on
plot(x_0(k),y_0(k),'ro','markersize',4,'linewidth',2)
string = strcat('\leftarrow (x,y,z) = (',num2str(x_y_z),')');
x = round(double(x_0(k)));
y = round(double(y_0(k)));
text(x+20,y,string,'fontweigh','bold','color','black')
legend('Matched points')
title('World location')
% finding position with in-built MATLAB function to check the accuracy of
% results
triangulate(loc0(k,:),loc1(k,:),p0',p1')
toc





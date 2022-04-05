close all
clc
clear
tic
cd 'C:\Users\User\Desktop\HW3'
% loadign positions of corners
load('checkerboard.mat')
% reading checjerboard image
img = imread('cameracalib.jpg');
gray = rgb2gray(img);
% finding edge
ed = edge(gray,'canny',[.2,.4]);
% showing results
imshow(img)
hold on
plot(points(:,1),points(:,2),'go','markersize',4,'linewidth',2)
legend('Chosen points')
figure
imshow(ed)
hold on
plot(points(:,1),points(:,2),'go','markersize',4,'linewidth',2)
legend('Chosen points')
x = points(:,1);
y = points(:,2);
% world positions
world = 28*[6,6,0;0,3,7;3,1,0;
         4,5,0;0,2,3;0,4,5];
A =[];
% creating A matrix
for i=1:6
    x = points(i,1);
    y = points(i,2);
    X = world(i,1);
    Y = world(i,2);
    Z = world(i,3);
    r1 = [X,Y,Z,1,0,0,0,0,-x*X,-x*Y,-x*Z,-x];
    r2 = [0,0,0,0,X,Y,Z,1,-y*X,-y*Y,-y*Z,-y];
    A = cat(1,A,r1);
    A = cat(1,A,r2); 
end
% calculating null space of A
[~,~,V] = svd(A);
P = V(:,end);
P = reshape(P,[4,3])';
% a sample world position
world_x = [0;28*6;28*6;1];
% estimated position in image
es_x = P*world_x;
es_x = es_x/es_x(3);
% real position in image
real_x = [1103,1900];
figure
% showing results
imshow(img)
hold on
plot(real_x(1),real_x(2),'go','markersize',4,'linewidth',2)
plot(es_x(1),es_x(2),'ro','markersize',4,'linewidth',2)
legend('Real point','Estimated point')
toc






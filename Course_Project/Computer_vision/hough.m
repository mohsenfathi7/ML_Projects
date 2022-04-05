%% hough
clc
clear
close all
% reading image
img = imread("pathway.jpg");
gray = rgb2gray(img);
[m,n] = size(gray);
% canny edge detecion
ed = edge(gray,'canny',[0.25,0.4]);
[x,y] = find(ed==1);
k=180;
% making theta vector
theta = linspace(-pi/2,pi/2,k);
% maximum possible rho
rmax = sqrt(m^2 + n^2);
rmax = round(rmax);
hough = zeros(2*rmax,k);
for i=1:k
    % hough transform
    r = x*cos(theta(i)) + y*sin(theta(i));
    r = round(r);
    for j=1:length(r)
        % filling hough space
        R = r(j)+rmax;
        hough(R,i) = hough(R,i)+1;
    end
end
% finding maximum theta and rho
maximum = max(hough(:));
% thresholding
th = 0.28 * maximum;
[x2,y2] = find(hough > th);
result = zeros(m,n);
% keeping points satisfying threshold
for i=1:length(x2)
    temp = x*cos(theta(y2(i))) + y*sin(theta(y2(i)));
    temp = round(temp);
    temp = temp - (x2(i)-rmax);
    % considering tolerance for points
    [x3,~] = find(abs(temp)<2);
    for j=1:length(x3)
        result(x(x3(j)),y(x3(j))) =1;
    end   
end
% showing the results
hough_show = imresize(hough,[1000,1000]);
imshow(hough_show,[])
title('Hough space')
xlabel('\theta','fontsize',15)
ylabel('\rho','fontsize',15)
axis on
figure
imshow(ed)
title('Canny edges')
figure
imshow(result)
title('Hough results')
%% RANSAC
clc
clear
% reading image
img = imread("horizons.jpg");
gray = rgb2gray(img);
% canny edge detecion
ed = edge(gray,'canny',[0.15,0.3]);
[x,y] = find(ed==1);
% counts of points
L= numel(x);
p1 = (1:L)';
p1 = repmat(p1,[1,L]);
p2 = p1';
% creating a matrix for checking points to
% avoid multiple choice of a pair point
pcheck = ones(1,L);
pcheck = diag(pcheck);
for i=1:1000
    % creating 2 random number
    p = randperm(L,2);
    while pcheck(p(1),p(2)) == 1
    p = randperm(L,2);
    end
    % filling checking matrix  to avoid
    % multiple choice of a pair point
    pcheck(p(1),p(2)) = pcheck(p(1),p(2))+1;
    pcheck(p(2),p(1)) = pcheck(p(2),p(1))+1;
    % coefficients of linear system
    coef = [x(p(1)),y(p(1));x(p(2)),y(p(2))];
    % solving the equation of system
    a = linsolve(coef,[-1;-1]);
    % finding the distance of other points
    % to the fitted line
    norm = sqrt(a(1)^2 + a(2)^2);
    dist = a(1)*x + a(2)*y + 1;
    dist = abs(dist);
    dist = dist/norm;
    % checking the counts of inlier 
    inlier = numel(find(dist<3));
    if inlier>700
        break
    end 
end
% showing results
pad = zeros(size(ed));
x = 1:size(img,2);
y = (a(2)*x +1)/(-a(1));
imshow(ed)
hold on
plot(x,y,'color','red')
legend('Fitted line')
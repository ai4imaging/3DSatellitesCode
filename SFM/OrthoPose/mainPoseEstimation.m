% SPDX-License-Identifier: GPL-3.0-or-later
% Copyright (c) 2017 Laura F. Julia <laura.fernandez-julia@enpc.fr>

function mainPoseEstimation(focal)
% focal must be the focal length in pixels of the camera.

im_path='simu2/';
image_names={'input_0.jpg','input_1.jpg','input_2.jpg'};
corresp_files={'01.txt','12.txt','02.txt'};

info=imfinfo(strcat(im_path,image_names{1}));
imsize=[info.Width;info.Height];
if ~isa(focal,'double')
    focal=str2double(focal);
end
CalM=repmat([focal,0,imsize(1)/2;0,focal,imsize(2)/2;0,0,1],3,1);

%% Read matches from files %%%

% read matches between image 1 and image 2
dataFile = fopen(strcat(im_path,corresp_files{1}),'r');
corresp12 = fscanf(dataFile,'%f');
fclose(dataFile);
corresp12=reshape(corresp12,4,[]);

% read matches between image 2 and image 3
dataFile = fopen(strcat(im_path,corresp_files{2}),'r');
corresp23 = fscanf(dataFile,'%f');
fclose(dataFile);
corresp23=reshape(corresp23,4,[]);

% read matches between image 1 and image 3 (optional)
dataFile = fopen(strcat(im_path,corresp_files{3}),'r');
corresp13 = fscanf(dataFile,'%f');
fclose(dataFile);
corresp13=reshape(corresp13,4,[]);


%% Compute tracks %%%
try
    Corresp=matches2triplets(corresp12,corresp23,corresp13);
catch ME
    file=fopen('demo_failure.txt','w');
    fprintf(file,ME.message);
    fclose(file);
    return
end

fprintf('%d tracks between the three images.\n',size(Corresp,2));

csvwrite("Corresp.csv", Corresp)
%% A C RANSAC with Orthographic model %%%
[inliers,Sol,ransac_th]=AC_RANSAC_Orthographic(Corresp,CalM,imsize);
fprintf('%d inliers were found by AC-RANSAC.\n',length(inliers));
csvwrite('inliers.csv',inliers)
%% Orthographic model with all inliers %%%
try
    [Sol1,Sol2]=OrthographicPoseEstimation(Corresp(:,inliers),CalM);
catch ME
    file=fopen('demo_failure.txt','w');
    fprintf(file,ME.message);
    fclose(file);
    return
end

%% B A for both possible solutions %%%
R_t_0=[Sol1{1},Sol1{2}]; Reconst0=Sol1{3};
[R_t_1,Reconst1,iter1,repr_err1]=BundleAdjustment(CalM,R_t_0,Corresp(:,inliers),Reconst0);
fprintf('Minimum reached for first solution with %d iterations. ',iter1);
fprintf('Final reprojection error is %f.\n',repr_err1);

R_t_0=[Sol2{1},Sol2{2}]; Reconst0=Sol2{3};
[R_t_2,Reconst2,iter2,repr_err2]=BundleAdjustment(CalM,R_t_0,Corresp(:,inliers),Reconst0);
fprintf('Minimum reached for second solution with %d iterations. ',iter2);
fprintf('Final reprojection error is %f.\n',repr_err2);
csvwrite("corr.csv",Corresp(:,inliers))
%% Choose solution with less repr. err. todo %%%
if repr_err1<repr_err2
    Solution=R_t_1;
    Reconst=Reconst1;
else
    Solution=R_t_2;
    Reconst=Reconst2;
end
%csvwrite('points1.csv',Reconst1.')
%csvwrite('camera1.csv',R_t_1)
%csvwrite('points2.csv',Reconst2.')
%csvwrite('camera2.csv',R_t_2)

%% Orientations %%%
R2=Solution(4:6,1:3); t2=Solution(4:6,4);
R3=Solution(7:9,1:3); t3=Solution(7:9,4);

%% PLY file %%%
Color=paintReconstruction(Corresp(1:2,:),strcat(im_path,image_names{1}));
writePLYreconstruction('simu2/reconstruction1.ply',CalM,R_t_1,Reconst1,Color);
writePLYreconstruction('simu2/reconstruction2.ply',CalM,R_t_2,Reconst2,Color);
writePLYreconstruction('simu2/reconstruction.ply',CalM,Solution,Reconst,Color);
writeOrientations('simu2/orientations.txt',Solution);
csvwrite('points.csv',Reconst.')
csvwrite('camera.csv',Solution)
dlmwrite('simu2/tracks.txt',Corresp.','delimiter',' ');
dlmwrite('simu2/inliers.txt',Corresp(:,inliers).','delimiter',' ');

outnames={'output_0.png','output_1.png','output_2.png'};
for i=1:3
    image=imread(strcat(im_path,image_names{i}));
    co = Corresp(2*i-1:2*i,:)';
    in = Corresp(2*i-1:2*i,inliers)';
    if exist('OCTAVE_VERSION', 'builtin')
        out = drawCircles(image,co,[255 0 0]);
        out = drawCircles(out,in,[0 255 0]);
    else
        out=insertMarker(image,co,'o','color','red');
        out=insertMarker(out,in,'o','color','green');
    end
    imwrite(out,strcat(im_path,outnames{i}));
end

end

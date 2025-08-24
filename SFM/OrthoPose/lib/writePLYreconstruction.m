% SPDX-License-Identifier: GPL-3.0-or-later
% Copyright (c) 2017 Laura F. Julia <laura.fernandez-julia@enpc.fr>

function writePLYreconstruction(file_name,CalM,R_t,Reconst,Color)
%WRITEPLYRECONSTRUCTION creates a ply file for the 3D reconstruction
% 
%  Input arguments:
%  file_name  - string containing the name of the file
%  CalM       - 3Mx3 containing the M calibration 3x3 matrices for
%               each camera concatenated.
%  R_t        - 3Mx4 matrix containing the global rotation and translation
%               [R,t] for each camera concatenated.
%  Reconst    - 3xN matrix containing the N 3D points.
%  Color      - 3xN matrix containing the color of each 3D point (optional).

N=size(Reconst,2);
M=size(R_t,1)/3;

col=nargin>4;

file = fopen(file_name, 'wt');
fprintf(file,'ply\nformat ascii 1.0\nelement vertex %d\n', N+5*M);
fprintf(file,'property float x\nproperty float y\nproperty float z\n');
fprintf(file,'property uchar red\nproperty uchar green\nproperty uchar blue\n');
fprintf(file,'element edge %d\nproperty int vertex1\nproperty int vertex2\n',8*M);
fprintf(file,'property uchar red\nproperty uchar green\nproperty uchar blue\n');
fprintf(file,'end_header\n');

% Reconstruction Points
for n=1:N
    fprintf(file,'%f %f %f ',Reconst(1,n),Reconst(2,n),Reconst(3,n));
    if col
        fprintf(file,'%d %d %d\n',Color(1,n),Color(2,n),Color(3,n));
    else
        fprintf(file,'255 255 255\n');
    end
end

meanDepth=mean(Reconst(3,:));
scale=0.3*meanDepth/CalM(1,1);
% Predefined colors for first cameras. Others are taken random.
colors=[255 0 0; 0 255 0; 0 0 255];
colsOut=[];
% Camera Points and vertices
for m=1:M
    R=R_t((m-1)*3+(1:3),1:3);
    O=-R'*R_t((m-1)*3+(1:3),4);
    col=[randi(255) randi(255) randi(255)];
    if m<=size(colors,1)
        col=uint8(colors(m,:));
    end
    colsOut=[colsOut;col];
    fprintf(file,'%f %f %f %d %d %d\n',O(1),O(2),O(3),col(1),col(2),col(3));
    frame=[CalM((m-1)*3+(1:2),3);CalM((m-1)*3+1,1)]*scale;
    for i=1:4
        P=R'*frame+O;
        fprintf(file,'%f %f %f %d %d %d\n',P(1),P(2),P(3),col(1),col(2),col(3));
        frame(1)=-frame(1);
        if i==2
            frame(2)=-frame(2);
        end
    end
end
for m=1:M
    ind=N+(m-1)*5;
    % Different colors for cameras, despite meshlab ignoring edge color
    col=colsOut(m,:);
    for i=1:4
        fprintf(file,'%d %d %d %d %d\n',ind, ind+i, col(1),col(2),col(3));
    end
    fprintf(file,'%d %d 0 0 0\n',ind+1, ind+2); %different color to distinguish
    fprintf(file,'%d %d %d %d %d\n',ind+2, ind+4, col(1),col(2),col(3));
    fprintf(file,'%d %d %d %d %d\n',ind+4, ind+3, col(1),col(2),col(3));
    fprintf(file,'%d %d %d %d %d\n',ind+3, ind+1, col(1),col(2),col(3));
end

fclose(file);

end

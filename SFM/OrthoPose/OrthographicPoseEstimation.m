% SPDX-License-Identifier: GPL-3.0-or-later
% Copyright (c) 2017 Laura F. Julia <laura.fernandez-julia@enpc.fr>

function [Sol1,Sol2]=OrthographicPoseEstimation(Corresp,CalM)
%ORTHOGRAPHICPOSEESTIMATION Pose estimation by the Scaled-Orthographic model
%
%  Computation of the orientation of M perspective cameras from N 
%  corresponding tracks of image points and their internal paramaters.
%  The method used is the SCALED ORTHOGRAPHIC model as described in the IPOL
%  submission "The Orthographic Projection Model for Pose Calibration of Long
%  Focal Images" by L. F. Julia, P. Monasse, M. Pierrot-Deseilligny and it
%  is based on the factorization method by C. J. Poelman and T. Kanade.
%
%  Input arguments:
%  Corresp  - 2MxN matrix containing in each column, the M projections of
%             the same space point onto the M images.
%  CalM     - 3Mx3 matrix containing the M calibration 3x3 matrices for
%             each camera concatenated.
%
%  Output arguments: the two possible solutions Sol1, Sol2, in format of
%                    1x5-cell. Sol={Rot,Trans,Reconst, R, T} where:
% 
%  Rot      - 3Mx3 matrix containing the M 3x3 rotation matrices for each
%             camera concatenated. The first will always be the identity.
%  Trans    - 3Mx1 vector containing the M translation vectors for each
%             camera concatenated.
%  Reconst  - 3xN matrix containing the 3D reconstruction of the
%             correspondences.
%  R, T     - 2Mx3 motion matrix and 2Mx1 translation matrix in terms of
%             the orthographic model

N=size(Corresp,2);      % number of correspondences
M=size(Corresp,1)/2;    % number of views

if N<4 || M<3
    error('At least 4 tracks and 3 views are needed for pose estimation.\n');
end

% focal lenghts and principal points
focalL= CalM(3*(1:M)-2,1);
ppalP = CalM(setdiff(1:3*M,3*(1:M)),3);

% center image points subtracting ppal point and compute mean
W=Corresp-repmat(ppalP,1,N);
T=mean(W,2);

% subtract the mean to obtain W* 
W_star=W-repmat(T,1,N);

% compute svd of W*
[U,D,V]=svd(W_star);

% impose rank deficiency and compute rank factorization
R_aux=U(:,1:3)*(D(1:3,1:3)^(1/2));
S_aux=(D(1:3,1:3)^(1/2))*(V(:,1:3)).';

% To find QQ' s.t. R=R_aux*Q and S=Q*S_aux we solve
%  the homogeneous linear system M*coef(QQ')=0
SystM=zeros(2*M,6);
for i=1:M
    m=R_aux(2*i-1,:);
    n=R_aux(2*i,:);
    % norm constraints
    A=2*(m.'*m-n.'*n);
    SystM(2*i-1,:)=[(1./2)*A([1 5 9]), A([2 3 6])];
    % perpendicularity constraints
    A=m.'*n+n.'*m;
    SystM(2*i,:)= [(1./2)*A([1 5 9]), A([2 3 6])];
end

% solution to the system
[~,~,v]=svd(SystM);
CoefQQ=v(:,end)*sign(v(1,end));
QQ=[CoefQQ(1), CoefQQ(4), CoefQQ(5);...
    CoefQQ(4), CoefQQ(2), CoefQQ(6);...
    CoefQQ(5), CoefQQ(6), CoefQQ(3)];
[Q,p]=chol(QQ); Q=Q.';
if p>0
    error('The resulting matrix is not positive semi-definite.\n');
end

% final rank decomposition
R=R_aux*Q; S=Q\S_aux;

% recover rotation parameters
norms=sqrt(sum(R.^2,2));
Rot=zeros(3*M,3);
Rot(setdiff(1:3*M,3*(1:M)),:)=R./repmat(norms,1,3);
Rot(3*(1:M),:)=cross(Rot(3*(1:M)-2,:),Rot(3*(1:M)-1,:),2);

% recover translation parameters
Trans=zeros(3*M,1);
Trans(setdiff(1:3*M,3*(1:M)))=T;
Trans(3*(1:M))=focalL;
s=reshape(repmat((sum(reshape(norms,2,M),1)/2),3,1),[],1);
Trans=Trans./s;

% depth ambiguity: second solution
a=[1;1;-1];
Rot2=diag(repmat(a,M,1))*Rot*diag(a);
S2=diag(a)*S;
R2=R*diag(a);
T2=T; Trans2=Trans;
Reconstr=S;
Reconstr2=S2;

% translation and rotation to bring the center of the first camera to the world origin
Reconstr=Rot(1:3,:)*Reconstr+repmat(Trans(1:3),1,N);
Reconstr2=Rot2(1:3,:)*Reconstr2+repmat(Trans2(1:3),1,N);
R=R/Rot(1:3,:);    R2=R2/Rot2(1:3,:);
T=T-R*Trans(1:3);  T2=T2-R2*Trans2(1:3);

Rot=Rot/Rot(1:3,:);
Trans=Trans-Rot*Trans(1:3);
Rot2=Rot2/Rot2(1:3,:);
Trans2=Trans2-Rot2*Trans2(1:3);

% scaling so that distance from the first camera to the second is 1
alpha=1/norm(Trans(4:6));   alpha2=1/norm(Trans2(4:6));
Trans=alpha*Trans;          Trans2=alpha2*Trans2;
R=(1/alpha)*R;              R2=(1/alpha2)*R2;
Reconstr=alpha*Reconstr;    Reconstr2=alpha2*Reconstr2;

Sol1={Rot,Trans,Reconstr,R,T+ppalP};
Sol2={Rot2,Trans2,Reconstr2,R2,T2+ppalP};
end

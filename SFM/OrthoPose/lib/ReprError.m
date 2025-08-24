% SPDX-License-Identifier: GPL-3.0-or-later
% Copyright (c) 2017 Laura F. Julia <laura.fernandez-julia@enpc.fr>

function error=ReprError(ProjM,Corresp,Points3D)
%REPRERROR Computes the reprojection error of N points for M perspective cameras
%
%  The error is computed by projecting the space points onto the M images
%  using their projective matrices and computing the mean of the distances
%  from the repojected point to their respective original image point.
%
%  Input arguments:
%  ProjM      - 1xM-cell of 3x4 projection matrices.
%  Corresp    - 2MxN matrix with the image points in each image or
%               3MxN matrix if homogeneous coordinates are used.
%  Points 3D  - 3xN matrix containing the space points if known. If the
%  (optional)   space points are not provided they will be triangulated
%               using the projective matrices.
%
%  Output arguments:
%  error      - reprojection error

% computing dimensions
N=size(Corresp,2);

if size(ProjM{1},2)==3
    CalM=ProjM{1}; R_t=ProjM{2};
    M=size(CalM,1)/3; ProjM=cell(1,M);
    for m=1:M
        ProjM{m}=CalM(3*(m-1)+(1:3),:)*R_t(3*(m-1)+(1:3),:);
    end
else
    M=size(ProjM,2);
end

% compute 3D triangulation if necessary
if nargin~=3
    Points3D_est=triangulation3D(ProjM,Corresp);
elseif size(Points3D,1)==3
    Points3D_est=[Points3D;ones(1,N)];
elseif size(Points3D,1)==4
    Points3D_est=Points3D;
end

% convert to affine coordinates and adapt Corresp matrix
if size(Corresp,1)==3*M
    Corresp=reshape(Corresp,3,N*M);
    Corresp=Corresp(1:2,:)./repmat(Corresp(3,:),2,1);
else
    Corresp=reshape(Corresp,2,N*M);
end
mask=~isnan(Corresp);

% reproject points
P=cell2mat(ProjM.');
Corresp_est=reshape(P*Points3D_est,3,M*N);
Corresp_est=Corresp_est(1:2,:)./repmat(Corresp_est(3,:),2,1);

% compute RMS of distances
error= sqrt(mean((Corresp(mask)-Corresp_est(mask)).^2)*2);
end

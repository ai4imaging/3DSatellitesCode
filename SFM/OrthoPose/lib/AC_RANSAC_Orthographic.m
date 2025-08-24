% SPDX-License-Identifier: GPL-3.0-or-later
% Copyright (c) 2017 Laura F. Julia <laura.fernandez-julia@enpc.fr>

function [inliers,Sol,ransac_th]=AC_RANSAC_Orthographic(Corresp,CalM,imsize,NFA_th,max_it)
%AC_RANSAC_ORTHOGRAPHIC A Contrario RANSAC applied to the Scaled Orthographic
% Model for finding the inlier tracks between three orthographic views.
%
%  The implementation of AC-RANSAC (A contrario random sample consensus)
%  is based on the paper of P. Moulon, P. Monasse, R. Marlet: Adaptive
%  Structure from Motion with a contrario model estimation.
% 
%  The pose estimation with the orthographic model relies on the function
%  OrthographicPoseEstimation.
%
%  Input arguments:
%  Corresp    - 6xN matrix containing in each column, the 3 projections of
%               the same space point onto the 3 images.
%  CalM       - 9x3 matrix containing the 3 calibration 3x3 matrices for 
%               each camera, concatenated.
%  imsize     - 2-vector indicating the dimension of the images.
%  NFA_th     - threshold for NFA that establishes the validity of a
%               model (1 by default).
%  max_it     - maximum number of iterations for ransac (100 by default).
%
%  Output arguments:
%  inliers    - indexes indicating the final inliers in Data.
%  Sol        - final orientation giving such set of inliers in the format
%               of a 1x5-cell, Sol={Rot,Trans,Reconst, R, T}, the same as
%               the output of function OrthographicPoseEstimation.
%  ransac_th  - threshold used for ransac given by the AC method and
%               used in the final model&inliers computation.

% default parameters max_it and NFA_th
if nargin<4 || isempty(max_it)
    max_it=100;
end
if nargin<5 || isempty(NFA_th)
    NFA_th=1;
end

N=size(Corresp,2);
n_sample=4; % minimal number of matching points for pose estimation
d=2;        % dimension of the error (repr. error used)
n_out=2;    % number of possible output orientations
a=pi/(imsize(1)*imsize(2)); % probability of having error 1 for nul hypothesis

k_inliers=n_sample+1; % max number of inliers found
inliers=[];           % set of inliers
ransac_th=inf;        % ransac threshold
Sol=[];

if N < n_sample
    return;
end

it=0; max_old=max_it;
while it<max_it
    it=it+1;
    sample=my_randsample(N,n_sample);
    
    % compute orientations with Orthographic model from this sample
    % (if the function fails, we start a new iteration)
    try
        [Sol1,Sol2]=OrthographicPoseEstimation(Corresp(:,sample),CalM);
    catch
        if max_it<2*max_old
            max_it=max_it+1;
        end
        continue;
    end

    % compute the residual error for both solutions and choose the one
    % with lower repr. error
    R=Sol1{4}; T=Sol1{5}; err_min=inf; Sol_it=Sol1;
    for j=1:2
        ind={1:4,5:6,[1 2 5 6],3:4,3:6,1:2}; err=zeros(1,N);
        for k=1:3
            % (orthographic) 3d reconstruction from one pair of views
            p3D=pinv(R(ind{2*k-1},:))*(Corresp(ind{2*k-1},:)-repmat(T(ind{2*k-1}),1,N));
            % reprojection to the remaining view and error
            error=sqrt(sum((R(ind{2*k},:)*p3D+repmat(T(ind{2*k}),1,N) - Corresp(ind{2*k},:)).^2,1));
            % take the max of the error
            err=max([err;error]);
        end

        if sum(err)<err_min
            vec_errors=err;
            err_min=sum(err);
            if j==2
                Sol_it=Sol2;
            end
        end
        R=Sol2{4}; T=Sol2{5};
    end

    % points not in the sample used
    nosample=setdiff(1:N,sample);
    
    % sort the list of errors
    [~,ind_sorted]=sort(vec_errors(nosample));
    
    % search for minimum of NFA(model,k)
    NFA_min=NFA_th; k_min=0; err_threshold=inf;
    factor=n_out*prod(N-n_sample:N)/factorial(n_sample);
    for k=n_sample+1:N
        factor=factor*( (N-(k-1))/(k-n_sample) )*a;
        NFA=factor*(vec_errors(nosample(ind_sorted(k-n_sample))))^(d*(k-n_sample));
        if NFA<=NFA_min
            NFA_min=NFA;
            k_min=k;
            err_threshold=vec_errors(nosample(ind_sorted(k-n_sample)));
        end
    end
    
    % If the found model has more inliers or the same number with less
    %  error than the previous we keep it
    if k_min>k_inliers || (k_min==k_inliers && err_threshold<ransac_th)
        k_inliers=k_min;
        inliers=[reshape(sample,1,[]), nosample(ind_sorted(1:k_inliers-n_sample))];
        ransac_th=err_threshold;
        Sol=Sol_it;
    end
end

end

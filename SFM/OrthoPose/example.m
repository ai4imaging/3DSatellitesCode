% SPDX-License-Identifier: GPL-3.0-or-later
% Copyright (c) 2017 Laura F. Julia <laura.fernandez-julia@enpc.fr>
%               2018 Pascal Monasse <monasse@imagine.enpc.fr>

% Example script for Orthographic Pose Estimation

% Octave/Matlab specifics
if exist('OCTAVE_VERSION', 'builtin')
    pkg load optim;
else
    rng('shuffle');
end

clear;
close all;

addpath('lib');

%% Dataset info %%%
im_path='simu2/';
image_names={'input_0.jpg','input_1.jpg','input_2.jpg'};
info=imfinfo(strcat(im_path,image_names{1}));
imsize=[info.Width;info.Height];

zoomFactor=8; % zoom-out factor from original images
pixPerMm=imsize(1)*zoomFactor/24; % Canon EOS Mark ii sensor: 24x36mm
focalMm=1000; % 1000mm focal length
focal=focalMm*pixPerMm;
focal=4122;
mainPoseEstimation(focal);

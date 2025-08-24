% SPDX-License-Identifier: GPL-3.0-or-later
% Copyright (c) 2019 Pascal Monasse <pascal.monasse@enpc.fr>

function out = drawCircles(image, centers, col)
%DRAWCIRCLES Draw color circles on an image
% Input arguments:
% image   - the input rgb image
% centers - the Nx2 center coordinates (x y)
% col     - 3-vector rgb

    radius = 3;
    n = ceil(2*pi*radius*1.5); % 1.5: supersample perimeter
    angles = linspace(0,2*pi,n);
    x = cos(angles); y = sin(angles);
    d = radius*[x;y];
    d = d(:);
    pts = repmat(centers,1,n) + repmat(d',size(centers,1),1);
    pts = round(pts);
    pts = max(pts,1);
    pts(:,1:2:end) = min(pts(:,1:2:end),size(image,2));
    pts(:,2:2:end) = min(pts(:,2:2:end),size(image,1));
    ind = reshape(pts',1,[]);
    s = size(image);
    ind = sub2ind(s(1:2),ind(2:2:end),ind(1:2:end));

    out = image;
    for i = 1:3
        c = image(:,:,i);
        c(ind)=col(i);
        out(:,:,i) = c;
    end
end

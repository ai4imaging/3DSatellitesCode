% SPDX-License-Identifier: GPL-3.0-or-later
% Copyright (c) 2017 Laura F. Julia <laura.fernandez-julia@enpc.fr>

function Color=paintReconstruction(Corresp,im_name)
%PAINTRECONSTRUCTION extracts the color of each correspondance point from
% a reference image.
%  
%  Input arguments:
%  Corresp  - 2xN matrix with the coordinates of the correspondences in the
%             image.
%  im_name  - string containing the path to the image.

I=imread(im_name);
index=round(Corresp);
Color=zeros(3,size(index,2));
for i=1:size(index,2)
    Color(:,i)=I(index(2,i),index(1,i),:);
end

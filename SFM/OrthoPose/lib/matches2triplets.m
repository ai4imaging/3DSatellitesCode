% SPDX-License-Identifier: GPL-3.0-or-later
% Copyright (c) 2017 Laura F. Julia <laura.fernandez-julia@enpc.fr>

function Corresp=matches2triplets(corresp12,corresp23,corresp13)
%MATCHES2TRIPLETS Computes the matching triplets of points for 3 views from
% the matching points between pairs of images.
%
%  Input arguments:
%  corresp12   - 4xN matrix containing in each column, the coordinates of a 
%                point in image 1 followed by the coordinates of the
%                matching point in image 2.
%  corresp23   - 4xN matrix containing in each column, the coordinates of a
%                point in image 2 followed by the coordinates of the
%                matching point in image 3.
%  corresp13   - 4xN matrix containing in each column, the coordinates of a
%                point in image 1 followed by the coordinates of the
%                matching point in image 3. Optional, for consistency check.
%
%  Output arguments:
%  Corresp     - 6xN matrix containing in each column, the 3 projections of
%                the same space point onto each image.

% intersection of matches
[~,i1,i2]=intersect(corresp12(3:4,:).',corresp23(1:2,:).','rows');
csvwrite('i1.csv',i1)
csvwrite('i2.csv',i2)
if ~isempty(i1)
   Corresp=[corresp12(:,i1); corresp23(3:4,i2)];
else
    error('No common matching points between the 3 images.\n');
end

% If a third file is provided for consistency check
if nargin==3
    % intersection of matches
    [~,i1,~]=intersect(Corresp([1 2 5 6],:).',corresp13.','rows');
    if ~isempty(i1)
       Corresp=Corresp(:,i1);
    else
        error('No common matching points between the 3 images.\n');
    end
end

end

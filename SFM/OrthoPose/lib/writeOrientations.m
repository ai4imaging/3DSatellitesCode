% SPDX-License-Identifier: GPL-3.0-or-later
% Copyright (c) 2017 Laura F. Julia <laura.fernandez-julia@enpc.fr>

function writeOrientations(file_name,R_t)
%WRITEORIENTATIONS creates a txt file with the rotations and translations
% of M views.
%
%  Input arguments:
%  file_name  - string containing the name of the file
%  R_t        - 3Mx4 matrix containing the global rotation and translation
%               [R,t] for each camera

M=size(R_t,1)/3;
file = fopen(file_name, 'wt');
fprintf(file,'%% [R t] matrices of %d cameras.\n',M);
for m=1:M
    fprintf(file, 'P%d=[\n', m-1);
    for j=1:3
        i=3*(m-1)+j;
        fprintf(file,'     %f %f %f %f\n', R_t(i,1),R_t(i,2),R_t(i,3),R_t(i,4));
    end
    fprintf(file, ']\n');
end
fclose(file);

end

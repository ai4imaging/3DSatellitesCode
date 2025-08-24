Pose estimation of 3 views based on the orthographic model.
The algorithm uses only points visible in all 3 views. It implements the Tomasi and Kanade factorization method.

Laura F. Julia, <laura.fernandez-julia@enpc.fr>, Univ. Paris Est, LIGM, ENPC, France
Pascal Monasse, <monasse@imagine.enpc.fr>, Univ. Paris Est, LIGM, ENPC, France

Version 1.2, January 2024

Future releases and updates:
https://github.com/pmonasse/OrthographicPE.git

This MATLAB(R) / GNU OCTAVE directory contains the code associated to the IPOL article "The Orthographic Projection Model for Pose Calibration of Long Focal Images" by Laura F. Julia, Pascal Monasse and Marc Pierrot-Deseilligny.
URL: https://doi.org/10.5201/ipol.2019.248


Files and folders
-----------------

data/                         - folder with 3 example images and their correspondences by pairs
lib/                          - folder with auxiliary Matlab functions
README.txt                    - this file
LICENSE.txt                   - license file
OrthographicPoseEstimation.m  - main function for pose estimation of orthographic views
mainPoseEstimation.m          - do I/O and call OrthographicPoseEstimation
example.m                     - example script


Setup and Usage
---------------

MATLAB
------
1. You must have MATLAB software installed on your computer.
2. The following toolbox is required:
  * optimization_toolbox
3. Run the example script example.m to easily use the code. It contains all the necessary steps for the pose estimation of three perspective views using the orthographic model. The data used as example is in the data/ folder.

OCTAVE
------
1. You must have GNU Octave installed on you computer. You can install it in Ubuntu systems running the following command:

 sudo apt-get install octave octave-optim

2. Run the example script example.m.


Copyright and Licence
---------------------

Copyright (c) 2017 Laura F. Julia <laura.fernandez-julia@enpc.fr>
              2018 Pascal Monasse <monasse@imagine.enpc.fr>
All rights reserved.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

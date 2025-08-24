

cameras = readtable("camera.csv");
points3d = readtable("points.csv");
corresp = readtable("corr.csv");


for index = 1:length(points3d)
    pointTracks(index) = pointTrack([1, 2, 3], [10 20; 30 40]);
end


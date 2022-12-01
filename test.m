clear all; 
close all;
directories;
addpath(data_directory);
addpath(code_directory);
addpath(training_directory);
s = filesep;
load classifiers.mat
addpath([training_directory s 'test_face_photos'])

face_horizontal = 100;
face_vertical = 100;
face_size = [100 100];

path = strcat(training_directory, s, "test_face_photos");
ds = imageDatastore(path);
imgs = readall(ds);
% init faces for testing dataset
test_faces = zeros(face_vertical,face_horizontal,size(imgs,1));
% convert from cell to matrix
for i=1:size(imgs,1)
    temp_img = imgs{i};
    tic; result = boosted_detector_demo(temp_img, .5:.1:1, boosted_classifier, weak_classifiers, [100 100], 2); toc
    figure(2); imshow(result, []);
end
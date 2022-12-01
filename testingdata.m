% CS4337.001 - Computer Vision - Dr. Metsis
% Group 3 Final Project
% Members: Robert Elizondo, Kevin Garcia Lopez, Jacob Lopez

%% preprocessing

clear all; 
close all;
s = filesep;
directories;
addpath(code_directory)

%% Create the training dataset and compute integral images

% i: create the faces dataset
% load training faces into cells
imds = imageDatastore([training_directory s 'training_faces' s '*.bmp']);
imgs = readall(imds);
% store number of faces for later
num_faces = size(imgs,1);
% get size of training images
[face_vertical, face_horizontal] = size(cell2mat(imgs(1)));
face_size = [face_vertical, face_horizontal];
% init faces for training dataset
faces = zeros(face_vertical,face_horizontal,size(imgs,1));
% init face integral images
face_integrals = zeros(face_vertical, face_horizontal, num_faces);
% convert from cell into matrix
for i=1:num_faces
    faces(:,:,i) = imgs{i};
    % compute integral images for faces
    face_integrals(:,:,i) = integral_image(imgs{i});
end
% FINAL MATRIX AND LABELS (TRAINING FACES)
%faces = uint8(faces);
%faces_labels = ones(1,size(imgs,1));

%                 ii: create the nonfaces dataset
% load training nonfaces into cells
imds = imageDatastore([training_directory s 'training_nonfaces' s '*.jpg']);
imgs = readall(imds);
% store number of nonfaces for later
num_nonfaces = size(imgs,1);
% number of windows to crop from each image (this was chosen arbitrarily)
num_windows = 30;
window = [face_vertical, face_horizontal];
% init nonfaces for training dataset; 3900 imgs; 130 imgs x 30 windows each
nonfaces = zeros(face_vertical,face_horizontal,3900);
% init nonfaces integral images
nonface_integrals = zeros(face_vertical, face_horizontal, num_nonfaces * num_windows);
% crop out sub-patches of nonface images
for i=1:size(imgs,1) % for each nonface img (130 total)
    for j=1:num_windows % grab 30 random windows
        input_size = size(imgs{i}); % get size of input img
        rect = randomWindow2d(input_size,window); % grab random sub window
        top = rect.YLimits(1); 
        bot = rect.YLimits(2);
        left = rect.XLimits(1); 
        right = rect.XLimits(2);
        nonfaces(:,:,i*j) = imgs{i}(top:bot, left:right);
        % compute integral images for nonfaces
        nonface_integrals(:,:,i*j) = integral_image(nonfaces(:,:,i*j));
    end
end

% save results
save trainingdataset.mat faces nonfaces face_integrals nonface_integrals

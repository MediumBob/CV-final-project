% CS4337.001 - Computer Vision - Dr. Metsis
% Group 3 Final Project
% Members: Robert Elizondo, Kevin Garcia Lopez, Jacob Lopez

% PART 1: Adaboost
% This script creates a training and a testing dataset from the provided
% images, and uses the provided Adaboost code to train the face detector

%% - preprocessing
unzip('training_test_data.zip');
addpath('training_test_data');

%% Create the training dataset

%                  i: create the faces dataset
% load training faces into cells
imds = imageDatastore('.\training_test_data\training_faces\*.bmp');
imgs = readall(imds);
% get size of training images
[face_rows, face_cols] = size(cell2mat(imgs(1)));
% init faces for training dataset
training_faces = zeros(face_rows,face_cols,size(imgs,1));
% covert from cell into matrix
for i=1:size(imgs,1)
    training_faces(:,:,i) = imgs{i};
end
% FINAL MATRIX AND LABELS (TRAINING FACES)
training_faces = uint8(training_faces);
training_faces_labels = ones(1,size(imgs,1));

%                 ii: create the nonfaces dataset
% load training nonfaces into cells
imds = imageDatastore('.\training_test_data\training_nonfaces\*.jpg');
imgs = readall(imds);
% number of windows to crop from each image (this was chosen arbitrarily)
num_windows = 10;
window = [face_rows, face_cols];
% init nonfaces for training dataset; 1300 b/c 130 imgs x 10 windows each
training_nonfaces = zeros(face_rows,face_cols,1300);
% crop out sub-patches of nonface images
for i=1:size(imgs,1) % for each nonface img (130 total)
    for j=1:num_windows % grab 10 random windows
        input_size = size(imgs{i}); % get size of input img
        rect = randomWindow2d(input_size,window); % grab random sub window
        top = rect.YLimits(1); 
        bot = rect.YLimits(2);
        left = rect.XLimits(1); 
        right = rect.XLimits(2);
        training_nonfaces(:,:,i*j) = imgs{i}(top:bot, left:right);
    end
end

% FINAL MATRIX AND LABELS (TRAINING NONFACES)
training_nonfaces = uint8(training_nonfaces);
training_nonfaces_labels = zeros(1,size(imgs,1)*num_windows);

figure(1); imshow(training_nonfaces(:,:,1300));

%                iii: create the final training dataset


%% NOT FINISHED WITH ABOVE SECTION

%% AdaBoost step 1: get rectangle filters


%% AdaBoost step 2: get weak classifiers

%% AdaBoost step 3: precompute responses of training data on all wc's

%% AdaBoost step 4: get strong/boosted classifier

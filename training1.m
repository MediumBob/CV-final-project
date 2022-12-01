% CS4337.001 - Computer Vision - Dr. Metsis
% Group 3 Final Project
% Members: Robert Elizondo, Kevin Garcia Lopez, Jacob Lopez

% PART 1: Adaboost
% This script creates a training and a testing dataset from the provided
% images, and uses the provided Adaboost code to train the face detector
s = filesep;
directories;
addpath(code_directory)

%% - preprocessing
% unzip('training_test_data.zip');
% addpath('training_test_data');

%% Create the training dataset
if (~isfile('trainingdataset.mat')) 
   testingdata;
else 
    load trainingdataset.mat
end

    % FINAL MATRIX AND LABELS (TRAINING NONFACES)
    % training_nonfaces = uint8(training_nonfaces);
    % training_nonfaces_labels = zeros(1,size(imgs,1)*num_windows);
    % 
    % figure(1); imshow(training_nonfaces(:,:,1300));

%                iii: create the final training dataset

%% NOT FINISHED WITH ABOVE SECTION

number = 1000;
face_horizontal = 100;
face_vertical = 100;
face_size = [100 100];
weak_classifiers = cell(1, number);
for i = 1:number
    weak_classifiers{i} = generate_classifier(face_vertical, face_horizontal);
end

example_number = size(faces, 3) + size(nonfaces, 3);
labels = zeros(example_number, 1);
labels (1:size(faces, 3)) = 1;
labels((size(faces, 3)+1):example_number) = -1;
examples = zeros(face_vertical, face_horizontal, example_number);
examples (:, :, 1:size(faces, 3)) = face_integrals;
examples(:, :, (size(faces, 3)+1):example_number) = nonface_integrals;

classifier_number = numel(weak_classifiers);

responses =  zeros(classifier_number, example_number);

for example = 1:example_number
    integral = examples(:, :, example);
    for feature = 1:classifier_number
        classifier = weak_classifiers {feature};
        responses(feature, example) = eval_weak_classifier(classifier, integral);
    end
    disp(example)
end

boosted_classifier = AdaBoost(responses, labels, 2);
%% AdaBoost step 1: get rectangle filters


%% AdaBoost step 2: get weak classifiers

%% AdaBoost step 3: precompute responses of training data on all wc's

%% AdaBoost step 4: get strong/boosted classifier
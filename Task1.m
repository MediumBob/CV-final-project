% CS4337.001 - Computer Vision - Dr. Metsis
% Group 3 Final Project
% Members: Robert Elizondo, Kevin Garcia Lopez, Jacob Lopez

% PART 1: Adaboost
% This script creates a training and a testing dataset from the provided
% images, and uses the provided Adaboost code to train the face detector

%% - preprocessing
clear all; close all;
s = filesep;
repo_path = 'C:\Users\Bobby King\Desktop\Computer Vision\Git\cs4337-fall2022'; % change this as needed
addpath([repo_path s 'Code' s '00_common' s '00_detection'])
addpath([repo_path s 'Code' s '00_common' s '00_images'])
addpath([repo_path s 'Code' s '00_common' s '00_utilities'])
addpath([repo_path s 'Code' s '17_boosting'])
addpath([repo_path s 'Data' s '00_common_data' s 'frgc2_b'])

load directories.mat


%% Create the training dataset and compute integral images

%                  i: create the faces dataset
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
imds = imageDatastore('.\training_test_data\training_nonfaces\*.jpg');
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

% FINAL MATRIX AND LABELS (TRAINING NONFACES)
%nonfaces = uint8(nonfaces);
%nonfaces_labels = zeros(1,size(imgs,1)*num_windows);

%%

%{
% testing evaluation of a random classifier of type 1.

wc = generate_classifier1(face_vertical, face_horizontal); 
disp(wc); 
disp(wc{1}); 
disp(wc{2});

% disp(wc{9});
%%%

index = 1;
face = faces(:,:,index);
integral = face_integrals(:, :, index);

response = eval_weak_classifier(wc, integral);

% verification:
top = wc{7};
left = wc{8};
bottom = top + wc{5} - 1;
right = left + 2 * wc{6} - 1;
rec_filter = wc{9};
response2 = sum(sum(face(top:bottom, left:right) .* rec_filter));


%%

% testing evaluation of a random classifier of type 2.

wc = generate_classifier2(face_vertical, face_horizontal); 
disp(wc); 
disp(wc{1}); 
disp(wc{2});

%%%

index = 1;
face = faces(:,:,index);
integral = face_integrals(:, :, index);

response = eval_weak_classifier(wc, integral);

% verification:
top = wc{7};
left = wc{8};
bottom = top + 2 * wc{5} - 1;
right = left + 1 * wc{6} - 1;
rec_filter = wc{9};
response2 = sum(sum(face(top:bottom, left:right) .* rec_filter));


%%

% testing evaluation of a random classifier of type 3.

wc = generate_classifier3(face_vertical, face_horizontal); 
disp(wc); 
disp(wc{1}); 
disp(wc{2});

%%%

index = 1;
face = faces(:,:,index);
integral = face_integrals(:, :, index);

response = eval_weak_classifier(wc, integral);

% verification:
top = wc{7};
left = wc{8};
bottom = top + 1 * wc{5} - 1;
right = left + 3 * wc{6} - 1;
rec_filter = wc{9};
response2 = sum(sum(face(top:bottom, left:right) .* rec_filter));


%%

% testing evaluation of a random classifier of type 4.

wc = generate_classifier4(face_vertical, face_horizontal); 
disp(wc); 
disp(wc{1}); 
disp(wc{2});

%%%

index = 1;
face = faces(:,:,index);
integral = face_integrals(:, :, index);

response = eval_weak_classifier(wc, integral);

% verification:
top = wc{7};
left = wc{8};
bottom = top + 3 * wc{5} - 1;
right = left + 1 * wc{6} - 1;
rec_filter = wc{9};
response2 = sum(sum(face(top:bottom, left:right) .* rec_filter));
%}
%%

% choosing a set of random weak classifiers

number = 1000;
weak_classifiers = cell(1, number);
for i = 1:number
    weak_classifiers{i} = generate_classifier(face_vertical, face_horizontal);
end


% save classifiers1000 weak_classifiers

%% precompute responses of all training examples on all weak classifiers

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
    %disp(example)
end

% save training1000 responses labels classifier_number example_number

%% verify that the computed responses are correct

% choose a classifier
a = random_number(1, classifier_number);
wc = weak_classifiers{a};

% choose a training image
b = random_number(1, example_number);
if (b <= size(faces, 3))
    integral = face_integrals(:, :, b);
else
    integral = nonface_integrals(:, :, b - size(faces,3));
end

% see the precomputed response
disp([a, b]);
disp(responses(a, b));
disp(eval_weak_classifier(wc, integral));

%%

%clear all;
%load training1000;
weights = ones(example_number, 1) / example_number;


%%
cl = random_number(1, 1000)
[error, thr, alpha] = weighted_error(responses, labels, weights, cl)

%%

weights = ones(example_number, 1) / example_number;
% next line takes about 8.5 seconds.
tic; [index error threshold] = find_best_classifier(responses, labels, weights); toc
disp([index error]);

%%

%clear all;
%load training1000;
%load classifiers1000;
boosted_classifier = AdaBoost(responses, labels, 15);

% the above line produces this output (second column is 
% error rate of current strong classifier):
%         round        error   best_error best_classifier     alpha    threshold
%             1       0.0295       0.0295          875      -1.7467      -403.81
%             2       0.0295     0.090954          426       -1.151      -193.24
%             3       0.0175     0.094024          373      -1.1327      -395.24
%             4        0.014      0.12683          283      0.96462       982.08
%             5        0.007      0.15141         1000     -0.86181      -397.68
%             6       0.0105      0.15362          864      0.85323        63.99
%             7       0.0055      0.15379          517       0.8526          159
%             8        0.004      0.18631          654     -0.73709      -148.65
%             9       0.0035       0.1776          242      0.76634       -72.45
%            10        0.001      0.17712          559      0.76798       255.69
%            11       0.0015      0.19225          369      0.71771       67.521
%            12       0.0005      0.20436          898      0.67962         1023
%            13            0      0.16552          542     -0.80886      -256.08
%            14            0      0.16625          686      0.80622       46.174
%            15            0      0.15135          780      0.86202       346.11

%%

%save boosted15 boosted_classifier

%%

%load faces1000;
%load nonfaces1000;

% Let's classify a couple of our face and non-face training examples. 
% A positive prediction value means the classifier predicts the input image to be
% a face. A negative prediction value means the classifier thinks it's not
% a face. Values farther away from zero means the classifier is more
% confident about its prediction, either positive or negative.

prediction = boosted_predict(faces(:, :, 200), boosted_classifier, weak_classifiers, 15)

prediction = boosted_predict(nonfaces(:, :, 500), boosted_classifier, weak_classifiers, 15)

%%

% load a photograph
photo = read_gray('faces4.bmp');

% rotate the photograph to make faces more upright (we 
% are cheating a bit, to save time compared to searching
% over multiple rotations).
photo2 = imrotate(photo, -10, 'bilinear');
photo2 = imresize(photo2, 0.34, 'bilinear');
figure(1); imshow(photo2, []);

% w1 and w2 are the locations of the faces, according to me.
% Used just for bookkeeping.
%w1 = photo2(40:87, 75:113);
%w2 = photo2(100:130, 47:71);

%%

tic; result = boosted_multiscale_search(photo2, 1, boosted_classifier, weak_classifiers, face_size); toc
figure(2); imshow(result, []);
figure(3); imshow(max((result > 4) * 255, photo2 * 0.5), [])

%%

tic; [result, boxes] = boosted_detector_demo(photo2, 1, boosted_classifier, weak_classifiers, face_size, 2); toc
figure(2); imshow(result, []);

%%

% load a photograph
photo = read_gray('faces4.bmp');

% rotate the photograph to make faces more upright (we 
% are cheating a bit, to save time compared to searching
% over multiple rotations).
photo2 = imrotate(photo, -10, 'bilinear');

% w1 and w2 are the locations of the faces, according to me.
% Used just for bookkeeping.
w1 = photo2(132:218, 221:290);
w2 = photo2(299:372, 133:192);

% apply the boosted detector, and get the 
% top 10 matches. Takes a few seconds on my desktop.
[result, boxes] = boosted_detector_demo(photo2, 1:0.5:3, boosted_classifier, weak_classifiers, face_size, 2);
figure(1); imshow(photo2, []);
figure(2); imshow(result, [])
% correct results are: w2 at rank 25, w1 at rank 38

%%

photo = read_gray('faces5.bmp');

% w1 is the location of the face, according to me.
% Used just for bookkeeping.
w1 = photo(110:148, 100:130);

% apply the boosted detector, and get the 
% top match.
[result, boxes] = boosted_detector_demo(photo, 1:0.5:3, boosted_classifier, weak_classifiers, face_size, 1);
figure(1); imshow(photo, []);
figure(2); imshow(result, [])
% rank of correct result using normalized correlation is 1.

%%

photo = read_gray('vjm.bmp');

% apply the boosted detector, and get the 
% top 3 matches.
tic; [result, boxes] = boosted_detector_demo(photo, 1:0.5:3., boosted_classifier, weak_classifiers, face_size, 3); toc
figure(1); imshow(photo, []);
figure(2); imshow(result, [])
% rank of correct result using normalized correlation is 1.

%%

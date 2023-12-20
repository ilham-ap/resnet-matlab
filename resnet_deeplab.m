clc; clear; close all;
%datasets https://s.id/datasetsResNet18
%pre train net
load resnet18v_lsmanew.mat
%retrain change to true,
doTraining = false;
imgDir = fullfile('train10');
imds = imageDatastore(imgDir);
I = readimage(imds,1);
I = histeq(I);
figure
imshow(I(:,:,[4 3 2]))
classes = [
 "vegetation"
 "soil"
 "water"
 ];
labelIDs = PixelLabelIDs1();
labelDir = fullfile('label6');
pxds = pixelLabelDatastore(labelDir,classes,labelIDs);
C = readimage(pxds,1);
cmap = ColorMap1;
B = labeloverlay(I(:,:,[4 3 2]),C,'ColorMap',cmap);
figure
imshow(B)
tbl = countEachLabel(pxds)
frequency = tbl.PixelCount/sum(tbl.PixelCount)
[imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionData(imds,pxds);
% Specify the network image size. This is typically the same as the traing image sizes.
imageSize = [2133 2738 3];
% Specify the number of classes.
numClasses = numel(classes);
% Create DeepLab v3+.
N = size(I,3);
% Create DeepLab v3+.
lgraph = deeplabv3plusLayers(imageSize, numClasses, "resnet18");
layers = lgraph.Layers;
%%
newlgraph = replaceLayer(lgraph,'data',imageInputLayer([size(I)],'Name','data'));
newlgraph = replaceLayer(newlgraph,'conv1',convolution2dLayer(7,64,'stride',[2 2],'padding',[3 3 3 3],'Name','conv1'));
lgraph = newlgraph;
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq
pxLayer = pixelClassificationLayer('Name','labels','Classes',tbl.Name,'ClassWeights',classWeights);
lgraph = replaceLayer(lgraph,"classification",pxLayer);
% Define validation data.
dsVal = combine(imdsVal,pxdsVal);
% Define training options. 
options = trainingOptions('sgdm', ...
 'LearnRateSchedule','piecewise',...
 'LearnRateDropPeriod',10,...
 'LearnRateDropFactor',0.3,...
 'Momentum',0.9, ...
 'InitialLearnRate',1e-3, ...
 'L2Regularization',0.005, ...
 'ValidationData',dsVal,...
 'MaxEpochs',30, ... 
 'MiniBatchSize',8, ...
 'Shuffle','every-epoch', ...
 'CheckpointPath', tempdir, ...
 'VerboseFrequency',2,...
 'Plots','training-progress',...
 'ValidationPatience', 4);
dsTrain = combine(imdsTrain, pxdsTrain);
xTrans = [-10 10];
yTrans = [-10 10];
dsTrain = transform(dsTrain, @(data)augmentImageAndLabel(data,xTrans,yTrans));
if doTraining 
 [net, info] = trainNetwork(dsTrain,lgraph,options);
 end
cmap = ColorMap1;
figure
I = readimage(imdsTest,1);
C = semanticseg(I, net);
B = labeloverlay(histeq(I(:,:,[4 3 2])),C,'Colormap',cmap,'Transparency',0);
imshow(B)
pixelLabelColorbar(cmap, classes);
%% function
function data = augmentImageAndLabel(data, xTrans, yTrans)
% Augment images and pixel label images using random reflection and
% translation.
for i = 1:size(data,1)
 
 tform = randomAffine2d(...
 'XReflection',true,...
 'XTranslation', xTrans, ...
 'YTranslation', yTrans);
 
 % Center the view at the center of image in the output space while
 % allowing translation to move the output image out of view.
 rout = affineOutputView(size(data{i,1}), tform, 'BoundsStyle', 'centerOutput');
 
 % Warp the image and pixel labels using the same transform.
 data{i,1} = imwarp(data{i,1}, tform, 'OutputView', rout);
 data{i,2} = imwarp(data{i,2}, tform, 'OutputView', rout);
 
end
end
function cmap = ColorMap1()
% Define the colormap used by CamVid dataset

cmap = [
 34 139 34 
 128 128 0
 59,104,250
 ];
% Normalize between [0 1].
cmap = cmap ./ 255;
end
function labelIDs = PixelLabelIDs1()
labelIDs = { ...
 
 
 [2;]
 
 [3;]
 [4;]
 };
end
function [imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionData(imds,pxds)
% Partition data by randomly selecting 60% of the data for training. The
% rest is used for testing.
 
% Set initial random state for example reproducibility.
rng(0); 
numFiles = numel(imds.Files);
shuffledIndices = randperm(numFiles);
% Use 60% of the images for training.
numTrain = round(0.60 * numFiles);
trainingIdx = shuffledIndices(1:numTrain);
% Use 20% of the images for validation
numVal = round(0.20 * numFiles);
valIdx = shuffledIndices(numTrain+1:numTrain+numVal);
% Use the rest for testing.
testIdx = shuffledIndices(numTrain+numVal+1:end);
% Create image datastores for training and test.
trainingImages = imds.Files(trainingIdx);
valImages = imds.Files(valIdx);
testImages = imds.Files(testIdx);
imdsTrain = imageDatastore(trainingImages);
imdsVal = imageDatastore(valImages);
imdsTest = imageDatastore(testImages);
% Extract class and label IDs info.
classes = pxds.ClassNames;
labelIDs = PixelLabelIDs1();
% Create pixel label datastores for training and test.
trainingLabels = pxds.Files(trainingIdx);
valLabels = pxds.Files(valIdx);
testLabels = pxds.Files(testIdx);
pxdsTrain = pixelLabelDatastore(trainingLabels, classes, labelIDs);
pxdsVal = pixelLabelDatastore(valLabels, classes, labelIDs);
pxdsTest = pixelLabelDatastore(testLabels, classes, labelIDs);
end
function pixelLabelColorbar(cmap, classNames)
% Add a colorbar to the current axis. The colorbar is formatted
% to display the class names with the color.
colormap(gca,cmap)
% Add colorbar to current figure.
c = colorbar('peer', gca);
% Use class names for tick marks.
c.TickLabels = classNames;
numClasses = size(cmap,1);
% Center tick labels.
c.Ticks = 1/(numClasses*2):1/numClasses:1;
% Remove tick mark.
c.TickLength = 0;
end

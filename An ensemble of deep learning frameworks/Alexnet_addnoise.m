addpath(genpath('D:\xxxxxx\MelSP')) 
addpath(genpath('D:\xxxxxx\MelSP\addnoise')) 

clear
close all

cough_datanum=3326;
noncough_datanum=3133;
%% Train CNN
%% Load
path_train = 'D:\xxxxxx\MelSP';

imds = imageDatastore(path_train, ...
    'IncludeSubfolders',true);

randlabel1=randperm(3326);
randlabel2=randperm(3133);

cough_temp=imds.Files(1:3326);
noncough_temp=imds.Files(3327:6459);

cough_temp=cough_temp(randlabel1);
noncough_temp=noncough_temp(randlabel2);

cough_temp=cough_temp(1:2661);
noncough_temp=noncough_temp(1:2506);


%% Load augmentation
path_train = 'D:\xxxxxx\MelSP\addnoise';

imds2 = imageDatastore(path_train, ...
    'IncludeSubfolders',true);

cough_temp2=imds2.Files(1:3326);
noncough_temp2=imds2.Files(3327:6459);

cough_temp2=cough_temp2(randlabel1);
noncough_temp2=noncough_temp2(randlabel2);

cough_temp2=cough_temp2(1:2661);
noncough_temp2=noncough_temp2(1:2506);

%% Merge
imds_label=[ones(2661,1);2*ones(2506,1);ones(2661,1);2*ones(2506,1)];
imds_label=categorical(imds_label);

imds.Files=[cough_temp;noncough_temp;cough_temp2;noncough_temp2];
imds.Labels=imds_label;

net = alexnet;
layers = net.Layers;

%% Train model
layers(23:25)=[
    fullyConnectedLayer(2,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

inputSize = layers(1).InputSize;
% pixelRange = [-30 30];
% imageAugmenter = imageDataAugmenter( ...
%     'RandXReflection',true, ...
%     'RandXTranslation',pixelRange);
% augimdsTrain = augmentedImageDatastore(inputSize(1:2),imds, ...
%             'DataAugmentation',imageAugmenter);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imds);

options = trainingOptions('adam', ...
    'MiniBatchSize',128, ...
    'MaxEpochs',30, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'ExecutionEnvironment','gpu', ...
    'Verbose',0);     

Mynet = trainNetwork(augimdsTrain,layers,options);

%% Extract depth features
layer = 'fc6';
deepTrainfeature = squeeze(activations(Mynet,augimdsTrain,layer,'OutputAs','rows','ExecutionEnvironment','gpu'));

%% Train SVM
train_label=[ones(2661,1);2*ones(2506,1);ones(2661,1);2*ones(2506,1)];
model = fitcsvm(deepTrainfeature,train_label,'Standardize',true,'KernelFunction','RBF', 'KernelScale','auto');

clear cough_temp noncough_temp randlabel1 randlabel2
%% Test
path_train = 'D:\xxxxxx\MelSP';

imds = imageDatastore(path_train, ...
    'IncludeSubfolders',true);

cough_temp=imds.Files(1:3326);
noncough_temp=imds.Files(3327:6459);

cough_temp=cough_temp(randlabel1);
noncough_temp=noncough_temp(randlabel2);

cough_temp=cough_temp(2662:3326);
noncough_temp=noncough_temp(2507:3133);

imds_label=[ones(665,1);2*ones(627,1)];
test_label=imds_label;
imds_label=categorical(imds_label);

imds.Files=[cough_temp;noncough_temp];
imds.Labels=imds_label;

augimdsTest = augmentedImageDatastore(inputSize(1:2),imds);
feature = squeeze(activations(Mynet,augimdsTest,layer,'OutputAs','rows','ExecutionEnvironment','gpu'));

[species,scores]=predict(model,feature);

%% Result
temp=find(test_label==1);
cough_species=species(temp);
temp=find(test_label==2);
noncough_species=species(temp);

TP2=length(find(cough_species==1));     
FP2=length(find(noncough_species==1));  
FN2=length(find(cough_species==2));   
TN2=length(find(noncough_species==2)); 

overall_accuracy=100*(TP2+TN2)/size(species,1);  
cough_accuracy=100*TP2/(TP2+FN2);      
noncough_accurayc=100*TN2/(TN2+FP2);    
precision=100*TP2/(TP2+FP2);            
F1_score=2*precision*cough_accuracy/(precision+cough_accuracy);  

final_result=[overall_accuracy,cough_accuracy,noncough_accurayc,precision,F1_score]


rmpath(genpath('D:\xxxxxx\MelSP')) 
rmpath(genpath('D:\xxxxxx\MelSP\addnoise')) 


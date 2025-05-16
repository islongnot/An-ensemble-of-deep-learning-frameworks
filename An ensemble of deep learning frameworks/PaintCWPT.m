
addpath(genpath('D:\xxxxxx\dataset'))
addpath(genpath('D:\xxxxxx\CWPT'))

clear
close all

fs=44.1e3;                        
FrameDuration=0.02;                 
OverlapLength=0.01;
wlen=floor(FrameDuration*fs/2)*2;    
inc=floor(wlen/2);                  
overlap=wlen-inc;
bandpass_filter=fir1(128,[100 16e3]*2/fs);

%% load
augmenter = audioDataAugmenter( ...
    "AugmentationMode","sequential", ...
    "NumAugmentations",1, ...
    ...
    "TimeStretchProbability",0, ...
    "SpeedupFactorRange", [0.75,1.25], ...
    ...
    "PitchShiftProbability",0, ...
    "SemitoneShiftRange",[-1,1], ...
    ...
    "VolumeControlProbability",0, ...
    "VolumeGainRange",[-1 1],...
    ...
    "AddNoiseProbability",1, ...
    "SNRRange",[0 2], ...
    ...
    "TimeShiftProbability",0);

folder='D:\xxxxxx\dataset\cough';
files = dir([folder '\*.wav']);
num=length(files);

for i=1:num
    filename=files(i).name;
    [temp,Fs]=audioread(filename);
    temp=filter(bandpass_filter,1,temp);     
    temp=filter([1 -0.9375],1,temp);         
    audioOut = augment(augmenter,temp,fs);
    coughdata(i)={audioOut};
end

folder='D:\xxxxxx\dataset\noncough';
files = dir([folder '\*.wav']);
num=length(files);
for i=1:num
    filename=files(i).name;
    [temp,Fs]=audioread(filename);
    temp=filter(bandpass_filter,1,temp);     
    temp=filter([1 -0.9375],1,temp);         
    audioOut = augment(augmenter,temp,fs);
    noncoughdata(i)={audioOut};
end

cough_data=coughdata';
noncough_data=noncoughdata';

%% paint
count=1;
for i=1:size(cough_data,1)
    y1=cough_data{i};
    fb = cwtfilterbank(SignalLength=size(y1,1), ...
    SamplingFrequency=fs, ...
    FrequencyLimits=[0 22050], ...
    wavelet="bump", ...
    VoicesPerOctave=12);

    [cfs,~] = wt(fb,y1);

    imagesc(abs(cfs))
    axis xy
    ylim([0,100])
    
    f=getframe(gca);
    imwrite(f.cdata,['D:\xxxxxx\CWPT\cough\','cough.',num2str(count),'.jpg'],"Quality",100)
    close all
    count=count+1;
end


count=1;
for i=1:size(noncough_data,1)
    y1=noncough_data{i};
    fb = cwtfilterbank(SignalLength=size(y1,1), ...
    SamplingFrequency=fs, ...
    FrequencyLimits=[0 22050], ...
    wavelet="bump", ...
    VoicesPerOctave=12);

    [cfs,~] = wt(fb,y1);

    imagesc(abs(cfs))
    axis xy
    ylim([0,100])

    f=getframe(gca);
    imwrite(f.cdata,['D:\xxxxxx\CWPT\noncough\','noncough.',num2str(count),'.jpg'],"Quality",100)
    close all
    count=count+1;
end


rmpath(genpath('D:\xxxxxx\dataset'))
rmpath(genpath('D:\xxxxxx\CWPT'))


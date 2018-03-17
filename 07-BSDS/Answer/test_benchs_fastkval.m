%% Examples of benchmarks for different input formats
addpath benchmarks
clear all;close all;clc;

%% 2.   morphological version for :boundary benchmark for results stored as contour images

%imgDir = 'G:\BSDS500\data\images\val';
%gtDir = 'G:\BSDS500\data\groundTruth\val';
%pbDir = 'G:\bench_fast\data\png';
%outDir = 'G:\bench_fast\eval\test_all_fast';
%mkdir(outDir);
%nthresh = 99;

%tic;
%boundaryBench_fast(imgDir, gtDir, pbDir, outDir, nthresh);
%toc;


%% 4. morphological version for : all the benchmarks for results stored as a cell of segmentations

imgDir = './Evaluacion/BSDS500/data/images/val';
gtDir = './Evaluacion/BSDS500/data/groundTruth/val';
inDir = './missegmentaciones/Validation/kmeans';
outDir = './val_eval_k';
mkdir(outDir);
nthresh = 99;

tic;
allBench_fast(imgDir, gtDir, inDir, outDir, nthresh);
toc;

%plot_eval(outDir,'r');

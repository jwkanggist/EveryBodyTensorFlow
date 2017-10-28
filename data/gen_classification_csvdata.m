% -------------------------------------------------------------------------
% filename: gen_classification_csvdata.m
% Objectives : this script generates data set for neural networks example
% - sprial data
% - clusterincluster data
% 
% Each functions to generate the data set are downloaded from 
% https://kr.mathworks.com/matlabcentral/fileexchange/41459-6-functions-for-generating-artificial-datasets
% which is written by Jeroen Kools
%
% written by jwkang 2017 Oct
% ------------------------------------------------------------------------



% clusterincluster_set = clusterincluster(N, r1, r2, w1, w2, arms);
N=5000;
clusterincluster_set = clusterincluster(N);

figure(1)
classzero_index = 1:floor(N/2);
classone_index = floor(N/2)+1:length(clusterincluster_set);
scatter(clusterincluster_set(classzero_index,1),clusterincluster_set(classzero_index,2),'b'); hold on
scatter(clusterincluster_set(classone_index,1),clusterincluster_set(classone_index,2),'r')
legend('class zero','class one')

filename =['clusterincluster_N' num2str(N) '.csv'] ;
csvwrite(filename,clusterincluster_set) ;
%-------------------------------------------
% spiral data generation 
%
% degrees controls the length of the spirals
% start determines how far from the origin the spirals start, in degrees
% noise displaces the instances from the spiral. 
%  0 is no noise, at 1 the spirals will start overlapping
N = 5000;
degrees = 570;
start = 90;
noise = 1;
twospirals_set = twospirals(N, degrees, start, noise);

figure(2)
classzero_index = 1:floor(N/2);
classone_index = floor(N/2)+1:length(twospirals_set);
scatter(twospirals_set(classzero_index,1),twospirals_set(classzero_index,2),'b'); hold on
scatter(twospirals_set(classone_index,1),twospirals_set(classone_index,2),'r')
legend('class zero','class one')

filename = ['twospirals_N' num2str(N) '.csv'];
csvwrite(filename,twospirals_set);
clc;clear;close all;

addpath(genpath('./'));
% addpath(genpath('../Methods/GeneralizedCoherentPointDrift-main/'))
% addpath(genpath('../Dataset/TOSCA/SGP_dataset_off/isometric/null/'))

%% 

file1 = 'dog1.off';
file2 = 'dog3.off';

k = 500;
S1 = MESH.preprocess(file1, 'IfComputeLB', true, 'numEigs', k,'IfFindNeigh',true,'IfFindEdge',true,'IfComputeNormals',true);
S2 = MESH.preprocess(file2, 'IfComputeLB', true, 'numEigs', k,'IfFindNeigh',true,'IfFindEdge',true,'IfComputeNormals',true);

[S1, S2] = surfaceNorm(S1, S2);

S1.area = sum(calc_tri_areas(S1.surface));
S2.area = sum(calc_tri_areas(S2.surface));

GT1 = load('dog1.gt');
GT2 = load('dog3.gt');
GT = zeros(length(GT1),1);
for ij=1:length(GT1)
    [~,GT(ij)] = min(abs(GT2-GT1(ij)));
end

% D = S2.Gamma;

%% SHOT matching
opts.shot_num_bins = 10;
opts.shot_radius = 5;
shot1 = calc_shot(S1.surface.VERT', S1.surface.TRIV', 1:S1.nv, opts.shot_num_bins, opts.shot_radius*sqrt(S1.area)/100, 3)';
shot2 = calc_shot(S2.surface.VERT', S2.surface.TRIV', 1:S2.nv, opts.shot_num_bins, opts.shot_radius*sqrt(S2.area)/100, 3)';
T0 = knnsearch(shot2, shot1, 'NSMethod','kdtree'); 

rgb=coord2rgb([S1.surface.X,S1.surface.Y,S1.surface.Z]);
mplot_mesh_rgb([S1.surface.X,S1.surface.Y,S1.surface.Z],S1.surface.TRIV,rgb); title('Source')
rgb=coord2rgb([S2.surface.X,S2.surface.Y,S2.surface.Z]);
mplot_mesh_rgb([S2.surface.X,S2.surface.Y,S2.surface.Z],S2.surface.TRIV,rgb); title('Reference')
mplot_mesh_rgb([S1.surface.X,S1.surface.Y,S1.surface.Z],S1.surface.TRIV,rgb(GT,:)); title('Ground Truth')
mplot_mesh_rgb([S1.surface.X,S1.surface.Y,S1.surface.Z],S1.surface.TRIV,rgb(T0,:));title('Shot Matching')

%% LOPR
Nf = 5;
max_iters = 7; 

[Tlopr, ~] = MWP_LOPR(S1, S2, 500, T0, Nf, max_iters,0.2);

mplot_mesh_rgb([S1.surface.X,S1.surface.Y,S1.surface.Z],S1.surface.TRIV,rgb(Tlopr,:)); title('LOPR ');

%% MWP
f = 5;
num_iters = 7;

[Tmwp, ~] = MWP2(S1, S2, 500, T0, f, num_iters);

mplot_mesh_rgb([S1.surface.X,S1.surface.Y,S1.surface.Z],S1.surface.TRIV,rgb(Tmwp,:)); title('MWP');

%% Sinkhorn
f = 5;
num_iters = 7;

[Tsinkhorn,~]=MWP_SinkHorn(S1,S2,T0,f,num_iters);

mplot_mesh_rgb([S1.surface.X,S1.surface.Y,S1.surface.Z],S1.surface.TRIV,rgb(Tsinkhorn,:)); title('Sinkhorn');

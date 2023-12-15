function [T_fin,C]=MWP_LOPR(s1_shape,s2_shape,k,T0,Nf,num_iters,tau)
% matching: s1_shape --> s2_shape
s1_shape.evecs = s1_shape.evecs(:,1:k);
s1_shape.evals = s1_shape.evals(1:k);
s2_shape.evecs = s2_shape.evecs(:,1:k);                
s2_shape.evals = s2_shape.evals(1:k);

% prepare filters, functions from gspbox:https://github.com/epfl-lts2/gspbox
g=gsp_design_meyer(s1_shape.evals(end),Nf);
ref_g=gsp_design_meyer(s2_shape.evals(end),Nf);

neighbors1 = s1_shape.vtx_neigh; neighbors2 = s2_shape.vtx_neigh;
% [neighbors1,neighbors2] = comput_neighbors(s1_shape.nv,s2_shape.nv,s1_shape.surface.TRIV,s2_shape.surface.TRIV);

% g=gsp_design_simple_tf(s1_shape.evals(end),Nf);
% ref_g=gsp_design_simple_tf(s2_shape.evals(end),Nf);

ref_k=size(s2_shape.evals,1);
k=size(s1_shape.evals,1);

T=T0; M_max = 0; n2=5;

for it=1:num_iters

    C_fmap=s1_shape.evecs\s2_shape.evecs(T,:);

    C=0;
    for s=1:Nf
        ref_fs=sparse(1:ref_k,1:ref_k,ref_g{s}(s2_shape.evals));
        fs=sparse(1:k,1:k,g{s}(s1_shape.evals));
        C=C+fs*C_fmap*ref_fs;
    end

    Tn=knnsearch(gpuArray(s2_shape.evecs*C'),gpuArray(s1_shape.evecs),'k',n2);

    Tn = gather(Tn);
    [T,M_num] = LPO(Tn,s1_shape,s2_shape,neighbors1,neighbors2,n2,tau);
    n2 = ceil(M_num./1000);

    T=gather(T);
    T=T(:);

    if M_num>M_max
        M_max = M_num; T_fin = T;
    else
        fprintf('Iteration %d - # filtered matches: %d\n',it-1,M_max);
        T_fin=T; break;
    end

end

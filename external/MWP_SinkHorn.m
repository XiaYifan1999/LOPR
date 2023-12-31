function [T,C]=MWP_SinkHorn(s1_shape,s2_shape,T0,Nf,num_iters)
% matching: s1_shape --> s2_shape

% prepare filters, functions from gspbox:https://github.com/epfl-lts2/gspbox 
g=gsp_design_meyer(s1_shape.evals(end),Nf);
ref_g=gsp_design_meyer(s2_shape.evals(end),Nf);

% g=gsp_design_simple_tf(s1_shape.evals(end),Nf);
% ref_g=gsp_design_simple_tf(s2_shape.evals(end),Nf);

ref_k=size(s2_shape.evals,1);
k=size(s1_shape.evals,1);

T=T0;

for it=1:num_iters
    C=0;
    % C_fmap=s1_shape.evecs'*s1_shape.A*s2_shape.evecs(T,:);
    C_fmap=s1_shape.evecs\s2_shape.evecs(T,:);
    
    for s=1:Nf
        ref_fs=sparse(1:ref_k,1:ref_k,ref_g{s}(s2_shape.evals));
        fs=sparse(1:k,1:k,g{s}(s1_shape.evals));
        
        C=C+fs*C_fmap*ref_fs;
        %     figure;imagesc(C);
    end
    
    [~,T,~] = fast_sinkhorn_filter(s2_shape.evecs*C',s1_shape.evecs);
    % T=knnsearch(s2_shape.evecs*C',s1_shape.evecs);% cpu version
    
    % gpu knnsearch
    % T=knnsearch(gpuArray(s2_shape.evecs*C'),gpuArray(s1_shape.evecs));%
    
    T=gather(T);
    T=T(:); 
    
end


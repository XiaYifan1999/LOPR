function [T,M_num] = LPO(Tn,s1_shape,s2_shape,neighbors1,neighbors2, k2,lambda)

if nargin < 7
    lambda = 0.2;
end

tol=1e-5;

vector1 = s1_shape.surface.VERT;
vector2 = s2_shape.surface.VERT;

M_lp = LoPres_filter_v1(neighbors1,neighbors2,Tn(:,1),lambda);

M_num=size(M_lp,1); 
rem_index1 = 1:s1_shape.nv; rem_index1(M_lp(:,1))=[];
M_rem = (zeros(length(rem_index1),2));
k1 = ceil(M_num./100);

neighborhood = knnsearch(gpuArray(vector1(M_lp(:,1),:)),gpuArray(vector1(rem_index1,:)),'k',k1);
neighborhood = gather(neighborhood);

Ineighborhood = reshape(M_lp(neighborhood,1),k1,size(M_rem,1))';
Tneighborhood = reshape(M_lp(neighborhood,2),k1,size(M_rem,1))';
W = zeros(k1,s1_shape.nv);

for i = 1:length(rem_index1)
    z = vector1(Ineighborhood(i,:),:) - repmat(vector1(rem_index1(i),:),k1,1);
    G = z*z';
    G = G + eye(k1,k1)* tol * trace(G);
    W(:,i) = G\ones(k1,1);
    W(:,i) = W(:,i)/sum(W(:,i));
    LL_rem=sum(vector2(Tneighborhood(i,:),:)'*W(:,i),2);
    LL_dist = sum((repmat(LL_rem',k2,1) - vector2(Tn(rem_index1(i),:),:)).^2,2);
    [~,LL_ind] = min(LL_dist);
    M_rem(i,:) = [rem_index1(i), Tn(rem_index1(i),LL_ind)];
end

M_rem = (M_rem);
M = sortrows([M_lp;M_rem]); 
T =M(:,2);

end
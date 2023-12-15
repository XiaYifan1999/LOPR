function T_lp = LoPres_filter_v1(Neighbors1,Neighbors2,T,lambda)

n1 = length(Neighbors1);
com_scores = zeros(1,n1);
for i=1:n1
    neigh1 = Neighbors1{i};
    neigh2 = Neighbors2{T(i)};
    com_scores(i) = length(intersect(T(neigh1),neigh2))./(length(neigh1)+length(neigh2));
end

fscores = zeros(1,n1);
for i = 1:n1
    fscores(i) = (com_scores(i)+sum(com_scores(Neighbors1{i})))/(length(Neighbors1{i})+1);
end

inds1 = find(fscores>=lambda); 
T_lp = [inds1',T(inds1)];
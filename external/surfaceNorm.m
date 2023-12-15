function [S1, S2] = surfaceNorm(S1, S2)
X = [S1.surface.X S1.surface.Y S1.surface.Z];
Y = [S2.surface.X S2.surface.Y S2.surface.Z];
[X, Y] = norm2(X, Y);
S1.surface.X = X(:,1);
S1.surface.Y = X(:,2);
S1.surface.Z = X(:,3);
S1.surface.VERT = X;
S2.surface.X = Y(:,1);
S2.surface.Y = Y(:,2);
S2.surface.Z = Y(:,3);
S2.surface.VERT = Y;
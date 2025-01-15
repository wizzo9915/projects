K=0.235;
R=5;
L=0.2;
J=0.00004;
B=0.12;
g=100000000;
G1=tf(K,[L R]);
G2=tf(1,[J B]);
G3=tf(K,1);
G12=series(G1,G2);
G=feedback(G12,G3,-1)
pzmap(G)
roots([8e-6 0.0242 0.6552])
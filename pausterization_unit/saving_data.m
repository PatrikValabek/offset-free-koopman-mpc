



%%
u1 = inputs{1}.Values.Data;
save('experimental_data/u1_ident_3','u1')

u2 = inputs{2}.Values.Data;
save('experimental_data/u2_ident_3','u2')

u3 = inputs{3}.Values.Data;
save('experimental_data/u3_ident_3','u3')

%%

T1 = Temperatures{1}.Values.Data;
T2 = Temperatures{2}.Values.Data;
T4 = Temperatures{4}.Values.Data;

save('experimental_data/T1_ident_3','T1')
save('experimental_data/T2_ident_3','T2')
save('experimental_data/T4_ident_3','T4')

%%

figure
hold on
plot(T1)
plot(T4)

figure
hold on
plot(T2)
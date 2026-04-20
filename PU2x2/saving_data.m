



%%
u1 = inputs{1}.Values.Data;
save('ident_data/u1_ident_12','u1')

u2 = inputs{2}.Values.Data;
save('ident_data/u2_ident_12','u2')

u3 = inputs{3}.Values.Data;
save('ident_data/u3_ident_12','u3')

%%

T1 = Temperatures{1}.Values.Data;
T2 = Temperatures{2}.Values.Data;
T4 = Temperatures{4}.Values.Data;

save('ident_data/T1_ident_12','T1')
save('ident_data/T2_ident_12','T2')
save('ident_data/T4_ident_12','T4')

%%

figure
hold on
%plot(T1)
plot(T4)

figure
hold on
plot(T2)
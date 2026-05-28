



%%
u1 = inputs{1}.Values.Data;
save('ident_data/u1_ident_16','u1')

u2 = inputs{2}.Values.Data;
save('ident_data/u2_ident_16','u2')

u3 = inputs{3}.Values.Data;
save('ident_data/u3_ident_16','u3')

%%

T1 = Temperatures{1}.Values.Data;
T2 = Temperatures{2}.Values.Data;
T4 = Temperatures{4}.Values.Data;

save('ident_data/T1_ident_16','T1')
save('ident_data/T2_ident_16','T2')
save('ident_data/T4_ident_16','T4')

%%

figure
hold on
%plot(T1)
plot(T4)

figure
hold on
plot(T2)
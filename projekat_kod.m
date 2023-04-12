clear
close all
clc

%% Ucitavanje podataka
T = readtable('genres.csv');
izlaz = T{:, 12}';
ulaz = T{:, 1:11};
N = length(izlaz);

%% Podela na klase
Rap = strcmp(izlaz', 'Rap')';
Pop = strcmp(izlaz', 'Pop')';
RnB = strcmp(izlaz', 'RnB')';
K1 = ulaz(Rap == 1, :);
K2 = ulaz(Pop == 1, :);
K3 = ulaz(RnB == 1, :);
izlazOH = [Rap;Pop;RnB];

%% Prikaz klasa
kategorije = categorical(izlaz);
figure
    histogram(kategorije, 'BarWidth', 0.7)
    title('Histogram klasa')
    xlabel('Klase')
    ylabel('Broj pojavljivanja klasa')
    grid on;

%% Podela podataka na odgovarajuce skupove po klasama
%podela za klasu 1
N1 = length(K1);
rng(50);
K1perm = K1(randperm(N1), :);
K1trening = K1perm(1 : round(0.7*N1), :);
K1test = K1perm(round(0.7*N1) + 1 : round(0.85*N1), :);
K1val = K1perm(round(0.85*N1) + 1 : N1, :);
%podela za klasu 2
N2 = length(K2);
rng(50);
K2perm = K2(randperm(N2), :);
K2trening = K2perm(1 : round(0.7*N2), :);
K2test = K2perm(round(0.7*N2) + 1 : round(0.85*N2), :);
K2val = K2perm(round(0.85*N2) + 1 : N2, :);
%podela za klasu 3
N3 = length(K3);
rng(50);
K3perm = K3(randperm(N3), :);
K3trening = K3perm(1 : round(0.7*N3), :);
K3test = K3perm(round(0.7*N3) + 1 : round(0.85*N3), :);
K3val = K3perm(round(0.85*N3) + 1 : N3, :);

%% Grupisanje u zajednocki trening, test i validacioni skup
%trening skup
ulazTrening = [K1trening;K2trening;K3trening];
v1 = [ones(1, round(0.7*N1));zeros(2, round(0.7*N1))];
v2 = [zeros(1, round(0.7*N2));ones(1, round(0.7*N2));zeros(1, round(0.7*N2))];
v3 = [zeros(2, round(0.7*N3));ones(1, round(0.7*N3))];
izlazTrening = [v1, v2, v3];
%test skup
ulazTest = [K1test;K2test;K3test];
v1 = [ones(1, round(0.15*N1));zeros(2, round(0.15*N1))];
v2 = [zeros(1, round(0.15*N2));ones(1, round(0.15*N2));zeros(1, round(0.15*N2))];
v3 = [zeros(2, round(0.15*N3));ones(1, round(0.15*N3))];
izlazTest = [v1, v2, v3];
%validacioni skup
ulazVal = [K1val;K2val;K3val];
izlazVal = izlazTest;

%% Zajednicki skup prosledjen neuralnoj mrezi
ulazSve = [ulazTrening; ulazVal];
izlazSve = [izlazTrening, izlazVal];

%% Permutovanje elemenata u skupovima
rng(50);
ind = randperm(length(ulazSve));
ulazSve = ulazSve(ind, :);
izlazSve = izlazSve(:, ind);
rng(50);
ind = randperm(length(ulazVal));
ulazVal = ulazVal(ind, :);
izlazVal = izlazVal(:, ind);
rng(50);
ind = randperm(length(ulazTest));
ulazTest = ulazTest(ind, :);
izlazTest = izlazTest(:, ind);

%% Krosvalidacija
% arhitektura = {[20 15 10], [5 10 15], [6 7 8]};
% arhitektura = {[5 10 15], [10 10 10], [10 10 5]};
arhitektura = {[20 25 20], [8 9 21], [7 5 12]};
transferFcns = ['poslin'; 'tansig'; 'logsig';];
numTransFcns = size(transferFcns);
numTransFcns = numTransFcns(1);
% r = [0.2, 0.5, 0.9];
% r = [0.1, 0.2, 0.3];
% r = [0.1, 0.3, 0.7, 0.9];
i = 1;
F1best = 0;
P1best = 0;
P2best = 0;
P3best = 0;
R1best = 0;
R2best = 0;
R3best = 0;
for reg = [0.05, 0.1, 0.15, 0.2, 0.7, 0.9]
    for t = 1:numTransFcns
        for arh = 1:length(arhitektura)
            for w = [0.5, 1, 2, 3]
                for lr = [0.1, 0.5, 0.8]
                    rng(50);
                    %kreiranje mreze
                    net = patternnet(arhitektura{arh});

                    net.divideFcn = 'divideind';
                    net.divideParam.trainInd = 1 : length(ulazTrening);
                    net.divideParam.valInd = length(ulazTrening) + 1 : length(ulazSve);
                    net.divideParam.testInd = [];

                    tf = transferFcns(t, :);
                    for k = 1:length(arhitektura{arh})
                        net.layers{k}.transferFcn = tf;
                    end
                    net.layers{k + 1}.transferFcn = 'softmax';

                    %net.trainParam.showWindow = false;
                    %net.trainParam.showCommandLine = false;

                    net.performParam.regularization = reg;

                    net.trainParam.lr = lr;
                    net.trainParam.epochs = 600;
                    net.trainParam.goal = 1e-5;
                    net.trainParam.min_grad = 1e-6;
                    net.trainParam.max_fail = 200;
                    
                    %net.trainFcn = 'traingd';
                    %net.trainFcn = 'traingda';
                    %net.trainFcn = 'traingdm';
                    net.trainFcn = 'trainscg';
                    
                    %druga klasa je mala i zelimo da joj posvetimo vise paznje
                    weight = ones(1, length(izlazSve));
                    weight(izlazSve(2,:) == 1) = w;

                    %treniranje mreze
                    net = train(net, ulazSve', izlazSve, [], [], weight);

                    %predikcija mreze
                    pred = sim(net, ulazVal');

                    [~, cm] = confusion(izlazVal, pred);
                    P1 = cm(1,1)/(cm(1,1)+cm(1,2)+cm(1,3));
                    R1 = cm(1,1)/(cm(1,1)+cm(2,1)+cm(3,1));
                    P2 = cm(2,2)/(cm(2,1)+cm(2,2)+cm(2,3));
                    R2 = cm(2,2)/(cm(1,2)+cm(2,2)+cm(3,2));
                    P3 = cm(3,3)/(cm(3,1)+cm(3,2)+cm(3,3));
                    R3 = cm(3,3)/(cm(1,3)+cm(2,3)+cm(3,3));
                    F11 = 2*P1*R1/(P1+R1);
                    F12 = 2*P2*R2/(P2+R2);
                    F13 = 2*P3*R3/(P3+R3);
                    F1 = 1/3*(F11+F12+F13);
                    
                    %provera da li je najbolja mreza trenutna
                    %kriterijumska funkcija je Fscore zbog nebalansiranih klasa
                    if F1 > F1best
                        F1best = F1;
                        Bestreg = reg;
                        BestTransferFunction = tf;
                        BestArh = arh;
                        best_w = w;
                        best_lr = lr;
                        
                        P1best = P1;
                        P2best = P2;
                        P3best = P3;
                        R1best = R1;
                        R2best = R2;
                        R3best = R3;
                    end 
                    i = i + 1;
                end
            end
        end
   end
end

%% Projektovanje najbolje mreze
rng(50);
net = patternnet(arhitektura{BestArh});

net.divideFcn = '';

for k = 1:length(arhitektura{arh})
    net.layers{k}.transferFcn = BestTransferFunction;
end
net.layers{k + 1}.transferFcn = 'softmax';

net.performParam.regularization = Bestreg;
net.trainParam.lr = best_lr;
net.trainParam.epochs = 600;
net.trainParam.goal = 1e-5;
net.trainParam.min_grad = 1e-6;
net.trainParam.max_fail = 100;
net.trainFcn = 'trainscg';

weight = ones(1, length(izlazSve));
weight(izlazSve(2,:) == 1) = best_w;

%treniranje mreze
net = train(net, ulazSve', izlazSve, [], [], weight);

%predikcija mreze
pred = sim(net, ulazTest');
pred = round(pred);
figure, plotconfusion(izlazTest, pred);
disp('---------KRAJ---------')

%% Prikaz najboljih vrednosti
disp('Najbolja arhitektura:')
disp(arhitektura{BestArh})
disp('Najbolja konstanta obucavanja:')
disp(best_lr)
disp('Najbolja konstanta regularizacije:')
disp(Bestreg)
disp('Najbolja tezina:')
disp(best_w)
disp('Najbolji broj epoha:')
disp(net.trainParam.epochs)
disp('Najbolji goal:')
disp(net.trainParam.goal)
disp('Najbolja minimalna promena gradijenta:')
disp(net.trainParam.min_grad)
disp('Najbolja dozvoljena maksimalna greska:')
disp(net.trainParam.max_fail)

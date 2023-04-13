clear
close all
clc

%% Data loading
T = readtable('genres.csv');
out = T{:, 12}';
in = T{:, 1:11};
N = length(out);

%% Division into classes
Rap = strcmp(out', 'Rap')';
Pop = strcmp(out', 'Pop')';
RnB = strcmp(out', 'RnB')';
K1 = in(Rap == 1, :);
K2 = in(Pop == 1, :);
K3 = in(RnB == 1, :);
outOH = [Rap;Pop;RnB];

%% Class view
category = categorical(out);
figure
    histogram(category, 'BarWidth', 0.7)
    title('Class histogram')
    xlabel('Classes')
    ylabel('Number of class occurrences')
    grid on;

%% Division of data into appropriate sets by classes
%first class
N1 = length(K1);
rng(50);
K1perm = K1(randperm(N1), :);
K1trening = K1perm(1 : round(0.7*N1), :);
K1test = K1perm(round(0.7*N1) + 1 : round(0.85*N1), :);
K1val = K1perm(round(0.85*N1) + 1 : N1, :);
%second class
N2 = length(K2);
rng(50);
K2perm = K2(randperm(N2), :);
K2trening = K2perm(1 : round(0.7*N2), :);
K2test = K2perm(round(0.7*N2) + 1 : round(0.85*N2), :);
K2val = K2perm(round(0.85*N2) + 1 : N2, :);
%third class
N3 = length(K3);
rng(50);
K3perm = K3(randperm(N3), :);
K3trening = K3perm(1 : round(0.7*N3), :);
K3test = K3perm(round(0.7*N3) + 1 : round(0.85*N3), :);
K3val = K3perm(round(0.85*N3) + 1 : N3, :);

%% Grouping into joint training, test and validation sets
%training
inTrening = [K1trening;K2trening;K3trening];
v1 = [ones(1, round(0.7*N1));zeros(2, round(0.7*N1))];
v2 = [zeros(1, round(0.7*N2));ones(1, round(0.7*N2));zeros(1, round(0.7*N2))];
v3 = [zeros(2, round(0.7*N3));ones(1, round(0.7*N3))];
outTrening = [v1, v2, v3];
%test
inTest = [K1test;K2test;K3test];
v1 = [ones(1, round(0.15*N1));zeros(2, round(0.15*N1))];
v2 = [zeros(1, round(0.15*N2));ones(1, round(0.15*N2));zeros(1, round(0.15*N2))];
v3 = [zeros(2, round(0.15*N3));ones(1, round(0.15*N3))];
outTest = [v1, v2, v3];
%validation
inVal = [K1val;K2val;K3val];
outVal = outTest;

%% Common set passed to neural network
inAll = [inTrening; inVal];
outAll = [outTrening, outVal];

%% Permutation of elements in sets
rng(50);
ind = randperm(length(inAll));
inAll = inAll(ind, :);
outAll = outAll(:, ind);
rng(50);
ind = randperm(length(inVal));
inVal = inVal(ind, :);
outVal = outVal(:, ind);
rng(50);
ind = randperm(length(inTest));
inTest = inTest(ind, :);
outTest = outTest(:, ind);

%% Cross validation
% architecture = {[20 15 10], [5 10 15], [6 7 8]};
% architecture = {[5 10 15], [10 10 10], [10 10 5]};
architecture = {[20 25 20], [8 9 21], [7 5 12]};
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
        for arh = 1:length(architecture)
            for w = [0.5, 1, 2, 3]
                for lr = [0.1, 0.5, 0.8]
                    rng(50);
                    %creating a network
                    net = patternnet(architecture{arh});

                    net.divideFcn = 'divideind';
                    net.divideParam.trainInd = 1 : length(inTrening);
                    net.divideParam.valInd = length(inTrening) + 1 : length(inAll);
                    net.divideParam.testInd = [];

                    tf = transferFcns(t, :);
                    for k = 1:length(architecture{arh})
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
                    
                    %second class is small and we want to pay more attention to it
                    weight = ones(1, length(outAll));
                    weight(outAll(2,:) == 1) = w;

                    %treniranje mreze
                    net = train(net, inAll', outAll, [], [], weight);

                    %prediction
                    pred = sim(net, inVal');

                    [~, cm] = confusion(outVal, pred);
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
                    
                    %checking if the best network is current
                    %the criterion function is Fscore due to unbalanced classes
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

%% Designing the best network
rng(50);
net = patternnet(architecture{BestArh});

net.divideFcn = '';

for k = 1:length(architecture{arh})
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

weight = ones(1, length(outAll));
weight(outAll(2,:) == 1) = best_w;

%network training
net = train(net, inAll', outAll, [], [], weight);

%network prediction
pred = sim(net, inTest');
pred = round(pred);
figure, plotconfusion(outTest, pred);
disp('---------The end---------')

%% Display of the best parameters
disp('Best architecture:')
disp(architecture{BestArh})
disp('Best training constant:')
disp(best_lr)
disp('Best regularization constant:')
disp(Bestreg)
disp('Best weight:')
disp(best_w)
disp('Best number of epochs:')
disp(net.trainParam.epochs)
disp('Best goal:')
disp(net.trainParam.goal)
disp('Best minimum gradient change:')
disp(net.trainParam.min_grad)
disp('Best the best allowed maximum error:')
disp(net.trainParam.max_fail)

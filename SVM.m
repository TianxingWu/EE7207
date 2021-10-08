clear
close all

%% Load Data
load('data_test.mat')
load('data_train.mat')
load('label_train.mat')

rng(0)
%% Traning
% SVM_Classifier = fitcsvm(data_train,label_train,'KernelFunction','gaussian','OptimizeHyperparameters','auto');
% pred_train = predict(SVM_Classifier,data_train);
% acc_train = sum(pred_train==label_train) / size(data_train,1);




%% Method 1: interation
% initial:
c_best = 15;
fprintf('Initial BoxConstraint:%.2f\n',c_best)

for i=1:3
    % 2. hyper-param:KernelScale
    s_min = 0;
    s_step = 0.1;
    s_max = 5;
    accs_train_tune_s = zeros((s_max-s_min)/s_step,1);
    accs_val_tune_s = zeros((s_max-s_min)/s_step,1);
    acc_val_best_s = 0;
    c = c_best;
    for i=1:(s_max-s_min)/s_step
        KernelScale = s_min+i*s_step;
        K = 5; % 5-fold-validation
        [accs_train_tune_s(i), accs_val_tune_s(i)] = train_SVM_Kfold(data_train, label_train, c, KernelScale, K);
        if accs_val_tune_s(i)>=acc_val_best_s
            s_best = KernelScale;
            acc_val_best_s = accs_val_tune_s(i);
        end
    end
    % plot
    ss = s_min+s_step:s_step:s_max;
    figure
    hold on
    plot(ss, accs_train_tune_s, 'r', 'LineWidth',2)
    plot(ss, accs_val_tune_s, 'b', 'LineWidth',2)
    legend('training set', 'validation set','Location','northeast')
    title(['SVM performance with different KernelScale (C=',num2str(c_best), ')']);

    xlabel('KernelScale');
    ylabel('classification accuracy');
    hold off

%     fprintf('Current C:\n%d\n', c_best)
%     fprintf('Best KernelScale:\n%d\n', s_best)
%     fprintf('Best acc with KernelScale:\n%f\n\n', acc_val_best_s)
    fprintf('BoxConstraint:%.2f, KernelScale:%.2f, acc:%.4f\n',c_best,s_best,acc_val_best_s)
    
    
    % 1. hyper-param: C

    c_min = 0;
    c_step = 0.5;
    c_max = 20;
    accs_train_tune_c = zeros((c_max-c_min)/c_step,1);
    accs_val_tune_c = zeros((c_max-c_min)/c_step,1);
    acc_val_best_c = 0;
    for i=1:(c_max-c_min)/c_step
        c = c_min+i*c_step;
        KernelScale = s_best;
        K = 5; % 5-fold-validation  
        [accs_train_tune_c(i), accs_val_tune_c(i)] = train_SVM_Kfold(data_train, label_train, c, KernelScale, K);
        if accs_val_tune_c(i) >= acc_val_best_c
            c_best = c;
            acc_val_best_c = accs_val_tune_c(i);
        end
    end
    % plot
    cs = c_min+c_step:c_step:c_max;
    figure
    hold on
    plot(cs, accs_train_tune_c, 'r', 'LineWidth',2)
    plot(cs, accs_val_tune_c, 'b', 'LineWidth',2)
    legend('training set', 'validation set','Location','northeast')

    title(['SVM performance with different C (KernelScale=',num2str(KernelScale), ')']);
    xlabel('BoxConstraint');
    ylabel('classification accuracy');
    hold off
%     fprintf('Current KernelScale:\n%d\n', KernelScale)
%     fprintf('Best c:\n%d\n', c_best)
%     fprintf('Best acc with c:\n%f\n\n', acc_val_best_c)
    fprintf('BoxConstraint:%.2f, KernelScale:%.2f, acc:%.4f\n',c_best,s_best,acc_val_best_s)

end

%% Method 2: Grid Search

%=========================================================================
% 1. Low Resolution
s_min = 0;
s_step = 0.1;
s_max = 5;
c_min = 0;
c_step = 0.5;
c_max = 20;
accs_train_grid = zeros((c_max-c_min)/c_step,(s_max-s_min)/s_step);
accs_val_grid = zeros((c_max-c_min)/c_step,(s_max-s_min)/s_step);
acc_val_best_grid = 0;
for i=1:(c_max-c_min)/c_step
    for j=1:(s_max-s_min)/s_step
        c = c_min+i*c_step;
        KernelScale = s_min+j*s_step;
        K = 5; % 5-fold-validation  
        [accs_train_grid(i,j), accs_val_grid(i,j)] = train_SVM_Kfold(data_train, label_train, c, KernelScale, K);
        if accs_val_grid(i,j) >= acc_val_best_grid
            c_best = c;
            s_best = KernelScale;
            acc_val_best_grid = accs_val_grid(i,j);
            accs_train_best_grid = accs_train_grid(i,j);
        end
        fprintf('BoxConstraint:%.2f, KernelScale:%.2f, acc:%.4f\n',c,KernelScale,accs_val_grid(i,j))
    end
end
fprintf('Low RES: best BoxConstraint:%.2f, best KernelScale:%.2f, best train_acc:%.4f\n, best val_acc:%.4f\n',c_best,s_best,accs_train_best_grid,acc_val_best_grid)

% plot
figure
[C,S] = meshgrid((c_min+c_step):c_step:c_max, (s_min+s_step):s_step:s_max);
C = C.';
S = S.';
% plot3(C, S, accs_val_grid, '.')
scatter3(C(:),S(:),accs_val_grid(:),30, accs_val_grid(:),'filled');
title(['Kernel SVM Grid Search (Resolution=(',num2str(c_step),',', num2str(s_step), '))'])
xlabel('BoxConstraint')
ylabel('KernelScale')
c = colorbar('eastoutside');
c.Label.String = 'Accuracy on Validation Set';
colormap(jet);
view(2)
% accs_val_grid

%=========================================================================
% 2. High Resolution
s_min = 1.5;
s_step = 0.05;
s_max = 3;
c_min = 0;
c_step = 0.1;
c_max = 5;
accs_train_grid = zeros((c_max-c_min)/c_step,(s_max-s_min)/s_step);
accs_val_grid = zeros((c_max-c_min)/c_step,(s_max-s_min)/s_step);
acc_val_best_grid = 0;
for i=1:(c_max-c_min)/c_step
    for j=1:(s_max-s_min)/s_step
        c = c_min+i*c_step;
        KernelScale = s_min+j*s_step;
        K = 5; % 5-fold-validation  
        [accs_train_grid(i,j), accs_val_grid(i,j)] = train_SVM_Kfold(data_train, label_train, c, KernelScale, K);
        if accs_val_grid(i,j) >= acc_val_best_grid
            c_best = c;
            s_best = KernelScale;
            acc_val_best_grid = accs_val_grid(i,j);
            accs_train_best_grid = accs_train_grid(i,j);
        end
        fprintf('BoxConstraint:%.2f, KernelScale:%.2f, acc:%.4f\n',c,KernelScale,accs_val_grid(i,j))
    end
end
fprintf('High RES: best BoxConstraint:%.2f, best KernelScale:%.2f, best train_acc:%.4f\n, best val_acc:%.4f\n',c_best,s_best,accs_train_best_grid,acc_val_best_grid)

% plot
figure
[C,S] = meshgrid((c_min+c_step):c_step:c_max, (s_min+s_step):s_step:s_max);
C = C.';
S = S.';
% plot3(C, S, accs_val_grid, '.')
scatter3(C(:),S(:),accs_val_grid(:),30, accs_val_grid(:),'filled');
title(['Kernel SVM Grid Search (Resolution=(',num2str(c_step),',', num2str(s_step), '))'])
xlabel('BoxConstraint')
ylabel('KernelScale')
c = colorbar('eastoutside');
c.Label.String = 'Accuracy on Validation Set';
colormap(jet);
view(2)



%% Final train and test: train on full data with optimal parameters
[best_SVM_Classifier, acc_train] = train_SVM(data_train, label_train, c_best, s_best);
[pred, ~] = test_SVM(data_test, nan, best_SVM_Classifier);
fprintf('Train on Full data:\n')
fprintf('Train_acc: %.3f\n', acc_train)
fprintf('Prediction on Test set:\n')
fprintf('%d  %d  %d  %d  %d  %d  %d  %d  %d  %d  %d  %d  %d  %d  %d  %d  %d  %d  %d  %d  %d\n\n',pred.');

%% Data analysis
[N, D] = size(data_train);
ax_x = 1:D;
ax_y = 1:D;
rho = corr(data_train, 'type', 'pearson');
figure
h = heatmap(ax_x,ax_y, abs(rho));
colormap('redbluecmap')

[show_dim_1, show_dim_2]= find(rho==min(min(abs(rho))));
show_dim_1 = show_dim_1(1);
show_dim_2 = show_dim_2(1);
x = data_train(:,show_dim_1);
y = data_train(:,show_dim_2);

sv = best_SVM_Classifier.SupportVectors;

figure
gscatter(data_train(:,show_dim_1),data_train(:,show_dim_2),label_train)
hold on
plot(sv(:,show_dim_1),sv(:,show_dim_2),'ko','MarkerSize',10)
legend('versicolor','virginica','Support Vector')
hold off


figure
plot(x((label_train==1)),y((label_train==1)),'r.','MarkerSize',20)
hold on
plot(x((label_train==-1)),y((label_train==-1)),'c.','MarkerSize',20)
plot(sv(:,show_dim_1),sv(:,show_dim_2),'ko','MarkerSize',10, 'LineWidth',1)
legend('class 1','class 2','support vector','Location','northeast')
hold off
 
%% Functions

function [acc_train_mean, acc_val_mean] = train_SVM_Kfold(x_orig, y_orig, c, KernelScale, K)
    N = size(x_orig,1);
    num = round(N/K);
    accs_train = zeros(K,1);
    accs_val = zeros(K,1);

    for i=1:K
        mask_val = logical(zeros(N,1));
        mask_val(num*(i-1)+1:num*i)=1;
        x_val = x_orig(mask_val, :);
        y_val = y_orig(mask_val, :);
        x_train = x_orig(~mask_val, :);
        y_train = y_orig(~mask_val, :);
        [SVM_Classifier, acc_train] = train_SVM(x_train, y_train, c, KernelScale);
        [~, acc_val] = test_SVM(x_val, y_val, SVM_Classifier);
        accs_train(i) = acc_train;
        accs_val(i) = acc_val;
    end
    acc_train_mean = mean(accs_train);
    acc_val_mean = mean(accs_val);
end

function [SVM_Classifier, acc_train] = train_SVM(x_train, y_train, c, KernelScale)
SVM_Classifier = fitcsvm(x_train,y_train,'KernelFunction','gaussian','BoxConstraint',c,'KernelScale',KernelScale);
pred = predict(SVM_Classifier,x_train);
acc_train = sum(pred==y_train) / size(x_train,1);
end

function [pred, acc_test] = test_SVM(x_test, y_test, SVM_Classifier)
pred = predict(SVM_Classifier,x_test);
acc_test = sum(pred==y_test) / size(x_test,1);
end



function [MSE_val_mean, acc_train_mean, acc_val_mean] = train_RBF_Kfold(x_orig, y_orig, m, sigma, K)
    N = size(x_orig,1);
    num = round(N/K);
    accs_train = zeros(K,1);
    accs_val = zeros(K,1);
    MSEs_val = zeros(K,1);

    for i=1:K
        mask_val = logical(zeros(N,1));
        mask_val(num*(i-1)+1:num*i)=1;
        x_val = x_orig(mask_val, :);
        y_val = y_orig(mask_val, :);
        x_train = x_orig(~mask_val, :);
        y_train = y_orig(~mask_val, :);
        [centers, weight, sigma, ~, acc_train] = train_RBF(x_train, y_train, m, sigma);
        [~, MSE_val, acc_val] = test_RBF(x_val, y_val, centers, weight, sigma);
        accs_train(i) = acc_train;
        accs_val(i) = acc_val;
        MSEs_val(i) = MSE_val;
    end
    acc_train_mean = mean(accs_train);
    acc_val_mean = mean(accs_val);
    MSE_val_mean = mean(MSEs_val);
end

function [centers, weight, sigma, MSE_train, acc_train] = train_RBF(x_train, y_train, m, sigma)
    % x_train shape: (N, D)
    % y_train shape: (N, 1)
    % calculate centers
    N = size(y_train,1);
    [~, centers] = kmeans(x_train, m); % centers shape: (m, D)
    % calculate sigma if not given (rough number)
    if isnan(sigma)
        sigma = max(pdist(centers)) / sqrt(2*m);
    end
    % calculate phi
    dists = pdist2(x_train, centers); % dists shape: (N, m)
    phi = exp(-dists.^2./(2*sigma^2)); % phi shape: (N, m)
    phi = [phi, ones(N,1)];  % phi shape: (N, m+1)
    % calculate weight
    weight = pinv(phi'* phi)*phi'*y_train;  % weight shape: (N, 1)
    % train accuracy
    y = phi*weight;
    pred = y;
    pred(pred>0) = 1;
    pred(pred<0) = -1;
    MSE_train = sum((y-y_train).^2)/N;
    acc_train = sum(pred==y_train) / N;
    
end

function [pred, MSE_test, acc_test] = test_RBF(x_test, y_test, centers, weight, sigma)
    N = size(x_test,1);
    dists = pdist2(x_test, centers); % dists shape: (N, m)
    phi = exp(-dists.^2./(2*sigma^2)); % phi shape: (N, m)
    phi = [phi, ones(N,1)];  % phi shape: (N, m+1)
    y = phi*weight;
    pred = y;
    pred(pred>0) = 1;
    pred(pred<0) = -1;
    if isnan(y_test)
        MSE_test = nan;
        acc_test = nan;
    else
        MSE_test = sum((y-y_test).^2)/N;
        acc_test = sum(pred==y_test) / N;
    end
end


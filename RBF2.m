clear
close all

%% Load Data
load('data_test.mat')
load('data_train.mat')
load('label_train.mat')

% fix random seed
rng(3)  % 0-5 mean diviation


%% Traning
% 1. hyper-param: m

METHOD = 3; % 1:kmeans, 2:random, 3:som

m_min = 1;
m_step = 1;
m_max = 150;
if METHOD==3
    ms = [4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169];
else
    ms = m_min+m_step:m_step:m_max;
end
accs_train_tune_m = zeros(length(ms),1);
accs_val_tune_m = zeros(length(ms)/m_step,1);

% MSEs_val_tune_m = zeros((m_max-m_min)/m_step,1);


MSE_val_best_m = nan;
% for i=1:(m_max-m_min)/m_step
for i=1:length(ms)
%     m = m_min+i*m_step;
    m = ms(i)
    sigma = nan;
    K = 5; % 5-fold-validation  
    [MSE_val, accs_train_tune_m(i), accs_val_tune_m(i)] = train_RBF_Kfold(data_train, label_train, m, sigma,METHOD, K);

%     MSEs_val_tune_m(i) = MSE_val;

    if isnan(MSE_val_best_m) || MSE_val <= MSE_val_best_m
        m_best = m;
        MSE_val_best_m = MSE_val;
    end
end
% plot
ms = m_min+m_step:m_step:m_max;
if METHOD==3
    ms = [4, 9, 16, 25, 36, 47, 64, 81, 100, 121, 144, 169];
end
figure
hold on
plot(ms, accs_train_tune_m, 'r', 'LineWidth',2)
plot(ms, accs_val_tune_m, 'b', 'LineWidth',2)
legend('training set', 'validation set','Location','northeast')

% title('RBF-Kmeans performance with different hidden neuron number');
xlabel('number of hidden neurons');
ylabel('classification accuracy');

% plot(ms, MSEs_val_tune_m, 'LineWidth',2)

hold off
fprintf('Best m:\n%d\n', m_best)
% fprintf('Current sigma under m:\n%d\n', sigma_best_1)
fprintf('Best acc with m:\n%f\n', max(accs_val_tune_m))


% 2. hyper-param:sigma
if METHOD==1
    
    s_min = 0;
    s_step = 0.1;
    s_max = 5;
    accs_train_tune_s = zeros((s_max-s_min)/s_step,1);
    accs_val_tune_s = zeros((s_max-s_min)/s_step,1);
    % acc_val_best_s = 0;
    MSE_val_best_s = nan;
    m = m_best;
    for i=1:(s_max-s_min)/s_step
        sigma = s_min+i*s_step;
        K = 5; % 5-fold-validation
        [MSE_val, accs_train_tune_s(i), accs_val_tune_s(i)] = train_RBF_Kfold(data_train, label_train, m, sigma,METHOD, K);
        if isnan(MSE_val_best_s) || MSE_val <= MSE_val_best_s
            sigma_best_2 = sigma;
            MSE_val_best_s = MSE_val;
        end
    end
    % plot
    ss = s_min+s_step:s_step:s_max;
    figure
    hold on
    plot(ss, accs_train_tune_s, 'r', 'LineWidth',2)
    plot(ss, accs_val_tune_s, 'b', 'LineWidth',2)
    legend('training set', 'validation set','Location','northeast')
    title('RBF-Kmeans performance with different sigma');

    xlabel('\sigma');
    ylabel('classification accuracy');
    hold off

    fprintf('Best sigma under m:\n%d\n', sigma_best_2)
    fprintf('Best acc with (sigma, m):\n%f\n', max(accs_val_tune_s))
end

% 3. train on full data with optimal parameters
if METHOD~=1
    sigma_best_2=nan;
end
[c_best, w_best, sigma_best, ~, ~] = train_RBF(data_train, label_train, m_best, sigma_best_2, METHOD);
if METHOD~=1
    sigma_best_2=sigma_best;
end

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
% % x_toDraw = -1:0.001:1;
% % y_toDraw = -1:0.001:1;
% % [a,b] = meshgrid(x_toDraw, y_toDraw);
% dists = pdist2(x_test, centers); % dists shape: (N, m)
%     phi = exp(-dists.^2./(2*sigma^2)); % phi shape: (N, m)
% z_toDraw = 
% z = sigma_best_2
% figure
% plot(x((label_train==1)),y((label_train==1)),'o','LineWidth',2)
% hold on
% plot(x((label_train==-1)),y((label_train==-1)),'x','LineWidth',2)
% 
% xx = c_best(:,show_dim_1);
% yy = c_best(:,show_dim_2);
% plot(xx,yy,'m*','LineWidth',1)
% 
% hold off

% figure
% gscatter(x,y,label_train)
% hold on
% plot(c_best(:,show_dim_1),c_best(:,show_dim_2),'ko','MarkerSize',10)
% legend('class 1','class 2','center vectors','Location','northeast')
% hold off

figure
plot(x((label_train==1)),y((label_train==1)),'r.','MarkerSize',20)
hold on
plot(x((label_train==-1)),y((label_train==-1)),'c.','MarkerSize',20)
plot(c_best(:,show_dim_1),c_best(:,show_dim_2),'ks','MarkerSize',10, 'LineWidth',1)
legend('class 1','class 2','centers','Location','northeast')
hold off

%% Testing
% acc_val = test_RBF(x_val, y_val, centers, weight, sigma);
[pred, ~,~] = test_RBF(data_test, nan, c_best, w_best, sigma_best_2);
fprintf('\nTest set prediction:\n')
fprintf('%d  %d  %d  %d  %d  %d  %d  %d  %d  %d  %d  %d  %d  %d  %d  %d  %d  %d  %d  %d  %d\n\n',pred.');



%% Functions
function [MSE_val_mean, acc_train_mean, acc_val_mean] = train_RBF_Kfold(x_orig, y_orig, m, sigma, method, K)
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
        [centers, weight, sigma, ~, acc_train] = train_RBF(x_train, y_train, m, sigma, method);
        [~, MSE_val, acc_val] = test_RBF(x_val, y_val, centers, weight, sigma);
        accs_train(i) = acc_train;
        accs_val(i) = acc_val;
        MSEs_val(i) = MSE_val;
    end
    acc_train_mean = mean(accs_train);
    acc_val_mean = mean(accs_val);
    MSE_val_mean = mean(MSEs_val);
end

function [centers, weight, sigma, MSE_train, acc_train] = train_RBF(x_train, y_train, m, sigma, method)
    % x_train shape: (N, D)
    % y_train shape: (N, 1)
    % calculate centers
    N = size(y_train,1);
    if method==1 % kmeans
        [~, centers] = kmeans(x_train, m); % centers shape: (m, D)
    elseif method==2 % random
        centers = x_train(randperm(size(x_train, 1), m),:);
    elseif method==3 % som (may have bug)
        a = round(sqrt(m));
        b = a;
        net = selforgmap([a b]);
        net = train(net,x_train.');
%         view(net)
%         c = net(data_train);
%         classes = vec2ind(c);
        weight = net.IW{1,1};
        centers = weight;
        for i=1:size(centers, 2)
            centers(:,i) = -1 + (centers(:,i)-min(centers(:,i)))./(max(centers(:,i))-min(centers(:,i))).*2;
        end
    end
    % calculate sigma if not given (rough number)
    if isnan(sigma)
        sigma = max(pdist(centers)) / sqrt(2*m);
    end
    % calculate phi

    dists = pdist2(x_train, centers); % dists shape: (N, m)
    phi = exp(-dists.^2./(2*sigma^2)); % phi shape: (N, m)
    phi = [phi, ones(N,1)];  % phi shape: (N, m+1)
    % calculate weight
    weight = pinv(phi.'* phi)*phi.'*y_train;  % weight shape: (N, 1)
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


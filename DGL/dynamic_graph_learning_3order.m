%% setup
clc;clear all;
%input Y: n*T
% W = cell(12,20);
% X_hat = cell(12,20);
% LOSS = cell(12,20);
load('neuron_data.mat');
load('ord_mat.mat');
load('grey_data.mat');
load('neuCoord_V1AL.mat');
load('ord_am.mat');
load('RDK_SIG2.mat');
% load('gen_sig10.mat');
WWW = cell(1,5);
% for ii = 6:10
%load('3rd_gen_sig5.mat')
% load(['3rd_gen_sig',num2str(ii),'.mat']);

% some other initializations
N = 5000; % iteration
gamma = 0.05;
eta = 0.05;
eta2 = 0.04;
alpha = 1e-3;
lamda = 1;
tau = 1e-3;
l_r = 1e1; %learning rate
l_r2 = 1e0;
num = 50;
num_l = 36;
K = 30;


w_3d = cell(1,num_l);
for i = 1:num_l
    w_3d{i} = zeros(num,num,num);
end

% time interval
% t_p = floor(427/num_l);
% t_p = 40;
t_p = 44;

% Y_t
q = 3;
Y2 = cell(num_l,1);
sig = SIG{q}(1:1584,ord)';
for i = 1:50
    sig(i,:) = (sig(i,:)-min(sig(i,:),[],'all'))/(max(sig(i,:),[],'all')-min(sig(i,:),[],'all'))';
end
for i = 1:num_l
    %Y2{i} = Y{3*q-2}((t_p*(i-1)+1:t_p*i),ord_am)';
    Y2{i} = sig(:,t_p*(i-1)+1:t_p*i)/5;
end

A = cell(num,num);
trace_mat = zeros(num,num,num_l);

for i = 1:num-1
    for j = i+1:num
        a = zeros(1,num);
        a(i) = 1;
        a(j) = -1;
        A{i,j} = a'*a;
        for k = 1:num_l
            trace_mat(i,j,k) = trace(Y2{k}'*A{i,j}*Y2{k});
        end
    end
end

% initialize w_t and L(w_t)
w = zeros(num*(num-1)/2,num_l);
for i = 1:num_l
    ord_w = randperm(1225);
    w(ord_w(1:K),i) = 1;
end
beta = rand(num*(num-1)/2,num_l-1)*alpha;
z = rand(num*(num-1)/2,num_l-1);
id_m = eye(num); %identity matrix
a_m = zeros(num,num*(num-1)/2);
L_w = zeros(num,num,num_l);
loss = zeros(1,N);
dw_m = zeros(1,N);
% construct vectors for each edge
sum_x_energy = zeros(num,num*(num-1)/2);
for k = 1:num_l
    for i = 1:num
        for j = i+1:num
            index = (j-i)+(i-1)*(100-i)/2;
            a_m(i,index) = 1;
            a_m(j,index) = -1;
            sum_x_energy(i,index) = 1;
            sum_x_energy(j,index) = 1;
            % initialize Laplacian matrix
            L_w(:,:,k) = L_w(:,:,k)+w(index,k)*a_m(:,index)*a_m(:,index)';
        end
    end
end
sum_x_energy = sum_x_energy';

% iterative solution
LOSS = cell(1,20);
X2 = cell(1,20);
W = cell(1,20);
W_3d = cell(1,20);
%for order3 = 1:2:5
order3 = 2;
% for q = 1:20
%     Y2 = cell(num_l,1);
% for i = 1:num_l
%     Y2{i} = Y{3*q-2}((t_p*(i-1)+  1:t_p*i),ord_am)';
% end
    w_3d = cell(1,num_l);
    for i = 1:num_l
        w_3d{i} = zeros(num,num,num);
    end
    fprintf('order = %d, q = %d\n\n',order3,q)
w = zeros(num*(num-1)/2,num_l);
for i = 1:num_l
    ord_w = randperm(1225);
    w(ord_w(1:K),i) = 1;
end
beta = rand(num*(num-1)/2,num_l-1)*alpha;
z = rand(num*(num-1)/2,num_l-1);
id_m = eye(num); %identity matrix
a_m = zeros(num,num*(num-1)/2);
L_w = zeros(num,num,num_l);
loss = zeros(1,N);
dw_m = zeros(1,N);
% construct vectors for each edge
sum_x_energy = zeros(num,num*(num-1)/2);
for k = 1:num_l
    for i = 1:num
        for j = i+1:num
            index = (j-i)+(i-1)*(100-i)/2;
            a_m(i,index) = 1;
            a_m(j,index) = -1;
            sum_x_energy(i,index) = 1;
            sum_x_energy(j,index) = 1;
            % initialize Laplacian matrix
            L_w(:,:,k) = L_w(:,:,k)+w(index,k)*a_m(:,index)*a_m(:,index)';
        end
    end
end
sum_x_energy = sum_x_energy';

for m = 1:N
    % update x_t
    for i = 1:num_l
        temp = zeros(num,num);
        for j = 1:num-2
            for k = j+1:num-1
                for l = k+1:num
                    %temp = temp+w_3d{i}(j,k,l)*(trace_mat(j,k,i)*(A{j,l}+A{k,l})+trace_mat(j,l,i)*(A{j,k}+A{k,l})+trace_mat(k,l,i)*(A{j,k}+A{j,l}));
                    temp = temp+w_3d{i}(j,k,l)*(trace_mat(k,l,i)*trace_mat(j,k,i)*A{j,l}+trace_mat(j,l,i)*trace_mat(j,k,i)*A{k,l}+trace_mat(k,l,i)*trace_mat(j,l,i)*A{j,k});
                end
            end
        end
        D_3d = extract_D(w_3d{i},num);
        X{i} = inv(id_m+gamma*L_w(:,:,i)-eta*(diag(diag(L_w(:,:,i))))+tau*temp-eta2*diag(D_3d))*Y2{i};
    end
    
    % updata w_t amd L(w_t)
    dw = zeros(num*(num-1)/2,num_l);
    L_w = zeros(num,num,num_l);
    
        
    % calculate derivative and updata laplacian with the new weight matrix
    for p = 1:num_l
%         temp_dw1 = diag(gamma*a_m'*X{p}*X{p}'*a_m);
        temp_dw1 = zeros(1225,1);
        for j = 1:1225
            temp_dw1(j) = gamma*a_m(:,j)'*X{p}*X{p}'*a_m(:,j);
        end
        temp_dw2 = eta*sum_x_energy*diag(X{p}*X{p}');
        if p == 1
            dw(:,p) = temp_dw1-temp_dw2+(-beta(:,p));
        elseif p == num_l
            dw(:,p) = temp_dw1-temp_dw2+(beta(:,p-1));
        else
            dw(:,p) = temp_dw1-temp_dw2+(beta(:,p-1)-beta(:,p));
        end
    end
    
    w = w-l_r*dw;
    dw_m(m) = norm(dw,2);
    % project w to w_hat with K constraint
    for i = 1:num_l
        w(:,i) = projection_3(w(:,i),K);
        temp_w = vector2adjacent(w(:,i),50);
        D = diag(sum(temp_w,1));
        L_w(:,:,i) = D-temp_w;
    end
    
    %use proximal method to updata z
    for i = 1:num_l-1
            z(:,i) = proximal_zt(z(:,i),alpha,beta(:,i),lamda);
            %norm_z(i,m) = norm(z(:,i),1);
            %diff_z_w(i,m) = norm(z(:,i)-w(:,i)+w(:,i+1),2);
    end
    
    %update beta
    for i = 1:num_l-1
        beta(:,i) = beta(:,i)+l_r2*(z(:,i)-w(:,i)+w(:,i+1))*alpha;
        %norm_beta(i,m) = norm(beta(:,i),2);
    end
    
    % update w_3d
    for l = 1:num_l
        X_energy = diag(X{l}*X{l}');
        for i = 1:num-2
            for j = i+1:num-1
                for k = j+1:num
                    w_3d{l}(i,j,k) = w_3d{l}(i,j,k)-l_r*tau*(trace_mat(i,j,l)*trace_mat(j,k,l)*trace_mat(i,k,l))+l_r*eta2*(X_energy(i)+X_energy(j)+X_energy(k));
%                     w_3d{l}(i,j,k) = w_3d{l}(i,j,k)-l_r*tau*(trace_mat(i,j,l)*trace_mat(i,k,l)+trace_mat(i,j,l)*trace_mat(j,k,l)+trace_mat(i,k,l)*trace_mat(j,k,l));
                end
            end
        end
        temp_v = threed2oned(w_3d{l},num);
        temp_v = projection_3(temp_v,order3);
        w_3d{l} = oned2threed(temp_v,num);
%         w_3d{l} = projection_3(w_3d{l},2);
%         w_3d{l}(w_3d{l}<0) = 0;
%         w_3d{l}(w_3d{l}>1) = 1;
    end
    
    
    % update trace_mat
    for l = 1:num_l
        for i = 1:num-1
            for j = i+1:num
                trace_mat(i,j,l) = trace(X{l}'*A{i,j}*X{l});
            end
        end
    end
    
    % calculate loss
    for i = 1:num_l
        if i == 1
            loss(m) = norm(Y2{i}-X{i},'fro')+trace(X{i}'*(gamma*L_w(:,:,i))*X{i});
        else
            loss(m) = loss(m)+(norm(Y2{i}-X{i},'fro')+trace(X{i}'*(gamma*L_w(:,:,i))*X{i}))+alpha*norm(w(:,i-1)-w(:,i),1);
        end
        for j = 1:num-2
            for k = j+1:num-1
                for l = k+1:num
                    loss(m) = loss(m)+tau*w_3d{i}(j,k,l)*(trace_mat(i,j,l)*trace_mat(i,k,l)+trace_mat(i,k,l)*trace_mat(j,k,l)+trace_mat(i,j,l)*trace_mat(j,k,l));
                end
            end
        end
    end
    loss(m) = loss(m)/num_l;
    if mod(m,10)==0
        fprintf('iter = %d\nloss = %.4f\n\n',m,loss(m))
    end
end
% LOSS{q} = loss;
% X2{q} = X;
% W{q} = w;
% W_3d{q} = w_3d;
% 
% end
% WWW{ii-5} = w;
% end

%% analyze
clc;clear all
file = load('3order_N_4.mat');
W_3d = file.W_3d;
W = file.W;
coord = file.neuCoord_V1AL;
ord_am = file.ord_am;

N = 4;
num = 8;

w = W{N};
w_3d = W_3d{N};

for i = 1:num
    figure(i),
    v = w(:,i);
    w_2d = vector2adjacent(v,50);
%     v_3d = threed2oned(w_3d{1},50);
    [pos,val] = extract_non_zero_index(w_3d{i},50);

    for j = 1:length(val)
        if val(j) > 0.5
            p = fill(coord(ord_am(pos(:,j)),1),coord(ord_am(pos(:,j)),2),'r');
            p.EdgeColor = 'None';
            hold on;

        end
    end


    G = graph(w_2d);
    p = plot(G);
    p.XData = coord(ord_am,1);
    p.YData = coord(ord_am,2);
    title(['Graph',int2str(i)])
    p.EdgeColor = [0,0,0];
    p.LineWidth = 1;
end

%% plot graph
load('gen_pos.mat');
w1 = w;
w1(w1<0.5) = 0;
w1(w1>0.5) = 1;
for i = 1:num_l
    w_m = vector2adjacent(w1(:,i),50);
    g =  graph(w_m);
    figure(i),p = plot(g);
    p.XData = pos(1,:);
    p.YData = pos(2,:);
end

%% analysis
load('neuCoord.mat')
w(w<0.5) = 0;
w(w>0.5) = 1;

for i = 1:num_l
    figure(i),
    temp_w = vector2adjacent(w(:,i),50);
    g = graph(temp_w);
    p = plot(g);
    hold on;
    p.XData = neuCoord(1,ord);
    p.YData = neuCoord(2,ord);
    [pos,val] = extract_non_zero_index(w_3d{i},50);

    for j = 1:length(val)
        if val(j) > 0.5
            p = fill(neuCoord(1,ord(pos(:,j))),neuCoord(2,ord(pos(:,j))),'r');
            p.EdgeColor = 'None';
            hold on;

        end
    end
end










































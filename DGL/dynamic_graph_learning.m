% This version use partial update method to do the optimization
% This method use less time for the optimization, and hope it may help
% get to the optimal point much faster.
%%
clc;clear all;
%input Y: n*T

W = cell(1,10);
X_hat = cell(1,10);
LOSS = cell(1,10);
load('neuron_data.mat');
load('ord_mat.mat');
load('grey_data.mat');
load('neuCoord_V1AL.mat');
load('ord_am.mat');
load('ord_nm1.mat');
load('AL_info.mat');
load('213_total_T3.mat');
load('ord_am_new.mat');
load('ord_nm1_new.mat');
for q = 1:20
% q = 3;
ord = ord_new(q,:);
mov_num = 3*q-2;
num_l = 8; % number of time interval
t_p = 53;

% Y2_AL = cell(num_l+5,1);
Y2_AL = cell(num_l,1);
Y2_V1 = cell(num_l,1);
%load('160809_Combo3_trace.mat')
Y2 = cell(num_l,1);
% for i = 1:num_l
%     if mod(i,2) == 0
%         Y2{i-1} = Y{3}(:,ord)';
%     else
%         Y2{i+1} = gray_movie{4}(:,ord)';
%     end
% end
% Y2{1} = Y{3}(1:108,ord)';
% Y2{2} = Y{3}(55:162,ord)';
% Y2{3} = Y{3}(109:216,ord)';
% Y2{4} = Y{3}(163:270,ord)';
% Y2{5} = Y{3}(217:324,ord)';
% Y2{6} = Y{3}(271:378,ord)';
% Y2{7} = Y{3}(325:427,ord)';

% Y2{1} = Y{3}(1:107,ord)';
% Y2{2} = Y{3}(108:214,ord)';
% Y2{3} = Y{3}(215:321,ord)';
% Y2{4} = Y{3}(322:427,ord)';
% Y2{5} = gray_movie{4}(:,ord)';
% rand_ord = randperm(8);
% norm_AL = norm(Y_AL{mov_num}(1:426,ord_AL_am(1:25)),2);
norm_V1 = norm(Y{mov_num}(1:426,ord),2);
for i = 1:num_l
%     Y2_AL{i} = Y_AL{mov_num}((t_p*(i-1)+1:t_p*i),ord_AL_am(1:25))';
    Y2_V1{i} = Y{mov_num}((t_p*(i-1)+1:t_p*i),ord)';
%     Y2_AL{i} = Y2_AL{i}/norm_AL;
%     Y2_V1{i} = Y2_V1{i}/norm_V1;
%     Y2{i} = zeros(50,t_p);
%     for q = 1:20
%         Y2{i} = Y2{i}+Y_AL{3*q-2}((t_p*(i-1)+1:t_p*i),ord_am)';
%     %Y2{i} = imresize(Y2{i},[50,5*t_p]);
%     end
%     Y2{i} = Y2{i}/20;
end
% for i = 1:5
%     Y2_AL{num_l+i} = gray_movie_AL{3*q-1}((t_p*(i-1)+1:t_p*i),ord_AL_am(1:25))';
% end
for i = 1:num_l
%     Y2{i} = [Y2_V1{i};Y2_AL{i}];
    Y2{i} = Y2_V1{i};
    %Y2{i} = Y2_AL{i};
end
% for i = 1:num_l
%     Y2{i} = Y{3*q-2}((t_p*(i-1)+1:t_p*i),ord_am)';
% %     Y2{i} = zeros(50,t_p);
% %     for q = 1:20
% %         Y2{i} = Y2{i}+Y_AL{3*q-2}((t_p*(i-1)+1:t_p*i),ord_am)';
% %     %Y2{i} = imresize(Y2{i},[50,5*t_p]);
% %     end
% %     Y2{i} = Y2{i}/20;
% end

% initialization
% min_{X,W} norm(Y-X,2)+trace(gamma*X'*L(w)*X+alpha*norm(w_t-w_{t+1},1))
% zt = w_t-w_{t+1}
X = cell(num_l,1);
R_y = cell(num_l,1);
N = 10000; % iteration
num = 50; % number of neurons
num_e = num*(num-1)/2;
num_update = 8;
K = 30; % number of edges
gamma = 0.05;
eta = 0.05;
alpha = 1e-3;
% alpha = 0;
lamda = 1;
kappa = 1e-4;
l_r = 1e0; %learning rate
l_r2 = 1e-1;

% parameters for watching convergence
size_sgn = zeros(1,num_l);
norm_z = zeros(num_l-1,N);
diff_z_w = zeros(num_l-1,N);
norm_beta = zeros(num_l-1,N);
diff_x_y = zeros(num_l,N);
norm_laplacian = zeros(1,N);
for i = 1:num_l
    size_sgn(i) = size(Y2{i},2);
end

% initialize the weight matrix
w = zeros(num*(num-1)/2,num_l);
for i = 1:num_l
    ord_w = randperm(num_e);
    w(ord_w(1:K),i) = 1;
end
% w = v_total;

% parameters for duality
beta = rand(num*(num-1)/2,num_l-1)*alpha;
z = rand(num*(num-1)/2,num_l-1);
id_m = eye(num); %identity matrix
a_m = zeros(num,num*(num-1)/2);
L_w = zeros(num,num,num_l);
loss = zeros(1,N);
dw_m = zeros(1,N);

% construct vectors for each edge, definition for laplacian matrix
sum_x_energy = zeros(num,num*(num-1)/2);
for k = 1:num_l
    for i = 1:num
        for j = i+1:num
            index = (j-i)+(i-1)*(2*num-i)/2;
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

% % record order of w for adjacent matrix
% m = 1;
% D = zeros(num,num);
% for i = 1:50
%     for j = i+1:50
%         D(i,j) = m;
%         m = m+1;
%     end
% end
% D = D+D';
% D_ord = zeros(49,49);
% for i = 1:50
%     temp = D(i,:);
%     temp(i) = [];
%     D_ord(i,:) = temp;
%     
% end

%maximize_{w} trace(Y'inv(I+gamma*L_w)Y)
temp_m = zeros(num,num,num_l);
for i = 1:num_l
        R_y{i} = Y2{i}*Y2{i}';
end

for m= 1:N
    update_ord = randperm(num_l);
    update_ord = update_ord(1:num_update);
    %calculate (I+gamma*L_w)^(-1)
    for i = 1:num_l
        temp_m(:,:,i) = inv(id_m+gamma*L_w(:,:,i)-eta*(diag(diag(L_w(:,:,i)))));
    end
    
    % calculate X
    for i = 1:num_l
        X{i} = temp_m(:,:,i)*Y2{i};
    end
    
    %initial derivative of W dw and L_w
    dw = zeros(num*(num-1)/2,num_l);
    %L_w = zeros(num,num,num_l);
    
        
    % calculate derivative and updata laplacian with the new weight matrix
    for a = 1:num_update
        p = update_ord(a);
%         temp_dw1 = diag(gamma*a_m'*X{p}*X{p}'*a_m);
        temp_dw1 = zeros(num_e,1);
        for j = 1:num_e
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
        %         for i = 1:num
%             for j = i+1:num
%                 index = (j-i)+(i-1)*(num*2-i)/2;
%                 if p == 1
%                     dw(index,p) = temp_dw1(index)-(temp_dw2(i)+temp_dw2(j))+(-beta(index,p));
%                 elseif p == num_l
%                     dw(index,p) = temp_dw1(index)-(temp_dw2(i)+temp_dw2(j))+(beta(index,p-1));
%                 else
%                     dw(index,p) = temp_dw1(index)-(temp_dw2(i)+temp_dw2(j))+(beta(index,p-1)-beta(index,p));
%                 end
%             end
%         end
    end
    
    %calculate w

    w = w-l_r*dw;
%     w = w-l_r*(dw+2*kappa*(w-v_total));
    dw_m(m) = norm(dw,2);
    
    % project w to w_hat with K constraint
    for a = 1:num_update
        i = update_ord(a);
%         w(:,i) = projection_5(w(:,i),v_total(:,i),K,kappa);
        w(:,i) = projection_3(w(:,i),K);
        L_w(:,:,i) = zeros(num,num);
%         D = zeros(num,num);
%         for j = 1:num
%             D(j,j) = sum(w(D_ord(j,:),i));
%         end
        temp_w = vector2adjacent(w(:,i),num);
        D = diag(sum(temp_w,1));
        L_w(:,:,i) = D-temp_w;
        % calculate new laplacian matrix
%         for j = 1:num*(num-1)/2
%             L_w(:,:,i) = L_w(:,:,i)+w(j,i)*a_m(:,j)*a_m(:,j)';
%         end
    end
    
    %use proximal method to updata z
    for i = 1:num_l-1
        z(:,i) = proximal_zt(z(:,i),alpha,beta(:,i),lamda);
        norm_z(i,m) = norm(z(:,i),1);
        diff_z_w(i,m) = norm(z(:,i)-w(:,i)+w(:,i+1),2);
    end
    
    %update beta
    for i = 1:num_l-1
        beta(:,i) = beta(:,i)+l_r2*(z(:,i)-w(:,i)+w(:,i+1))*alpha;
        norm_beta(i,m) = norm(beta(:,i),2);
    end
    
    %calculate loss
    for i = 1:num_l
        if i == 1
            loss(m) = norm(Y2{i}-X{i},'fro')+trace(X{i}'*(gamma*L_w(:,:,i)-eta*(diag(diag(L_w(:,:,i)))))*X{i});
        else
            loss(m) = loss(m)+(norm(Y2{i}-X{i},'fro')+trace(X{i}'*(gamma*L_w(:,:,i)-eta*(diag(diag(L_w(:,:,i)))))*X{i}))+alpha*norm(w(:,i-1)-w(:,i),1);
        end
        diff_x_y(i,m) = norm(Y2{i}-X{i},'fro');
        norm_laplacian(m) = norm_laplacian(m)+trace(X{i}'*(gamma*L_w(:,:,i)-eta*(diag(diag(L_w(:,:,i)))))*X{i});
    end
%     loss(m) = (loss(m)+norm(kappa*(w-v_total),2))/num_l;
    loss(m) = loss(m)/num_l;
    %print number
    if mod(m,100)==0
        m/100
        loss(m)
    end
end
for i = 1:num_l
%     w(:,i) = projection_5(w(:,i),v_total(:,i),K,kappa);
    w(:,i) = projection_3(w(:,i),K);
end
W{q} = w;
X2 = [];
for i = 1:num_l
    X2 = [X2,X{i}];
end
X_hat{q} = X2;
LOSS{q} = loss;
end
%%
Y3 = [];
X2 = [];
for i = 1:num_l
    Y3 = [Y3,Y2{i}];
    X2 = [X2,X{i}];
end
figure(1),
subplot(4,1,1)
%imshow(mat2gray(Y3))
imagesc(Y3)

subplot(4,1,2)
%imshow(mat2gray(X2))
imagesc(X2)
subplot(4,1,3)
plot(loss)
%imagesc(ans)
%imagesc([Y{7}(1:426,ord_am(1:25))';Y_AL{7}(1:426,ord_AL_am(1:25))'])
axis normal
title('cost')
subplot(4,1,4)
plot(dw_m)
axis normal
title('sum of 2-norm of derivative over w_t')

%%
pat1 = imread('p1.jpg');
pat2 = imread('p2.jpg');
pat3 = imread('p3.jpg');
pat4 = imread('p4.jpg');
pat5 = imread('p5.jpg');
pat6 = imread('p6.jpg');
pat7 = imread('p7.jpg');
pat8 = imread('p8.jpg');
w1 = w;
% w2 = v_total;
w1(w1<0.5) = 0;
% G = cell(num_l,1);
% im = cell(1,num_l);
% TestVideo = VideoWriter('TestVideo3_K_10.avi');
% TestVideo.FrameRate = 1;
% open(TestVideo)
for i = 1:num_l
    %figure(i),
    %figure(i),
%     eval(['G',num2str(i), '= graph(vector2adjacent(w1(:,i),50));'])
%     %eval(['G',num2str(i), '= graph(vector2adja             cent(w1,50));'])
%     eval(['p',num2str(i), '= plot(G',num2str(i),');'])
%     eval(['p',num2str(i),'.XData = neuCoord_V1AL(ord_am,1);'])
%     eval(['p',num2str(i),'.YData = neuCoord_V1AL(ord_am,2);'])
    clf;
    G1 = graph(vector2adjacent(w1(:,i),num));
%     G2 = graph(vector2adjacent(w2(:,i),50));
    subplot(8,8,1:56),p1 = plot(G1);
%    hold on;
%    subplot(8,8,1:56),p2 = plot(G2);
    p1.XData(1:25) = neuCoord_V1AL(ord_am(1:25),1);
    p1.YData(1:25) = neuCoord_V1AL(ord_am(1:25),2);
%     p1.XData(26:50) = neuCoord_V1AL(ord_AL_am(1:25)+590,1);
%     p1.YData(26:50) = neuCoord_V1AL(ord_AL_am(1:25)+590,2);
%     p2.XData(1:25) = neuCoord_V1AL(ord_am(1:25),1);
%     p2.YData(1:25) = neuCoord_V1AL(ord_am(1:25),2);
%     p2.XData(26:50) = neuCoord_V1AL(ord_AL_am(1:25)+590,1);
%     p2.YData(26:50) = neuCoord_V1AL(ord_AL_am(1:25)+590,2);
%     p.XData = neuCoord_V1AL(ord_am,1);
%     p.YData = neuCoord_V1AL(ord_am,2);
    title(['graph ',num2str(i)]);
%     if i<54
%         subplot(8,8,[63,64]),p2 = imshow(pat1);
%     elseif i<107
%         subplot(8,8,[63,64]),p2 = imshow(pat2);
%     elseif i<160
%         subplot(8,8,[63,64]),p2 = imshow(pat3);
%     elseif i<214
%         subplot(8,8,[63,64]),p2 = imshow(pat4);
%     elseif i<267
%         subplot(8,8,[63,64]),p2 = imshow(pat5);
%     elseif i<320
%         subplot(8,8,[63,64]),p2 = imshow(pat6);
%     elseif i<373
%         subplot(8,8,[63,64]),p2 = imshow(pat7);
%     else
%         subplot(8,8,[63,64]),p2 = imshow(pat8);
%     end
%     drawnow
%     
     drawnow
%     frame = getframe(figure(1));
%     writeVideo(TestVideo,frame);
%     im{i} = frame2im(frame);
     pause(0.5)
     
    %eval(['saveas(p',num2str(i),',"graph',num2str(i),'_213interval","jpg");'])
end
close(TestVideo)
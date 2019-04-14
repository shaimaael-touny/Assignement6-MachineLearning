clc
clear all
close all
ds = datastore('house_prices_data_training_data.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',17999);
T=read(ds);
x=T(:,4:21); % training set from column 4 to 21
x=x{:,:}; 
x_cor=corr(x);
x_cov=cov(x);
[U S V]= svd(x_cov);

%normalization
[n m]=size(x);
for w=1:m
    if max(abs(x(:,w)))~=0;
        x(:,w)=(x(:,w)-mean((x(:,w))))./std(x(:,w));
        
    end
end

%to get K
K=0;
alpha=0.5;
while (alpha>=0.001)
    K=K+1;
    lamdas(K,:)=sum(max(S(:,1:K)));
    lamdass=sum(max(S));
    alpha=1-lamdas./lamdass;    
end
R=U(:, 1:K)'*(x)'; 
app_data=U(:,1:K)*R;
error=(1/m)*(sum(app_data-x'));

%linear regression 
Alpha=0.01;
lamda=0.001;
h=1;
Theta=zeros(m,1);
k=1;
Y=T{:,3}/mean(T{:,3});
E(k)=(1/(2*m))*sum((app_data'*Theta-Y).^2); %cost function
while h==1
    Alpha=Alpha*1;
    Theta=Theta-(Alpha/m)*app_data*(app_data'*Theta-Y);
    k=k+1;
    E(k)=(1/(2*m))*sum((app_data'*Theta-Y).^2);
        %Regularization
    Reg(k)=(1/(2*m))*sum((app_data'*Theta-Y).^2)+(lamda/(2*m))*sum(Theta.^2);
    %
    if E(k-1)-E(k)<0;
        break
    end
    q=(E(k-1)-E(k))./E(k-1);
    if q <.001;
        h=0;
    end
end

%running kmeans
% figure()
% plotScree(x,10);
X=x;
for K=1:10
for i=1:10
  centroids = initCentroids(X, K);
  indices = getClosestCentroids(X, centroids);
  centroids = computeCentroids(X, indices, K);
  iterations = 0;
        for ii = 1 :K
            clustering = X(find(indices == ii), :);
            cost = 0;
            for z = 1 : size(clustering,1)
                cost = cost + sum((clustering(z,:) - centroids(ii,:)).^2)/17999;
            end
            costVec(1,K) = cost;
            
        end
end
end       
[ o bestKvalue ] = min(costVec);
noClusters = 1:10;
plot(noClusters, costVec);

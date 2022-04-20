%%%%%%%%%%%%%%%%%%%%%%%数据集可视化%%%%%%%%%%%%%%%%%%%%%%
filename = 'housing.txt';
delimiterIn = ' ';
BO = importdata(filename,delimiterIn);
figure(1);
plotmatrix(BO,'g');
figure(2);
R= corrcoef(BO(:,:));
imagesc(R);
caxis([-1,1]);
colorbar

%%%%%%%%%%%%%%%%%%%%%%BP神经网络预测%%%%%%%%%%%%%%%%%%%%%%
%导入数据到housing
formatSpec = '%8f%7f%8f%3f%8f%8f%7f%8f%4f%7f%7f%7f%7f%f%[^\n\r]'; 
fileID = fopen(filename,'r'); 
dataArray = textscan(fileID, formatSpec, 'Delimiter', '', 'WhiteSpace', '', 'ReturnOnError', false); 
fclose(fileID); 
housing = table(dataArray{1:end-1}, 'VariableNames', {'VarName1','VarName2','VarName3','VarName4','VarName5','VarName6','VarName7','VarName8','VarName9',... 
'VarName10','VarName11','VarName12','VarName13','VarName14'}); 
%清空临时变量
clearvars filename formatSpec fileID dataArray ans; 
%housing加特征名
inputNames = {'CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT'};
outputNames = {'MEDV'};
housingAttributes = [inputNames,outputNames];
housing.Properties.VariableNames = housingAttributes; 
X = housing{:,inputNames}; %X是特征506*3 Y是label 506*1
y = housing{:,outputNames};
len = length(y);
index = randperm(len);%生成1~len 的随机数

%%产生训练集和数据集
%训练集——前70%
p_train = X(index(1:round(len*0.7)),:);%训练样本输入354*13
t_train = y(index(1:round(len*0.7)),:);%训练样本输出354*1
%测试集——后30%
p_test = X(index(round(len*0.7)+1:end),:);%测试样本输入152*13
t_test =y(index(round(len*0.7)+1:end),:);%测试样本输出152*1

%%数据归一化
%输入样本归一化
[pn_train,ps1] = mapminmax(p_train');%pn_train对应p_train 13*354 [-1,1]
pn_test = mapminmax('apply',p_test',ps1);%pn_test对应p_test 13*152 [-1,1]
%输出样本归一化
[tn_train,ps2] = mapminmax(t_train');%tn_train对应t_train 1*354 [-1,1]
%tn_test = mapminmax('apply',t_test',ps2);%不用把测试集输出归一化，因为后文我们要将测试集预测输出反归一化

%网络创建和训练
net = feedforwardnet(5,'trainlm');%创建网络
net.trainParam.epochs = 5000;%设置训练次数
net.trainParam.goal=0.0000001;%设置收敛误差
[net,tr]=train(net,pn_train,tn_train);%训练网络

%网络预测
b=sim(net,pn_test);%b为测试数据的预测值 1*152

%%预测结果反归一化
predict_prices = mapminmax('reverse',b,ps2);

%%结果分析
t_test = t_test';
err_prices = t_test-predict_prices;
%输出BP网络预测结果误差的均值与方差
[mean(err_prices) std(err_prices)]
figure(3);
plot(t_test);
hold on;
%输出预测值与真实值的图线图
plot(predict_prices,'r');
xlim([1 length(t_test)]);
hold off;
legend({'Actual','Predicted'})
xlabel('Training Data point');
ylabel('Median house price');


%%%%%%%%%%%%%%%%%%%%%%线性回归预测%%%%%%%%%%%%%%%%%%%%%%
%采用另一种划分数据集办法，引入随机性
[M,N]= size(BO);
r = randperm(M,450);
%随机抽取450例作为训练集，余下56例作为测试集
training_data= BO(r, :);    %训练集450*14
nr = setdiff( 1:M , r);
test_data= BO(nr,:);        %测试集56*14
trainingx=  training_data(:,1:N-1); %训练集输入450*13
trainingy= training_data(:,N);      %训练集输出450*1
testx=  test_data(:,1:N-1);         %测试集输入56*13
testy= test_data(:,N);              %测试集输出56*1
%用13个特征拟合线性模型
lm = fitlm(trainingx,trainingy,'linear') %生成线性模型lm
figure(5);
plot(lm)
testyhatL= feval(lm,testx);              %线性模型预测输出testyhatL 56*1
testerrorL= (1/56)*sum((testy-testyhatL).^2);%误差为平方和的均值
%挑选第1 2 4 5 6 8 10 11 12 13号特征拟合线性模型
trainingxnew= horzcat(trainingx(:,1:2),trainingx(:,4:6),trainingx(:,8),trainingx(:,10:end));
testxnew= horzcat(testx(:,1:2),testx(:,4:6),testx(:,8),testx(:,10:end));
lmnew = fitlm(trainingxnew,trainingy,'linear');
figure(6);
plot(lmnew)
testyhatLnew= feval(lmnew,testxnew);
testerrorLnew= (1/56)*sum((testy-testyhatLnew).^2);
%改变所得拟合模型所包含的参数
lmQ = fitlm(trainingx,trainingy,'quadratic')
figure(7);
plot(lmQ)
testyhatLQ= feval(lmQ,testx);
testerrorLQ= (1/56)*sum((testy-testyhatLQ).^2);
figure(4)
plot(testy)
hold on
plot(testyhatL,'r')
hold on
plot(testyhatLnew,'g')
hold on
plot(testyhatLQ,'y')
hold off
legend({'testy','testyhatL','testyhatLnew','testyhatLQ'})


%%%%%%%%%%%%%%%%%%%%%%房价分类%%%%%%%%%%%%%%%%%%%%%%
%价值小于15的房屋分类为1(便宜)，15到30之间的为2(中等)，超过30的为3(昂贵)
BOS=BO;
for i=1:506
if BO(i,14)<15
 BOS(i,14)=1;   
elseif BO(i,14)>=15 && BO(i,14)<=30
   BOS(i,14)=2;  
else
   BOS(i,14)=3;  
end
end

training_dataC= BOS(r, :);%训练集450*14 前13个没变，房价从连续值按以上规则改为分类值
nr = setdiff( 1:M , r);
test_dataC= BOS(nr,:);%测试集56*14 前13个没变，房价从连续值按以上规则改为分类值
trainingxC=  training_dataC(:,1:N-1);%训练集输入
trainingyC= training_dataC(:,N);%训练集输出
testxC=  test_dataC(:,1:N-1);%测试集输入
testyC= test_dataC(:,N);%测试集输出（真实值）

mdlknn= fitcknn(trainingxC,trainingyC,'NumNeighbors',5,'Standardize',1);
testyChat= predict(mdlknn, testxC);%预测值
figure(9);
C=confusionmat(testyC,testyChat);
confusionchart(C);
loss= ones(1,10);
kloss=ones(1,10);
for i= 1:10
mdlknn= fitcknn(trainingxC,trainingyC,'NumNeighbors',i,'Standardize',1)
    loss(i) = resubLoss(mdlknn);
CVmdlknn = crossval(mdlknn,'KFold',5);
kloss(i) = kfoldLoss(CVmdlknn);
end
figure(8);
plot(1:10,loss,'r');
hold on
plot(1:10,kloss);
legend({'loss','kloss'})




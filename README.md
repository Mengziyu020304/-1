# end
clc;clear;close all;	
load('R_04_Jun_2025_22_22_55.mat')	
random_seed=G_out_data.random_seed ;  %界面设置的种子数 	
rng(random_seed)  %固定随机数种子 	
	
data_str=G_out_data.data_path_str ;  %读取数据的路径 	
dataO=readtable(data_str,'VariableNamingRule','preserve'); %读取数据 	
data1=dataO(:,2:end);test_data=table2cell(dataO(1,2:end));	
for i=1:length(test_data)	
      if ischar(test_data{1,i})==1	
          index_la(i)=1;     %char类型	
      elseif isnumeric(test_data{1,i})==1	
          index_la(i)=2;     %double类型	
      else	
        index_la(i)=0;     %其他类型	
     end 	
end	
index_char=find(index_la==1);index_double=find(index_la==2);	
 %% 数值类型数据处理	
if length(index_double)>=1	
    data_numshuju=table2array(data1(:,index_double));	
    index_double1=index_double;	
	
    index_double1_index=1:size(data_numshuju,2);	
    data_NAN=(isnan(data_numshuju));    %找列的缺失值	
    num_NAN_ROW=sum(data_NAN);	
    index_NAN=num_NAN_ROW>round(0.2*size(data1,1));	
    index_double1(index_NAN==1)=[]; index_double1_index(index_NAN==1)=[];	
    data_numshuju1=data_numshuju(:,index_double1_index);	
    data_NAN1=(isnan(data_numshuju1));  %找行的缺失值	
    num_NAN__COL=sum(data_NAN1');	
    index_NAN1=num_NAN__COL>0;	
    index_double2_index=1:size(data_numshuju,1);	
    index_double2_index(index_NAN1==1)=[];	
    data_numshuju2=data_numshuju1(index_double2_index,:);	
    index_need_last=index_double1;	
 else	
    index_need_last=[];	
    data_numshuju2=[];	
end	
%% 文本类型数据处理	
	
data_shuju=[];	
 if length(index_char)>=1	
  for j=1:length(index_char)	
    data_get=table2array(data1(index_double2_index,index_char(j)));	
    data_label=unique(data_get);	
    if j==length(index_char)	
       data_label_str=data_label ;	
    end    	
	
     for NN=1:length(data_label)	
            idx = find(ismember(data_get,data_label{NN,1}));  	
            data_shuju(idx,j)=NN; 	
     end	
  end	
 end	
label_all_last=[index_char,index_need_last];	
[~,label_max]=max(label_all_last);	
 if(label_max==length(label_all_last))	
     str_label=0; %标记输出是否字符类型	
     data_all_last=[data_shuju,data_numshuju2];	
     label_all_last=[index_char,index_need_last];	
 else	
    str_label=1;	
    data_all_last=[data_numshuju2,data_shuju];	
    label_all_last=[index_need_last,index_char];     	
 end	
 data=data_all_last;	
 data_biao_all=data1.Properties.VariableNames;	
 for j=1:length(label_all_last)	
    data_biao{1,j}=data_biao_all{1,label_all_last(j)};	
 end	
	
% 异常值检测	
data=data;	
	
%%  特征处理 特征选择或者降维	
	
 A_data1=data;	
 data_biao1=data_biao;	
 select_feature_num=G_out_data.select_feature_num1;   %特征选择的个数	
index_name=data_biao1;	
print_index_name=[]; 	
[B,~] = lasso(A_data1(:,1:end-1),A_data1(:,end),'Alpha',1); 	
L_B=(B~=0);   SL_B=sum(L_B); [~,index_L1]=min(abs(SL_B-select_feature_num)); 	
feature_need_last=find(L_B(:,index_L1)==1);  	
data_select=[A_data1(:,feature_need_last),A_data1(:,end)];	
feature_need_last=find(L_B(:,index_L1)==1);	
	
for NN=1:length(feature_need_last) 	
   print_index_name{1,NN}=index_name{1,feature_need_last(NN)};	
end 	
disp('选择特征');disp(print_index_name)  	
	
	
	
%% 数据划分	
x_feature_label=data_select(:,1:end-1);    %x特征	
y_feature_label=data_select(:,end);          %y标签	
index_label1=randperm(size(x_feature_label,1));	
index_label=G_out_data.spilt_label_data;  % 数据索引	
if isempty(index_label)	
     index_label=index_label1;	
end	
spilt_ri=G_out_data.spilt_rio;  %划分比例 训练集:验证集:测试集	
train_num=round(spilt_ri(1)/(sum(spilt_ri))*size(x_feature_label,1));          %训练集个数	
vaild_num=round((spilt_ri(1)+spilt_ri(2))/(sum(spilt_ri))*size(x_feature_label,1)); %验证集个数	
 %训练集，验证集，测试集	
train_x_feature_label=x_feature_label(index_label(1:train_num),:);	
train_y_feature_label=y_feature_label(index_label(1:train_num),:);	
vaild_x_feature_label=x_feature_label(index_label(train_num+1:vaild_num),:);	
vaild_y_feature_label=y_feature_label(index_label(train_num+1:vaild_num),:);	
test_x_feature_label=x_feature_label(index_label(vaild_num+1:end),:);	
test_y_feature_label=y_feature_label(index_label(vaild_num+1:end),:);	
%Zscore 标准化	
%训练集	
x_mu = mean(train_x_feature_label);  x_sig = std(train_x_feature_label); 	
train_x_feature_label_norm = (train_x_feature_label - x_mu) ./ x_sig;    % 训练数据标准化	
y_mu = mean(train_y_feature_label);  y_sig = std(train_y_feature_label); 	
train_y_feature_label_norm = (train_y_feature_label - y_mu) ./ y_sig;    % 训练数据标准化  	
%验证集	
vaild_x_feature_label_norm = (vaild_x_feature_label - x_mu) ./ x_sig;    %验证数据标准化	
vaild_y_feature_label_norm=(vaild_y_feature_label - y_mu) ./ y_sig;  %验证数据标准化	
%测试集	
test_x_feature_label_norm = (test_x_feature_label - x_mu) ./ x_sig;    % 测试数据标准化	
test_y_feature_label_norm = (test_y_feature_label - y_mu) ./ y_sig;    % 测试数据标准化  	
	
%% 参数设置	
num_pop=G_out_data.num_pop1;   %种群数量	
num_iter=G_out_data.num_iter1;   %种群迭代数	
method_mti=G_out_data.method_mti1;   %优化方法	
BO_iter=G_out_data.BO_iter;   %贝叶斯迭代次数	
min_batchsize=G_out_data.min_batchsize;   %batchsize	
max_epoch=G_out_data.max_epoch1;   %maxepoch	
hidden_size=G_out_data.hidden_size1;   %hidden_size	
attention_label=G_out_data.attention_label;   %注意力机制标签	
attention_head=G_out_data.attention_head;   %注意力机制设置	
	
	
	
%% 算法处理块	
	
	
	
	
disp('随机森林分类') 	
t1=clock; 	
 num_tree=50;   %集成树的棵树	
 [Mdl]  = optimizebaye_fitCTreeBagger(train_x_feature_label_norm,train_y_feature_label,vaild_x_feature_label_norm,vaild_y_feature_label,BO_iter) ;  	
	
	
	
	
y_train_predict=RF_process(predict(Mdl,train_x_feature_label_norm));  %训练集预测结果	
y_vaild_predict=RF_process(predict(Mdl,vaild_x_feature_label_norm));  %验证集预测结果	
y_test_predict=RF_process(predict(Mdl,test_x_feature_label_norm));  %测试集预测结果	
t2=clock;	
 Time=t2(3)*3600*24+t2(4)*3600+t2(5)*60+t2(6)-(t1(3)*3600*24+t1(4)*3600+t1(5)*60+t1(6));       	
	
	
disp(['运行时长: ',num2str(Time)])	
confMat_train = confusionmat(train_y_feature_label,y_train_predict);	
TP_train = diag(confMat_train);      TP_train=TP_train'; % 被正确分类的正样本 True Positives	
FP_train = sum(confMat_train, 1)  - TP_train;  %被错误分类的正样本 False Positives	
FN_train = sum(confMat_train, 2)' - TP_train;  % 被错误分类的负样本 False Negatives	
TN_train = sum(confMat_train(:))  - (TP_train + FP_train + FN_train);  % 被正确分类的负样本 True Negatives	
	
disp('训练集*******************************************************************************')	
accuracy_train = sum(TP_train) / sum(confMat_train(:)); accuracy_train(isnan(accuracy_train))=0; disp(['训练集accuracy：',num2str(mean(accuracy_train))])% Accuracy 	
precision_train = TP_train ./ (TP_train + FP_train); precision_train(isnan(precision_train))=0; disp(['训练集precision_train：',num2str(mean(precision_train))]) % Precision	
recall_train = TP_train ./ (TP_train + FN_train);recall_train(isnan(recall_train))=0; disp(['训练集recall_train：',num2str(mean(recall_train))])  % Recall / Sensitivity	
F1_score_train = 2 * (precision_train .* recall_train) ./ (precision_train + recall_train); F1_score_train(isnan(F1_score_train))=0;  disp(['训练集F1_score_train：',num2str(mean(F1_score_train))])   % F1 Score	
specificity_train = TN_train ./ (TN_train + FP_train); specificity_train(isnan(specificity_train))=0; disp(['训练集specificity_train：',num2str(mean(specificity_train))])  % Specificity	
	
disp('验证集********************************************************************************')	
confMat_vaild = confusionmat(vaild_y_feature_label,y_vaild_predict);	
TP_vaild = diag(confMat_vaild);      TP_vaild=TP_vaild'; % 被正确分类的正样本 True Positives	
FP_vaild = sum(confMat_vaild, 1)  - TP_vaild;  %被错误分类的正样本 False Positives	
FN_vaild = sum(confMat_vaild, 2)' - TP_vaild;  % 被错误分类的负样本 False Negatives	
TN_vaild = sum(confMat_vaild(:))  - (TP_vaild + FP_vaild + FN_vaild);  % 被正确分类的负样本 True Negatives	
accuracy_vaild = sum(TP_vaild) / sum(confMat_vaild(:)); accuracy_vaild(isnan(accuracy_vaild))=0; disp(['验证集accuracy：',num2str(accuracy_vaild)])% Accuracy 	
precision_vaild = TP_vaild ./ (TP_vaild + FP_vaild); precision_vaild(isnan(precision_vaild))=0; disp(['验证集precision_vaild：',num2str(mean(precision_vaild))]) % Precision	
recall_vaild = TP_vaild ./ (TP_vaild + FN_vaild); recall_vaild(isnan(recall_vaild))=0;  disp(['验证集recall_vaild：',num2str(mean(recall_vaild))])  % Recall / Sensitivity	
F1_score_vaild = 2 * (precision_vaild .* recall_vaild) ./ (precision_vaild + recall_vaild);  F1_score_vaild(isnan(F1_score_vaild))=0;  disp(['验证集F1_score_vaild：',num2str(mean(F1_score_vaild))])   % F1 Score	
specificity_vaild = TN_vaild ./ (TN_vaild + FP_vaild); specificity_vaild(isnan(specificity_vaild))=0; disp(['验证集specificity_vaild：',num2str(mean(specificity_vaild))])  % Specificity	
disp('测试集********************************************************************************') 	
confMat_test = confusionmat(test_y_feature_label,y_test_predict);	
TP_test = diag(confMat_test);      TP_test=TP_test'; % 被正确分类的正样本 True Positives	
FP_test = sum(confMat_test, 1)  - TP_test;  %被错误分类的正样本 False Positives	
FN_test = sum(confMat_test, 2)' - TP_test;  % 被错误分类的负样本 False Negatives	
TN_test = sum(confMat_test(:))  - (TP_test + FP_test + FN_test);  % 被正确分类的负样本 True Negatives	
	
accuracy_test = sum(TP_test) / sum(confMat_test(:)); accuracy_test(isnan(accuracy_test))=0; disp(['测试集accuracy：',num2str(accuracy_test)])% Accuracy	
precision_test = TP_test ./ (TP_test + FP_test);  precision_test(isnan(precision_test))=0; disp(['测试集precision_test：',num2str(mean(precision_test))]) % Precision	
recall_test = TP_test ./ (TP_test + FN_test); recall_test(isnan(recall_test))=0; disp(['测试集recall_test：',num2str(mean(recall_test))])  % Recall / Sensitivity	
F1_score_test = 2 * (precision_test .* recall_test) ./ (precision_test + recall_test); F1_score_test(isnan(F1_score_test))=0; disp(['测试集F1_score_test：',num2str(mean(F1_score_test))])   % F1 Score	
specificity_test = TN_test ./ (TN_test + FP_test); specificity_test(isnan(specificity_test))=0; disp(['测试集specificity_test：',num2str(mean(specificity_test))])  % Specificity	
	
	
	
%% 绘制ROC曲线	
[~,score_train]=predict(Mdl,train_x_feature_label_norm);  %训练集预测结果	
	
[~,score_vaild]=predict(Mdl,vaild_x_feature_label_norm);  %验证集预测结果	
	
[~,score_test]=predict(Mdl,test_x_feature_label_norm);  %测试集预测结果	
	
	
	
[X_ROC_train,Y_ROC_train,T_ROC_train,AUC_ROC_train] = perfcurve(train_y_feature_label,score_train(:,1),1);	
rocObj_train = rocmetrics(train_y_feature_label,score_train(:,1),1);	
	
figure	
plot(rocObj_train)	
title('Train ROC')	
%	
[X_ROC_vaild,Y_ROC_vaild,T_ROC_vaild,AUC_ROC_vaild] = perfcurve(vaild_y_feature_label,score_vaild(:,1),1);	
rocObj_vaild = rocmetrics(vaild_y_feature_label,score_vaild(:,1),1);	
	
figure	
plot(rocObj_vaild)	
title('Vaild ROC')	
%	
[X_ROC_test,Y_ROC_test,T_ROC_test,AUC_ROC_test] = perfcurve(test_y_feature_label,score_test(:,1),1);	
rocObj_test = rocmetrics(test_y_feature_label,score_test(:,1),1);	
figure	
plot(rocObj_test)	
title('Test ROC')	
	
	
%% K折验证	
x_feature_label_norm_all=(x_feature_label-x_mu)./x_sig;    %x特征	
y_feature_label_norm_all=y_feature_label;	
Kfold_num=G_out_data.Kfold_num;	
cv = cvpartition(size(x_feature_label_norm_all, 1), 'KFold', Kfold_num); % Split into K folds	
for k = 1:Kfold_num	
    trainingIdx = training(cv, k);	
    validationIdx = test(cv, k);	
     x_feature_label_norm_all_traink=x_feature_label_norm_all(trainingIdx,:);	
   y_feature_label_norm_all_traink=y_feature_label_norm_all(trainingIdx,:);	
	
   x_feature_label_norm_all_testk=x_feature_label_norm_all(validationIdx,:);	
   y_feature_label_norm_all_testk=y_feature_label_norm_all(validationIdx,:);	
	
  Mdlkf=TreeBagger(Mdl.NumTrees ,x_feature_label_norm_all_traink,y_feature_label_norm_all_traink,'Method','classification','MinLeafSize',Mdl.MinLeafSize);	
	
   Mdl_kfold{1,k}=Mdlkf;	
	
    y_test_predict_norm_all_testk=predict(Mdlkf,x_feature_label_norm_all_testk);  %测试集预测结果	
	
    y_test_predict_all_testk=RF_process(y_test_predict_norm_all_testk);	
	
   test_kfold=sum((y_test_predict_all_testk==y_feature_label_norm_all_testk))/length(y_feature_label_norm_all_testk);	
    AUC_kfold(k)=test_kfold;	
	
	
 end	
	
	
% k折验证结果绘图	
figure('color',[1 1 1]);	
	
color_set=[0.4353    0.5137    0.7490];	
plot(1:length(AUC_kfold),AUC_kfold,'--p','color',color_set,'Linewidth',1.3,'MarkerSize',6,'MarkerFaceColor',color_set,'MarkerFaceColor',[0.3,0.4,0.5]);	
grid on;	
box off;	
grid off;	
ylim([0.92*min(AUC_kfold),1.2*max(AUC_kfold)])	
xlabel('kfoldnum')	
ylabel('accuracy')	
xticks(1:length(AUC_kfold))	
set(gca,'Xgrid','off');	
set(gca,'Linewidth',1);	
set(gca,'TickDir', 'out', 'TickLength', [.005 .005], 'XMinorTick', 'off', 'YMinorTick', 'off');	
yline(mean(AUC_kfold),'--')	
%小窗口柱状图的绘制	
axes('Position',[0.6,0.65,0.25,0.25],'box','on'); % 生成子图	
GO = bar(1:length(AUC_kfold),AUC_kfold,1,'EdgeColor','k');	
GO(1).FaceColor = color_set;	
xticks(1:length(AUC_kfold))	
xlabel('kfoldnum')	
ylabel('accuracy')	
disp('****************************************************************************************') 	
disp([num2str(Kfold_num),'折验证预测准确率accuracy结果：'])	
disp(AUC_kfold) 	
disp([num2str(Kfold_num),'折验证  ','accuracy均值为： ' ,num2str(mean(AUC_kfold)),'    accuracy标准差为： ' ,num2str(std(AUC_kfold))]) 	
# class
clc;clear;close all
%% example_2_class_classification1.m
% 主要是用于比较2分类的
data_pre_all=[]; %记录预测数据
x_roc=[];y_roc=[];i=1;
load('训练结果 贝叶斯优化 随机森林分类  23_Apr_11_27_20 train_result_train_vaild_test.mat')
data1=data_Oriny_prey.y_test_predict;data_pre_all=[data_pre_all,data1];
x_roc{1,i}=data_Oriny_prey.X_ROC_test; y_roc{1,i}=data_Oriny_prey.Y_ROC_test; i=i+1;
data_true=data_Oriny_prey.test_y;

load('训练结果 贝叶斯优化 MLP分类  25_Apr_16_06_54 train_result_train_vaild_test.mat')
data6=data_Oriny_prey.y_test_predict;data_pre_all=[data_pre_all,data6];
x_roc{1,i}=data_Oriny_prey.X_ROC_test; y_roc{1,i}=data_Oriny_prey.Y_ROC_test; i=i+1;

load('训练结果 贝叶斯优化 最近邻分类  23_Apr_11_55_50 train_result_train_vaild_test.mat')
data2=data_Oriny_prey.y_test_predict;data_pre_all=[data_pre_all,data2];
x_roc{1,i}=data_Oriny_prey.X_ROC_test; y_roc{1,i}=data_Oriny_prey.Y_ROC_test; i=i+1;

load('训练结果 贝叶斯优化 LSTM分类  23_Apr_12_11_41 train_result_train_vaild_test.mat')
data5=data_Oriny_prey.y_test_predict;data_pre_all=[data_pre_all,data5];
x_roc{1,i}=data_Oriny_prey.X_ROC_test; y_roc{1,i}=data_Oriny_prey.Y_ROC_test; i=i+1;

load('训练结果 贝叶斯优化 CNN分类  23_Apr_13_15_02 train_result_train_vaild_test.mat')
data4=data_Oriny_prey.y_test_predict;data_pre_all=[data_pre_all,data4];
x_roc{1,i}=data_Oriny_prey.X_ROC_test; y_roc{1,i}=data_Oriny_prey.Y_ROC_test; i=i+1;

load('训练结果 贝叶斯优化 朴素贝叶斯分类  23_Apr_11_57_44 train_result_train_vaild_test.mat')
data3=data_Oriny_prey.y_test_predict;data_pre_all=[data_pre_all,data3];
x_roc{1,i}=data_Oriny_prey.X_ROC_test; y_roc{1,i}=data_Oriny_prey.Y_ROC_test; i=i+1;


load('训练结果 贝叶斯优化 决策树分类  23_Apr_13_09_16 train_result_train_vaild_test.mat')
data7=data_Oriny_prey.y_test_predict;data_pre_all=[data_pre_all,data7];
x_roc{1,i}=data_Oriny_prey.X_ROC_test; y_roc{1,i}=data_Oriny_prey.Y_ROC_test; i=i+1; 





%%load('训练结果 贝叶斯优化 SVM分类  23_Apr_13_24_15 train_result_train_vaild_test.mat')
%%data8=data_Oriny_prey.y_test_predict;data_pre_all=[data_pre_all,data8];
%%x_roc{1,i}=data_Oriny_prey.X_ROC_test; y_roc{1,i}=data_Oriny_prey.Y_ROC_test; i=i+1;
%%
str={'随机森林','MLP','最近邻','LSTM','CNN','朴素贝叶斯','决策树'};
%% ROC曲线
color=    [0.1569    0.4706    0.7098
    0.6039    0.7882    0.8588
    0.9725    0.6745    0.5490
    0.8549    0.9373    0.8275   
    0.7451    0.7216    0.8627
    0.7843    0.1412    0.1373
    1.0000    0.5333    0.5176
      0.5569    0.8118    0.7882
       1.0000    0.5333    0.5176];
figure('Position',[200,200,700,400])
for i=1:length(str)
    plot(x_roc{1,i},y_roc{1,i},'Color',color(i,:),"LineWidth",1.5)
    hold on
end
grid on
legend(str,'Location','best')
set(gca,"FontSize",12,"LineWidth",1.5)
box off
legend box off
xlabel('False positive rate')
ylabel('True positive rate')
%% 比较个算法的误差并绘图
Test_all=[];
for j=1:size(data_pre_all,2)
    y_test_predict=data_pre_all(:,j);
    test_y=data_true;
    confMat = confusionmat(test_y,y_test_predict);

    % Calculate metrics
    TP = diag(confMat);      TP=TP'; % 被正确分类的正样本 True Positives
    FP = sum(confMat, 1)  - TP;  %被错误分类的正样本 False Positives
    FN = sum(confMat, 2)' - TP;  % 被错误分类的负样本 False Negatives
    TN = sum(confMat(:))  - (TP + FP + FN);  % 被正确分类的负样本 True Negatives

    accuracy = sum(TP) / sum(confMat(:));  % Accuracy
    precision = mean(TP ./ (TP + FP));  % Precision
    recall = mean(TP ./ (TP + FN));  % Recall / Sensitivity
    F1_score = mean(2 * (precision .* recall) ./ (precision + recall));  % F1 Score
    specificity = mean(TN ./ (TN + FP));  % Specificity

 Test_all=[Test_all;accuracy precision recall F1_score  specificity];
end
%%
% str={'真实值','INFO加权向量均值优化CNN','CGO混沌博弈优化随机森林','RIME霜冰优化XGBoost' ,'CNN-GRU-SE注意力机制','KOA开普勒优化CBiLSTM-att-RF'};
str1=str;
str2={'accuracy','precision','recall','F1 score','specificity'};
data_out=array2table(Test_all);
data_out.Properties.VariableNames=str2;
data_out.Properties.RowNames=str1;
disp(data_out)
%% 柱状图 MAE MAPE RMSE 柱状图适合量纲差别不大的
% color=[
% 0.4431    0.4471    0.3765
%     0.6588    0.6392    0.5216
%     0.8745    0.8275    0.6706
%     0.7294    0.7843    0.4902
%     0.5961    0.7412    0.3255];

color=    [0.1569    0.4706    0.7098
    0.6039    0.7882    0.8588
    0.9725    0.6745    0.5490
    0.8549    0.9373    0.8275   
    0.7451    0.7216    0.8627
    0.7843    0.1412    0.1373
    1.0000    0.5333    0.5176
      0.5569    0.8118    0.7882
       1.0000    0.5333    0.5176];

figure('Units', 'pixels', ...
    'Position', [300 300 660 375]);
plot_data_t=Test_all(:,:)';
b=bar(plot_data_t,0.8);
hold on
x_data=[];
for i = 1 : size(plot_data_t,2)
    x_data(:, i) = b(i).XEndPoints'; 
end

for i =1:size(plot_data_t,2)
b(i).FaceColor = color(i,:);
b(i).EdgeColor=[0.6353    0.6314    0.6431];
b(i).LineWidth=1.2;
end

for i = 1 : size(plot_data_t,1)-1
    xilnk=(x_data(i, end)+ x_data(i+1, 1))/2;
    b1=xline(xilnk,'--','LineWidth',1.2);
    hold on
end 
ylim([min(min(plot_data_t))-0.1,1])
ax=gca;
legend(b,str1,'Location','best')
ax.XTickLabels =str2;
set(gca,"FontSize",12,"LineWidth",2)
box off
legend box off
%% 横向柱状图


figure('Units', 'pixels', ...
    'Position', [200 200 460 575]);
plot_data_t=Test_all(:,:)';
b=barh(plot_data_t,0.8);
hold on
x_data=[];
for i = 1 : size(plot_data_t,2)
    x_data(:, i) = b(i).XEndPoints'; 
end

for i =1:size(plot_data_t,2)
b(i).FaceColor = color(i,:);
b(i).EdgeColor=[0.6353    0.6314    0.6431];
b(i).LineWidth=1.2;
end

for i = 1 : size(plot_data_t,1)-1
    xilnk=(x_data(i, end)+ x_data(i+1, 1))/2;
    b1=yline(xilnk,'--','LineWidth',1.2);
    hold on
end 
xlim([min(min(plot_data_t))-0.1,1])
ax=gca;
legend(b,str1,'Location','best')
ax.YTickLabels =str2;
set(gca,"FontSize",12,"LineWidth",2)
box off
legend box off
%% 三维图
   % color=[0.3529    0.3725    0.6392
   %  0.5647    0.5961    0.7882
   %  0.7529    0.7569    0.8745
   %  0.8824    0.8235    0.8980
   %  0.9098    0.7922    0.8863];
% color=[
% 0.4431    0.4471    0.3765
%     0.6588    0.6392    0.5216
%     0.8745    0.8275    0.6706
%     0.7294    0.7843    0.4902
%     0.5961    0.7412    0.3255];

figure('Units', 'pixels', ...
    'Position', [200 200 560 575]);
plot_data_t=Test_all(:,:)';
b=bar3(plot_data_t,0.8);
hold on

for i =1:size(plot_data_t,2)
b(i).FaceColor = color(i,:);
b(i).EdgeColor=[0.7    0.7    0.7];
b(i).LineWidth=1.2;
end

zlim([min(min(plot_data_t))-0.1,1])
ax=gca;
ax.YTickLabels =str2;
ax.XTickLabels =str1;
set(gca,"FontSize",12,"LineWidth",2)
box off

%% 二维图
figure
plot_data_t1=Test_all(:,[3,5])';
MarkerType={'s','o','pentagram','^','v','s','o','pentagram','^','v'};
for i = 1 : size(plot_data_t1,2)
   scatter(plot_data_t1(1,i),plot_data_t1(2,i),120,MarkerType{i},"filled")
   hold on
end
set(gca,"FontSize",12,"LineWidth",2)
box off
legend box off
legend(str1,'Location','best')
xlabel('precision')
ylabel('specificity')
grid on


%% 雷达图
figure('Units', 'pixels', ...
    'Position', [150 150 520 500]);
Test_all1=Test_all./sum(Test_all);  %把各个指标归一化到一个量纲
Test_all1(:,end)=1-Test_all(:,end);
RC=radarChart(Test_all1);
% str3={'A-MAE','A-MAPE','A-MSE','A-RMSE','1-R2'};
RC.PropName=str2;
RC.ClassName=str1;
RC=RC.draw(); 
RC.legend();
load('color_list.mat')
colorList=color_list(randperm(150,length(str)),:);
% colorList=[78 101 155;
%           138 140 191;
%           184 168 207;
%           231 188 198;
%           253 207 158;
%           239 164 132;
%           182 118 108]./255;

colorList=[0.960784314000000	0.772549020000000	0.784313725000000
0.878431373000000	0.713725490000000	0.239215686000000
0.325490196000000	0.733333333000000	0.745098039000000
0.631372549000000	0.698039216000000	0.800000000000000
0	0.639215686000000	0.545098039000000
0.3529    0.3725    0.6392
    0.5647    0.5961    0.7882
    0.7529    0.7569    0.8745
    0.8824    0.8235    0.8980
    0.9098    0.7922    0.8863];


for n=1:RC.ClassNum
    RC.setPatchN(n,'Color',colorList(n,:),'MarkerFaceColor',colorList(n,:));
end

%% 罗盘图
figure('Units', 'pixels', ...
    'Position', [150 150 920 600]);
t = tiledlayout('flow','TileSpacing','compact');
for i=1:length(Test_all(1,:))
    nexttile
    th1 = linspace(2*pi/length(Test_all(:,1))/2,2*pi-2*pi/length(Test_all(:,1))/2,length(Test_all(:,1)));
    r1 = Test_all(:,i)';
    [u1,v1] = pol2cart(th1,r1);
    M=compass(u1,v1);
    for j=1:length(Test_all(:,1))
        M(j).LineWidth = 2;
        M(j).Color = colorList(j,:);

    end
    title(str2{i})
    set(gca,"FontSize",10,"LineWidth",1)
end
legend(M,str1,"FontSize",10,"LineWidth",1,'Box','off','Location','best')

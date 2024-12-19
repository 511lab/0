clc
clear all;
% datasets_name={'Cmc','Breast','breath-cancer','Ilpd','Australian','YaleB','ORL','FERET32x32','Palm','AR','German','Pima','Glass','Diabetes','Mpeg7uni'};
datasets_name={'Solar'}
% datasets_name={'Australian','Breast','breath-cancer','BUPA','Cars','Cleve','Cmc','Diabetes','German','Haberman','Ilpd','Pima','Solar','YaleB'}
ssc_name={'STDEMB'};%'STDEMB','xiaorong3'
% ssc_name={'STDEMB'};
for ds=1:length(datasets_name)
    dataset=datasets_name{ds};
    file=strcat('D:\Users\Documents\MATLAB\ZMW\datasets\',dataset,'.mat');
    load(file);
    % X=XX;
    % Y=YY;
    for cf=1:length(ssc_name)
        sscf=ssc_name{cf};
        result=[];
        filename=strcat('noise1124\',dataset,"_",sscf,"_(10%).xlsx");
        % if exist(filename,'file')
        %     disp("跳过");
        %     continue;
        % end
        class=length(unique(Y));  
        
        for asp=1:10
            measure=[];
            for count = 1:30%求10次的平均值作为实验结果，
                disp(asp);
                disp(count);
                [y1,YL,idx_l]=Random_sampling(Y,10,'all');%应该分层抽样
                [yt,~,idx_Ul]=noise_samples(YL,asp,'all',y1,idx_l);
                t0=cputime;
                if strcmp(sscf,"SETRED")
                    theta=0.1;
                    iters=50;
                    %训练
                    [lbl,mdl]=SETRED(X,yt,theta,iters);
                    %测试
                    pred_lbl=predict(mdl,X);
                elseif strcmp(sscf,"STDP")
                    alpha=2;
                    %训练
                    [lbl,mdl]=STDP(X,yt,alpha);
                    %测试
                    pred_lbl=predict(mdl,X);
                elseif strcmp(sscf,'ENN3' )
                    [ES,YS]=ENN3(X,yt,idx_Ul);
                    mdl=fitcknn(ES,YS,'NumNeighbors', 3, 'Distance','euclidean');
                    pred_lbl=predict(mdl,X);
                elseif strcmp(sscf,"LSEdit")
                        [n,~]=size(X);
                        %整合数据集
                        % X_hy=[];
                        idx_L=idx_Ul';
                        % idx_L=find(Y>-1); %有标签样本的索引
                        idx_X1=[1:n];
                        idx_X=idx_X1';
                        idx_L=idx_Ul';
                        X_L=X(idx_L,:);%根据索引获取样本
                        Y_L=Y(idx_L,:);
                        idx_U=setdiff(idx_X,idx_L);
                        XU=X(idx_U,:);
                        Y_U=Y(idx_U,:);  
                        mdl=fitcknn(X_L,Y_L,'NumNeighbors', 3, 'Distance','euclidean');
%                         mdl= fitctree(X_L,Y_L);
                        pred_lbl = predict(mdl,XU);

                        idxLU=[idx_L;idx_U];
                        idx_LU=sort( idxLU);
                        Y_LU=Y( idx_LU,:);
                        for l=1:length(idx_U)
                            Y_LU(idx_U(l))=pred_lbl(l);
                        end   
                        [ES,YS]=LSEdit(X,Y_LU);
                         mdl=fitcknn(ES,YS,'NumNeighbors', 3, 'Distance','euclidean');
%                         mdl= fitctree(ES,YS);
                        %测试
                        pred_lbl=predict(mdl,X);
                    elseif strcmp(sscf,"DE")

                        K=3;
                        k=2;
                        [n,~]=size(X);
                        %整合数据集
                        % X_hy=[];
                        idx_L=idx_Ul';
                        % idx_L=find(Y>-1); %有标签样本的索引
                        X_L=X(idx_L,:);%根据索引获取样本
                        Y_L=yt(idx_L,:);
                        idx_U=find(yt==-1);
                        XU=X(idx_U,:);
                        Y_U=yt(idx_U,:);
%                         mdl= fitctree(X_L,Y_L);
                        mdl=fitcknn(X_L,Y_L,'NumNeighbors', 3, 'Distance','euclidean');
                        pred_lbl = predict(mdl,XU);
                        idxLU=[idx_L;idx_U];
                        idx_LU=sort( idxLU);
                        Y_LU=Y( idx_LU,:);
                        for l=1:length(idx_U)
                            Y_LU(idx_U(l))=pred_lbl(l);
                        end
                        %训练
                        [newX,newY]=DE(X,Y_LU,K,k);
                        mdl=fitcknn(newX,newY,'NumNeighbors', 3, 'Distance','euclidean');
%                         mdl= fitctree(newX,newY);
                        %测试
                        pred_lbl=predict(mdl,X);
                elseif strcmp(sscf,"ENN")


                    %训练
                    [ES,YS]=ENN(X,yt,idx_Ul);
                    mdl = ClassificationKNN.fit(ES,YS,'NumNeighbors',1);

                    %测试
                    pred_lbl=predict(mdl,X);
                elseif strcmp(sscf,"STDPCEW")
                    alpha=2;
                    theta=0.1;
                    %训练
                    [lbl,mdl]=STDPCEW(X,yt,alpha,theta);
                    %测试
                    pred_lbl=predict(mdl,X);
                elseif strcmp(sscf,"SMNF")

                    %训练
                    Dc=0.5;

                    %                     [lbl, mdl] = COSA(X,yt,idx_Ul,pct);
                    [mdl]=SMNF(X,yt,Dc);
                    %                     [lbl, mdl] = BE(X,yt,idx_Ul);
                    %测试
                    pred_lbl=predict(mdl,X);
                elseif strcmp(sscf,"xiaorong3")
                    %训练
                    [lbl,mdl] = xiaorong3(X,yt);
                    %测试
                    pred_lbl=predict(mdl,X);
                elseif strcmp(sscf,"STDPNaN")
                    %训练
                    [lbl,mdl] = STDPNaN(X,yt);
                    %测试
                    pred_lbl=predict(mdl,X);
                elseif strcmp(sscf,"STSFCM")
                    eps1=0.5;
                    eps2=0.4;
                    N=10;
                    iters=200;
                    %训练
                    [lbl,svm_mdl]=STSFCM(X,yt,eps1,eps2,N,iters);
                    %测试
                    [pred_lbl,~,prob] = svmpredict(Y,X,svm_mdl,'-b 1');

                elseif strcmp(sscf,"STDPNF")
                    idx_L=idx_Ul;
                    label_x_t=yt(idx_L,:);
                    label_x=X(idx_L,:);
                    unlabel_x=X(find(yt==-1),:);
                    Dc=2;
                    [~,~,mdl]=STDPNF(label_x,label_x_t,unlabel_x,Dc);
                    %测试
                    pred_lbl=predict(mdl,X);
                elseif strcmp(sscf,"EBSA")

                    %训练
                    [lbl, mdl] = EBSA(X,yt,idx_Ul);
                    %测试
                    pred_lbl=predict(mdl,X);
                    % pred_lbl=svmpredict(Y,X,mdl,'-b 1');
                  elseif strcmp(sscf,"STLN2")
                    %训练
                    [lbl,mdl] = STLN2(X,yt);
                    %测试
                    pred_lbl=predict(mdl,X);
                 elseif strcmp(sscf,'STDEMB' )
                    [~,mdl] = STDEMB(X,yt,6,0.6);
                    %测试
                    pred_lbl=predict(mdl,X);
                 elseif strcmp(sscf,'ST_SFCM')
                            idx_L=idx_Ul;
                            label_x_t=yt(idx_L,:);
                            label_x=X(idx_L,:);
                            unlabel_x=X(find(yt==-1),:);
                            p=1/length(unique(Y))
                            [mdl] =ST_SFCM(label_x, label_x_t,unlabel_x,p);
                            pred_lbl=predict(mdl,X);   
                  elseif strcmp(sscf,'ST_OPF')
                        idx_L=idx_Ul;
                        label_x_t=yt(idx_L,:);
                        label_x=X(idx_L,:);
                        unlabel_x=X(find(yt==-1),:);
                        [mdl] =ST_OPF(label_x, label_x_t,unlabel_x);
                        pred_lbl=predict(mdl,X);   

                end
                timep=cputime-t0;%程序运行时间
                [out] = classification_evaluation(Y',pred_lbl');
                measure=[measure;out.avgAccuracy,out.fscoreMicro,timep];%out.fscoreMacro,out.fscoreMicro
            end
            mean_measure=mean(measure,1);
            std_measure=std(measure(:,1:2),1);
% %             var_measure=var(measure,1);
% %             result=[result;mean_measure,var_measure,rslpct];
%             var_measure=var(measure(:,1:2),1);
%             sqrt_measure=sqrt(var_measure);
            result=[result;mean_measure,std_measure,asp]; %#ok<AGROW>
        end
                %             result=[result;mean_measure,rslpct];
        xlswrite(filename,result)
    end
end
disp("参数已保存！")
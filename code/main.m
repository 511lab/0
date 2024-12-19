clc
clear all;
% load heart_scale;
% datasets_name={'Zoo','Yeast','autro_uni','Balance','Banknote','Cars','chess_uni','Cleve','crx_uni','Dermatology','Ecoli','Haberman','Ionosphere','Isolet','MSRA25','PIE32x32','UPS'};% 
% datasets_name={'Cmc','Breast','breath-cancer','Ilpd','Australian','YaleB','ORL','FERET32x32','Palm','AR','German','Pima','Glass','Diabetes','Mpeg7uni'};%'Transfusion',''Transfusion'''Cars'
% datasets_name={'breath-cancer'}
  datasets_name={'Australian','Breast','breath-cancer','BUPA','Cars','Cleve','Cmc','Diabetes','German','Haberman','Ilpd','Pima','Solar','YaleB'}
% datasets_name={'Cmc'}
% ssc_name={'STLN'};
%'Yeast','Yale','YaleB'  ,'Wine','Segment','Palm','ORL','Isolet','FERET32x32','Waveform','Vehicle','UPS','SPECTFheart','Sonar','Solar'
ssc_name={'STTSC'};% 'xiaorong3','SETRED','STDPCEW','STDPNaN','STDEMB','STDP','STDPNaN','STDP','ST_OPF',
for ds=1:length(datasets_name)%dataset_name
    dataset=datasets_name{ds};
    file=strcat('D:\Users\Documents\MATLAB\ZMW\datasets\',dataset,'.mat');
    load(file);
%     [~,d_x]=size(X);
%     if d_x>8
%         %����10ά
%         X=pca(X,10);
%     end
    % X=XX;
    % Y=YY;
    for cf=1:length(ssc_name)
        sscf=ssc_name{cf};
        % ��Ž��
        result=[];
        % unique(Y)ȥ�� 
        class=length(unique(Y));
        filename=strcat('D:\Users\Documents\MATLAB\ZMW\STTSC\acc\chushihuafanwei\0.25\',dataset,"_",sscf,"1.xlsx");
        % if exist(filename,'file')
        %     disp("����");
        %     continue;
        % end
        %��ʼ�б����������
        for rslpct=2:2:20
            % ���acc,F-score,temep
            measure=[];
            for count =1:100 %��5�ε�ƽ��ֵ��Ϊʵ������
                disp(rslpct);
                disp(count);
                [yt,~,idx_Ul]=Random_sampling(Y,rslpct,'all');%Ӧ�÷ֲ����
                t0=cputime;
                if strcmp(sscf,"SETRED")
                    theta=0.1;
                    iters=50;
                    %ѵ��
                    [lbl,mdl]=SETRED(X,yt,theta,iters);
                    %����
                    pred_lbl=predict(mdl,X);
                elseif strcmp(sscf,"STDP")
                    %ѵ��
                    [~,mdl]=STDP(X,yt,2);
                    pred_lbl=predict(mdl,X);
                elseif strcmp(sscf,"LSEdit")
                    [n,~]=size(X);
                    %�������ݼ�
                    % X_hy=[];
                    idx_L=idx_Ul';
                    % idx_L=find(Y>-1); %�б�ǩ����������
                    idx_X1=[1:n];
                    idx_X=idx_X1';
                    idx_L=idx_Ul';
                    X_L=X(idx_L,:);%����������ȡ����
                    Y_L=Y(idx_L,:);
                    idx_U=setdiff(idx_X,idx_L);
                    XU=X(idx_U,:);
                    Y_U=Y(idx_U,:);  
                    mdl=fitcknn(X_L,Y_L,'NumNeighbors', 3, 'Distance','euclidean');
                    pred_lbl = predict(mdl,XU);

                    idxLU=[idx_L;idx_U];
                    idx_LU=sort( idxLU);
                    Y_LU=Y( idx_LU,:);
                    for l=1:length(idx_U)
                        Y_LU(idx_U(l))=pred_lbl(l);
                    end   
                    [ES,YS]=LSEdit(X,Y_LU);
                    mdl=fitcknn(ES,YS,'NumNeighbors', 3, 'Distance','euclidean');
                    mdl= fitctree(ES,YS);
                    %����
                    pred_lbl=predict(mdl,X);
                elseif strcmp(sscf,"DE")
                    K=3;
                    k=2;
                    [n,~]=size(X);
                    %�������ݼ�
                    % X_hy=[];
                    idx_L=idx_Ul';
                    % idx_L=find(Y>-1); %�б�ǩ����������
                    idx_X1=[1:n];
                    idx_X=idx_X1';
                    idx_L=idx_Ul';
                    X_L=X(idx_L,:);%����������ȡ����
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
                    %ѵ��
                    [newX,newY]=DE(X,Y_LU,K,k);
                    mdl=fitcknn(newX,newY,'NumNeighbors', 3, 'Distance','euclidean');
%                         mdl= fitctree(newX,newY);
                    %����
                    pred_lbl=predict(mdl,X);
                elseif strcmp(sscf,"STDPCEW")
                    alpha=2;
                    theta=0.1;
                    %ѵ��
                    [lbl,mdl]=STDPCEW(X,yt,alpha,theta);
                    %����
                    pred_lbl=predict(mdl,X);
                elseif strcmp(sscf,"MLSTE")
                    k=10;
                    iters=200;
                    %ѵ��
                    [XL,YL,Prior,PriorN,Cond,CondN]=MLSTE(X,yt,k,idx_Ul,iters);
                    %����
                    [~,~,~,~,~,Outputs,pred_lbl]=MLKNN_test(XL,YL,X,Y,1,Prior,PriorN,Cond,CondN);
                elseif strcmp(sscf,"STDPNaN")
                    %ѵ��
                    [lbl,mdl] = STDPNaN(X,yt);
                    %����
                    pred_lbl=predict(mdl,X);
               elseif strcmp(sscf,"STLN")
                    %ѵ��
                    [lbl,mdl] = STLN(X,yt);
                    %����
                    pred_lbl=predict(mdl,X);
                elseif strcmp(sscf,"STLN1")
                    %ѵ��
                    [lbl,mdl] = STLN1(X,yt);
                    %����
                    pred_lbl=predict(mdl,X);
                elseif strcmp(sscf,"STLN2")
                    %ѵ��
                    [lbl,mdl] = STLN2(X,yt);
                    %����
                    pred_lbl=predict(mdl,X);
                elseif strcmp(sscf,"STTSC")
                    %ѵ��
                    [lbl,mdl] = STTSC(X,yt);
                    %����
                    [pred_lbl]=predict(mdl,X);
                 elseif strcmp(sscf,"LTSC")
                    %ѵ��
                    [lbl,mdl] = LTSC(X,yt);
                    %����
                    pred_lbl=predict(mdl,X);
                 elseif strcmp(sscf,"xiaorong33")
                    %ѵ��
                    [lbl,mdl] = xiaorong33(X,yt);
                    %����
                    pred_lbl=predict(mdl,X);
                elseif strcmp(sscf,"SMNF")
                    %ѵ��
                    Dc=0.5;
                    % [lbl, mdl] = COSA(X,yt,idx_Ul,pct);
                    [mdl]=SMNF(X,yt,Dc);
                    % [lbl, mdl] = BE(X,yt,idx_Ul);
                    %����
                    pred_lbl=predict(mdl,X);
                    % pred_lbl=svmpredict(Y,X,mdl,'-b 1');
                elseif strcmp(sscf,"ELS")
                    [n,~]=size(X);
                    %�������ݼ�
                    % X_hy=[];
                    idx_L=idx_Ul';
                    % idx_L=find(Y>-1); %�б�ǩ����������
                    idx_X1=[1:n];
                    idx_X=idx_X1';
                    idx_L=idx_Ul';
                    X_L=X(idx_L,:);%����������ȡ����
                    Y_L=Y(idx_L,:);
                    idx_U=setdiff(idx_X,idx_L);
                    XU=X(idx_U,:);
                    Y_U=Y(idx_U,:);
                    mdl= fitctree(X_L,Y_L);
                    pred_lbl = predict(mdl,XU);
                    idxLU=[idx_L;idx_U];
                    idx_LU=sort( idxLU);
                    Y_LU=Y( idx_LU,:);
                    for l=1:length(idx_U)
                        Y_LU(idx_U(l))=pred_lbl(l);
                    end
                    [ES,YS]=ELS(X,Y_LU);
                    mdl=fitcknn(ES,YS,'NumNeighbors', 3, 'Distance','euclidean');
                    %����
                    pred_lbl=predict(mdl,X);
                  elseif strcmp(sscf,'ENN3' )
                    %ѵ��
                    [ES,YS]=ENN3(X,yt,idx_Ul);
                    mdl=fitcknn(ES,YS,'NumNeighbors', 3, 'Distance','euclidean');
                    %����
                    pred_lbl=predict(mdl,X);
                  % elseif strcmp(sscf,'STDEMB' )
                  %   [~,mdl] = STDEMB(X,yt,0.5);
                  %   %����
                  % 
                  %   pred_lbl=predict(mdl,X);
                  elseif strcmp(sscf,'STDEMB' )
                   [~,mdl] = STDEMB(X,yt,6,0.6);
                    %����
    
                    pred_lbl=predict(mdl,X);
                  % elseif strcmp(sscf,'STDPNF' )
                  %   mdl = STDPNF(X,yt,0.5);
                  %   %����
                  %   pred_lbl=predict(mdl,X);
                  % 
                  elseif strcmp(sscf,'STDPNF' )
                        idx_L=idx_Ul;
                        label_x_t=yt(idx_L,:);
                        label_x=X(idx_L,:);
                        unlabel_x=X(find(yt==-1),:);
                        [~,~,mdl] =STDPNF(label_x, label_x_t,unlabel_x,2);
                         pred_lbl=predict(mdl,X);  
                  elseif strcmp(sscf,'massSTDP' )
                    [~,mdl] = massSTDP(X,yt,0.5);
                    %����
                   
                    pred_lbl=predict(mdl,X);
                    elseif strcmp(sscf,'ESTDEMB' )
                    [~,mdl] = ESTDEMB(X,yt,0.5);
                    %����
 
                    pred_lbl=predict(mdl,X);
                 elseif strcmp(sscf,'ST_SFCM')
                            idx_L=idx_Ul;
                            label_x_t=yt(idx_L,:);
                            label_x=X(idx_L,:);
                            unlabel_x=X(find(yt==-1),:);
                            p=1/length(unique(Y));
                            [mdl] =ST_SFCM(label_x, label_x_t,unlabel_x,p);
                            pred_lbl=predict(mdl,X);   
                  elseif strcmp(sscf,'ST_OPF')
                        idx_L=idx_Ul;
                        label_x_t=yt(idx_L,:);
                        label_x=X(idx_L,:);
                        unlabel_x=X(find(yt==-1),:);
                        [mdl] =ST_OPF(label_x, label_x_t,unlabel_x);
                         pred_lbl=predict(mdl,X);
                  elseif strcmp(sscf,'1NN')
                        idx_L=idx_Ul;
                        mdl=fitcknn(X(idx_L,:),Y(idx_L,:),'NumNeighbors',1, 'Distance','euclidean');
                         pred_lbl=predict(mdl,X);
                
              
                end

                
                

                timep=cputime-t0;%��������ʱ��
                [out] = classification_evaluation(Y',pred_lbl');
                measure=[measure;out.avgAccuracy,out.fscoreMicro,timep];%out.fscoreMacro,out.fscoreMicro
            end
            % ��measure��ƽ��ֵ
            mean_measure=mean(measure,1);
            % ��ȡmeasureǰ���У��󷽲�
            std_measure=std(measure(:,1:2),1);
            % result�ϲ�
            result=[result;mean_measure,std_measure,rslpct];
%             if exist(filename,'file')%������ɾ��
%                 delete(filename)
%             end
        end
        xlswrite(filename,result);
    end
end
disp("�����ѱ��棡")
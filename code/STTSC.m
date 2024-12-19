function [Y,mdl] = STTSC(X,Y)  
    D=pdist2(X,X,'Euclidean'); 
    idx_L=find(Y>-1);
    class=tabulate(Y(idx_L));
    class=class(:,1);
    cluster=length(unique(class));
    % DD=0
    while true
        idx_U=find(Y==-1);
        XL=X(idx_L,:);%根据索引获取样本
        YL=Y(idx_L,:);
        XU=X(idx_U,:);
        mdl=fitcknn(XL, YL,'NumNeighbors',3,'Distance','euclidean'); 
        %预测无标记近亲结点标签
        Y_UL=predict(mdl,XU);
        S=[];
        for v=1:length(idx_U)
            r=mean(D(idx_U(v),idx_L));
            % flag=zeros(cluster,1);

            prevLN1 = zeros(1,1); 
            aaaaa = 0;     % 连续出现相同类别标签的次数的初始值为0
            for theta=1.25:-0.1:0.75
                
                t=theta*r;
                e1=find(D(idx_U(v),idx_L)<=t);
                if isempty(e1)
                    continue;
                end
                ex1=Y(idx_L(e1));              
                LN1=mode(ex1);
                
                % 判断当前的类别标签LN1是否与上一次循环的相同
                if LN1==prevLN1
                    aaaaa = aaaaa + 1;  % 相同则计数器加1
                else
                    aaaaa = 1;         % 不相同则计数器清零
                    prevLN1 = LN1;     % 更新上一次循环的类别标签LN1的值
                end
                
                % 判断连续出现相同类别标签的次数是否达到三次，如果是则退出循环
                if aaaaa == 3
                    break;
                end
                
            end
            if Y_UL(v)==LN1
                S=[S;idx_U(v)];
            end 
        end
        if isempty(S)
            break;
        end

        X_UL=X(S,:);
        Y_NL=predict(mdl,X_UL);
        for u=1:length(Y_NL)
            Y(S(u))=Y_NL(u);
        end
        idx_L=sort([idx_L;S]);
        % DD=DD+1
        
    end

    mdl=fitcknn(X(idx_L,:),Y(idx_L,:),'NumNeighbors',3, 'Distance','euclidean');

end



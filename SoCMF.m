tic;
% 梯度下降算法计算得出用户偏好矩阵U和物品特征矩阵V


k1=20;

U=0.5*ones(usercount,k1);
V= rand(itemcount,k1);
yi= rand(usercount,k1);
bj=0.2*rand(1,itemcount);
bu=0.2*rand(1,usercount);
eta=0.001;lambda=0.0005;

epsilon=0.5*ones(usercount,itemcount);

iteraTime=90;
lambda_t=3;

 
alpha=1.5;

gammas=2*ones(1,itemcount); 

 
 for i=1:1:iteraTime
    
 
    for j=1:traincount 
   
        currentuser=train_rating(j,1);
        currentitem=train_rating(j,2);
        if(train_rating(j,3)~=0)
    ep=epsilon(currentuser,currentitem);

    originalrating=(U(currentuser,:)-V(currentitem,:))*(U(currentuser,:)-V(currentitem,:))';
    left= exp(ep*originalrating);
    right=  bu(currentuser)+bj(currentitem)+ U(currentuser,:)*V(currentitem,:)' ;
    
    
    %对商品已评分用户的隐式影响
      tcount=itemrating(currentitem,1);
      neiindex=itemrating(currentitem,2:tcount+1);
      neigh=sum(yi(neiindex,:),1);
     

  gamma=gammas(currentitem);
 
  right2=alpha* neigh*V(currentitem,:)'/(tcount+1)^gamma;
 
  right=right+right2;


  e=  (1-1/(1+left))*right-train_rating(j,3);  
   

    deri= 1/(1+left)^2;  %pr(u,j)的导数
            
       


            
            tempU =   e*(deri* left*ep*2*(U(currentuser,:)-V(currentitem,:))*right+(1-1/(1+left))*V(currentitem,:)) ;
            tempV =   e*(deri* left*ep*2*(V(currentitem,:)-U(currentuser,:))*right+(1-1/(1+left))*(U(currentuser,:)+alpha/(tcount+1)^gamma*neigh));
   
         
           tempep=   e*deri*left*originalrating*right ;
           tempbj=  e*(1-1/(1+left));
     
            tempbu=   e*(1-1/(1+left));
           
             

            tempgamma= -e*(1-1/(1+left))*right2*log(tcount+1);

       
              tempyi=   e*alpha*(1-1/(1+left))/(tcount+1)^gamma*ones(tcount,1)*V(currentitem,:);

            
            if(e>1000)
               return;
            end
 
                U(currentuser,:)=U(currentuser,:)-eta*tempU;
                V(currentitem,:)=V(currentitem,:)-eta*tempV;
                 bj(currentitem)=bj(currentitem)-eta*tempbj;

                bu(currentuser)=bu(currentuser)-eta*tempbu;

                      epsilon(currentuser,currentitem)=epsilon(currentuser,currentitem)-eta*tempep ;

                  yi(neiindex,:)=yi(neiindex,:)-eta*tempyi;
 
                     gammas(currentitem)=gammas(currentitem)- eta*tempgamma;


           
        end
         
    end
    
    U=U-eta*lambda*U;
    V=V-eta*lambda*V;
    bu=bu-eta*lambda*bu;
    yi=yi-eta*lambda*yi;
    bj=bj-eta*lambda*bj;
 

  userindex=randperm(usercount);


 for alluser=1:usercount 

      currentneighbor=zeros(1,k1);
      ijk=userindex(alluser);
       for j=1:network(ijk,1)
           currentuser=network(ijk,j+1);
           dist=min((U(ijk,:)-U(currentuser,:))*(U(ijk,:)-U(currentuser,:))',1);
           tempneighbor=   (1-0.5*dist^2)*(U(ijk,:)-U(currentuser,:));
           currentneighbor=currentneighbor-eta*(lambda_t *tempneighbor );  
       end

 U(ijk,:)=U(ijk,:)+currentneighbor;


      
 end
 
  end
    
    

 
%% 计算预测评分
%  UI_predict=U*V';
%% 在测试集test_ratings(20000×4)上计算MAE
 s=0;
 total=0;
for i=1:testcount
    userid=test_rating(i,1);
    itemid=test_rating(i,2);
    Grade=test_rating(i,3);
    tcount=itemrating(itemid,1);
    neiindex=itemrating(itemid,2:tcount+1);
    neigh=sum(yi(neiindex,:),1);
    gamma=gammas(itemid);
    ep=epsilon(userid,itemid);   
  score= (1-1/(1+exp(ep* (U(userid,:)-V(itemid,:))*(U(userid,:)-V(itemid,:))')))*(bu(userid)+bj(itemid)+U(userid,:)*V(itemid,:)'+alpha*neigh*V(itemid,:)'/(tcount+1)^gamma);
    s=s+abs(Grade-score);
    total=total+(Grade-score)^2;
end
MAE=s/testcount
RMSE=sqrt(total/testcount) 




s=0;
total=0;
coldcount=0;
for i=1:testcount
    userid=test_rating(i,1);
    itemid=test_rating(i,2);
    if(userrating(userid,1)>5)
        continue;
    end
    coldcount=coldcount+1;
    Grade=test_rating(i,3);
    tcount=itemrating(itemid,1);
    neiindex=itemrating(itemid,2:tcount+1);
    neigh=sum(yi(neiindex,:),1);
    gamma=gammas(itemid);
    ep=epsilon(userid,itemid);   
  score= (1-1/(1+exp(ep* (U(userid,:)-V(itemid,:))*(U(userid,:)-V(itemid,:))')))*(bu(userid)+bj(itemid)+U(userid,:)*V(itemid,:)'+alpha*neigh*V(itemid,:)'/(tcount+1)^gamma);
    s=s+abs(Grade-score);
    total=total+(Grade-score)^2;
end
MAE=s/coldcount
RMSE=sqrt(total/coldcount) 
 
toc;
%     
    
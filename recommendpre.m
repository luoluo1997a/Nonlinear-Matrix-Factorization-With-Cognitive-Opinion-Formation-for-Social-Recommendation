usercount=length(unique(rating(:,1)));
itemcount=length(unique(rating(:,2)));

lll=randperm(size(rating,1));
traincount=ceil(size(rating,1)*0.8);
testcount=size(rating,1)-traincount;
train_rating=rating(lll(1:traincount),:);
test_rating=rating(lll(traincount+1:size(rating,1)),:);


UI_train_matrix=zeros(usercount,itemcount);
for i=1:size(train_rating,1)   
   UI_train_matrix(train_rating(i,1),train_rating(i,2))=train_rating(i,3);  
end


userrating=zeros(usercount,20);
for i=1:usercount
   currentindex=find(UI_train_matrix(i,:)>0);
   userrating(i,1)=length(currentindex);
   userrating(i,2:userrating(i,1)+1)=currentindex;
    
end
itemrating=zeros(itemcount,20);
for i=1:itemcount
   currentindex=find(UI_train_matrix(:,i)>0);
   itemrating(i,1)=length(currentindex);
  itemrating(i,2:itemrating(i,1)+1)=currentindex;
    
end
network=zeros(usercount,20);
trustcount=size(trustnetwork,1);
for i=1:trustcount
   currentindex=trustnetwork(i,1);
   network(currentindex,1)=network(currentindex,1)+1;
   network(currentindex,network(currentindex,1)+1)=trustnetwork(i,2);
    
end
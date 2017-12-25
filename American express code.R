bank =read.csv("american.csv")
Final =read.csv("Final_Dataset.csv")
 is.na(bank$mvar3)= !bank$mvar3
is.na(bank$mvar9)=!bank$ mvar9,
is.na(lead$mvar3 = !lead$ mvar3,
       
is.na(lead$mvar3) = !lead$ mvar3,
is.na(lead$mvar9) =!lead$ mvar9,
        
is.na(lead$mvar3) = !lead$ mvar3
is.na(lead$mvar9) =!lead$ mvar9
ind =which(is.na(bank$mvar3), arr.ind =TRUE)
index = which(is.na(bank$mvar9), arr.ind =TRUE)
indl =which(is.na(lead$mvar3), arr.ind =TRUE)
indexl=which(is.na(lead$mvar9), arr.ind =TRUE)
bank$mvar3[ind] = mean(bank$mvar3, na.rm =TRUE)
bank$mvar9[index] = mean(bank$mvar9, na.rm =TRUE)
lead$mvar3[indl]= mean(lead$mvar3,na.rm =TRUE)
lead$mvar9[indexl] =mean(lead$mvar9 ,na.rm =TRUE)

bank =mutate(bank ,total_spend = mvar36+mvar37+mvar38+mvar39)
bank =mutatae(bank ,qt1 =mvar20+mvar24+mvar28+mvar32+mvar16)
bank =mutate(bank ,qt1 =mvar20+mvar24+mvar28+mvar32+mvar16)
bank =mutate(bank , qt2 =mvar17+mvar21+mvar25+mvar29+mvar33)
bank =mutate(bank ,qt3 =mvar34+mvar30+mvar26+mvar22+mvar18)
bank =mutate(bank ,qt4 =mvar34+mvar31+mvar27+mvar23+mvar19)
        
        
        
        
test1 = mutate(test1 ,total_spend =mvar36+mvar37+mvar38+mvar39)
test1 = mutate(test1 ,qt1=mvar20+mvar16+mvar24+mvar28+mvar32)
test1 =mutate(test1, qt2=mvar17+mvar21+mvar25+mvar29+mvar33)
test1 =mutate(test1, qt3 =mvar18+mvar22+mvar26+mvar30+mvar34)
test1= mutate(test1 ,qt4 =mvar19+mvar23+mvar27+mvar31+mvar35)

#DATA IS THE TRAINING DATA    &    LEAD IS THE TEST DATA.

#NEXT STEP IS APPLYING RANDOMFOREST MODEL FOR PREDICTING MVAR46, MVAR47 ,MVAR48
#AFTER THIS ADD THE PREDICTIONS COLOUMNS INTO DATA & LEAD DATASETS

#Pre processing data to apply Principle component analysis

train = select(data, cm_key,mvar2,mvar3,mvar4,mvar5,mvar6,mvar7,mvar8,mvar9,mvar10,mvar11,mvar13,mvar14,mvar15,mvar40,mvar41,mvar42,mvar43,mvar44,mvar45,total_spend,qt1,qt2,qt4,qt3,,mvar46,mvar48,mvar49,mvar50,mvar51,y)

#log transform
log.train = log(train[,2:19])
log.test= log(train[,20:25])
ir.pca = prcomp(log.train ,
                center =T,
                scale.=T)

#datasets for Supp card
train_supp =select(train, cm_key,mvar2,mvar3,mvar4,mvar5,mvar7,mvar10,mvar11,mvar13,mvar14,qt1,qt2,qt3,qt4,mvar40,mvar41,mvar42,mvar43,mvar44,mvar45,mvar46,mvar49)
Supp_lead = select(lead, cm_key,mvar2,mvar3,mvar4,mvar5,mvar6,mvar7,mvar8,mvar9,mvar10,mvar11,mvar13,mvar14,mvar15,qt1,qt2,qt3,qt4,mvar40,mvar41,mvar42,mvar43,mvar44,mvar45)

#model for supp card

model1 =randomForest(mvar46 ~ mvar2+mvar3+mvar4+mvar5+mvar6+mvar7+mvar8+mvar9+mvar10+mvar11+mvar13+mvar14+mvar15+qt1+qt2+qt3+qt4+mvar40+mvar41+mvar42+mvar43+mvar44+mvar45, data =train ,nodesize =750,ntree =800)
Supp =predict(model1,newdata=Supp_lead)

#dataset for Elite card
model2 =randomForest(y ~ mvar2+mvar3+mvar4+mvar5+mvar7+mvar10+mvar11+mvar13+mvar14+qt1+qt2+qt3+qt4+mvar40+mvar41+mvar42+mvar43+mvar44+mvar45, data= train ,nodesize =800, ntree =850)

train$mvar47 =as.factor(train$y)
model2 =randomForest(y ~ mvar2+mvar3+mvar4+mvar5+mvar7+mvar10+mvar11+mvar13+mvar14+qt1+qt2+qt3+qt4+mvar40+mvar41+mvar42+mvar43+mvar44+mvar45, data= train ,nodesize =800, ntree =850)
Elite =predict(model2,newdata = Supp_lead)
summary(Elite)

train$mvar48 = as.factor(train$mvar48)
model3= randomForest(mvar48 ~ mvar2+mvar3+mvar4+mvar5+mvar7+mvar10+mvar11+mvar13+mvar14+qt1+qt2+qt3+qt4+mvar40+mvar41+mvar42+mvar43+mvar44+mvar45, data= train, nodesize =850,ntree =900)
Credit =predict(model3, newdata =Supp_lead)
summary(Credit)
 
submit =data.frame(Customer = Supp_lead$cm_key, Card1 =Supp, Card2= Elite, Card3 =Credit)
write.csv(submit, file = "Am.csv")
ind= which(Supp==1)
submit =data.frame(Customerid =Supp_lead$cm_key[ind], Card1 =Supp[ind])
write.csv(submit ,file ="Supp.csv")
inde =which(Elite==1)
submit =data.frame(Customerid =Supp_lead$cm_key[inde], Card1 =Elite[inde])
index= which(Credit ==1)
submit1 =data.frame(Customer =Supp_lead$cm_key[index], Card =Credit[index])
write.csv(submit ,file="Elite.csv")
write.csv(submit1 ,file="Credit.csv")



Supp =predict(model, newdata =check)
Elite =predict(model1 ,newdata =check)
Credit =predict(model2 ,newdata =check)

ind =which(Supp==1)
sub =data.frame(Customer =check$cm_key[ind], Card =Supp[ind])
inde = which(Elite==1)
subm =data.frame(Customer =check$cm_key[inde], Card=Elite[inde])
index= which(Credit==1)
submit = data.frame(Customer =check$cm_key[index], Card= Credit[index])
write.csv(sub ,file="Supp.csv")
write.csv(subm ,file= "Elite.csv")
write.csv(submit, file="Credit.csv")


#FINAL Dind = which(testF$mvar3==0)
testF$mvar3[ind]= mean(testF$mvar3)
inde = which(testF$mvar9==0)
tetsF$mvar9[inde] = mean(testF$mvar9)


testF = mutate(testF ,qt1=mvar20+mvar16+mvar24+mvar28+mvar32)
testF =mutate(testF, qt2=mvar17+mvar21+mvar25+mvar29+mvar33)
testF =mutate(testF, qt3 =mvar18+mvar22+mvar26+mvar30+mvar34)
testF= mutate(testF ,qt4 =mvar19+mvar23+mvar27+mvar31+mvar35)

str(tetsF)


str(finalcheck)


Supp = predict(model, newdata= finalcheck)
Elite =predict(model1, newdata =finalcheck)
Credit = predict(model2, newdata =finalcheck)

ind= which(Supp==1)
inde= which(Elite ==1)
index= which(Credit ==1)

sub =data.frame(Customer =finalcheck$cm_key[ind], card =Supp[ind])
subn =data.frame(Customer =finalcheck$cm_key[inde], card =Elite[inde])
submit = data.frame(Customer =finalcheck$cm_key[index], card =Credit[index])

write.csv(sub ,file="Supp.csv")
write.csv(subn ,file= "Elite.csv")
write.csv(submit, file="Credit.csv")

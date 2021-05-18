
library(randomForest)
library(caret)
library(e1071)
library(ggplot2)
library(ROCR)
library(class)
library(pROC)

################################### SECTION ONE #########################################
#Importing data
data_12_df = read.csv(file.choose())
View(data_12_df)

#Checking data
names(data_12_df)
colnames(data_12_df)[1]<-c("Label")
nrow(data_12_df)
unique(data_12_df$Label)
labels = as.vector(data_12_df$Label)

#Creating binary labels (math symbol=1 and not math symbol=0)
bin_labels = c()
for(i in 1:nrow(data_12_df)){
  if(labels[i] == "approxequal"||labels[i] == "equal"||labels[i] == "greater"|| labels[i] == "greaterequal"||labels[i] == "less" || labels[i] == "lessequal"||labels[i] == "notequal"){
    bin_labels = append(bin_labels,1)
    print(labels[i])
  }
  else{
    bin_labels = append(bin_labels,0)
  }
}
table(bin_labels)
table(data_12_df$Label)
data_12_df = data.frame(data_12_df,bin_labels)
names(data_12_df)

##### 1.1 ######
log.fit = glm(bin_labels ~ nr_pix,data = data_12_df,family = binomial(link='logit'))
summary(log.fit)
log.fit$coefficients
predict(log.fit,data_12_df[1:5,"nr_prix"],type="response")

##### 1.2 ######
set.seed(12)
# Run the model with 5-fold cross validation
trControl_sec1 = trainControl(method = "cv",
                         number = 5,
                         savePredictions = T )
log.CV = train(bin_labels ~ nr_pix, data = data_12_df,method = "glm",
                trControl = trControl_sec1,family=binomial(link='logit'))
predictions = log.CV$pred
predictions = predictions[order(predictions$rowIndex),]
pred_labs = ifelse(predictions$pred>=0.5,1,0)
pred_labs
confusion_matrix = table(pred_labs,data_12_df$bin_labels)
TN = confusion_matrix[1,1] #True negatives
TP = confusion_matrix[2,2] #True positives
FP = confusion_matrix[1,2] #False positives
FN = confusion_matrix[2,1] #False negatives
#accuracy
accuracy = (TP+TN)/(TP+TN+FP+FN)
#true positive rate
TPR = TP/(TP + FN)
#false positive rate
FPR = FP/(FP + TN)
#precision
precision = TP/(TP+FP)
#recall
recall = TP/(TP+FN)
#F1-score 
F1 = (2*TP)/((2*TP)+FP+FN)

#ROC curve
roc_log = roc(data_12_df$bin_labels, predictions$pred)
ggroc(roc_log,colour = 'steelblue', size = 2,legacy.axes=TRUE)+
  ggtitle(paste0('ROC Curve ', '(AUC = ', accuracy, ')'))+
  geom_segment(aes(x = 0, xend = 1, y = 0, yend = 1))

################################## SECTION TWO ##########################################

###### 2.1 #######
feats6 = c( "nr_pix", "rows_with_2" ,"cols_with_2","rows_with_3p","cols_with_3p", "height" )
#Preparing the 3-label response
labels3 = c()
for(i in 1:nrow(data_12_df)){
  if(labels[i] == "approxequal"||labels[i] == "equal"||labels[i] == "greater"|| labels[i] == "greaterequal"||labels[i] == "less" || labels[i] == "lessequal"||labels[i] == "notequal"){
    labels3 = append(labels3,"Math")
    }
  else if(labels[i]=="five" ||labels[i]=="four"||labels[i]=="one"||labels[i]=="seven"||labels[i]=="six"||labels[i]=="three"||labels[i]=="two"){
    labels3 = append(labels3,"Digit")
  }
  else{
    labels3 = append(labels3,"Letter")
  }
}
table(labels3)#Checking if the number of new labels equal the size of the data
data_12_df = data.frame(data_12_df,"label3"=labels3)# Adding new label to data
k_vec = seq(3,25,2)
accuracy_vec = c()

#Training the KNN
for(K in k_vec){
  knn = knn(train=data_12_df[,feats6], cl=data_12_df$label3, test =data_12_df[,feats6] , k=K)
  accuracy_vec = append(accuracy_vec, sum(data_12_df$label3 == knn)/nrow(data_12_df)*100)
} 
acc_k.df = data.frame("Accuracy"=accuracy_vec,"K"=k_vec)
ggplot(data=acc_k.df, aes(x=K, y=Accuracy, group=1)) +
  geom_line()+
  geom_point()+labs(  # annotations layer
    title = "Accuracy vs K")

##### 2.2 ######
set.seed(12)
feats7 = c("label3", "nr_pix", "rows_with_2" ,"cols_with_2","rows_with_3p","cols_with_3p", "height" )

# Initialising the cross validator
trControl_knn2 = trainControl(method  = "cv",
                             number  = 5,
                             search='grid')
tuneGrid_knn2 = expand.grid( k= k_vec) #Creating a grid of ks

#Training the K-nearest neighbors model
knn_grid2 = train(label3~., data = data_12_df[,feats7],method = "knn", metric = "Accuracy",
                  tuneGrid = tuneGrid_knn2, trControl = trControl_knn2) #
#Results
acc_k.df2 = data.frame("Training_accuracy"=accuracy_vec,"CV_accuracy"=knn_grid2$results$Accuracy*100,
                       "K"=k_vec, "best_accuracy"=max(acc_k.df2$CV_accuracy))
acc_k.df2[3] = (acc_k.df2$K)^-1
#Plot
colors  <- c("Training" = "red", "5-fold CV" = "blue")
ggplot_knn2 = ggplot(acc_k.df2, aes(x=K)) +       # Create ggplot2 plot
  geom_line(aes(y = CV_accuracy, color = "5-fold CV"),size=1.5)+
  geom_line(aes(y =Training_accuracy , color = "Training"),size=1.5) +
  geom_line(aes(y=best_accuracy,color = "Best validation accuracy"),size=1.5)+
  labs(x = "1/K",y = "Accuracy", color = "Legend")
ggplot_knn2    

max(knn_grid2$results$Accuracy)
knn_grid2$results$k


################################# SECTION THREE #####################################

#Importing data
all_features <- read.delim("C:/Users/eugen/Desktop/Work/atizas/all_features.csv", header=FALSE)
View(all_features)

#Checking uploaded dataset
names(all_features) 
unique(all_features[,"V1"])
nrow(all_features)

####### 3,1 #######
set.seed(12)
# Run the model
trControl = trainControl(method = "cv",
                          number = 5,
                          search = "grid")
tuneGrid_pred = expand.grid(mtry = c(2, 4, 6, 8)) #Setting grid for random forest which takes only "mtry"
Ntrees = seq(25,400,25) #List of trees to use for tuning
trees_used = c() #Iitialising the vector for number of trees to use in every iteration
grid_accuracies = c() #Best accruacy 
best_feats = c()

#Loop over the tree number hyperparameter to tune model
for(Ntree in Ntrees){
  #training
  rf_grid = train(V1~., data = all_features,method = "rf", ntree=Ntree, metric = "Accuracy",tuneGrid = tuneGrid_pred, 
                  trControl = trControl,importance = TRUE)
  
  best_feats = append(best_feats,rf_grid$bestTune$mtry) #Best accuracy in each loop
  
  trees_used = append(trees_used,Ntree) #Trees used per iteraction

  grid_accuracies = append(grid_accuracies, max(rf_grid$results$Accuracy)) #Best accuracy for each trees and feature combination from 5-fold CV
}
#Creating the dataframe of the results
models_df = data.frame(best_feats ,trees_used, grid_accuracies)
colnames(models_df) = c("Number_of_features","Number_of_trees","Best_accuracy")
View(models_df)

#Extracting the best hyper parameter combination and its accuracy
best_df = subset(models_df,Best_accuracy==max(models_df$Best_accuracy))
best_features = best_df$Number_of_features
best_trees = best_df$Number_of_trees
best_accuracy = best_df$Best_accuracy

####### 3.2 ######
#Setting the 5-fold CV 
trControls_random = trainControl(method = "cv",
                         number = 5,
                         search = "grid")
#Setting the grid with only the best number of features from 3.1
tuneGrid_random = expand.grid(mtry = best_features)
random_accuracy = c() #Initializing the vector to store accuracies from refitting the random forest classifier

#Refitting the model 15 times and storing its accuracies in the vector that was initialized
for(i in 0:14){
  set.seed(i)
  rf_random = train(V1~., data = all_features,method = "rf", ntree=best_trees, metric = "Accuracy", 
                  trControl = trControls_random,importance = TRUE, tuneGrid=tuneGrid_random)
  random_accuracy = append(random_accuracy,max(rf_random$results$Accuracy))
}
mean_accuracy = mean(random_accuracy) #Mean accuracy
st_accuracy = sd(random_accuracy) #Standard deviation of the accuracies

#Data visualization of the refits
data_random = data.frame("Number_of_fits"=1:15,"Accuracy"=random_accuracy)
ggplot(data=data_random, aes(x=Number_of_fits, y=Accuracy, group=1)) +
  geom_line()+
  geom_point()

####### 3.3 #######
#modelLookup("knn")
set.seed(12)
## Run the K-nearest neighbors model ##
trControl_knn = trainControl(method  = "cv",
                             number  = 5,
                             search='grid')
grid_accuracies_knn = c() #Best accruacy 
tuneGrid_knn = expand.grid( k= c(3,5,7,9,11,13,15))

knn_grid = train(V1~., data = all_features,method = "knn", metric = "Accuracy",tuneGrid = tuneGrid_knn, 
                  trControl = trControl_knn)
  
max(knn_grid$results$Accuracy) #Best accuracy for each trees and feature combination from 5-fold CV
knn_grid$bestTune$k

## Run the gradient boost classifier ##
library(gbm)
trControl_gbm = trainControl(method  = "CV",
                             number  = 5)
#Hyparameters
Trees = c(50,75)
Depth = c(4,6)
Shrinkage = c(0.01,0.1) #Learning rate
N.minobsinnode=c(3,5)
combinations = c()
accuracies_gbm =c()
#Estimating gradient boost classifer using grid search
for (tree in Trees){
  for (depth in Depth){
    for(shrinkage in Shrinkage){
      for(minobs in N.minobsinnode){
        combinations = append(combinations,paste(toString(tree)," ",toString(depth)," ",toString(shrinkage)," ",toString(minobs)))
        tuneGrid_gbm_i = expand.grid( n.trees = c(tree),
                                    interaction.depth=c(depth),
                                    shrinkage=c(shrinkage),
                                    n.minobsinnode=c(minobs))
        gbm_grid_i <- train(V1 ~ .,
                          data = all_features,
                          trControl = trControl_gbm,
                          tuneGrid = tuneGrid_gbm,
                          method = "gbm",
                          metric = "Accuracy",
                          distribution = "multinomial",
                          verbose = FALSE)
        print(max(gbm_grid_i$results$Accuracy))
        accuracies_gbm = append(accuracies_gbm,max(gbm_grid_i$results$Accuracy))
        
      }
    }
  }
}
combinations[accuracies_gbm==max(accuracies_gbm)]
max(accuracies_gbm)

## Running the decision tree ##
library(rpart.plot)

trControl_DT = trainControl(method = "CV", number = 5)
tuneGrid_DT = expand.grid( cp= c(0.001,0.01,0.1)) #Grid of cost complexity values
min_plits_vec = c(5,10,15) #grid of 
combinations_DT = c() #Initialising the vector to store hyperparameter combinations
accuracies_DT = c() #Initialising vector to store accuracy per best accuracy
i = 0
#Training process
for(Minsplit in min_plits_vec){
  i=i+1
  dtree_fit = train(V1 ~., data = all_features, method = "rpart",
                    trControl=trControl_DT,
                    tuneGrid=tuneGrid_DT,
                    minsplit= Minsplit)
  combinations_DT = append(combinations_DT,paste(toString(tuneGrid_DT[i,])," ",toString(Minsplit)))
  accuracies_DT = append(accuracies_DT,max(dtree_fit$results$Accuracy))
}

max(accuracies_DT)
combinations_DT[accuracies_DT==max(accuracies_DT)]

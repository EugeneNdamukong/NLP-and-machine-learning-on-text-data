library(tm)
library(SnowballC)
library(caret) 
library(e1071)
library(caTools)
library(textmineR)
library(readr)
library(dplyr)
library(randomForest)
library(MLmetrics)
library(caTools)

#Source of data: https://www.kaggle.com/c/spooky-author-identification/overview

#Loading the data of texts
data1 <- read_csv("C:/Users/eugen/Desktop/Work/train.csv")
glimpse(data1)
#EAP: Edgar Allan Poe, HPL: HP Lovecraft; MWS: Mary Wollstonecraft Shelley
unique(data1$author)
table(data1$author)

#Randomize the data prior to splitting
data_sh = data1[sample(1:nrow(data1),nrow(data1)),]

#Creation of subsection of data from whole data
train_EAP = subset(data_sh,author=="EAP")[1:700,]
train_size_EAP = nrow(train_EAP)

train_HPL = subset(data_sh,author=="HPL")[1:700,]
train_size_HPL = nrow(train_HPL)

train_MWS = subset(data_sh,author=="MWS")[1:700,]
train_size_MWS = nrow(train_MWS)

train_txt = rbind(train_EAP[1:train_size_EAP,],
                  train_HPL[1:train_size_HPL,],
                  train_MWS[1:train_size_MWS,])
train_txt[1:5,]
glimpse(train_txt)
table(train_txt$author)

###### Preparing data for classification ###########
texts = c()
for(i in 1:nrow(train_txt)){
  texts = append(texts,train_txt$text[i])
}

#Step1 : Creating a corpus of all text
corpus = Corpus(VectorSource(texts)) 
corpus[[2]][1] #Viewing an element in the corpus

# Step 2 : Converting all characters to lowercase
corpus = tm_map(corpus, PlainTextDocument)
corpus = tm_map(corpus, tolower)
corpus[[2]][1]  

# Step 3 : Removal of punctuations
corpus = tm_map(corpus, removePunctuation)
corpus[[2]][1]

# Step 4 : Removal of stopwords
corpus = tm_map(corpus, removeWords, c(stopwords("english")))
corpus[[2]][1]

# Step 5 : Stemming 
corpus = tm_map(corpus, stemDocument)
corpus[[2]][1]  

# Step 6 : Creation of document-term matrix
doc_term_mat = DocumentTermMatrix(corpus)

# Step 6 : Taking care of sparsiness
sparse = removeSparseTerms(doc_term_mat,0.99)

# Step 7 : Convertinh the document-term matrix to dataframe
doc_df = as.data.frame(as.matrix(sparse))
colnames(doc_df) = make.names(colnames(doc_df))
doc_df$author = as.factor(train_txt$author)

nrow(doc_df) #Checking number of texts

#Splitting data into 50% "known" and 50% "Unkown"
set.seed(12)
split = sample.split(1:nrow(doc_df), SplitRatio = 0.5)
known = subset(doc_df, split==TRUE)
unknown = subset(doc_df, split==FALSE)

#Checking if there are at least 100 texts per author
table(known$author)
table(unknown$author)

prop.table(table(known$author)) #>30% is the baseline accuracy for known texts
prop.table(table(unknown$author)) #>30% is the baseline accuracy for known texts

##### Creating machine learning model #####

#Training random forest models
RF_model = randomForest(author ~ ., data=known,ntree = 500,
                        mtry = 15,#Used rule of thumb sqt(number trees)
                        proximity = TRUE,
                        importance = TRUE)

predictRF = predict(RF_model, newdata=known)#Predicition on the "known" writings
confusion_known = table(predictRF,known$author) #Confusing matrix showing classification of text to authors
Error_rate_k = 1 - Accuracy( predictRF,known$author) #Error rate on train data


### Using the model to make prediction on "unkown" texts ###
predictRF2 = predict(RF_model, newdata=unknown)#Predicition on the "unknown" writings
confusion_unkown = table(unknown$author,predictRF2) #Confusing matrix showing classification of text to authors
Error_rate_u = 1 - Accuracy( predictRF,unknown$author) #Error rate on train data

#General true and false positives 
TP = confusion_unkown[1,1] + confusion_unkown[2,2] + confusion_unkown[3,3] #True positives
prop_TP= TP*(1/nrow(unknown)) #Proportion of true positives

FP = confusion_unkown[1,2] + confusion_unkown[1,3] + 
  confusion_unkown[2,1] + confusion_unkown[2,3] +
  confusion_unkown[3,1] + confusion_unkown[3,2]#False positives
prop_FP = FP*(1/nrow(unknown)) #Proportion of false positives

#True and false positives of each author
TP_EAP = confusion_unkown[1,1] #True positives of EAP
prop_TP_EAP = TP_EAP / TP #Proportion of TP of EAP

TP_HPL = confusion_unkown[2,2] #True positives of HPL
prop_TP_HPL = TP_HPL / TP #Proportion of TP of HPL

TP_MWS = confusion_unkown[3,3] #True positives of MWS
prop_TP_MWS = TP_MWS / TP #Proportion of TP of MWS

FP_EAP = confusion_unkown[2,1] + confusion_unkown[3,1] #False positives of EAP
prop_FP_EAP = FP_EAP / FP #Proportion of FP of EAP

FP_HPL = confusion_unkown[1,2] + confusion_unkown[3,2] #False positives of HPL
prop_FP_HPL = FP_HPL / FP #Proportion of FP of HPL

FP_MWS = confusion_unkown[1,3] + confusion_unkown[2,3] #False positives of MWS
prop_FP_MWS = FP_MWS / FP #Proportion of FP of  MWS





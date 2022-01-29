# Download link for dataset
# https://www.kaggle.com/vagifa/ethereum-frauddetection-dataset/download

# Installing necessary packages
install.packages('DMwR')
install.packages('xts')
install.packages('quantmod')
install.packages('zoo')
install.packages('ROCR')
install.packages('pROC')
install.packages('corrgram')
install.packages("naniar")

# Initializing necessary libraries
library(data.table)
library(kernlab)
library(caret)
library(ggplot2)
library(zoo)
library(xts)
library(ROCR)
library(quantmod)
library(kernlab)
library(DiscriMiner)
library(mlbench)
library(dplyr)
library(GGally)
library(lattice)
library(reshape)
library(DMwR)
library(pROC)
library(corrgram)
library(naniar)

# Importing the data set (taken from Kaggle)
df=fread('transaction_dataset.csv')

# Checking the columns of the data frame
colnames(df)

# Viewing the data frame summary for insights to unknown data
summary(df)

# creating feature wise box plots to analyze the distribution
boxplot(df$`ERC20 avg val sent contract`)
boxplot(df$`ERC20 max val sent contract`)
boxplot(df$`ERC20 min val sent contract`)
boxplot(df$`ERC20 avg val sent`)
boxplot(df$`ERC20 max val sent`)
boxplot(df$`ERC20 min val sent`)
boxplot(df$`ERC20 max val rec`)
boxplot(df$`ERC20 avg time between contract tnx`)
boxplot(df$`ERC20 avg time between rec 2 tnx`)
boxplot(df$`ERC20 avg time between rec tnx`)
boxplot(df$`ERC20 avg time between sent tnx`)
boxplot(df$`ERC20 uniq rec contract addr`)
boxplot(df$`ERC20 uniq sent addr.1`)
boxplot(df$`ERC20 uniq sent addr`)
boxplot(df$`ERC20 total Ether received`)
boxplot(df$`Total ERC20 tnxs`)
boxplot(df$`total ether balance`)
boxplot(df$`total ether sent contracts`)
boxplot(df$`total ether received`)
boxplot(df$`total Ether sent`)
boxplot(df$`total transactions (including tnx to create contract`)
boxplot(df$`avg value sent to contract`)
boxplot(df$`max val sent to contract`)
boxplot(df$`min val sent to contract`)
boxplot(df$`min value sent to contract`)
boxplot(df$`min value sent to contract`)
boxplot(df$`avg val sent`)
boxplot(df$`max val sent`)
boxplot(df$`min val sent`)
boxplot(df$`avg val received`)
boxplot(df$`max value received`)
boxplot(df$`min value received`)
boxplot(df$`Unique Sent To Addresses`)
boxplot(df$`Unique Received From Addresses`)
boxplot(df$`Number of Created Contracts`)
boxplot(df$`Received Tnx`)
boxplot(df$`Sent tnx`)
boxplot(df$`Time Diff between first and last (Mins)`)
boxplot(df$`Avg min between received tnx`)
boxplot(df$`Avg min between sent tnx`)
a=colnames(df)

# plotting the heatmap depicting NA values in the dataset
vis_miss(df[, 4:51])

# Looping to standardize the column names (replacing " " with "_")
for (i in 1:length(a))
{
  a[i]=gsub(" ", "_", a[i])
}

a[7]="Time_Diff_between_first_and_last_Mins"
a[22]="total_transactions_including_tnx_to_create_contract"

colnames(df)=a

# dropping the columns based on distribution insights from the box plots
df1=df[,-c('ERC20_min_val_sent_contract','ERC20_max_val_sent_contract',
        'ERC20_avg_val_sent_contract','ERC20_avg_val_sent','ERC20_min_val_sent',
        'ERC20_max_val_sent','ERC20_avg_time_between_contract_tnx',
        'ERC20_avg_time_between_rec_2_tnx','ERC20_avg_time_between_rec_tnx',
        'ERC20_avg_time_between_sent_tnx','ERC20_uniq_sent_addr.1','total_ether_sent_contracts',
        'avg_value_sent_to_contract','max_val_sent_to_contract','min_value_sent_to_contract')]


# Dropping irrelevant features to the study
data=df1[,4:34]

# Viewing the characteristics of the data frame post all necessary removals
summary(data)

data <- as.data.frame(data)

c = table(data$FLAG)
# counting the no. of fraud cases
fraudCases = c[names(c) == 1]
nonFraudCases = c[names(c) == 0]
totalRows = fraudCases + nonFraudCases

pieChart <- data.frame(
  Groups = c("Non-Fraud", "Fraud"),
  value = c(nonFraudCases, fraudCases),
  percent = c(round(nonFraudCases*100/totalRows, 2), round(fraudCases*100/totalRows, 2))
)

# plotting a pie-chart denoting the target data distribution
ggplot(pieChart, aes(x="", y=value, fill=Groups)) +
  geom_bar(stat="identity", width=1, color="white") +
  geom_text(aes(label = paste0(percent, "%")), position = position_stack(vjust=0.5)) +
  coord_polar("y", start=0) +
  theme_void() + # remove background, grid, numeric labels
  scale_fill_manual(values=c("#f84444", "#9eda62"))

# Imputing missing data using median method for each columns with missing data
data$ERC20_uniq_rec_token_name[is.na(data$ERC20_uniq_rec_token_name)] = median(data$ERC20_uniq_rec_token_name,na.rm=T)
data$ERC20_uniq_sent_token_name[is.na(data$ERC20_uniq_sent_token_name)] = median(data$ERC20_uniq_sent_token_name,na.rm=T)
data$ERC20_avg_val_rec[is.na(data$ERC20_avg_val_rec)] = median(data$ERC20_avg_val_rec,na.rm=T)
data$ERC20_max_val_rec[is.na(data$ERC20_max_val_rec)] = median(data$ERC20_max_val_rec,na.rm=T)
data$ERC20_min_val_rec[is.na(data$ERC20_min_val_rec)] = median(data$ERC20_min_val_rec,na.rm=T)
data$ERC20_uniq_rec_contract_addr[is.na(data$ERC20_uniq_rec_contract_addr)] = median(data$ERC20_uniq_rec_contract_addr,na.rm=T)
data$ERC20_uniq_rec_addr[is.na(data$ERC20_uniq_rec_addr)] = median(data$ERC20_uniq_rec_addr,na.rm=T)
data$ERC20_uniq_sent_addr[is.na(data$ERC20_uniq_sent_addr)] = median(data$ERC20_uniq_sent_addr,na.rm=T)
data$ERC20_total_Ether_sent_contract[is.na(data$ERC20_total_Ether_sent_contract)] = median(data$ERC20_total_Ether_sent_contract,na.rm=T)
data$ERC20_total_ether_sent[is.na(data$ERC20_total_ether_sent)] = median(data$ERC20_total_ether_sent,na.rm=T)
data$ERC20_total_Ether_received[is.na(data$ERC20_total_Ether_received)] = median(data$ERC20_total_Ether_received,na.rm=T)
data$Total_ERC20_tnxs[is.na(data$Total_ERC20_tnxs)] = median(data$Total_ERC20_tnxs,na.rm=T)

# Viewing data characteristics post imputation
summary(data)

#Saving a copy of cleaned data
write.csv(data, file='median_imputed_and_cleaned.csv')

#Correlation plot for independent features after cleaning
corrgram(data[,2:31], order = F, upper.panel=panel.cor,text.panel=panel.txt, main = "Correlation Plot")

# Dropping features because of high correlation (>7.0)
data = data[ , -c('total_transactions_including_tnx_to_create_contract','total_ether_received','ERC20_total_Ether_received','ERC20_uniq_rec_contract_addr','ERC20_max_val_rec','ERC20_uniq_sent_token_name','Total_ERC20_tnxs')]


# Storing FLAG data in separate subset (i.e., flag)
flag = data$FLAG

# data scaling for data other than the FLAG column
data = scale(data[,2:24]) # scaled data
fin = cbind(flag, data) # binding the "flag" subset to the "data" subset
x <- as.data.frame(fin) # creating a data frame "x"
x$flag = as.factor(x$flag) # factorizing the FLAG column for catagorizing
old <- table(x$flag) # To check how many zeros and ones

# using SMOTE for oversampling the minority class
newdata <- SMOTE(flag~., x, perc.over = 100)

prop.table(table(newdata$flag))

# storing non-biased oversampled data
newfd <- table(newdata$flag)
training = createDataPartition(y= newdata$flag, p=0.80, list=FALSE)
train_set = newdata[training, ]
test_set = newdata[-training, ]

set.seed(1234)

# applying different machine learning techniques
#Bagging
model_bagging = train(data=train_set, flag~., method="treebag")
confusionMatrix(predict(model_bagging, newdata= train_set), train_set$flag)
confusionMatrix(predict(model_bagging, newdata= test_set), test_set$flag)
#Random_FOREST
model_forest = train(data=train_set, flag~., method="rf", prox=TRUE)
confusionMatrix(predict(model_forest, newdata= train_set), train_set$flag)
confusionMatrix(predict(model_forest, newdata= test_set), test_set$flag)
#Boosting
model_boosting = train(data=train_set, flag~., method="gbm", verbose=FALSE)
confusionMatrix(predict(model_boosting, newdata= train_set), train_set$flag)
confusionMatrix(predict(model_boosting, newdata= test_set), test_set$flag)
#CART
model_cart = train(data=train_set, flag~., method="rpart")
confusionMatrix(predict(model_cart, newdata= train_set), train_set$flag)
confusionMatrix(predict(model_cart, newdata= test_set), test_set$flag)

###################

# plotting the ROC curve for the boosting technique
f = predict(model_boosting, newdata=test_set, type = "prob")
f_roc=roc(test_set$flag, f[,"0"])
f_roc
plot(f_roc)

library(xgboost)
library(caret)

set.seed(38623919) # keeps consistency in result

df <- read.csv("cleaned_telco_churn_data.csv", header = TRUE, sep = ",")

View(content)

hist(df$Churn) # to view the distribution of churn=0 vs churn=1

# Here we do an 80/20 split of the train/test data
trainIndex <- createDataPartition(df$Churn, p = .8, list = FALSE)

# We split the dataframe into training and testing
train_df = df[trainIndex,]
test_df = df[-trainIndex,]

# Then split these dataframes into data and labels
train_data = data.matrix(train_df[, -10])
train_label  = train_df[,10]

test_data = data.matrix(test_df[,-10])
test_label = test_df[,10]

# We use XGBoost to create dense matrixes for the train and test data
xgb_train = xgb.DMatrix(data = train_data, label = train_label)
xgb_test = xgb.DMatrix(data = test_data, label=test_label)

# Used for evaluating the model as we train it
watchlist = list(train=xgb_train, test = xgb_test)

# Trains the model
xgb_model = xgb.train(data = xgb_train, max.depth = 3, watchlist=watchlist, 
                      nthread=2, nrounds=1000)

pred <- predict(xgb_model, test_data) # result is currently not 0-1, but decimal
prediction <- as.numeric(pred > 0.5)


# amount of Churns=0 and Churns=1 just to compare to predicted amount
length(test_label[test_label==0])
length(test_label[test_label==1])

# calculate average error of model
error <- mean(prediction != test_label)
paste("test-error=", error)

# calculate other accuracy predictions
paste("mean-squared-error=", mean((test_label - prediction)^2)); 
paste("mean-absolute-error=", caret::MAE(test_label, prediction));
paste("root-mean-squared-error=", caret::RMSE(test_label, prediction));

# confusion matrix as a table
table(test_label, prediction) 



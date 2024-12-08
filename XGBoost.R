library(xgboost)
library(caret)

set.seed(38623919) # seed used to keep consistency in result

df <- read.csv("cleaned_telco_churn_data.csv", header = TRUE, sep = ",")
hist(df$Churn) # to view the distribution of churn=0 vs churn=1

# Here we do an 80/20 split of the train/test data
trainIndex <- createDataPartition(df$Churn, p = .8, list = FALSE)

# We split the dataframe into training&testing then data&labels
train_df = df[trainIndex,]
test_df = df[-trainIndex,]

train_data = data.matrix(train_df[, -10])
train_label  = train_df[,10]
test_data = data.matrix(test_df[,-10])
test_label = test_df[,10]

# We use XGBoost to create dense matrixes for the train and test data
xgb_train = xgb.DMatrix(data = train_data, label = train_label)
xgb_test = xgb.DMatrix(data = test_data, label=test_label)

# Trains the model, using watchlist to evaluate as it trains
watchlist = list(train=xgb_train, test = xgb_test)
xgb_model = xgb.train(data = xgb_train, max.depth=4, watchlist=watchlist, 
                      nthread=2, nrounds=700, objective = "binary:logistic")

# calculate accuracy predictions
pred <- predict(xgb_model, test_data) 
prediction <- as.numeric(pred > 0.5)

paste("average-error=", mean(prediction != test_label))
paste("mean-squared-error=", mean((test_label - prediction)^2)); 
paste("mean-absolute-error=", caret::MAE(test_label, prediction));
paste("root-mean-squared-error=", caret::RMSE(test_label, prediction));

# confusion matrix (CM) as a table
table(test_label, prediction)
# https://www.damianoperri.it/public/confusionMatrix is used to visualize CM



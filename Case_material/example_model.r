library(Metrics)


# Define root path to data (INSERT PATH HERE)
setwd("")


# Import data sources
X_train = read.csv('./train/X_train.csv')
X_test = read.csv('./test/X_test.csv')
y_train = read.csv('./train/y_train.csv')


# Fit linear model
model <- lm(y_train[, 2] ~ ., X_train[, -1])


# Test to ensure predictions are formatted correctly
y_pred_train <- predict(model, X_train[, -1])
print(paste0("Mean absolute error is: ", mae(y_train[, 2], y_pred_train)))


# Create test predictions
y_pred_test <- predict(model, X_test[, -1])
y_pred_test <- data.frame(Prediction=y_pred_test)


# Save predictions to csv
write.csv(y_pred_test, file="./predictions.csv", row.names=F)


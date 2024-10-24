rm(list=ls())

library(lme4)
library(lmerTest)
library(emmeans)
library(msm)
library(vctrs)
library(ipw)
library(tidyverse)
library(survival)
library(xgboost)
library(caret)
library(MASS)
library(Matrix)
library(foreach)
library(doParallel)
library(data.table)
library(tidymodels)

memory.limit(500000000)
gc()

# 定义工作目录
setwd("xx")
# 读取数据
lipidsdata <- fread(
  "xx.csv",
  stringsAsFactors=FALSE,
  encoding = "UTF-8"
)

lipidsdata$outcome <-  as.numeric(
  as.character(lipidsdata$outcome))

# 调参网格
grid <- expand.grid(
  nrounds = 100
  max_depth = c(3,4,5,6,7,8),
  eta = c(0.01,0.02,0.03,0.06,0.1,0.2,0.3),
  gamma = c(0,0.01,0.05,0.1,0.2,0.5,1,2),
  colsample_bytree =  c(0.5,0.6,0.7,0.8,0.9,1),
  min_child_weight = c(1:6),
  subsample =  c(0.5,0.6,0.7,0.8,0.9,1)
)

# 划分数据集，80%为训练集，20%为测试集
set.seed(123)
lipidsdata_split <- initial_split(lipidsdata, prop = 0.80)
lipidsdata_train <- training(lipidsdata_split)
lipidsdata_test <- testing(lipidsdata_split)

# 矩阵数据加权

## 训练集

lipidsdata_train_td <- data.matrix(lipidsdata_train[, x1:x2]) %>%
  Matrix(., sparse = T) %>%
  list(data = ., y = lipidsdata_train$outcome)

lipidsdata_train_matrix <- xgb.DMatrix(
  data = lipidsdata_train_td$data,
  label = lipidsdata_train_td$y,
  weight = lipidsdata_train$Weight1
)

## 测试集

lipidsdata_test_td <- data.matrix(lipidsdata_test[, x1:x2]) %>%
  Matrix(., sparse = T) %>%
  list(data = ., y = as.factor(lipidsdata_test$outcome))

lipidsdata_test_matrix <- xgb.DMatrix(
  data = lipidsdata_test_td$data,
  label = lipidsdata_test_td$y,
  weight = lipidsdata_test$Weight1
)

# 循环 ----------------------------------------------------------------------

options(warn = -1)

cl <- makePSOCKcluster(10) # 依据个人电脑定线程数
registerDoParallel(cl)

results_all <- map(
  1:1000,
  function(i) {
    
    print(str_glue("开始种子数为{i}的交叉验证--------------------------------"))
    
    # 交叉验证
    
    set.seed(i)
    random_search <- train(
      outcome ~Covariates,
      data = lipidsdata_train,
      method = "xgbTree",
      tuneGrid = grid,
      metric = "RMSE",       # 选择评价指标
      trControl = trainControl(
        method = "cv",       # 交叉验证
        number = 10,         # 折数
        allowParallel = T    # 允许并行
      )
    )
    best_params <- random_search$bestTune      # 最终模型参数
    tune_results <- random_search$results %>%  # 所有评价指标
      mutate(seed = i)
    
    # 拟合最终模型
    
    mod <- xgb.train(
      data = lipidsdata_train_matrix,
      objective = "binary:logistic",
      nrounds = best_params$nrounds,
      max_depth = best_params$max_depth,
      eta = best_params$eta,
      gamma = best_params$gamma,
      colsample_bytree = best_params$colsample_bytree,
      min_child_weight = best_params$min_child_weight,
      subsample = best_params$subsample
    )
    
    # 训练集预测
    
    predicted_train_f <- predict(
      mod, newdata = lipidsdata_train_matrix, weights = lipidsdata$Weight1
    ) %>%
      as_tibble() %>%
      `colnames<-`(str_c("prediction_", i))
    
    # 测试集预测
    predicted_test_f <- predict(
      mod, newdata = lipidsdata_test_matrix, weights = lipidsdata$Weight1
    ) %>%
      as_tibble() %>%
      `colnames<-`(str_c("prediction_", i))
    
    # 模型预测值和评价指标输出
    
    results_f <- list(
      "训练集预测" = predicted_train_f,
      "测试集预测" = predicted_test_f,
      "评价指标" = tune_results
    )
    
    print(str_glue("完成种子数为{i}的交叉验证-----------------------------------"))
    
    return(results_f)
  })

stopCluster(cl)

options(warn = 1) # 打开warning

# 合并结果
## 预测值
predicted_train_results <- lipidsdata_train %>%
  bind_cols(
    results_all %>%
      map(~ .x[["训练集预测"]]) %>%
      reduce(bind_cols))

predicted_test_results <- lipidsdata_test %>%
  bind_cols(
    results_all %>%
      map(~ .x[["测试集预测"]]) %>%
      reduce(bind_cols))

## 模型拟合结果
metric_all <- results_all %>%
  map(~ .x[["评价指标"]]) %>%
  reduce(bind_rows)

# 保存数据
write.csv(predicted_train_results, "xx.csv" # 填写文件地址)
write.csv(metric_all,"xx.csv" # 填写文件地址)
write.csv( predicted_test_results,"xx.csv" # 填写文件地址)


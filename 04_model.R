# Modeling and Evaluation

library(glmnet)
library(dplyr)
library(lme4)
library(tidyr)
library(pROC)
library(tidyverse)
library(patchwork)
library(Hmisc)
library(caret)

########################### data preparation #########################

# initial data
load("E:/EntroLLM/data_wide.RData") 

data_wide$Gender <- as.character(data_wide$Gender)
data_wide$Race <- as.character(data_wide$Race)
data_wide$Education <- as.character(data_wide$Education)
data_wide$Married <- as.character(data_wide$Married)

# dataset with only demographic variables
data_demo_only = data_wide %>% 
  select(SEQN,Gender,Age,Race,Education,Married,PIR,BMI)

# GPT with 1536 embedding dimension
data_gpt1536 <- read.csv("E:/EntroLLM/data_wide_embedding_gpt1536.csv") %>%
  dplyr::select(-combined,-X)

data_gpt1536$embedding <- gsub("\\[", "", data_gpt1536$embedding)
data_gpt1536$embedding <- gsub("\\]", "", data_gpt1536$embedding)

data_gpt1536 <- data_gpt1536 %>%
  separate(embedding, into = paste0("var", 1:1536), sep = ",\\s*", convert = TRUE) %>%
  mutate(across(starts_with("var"), as.numeric)) 

data_gpt1536$Gender <- as.character(data_gpt1536$Gender)
data_gpt1536$Race <- as.character(data_gpt1536$Race)
data_gpt1536$Education <- as.character(data_gpt1536$Education)
data_gpt1536$Married <- as.character(data_gpt1536$Married)

# GPT with 50 embedding dimension
data_gpt50 <- read.csv("E:/EntroLLM/data_wide_embedding_gpt50.csv") %>% 
  dplyr::select(-combined,-n_tokens) 

data_gpt50$Gender <- as.character(data_gpt50$Gender)
data_gpt50$Race <- as.character(data_gpt50$Race)
data_gpt50$Education <- as.character(data_gpt50$Education)
data_gpt50$Married <- as.character(data_gpt50$Married)

# BERT with 768 embedding dimension
data_bert768 <- read.csv("E:/EntroLLM/data_wide_embedding_bert768.csv") %>% 
  dplyr::select(-X,-combined)

data_bert768$embedding <- gsub("\\[", "", data_bert768$embedding)
data_bert768$embedding <- gsub("\\]", "", data_bert768$embedding)
data_bert768$embedding <- gsub("\n", " ", data_bert768$embedding)

data_bert768 <- data_bert768 %>%
  mutate(embedding = str_trim(embedding),  # Remove leading and trailing spaces
         embedding = str_replace_all(embedding, "\\s+", " ")) # Replace multiple spaces with a single space

data_bert768 <- data_bert768 %>%
  separate(embedding, into = paste0("var", 1:768), sep = "\\s+", convert = TRUE) %>%
  mutate(across(starts_with("var"), as.numeric))

data_bert768$Gender <- as.character(data_bert768$Gender)
data_bert768$Race <- as.character(data_bert768$Race)
data_bert768$Education <- as.character(data_bert768$Education)
data_bert768$Married <- as.character(data_bert768$Married)

# BERT with 50 embedding dimension
data_bert50 <- read.csv("E:/EntroLLM/data_wide_embedding_bert50.csv") %>% 
  dplyr::select(-X,-combined,-n_tokens)

data_bert50$Gender <- as.character(data_bert50$Gender)
data_bert50$Race <- as.character(data_bert50$Race)
data_bert50$Education <- as.character(data_bert50$Education)
data_bert50$Married <- as.character(data_bert50$Married)


# Cohere with 1024 embedding dimension
data_cohere1024 <- read.csv("E:/EntroLLM/data_wide_embedding_cohere1024.csv") %>% 
  dplyr::select(-X,-n_tokens)

data_cohere1024$embedding <- gsub("\\[", "", data_cohere1024$embedding) 
data_cohere1024$embedding <- gsub("\\]", "", data_cohere1024$embedding)
data_cohere1024$embedding <- gsub(",", " ", data_cohere1024$embedding)

data_cohere1024 <- data_cohere1024 %>%
  mutate(embedding = str_trim(embedding),  # Remove leading and trailing spaces
         embedding = str_replace_all(embedding, "\\s+", " ")) # Replace multiple spaces with a single space

data_cohere1024 <- data_cohere1024 %>%
  separate(embedding, into = paste0("var", 1:1024), sep = "\\s+", convert = TRUE) %>% 
  mutate(across(starts_with("var"), as.numeric))

data_cohere1024$Gender <- as.character(data_cohere1024$Gender)
data_cohere1024$Race <- as.character(data_cohere1024$Race)
data_cohere1024$Education <- as.character(data_cohere1024$Education)
data_cohere1024$Married <- as.character(data_cohere1024$Married)

# Cohere with 50 embedding dimension
data_cohere50 <- read.csv("E:/EntroLLM/data_wide_embedding_cohere50.csv") %>% 
  dplyr::select(-X,-combined,-n_tokens)

data_cohere50$Gender <- as.character(data_cohere50$Gender)
data_cohere50$Race <- as.character(data_cohere50$Race)
data_cohere50$Education <- as.character(data_cohere50$Education)
data_cohere50$Married <- as.character(data_cohere50$Married)

# Entropy
data_entropy <- read.csv("E:/EntroLLM/data_wide_entropy.csv") %>% 
  dplyr::select(-X)

data_entropy$Gender <- as.character(data_entropy$Gender)
data_entropy$Race <- as.character(data_entropy$Race)
data_entropy$Education <- as.character(data_entropy$Education)
data_entropy$Married <- as.character(data_entropy$Married)

# GPT1536 + entropy
data_gpt1536_entropy=data_entropy %>% 
  dplyr::select(SEQN,Entropy_Day1:Entropy_Day7) %>% 
  inner_join(data_gpt1536,by="SEQN")

# GPT50 + entropy
data_gpt50_entropy=data_entropy %>% 
  dplyr::select(SEQN,Entropy_Day1:Entropy_Day7) %>% 
  inner_join(data_gpt50,by="SEQN")

# BERT768 + entropy
data_bert768_entropy=data_entropy %>% 
  dplyr::select(SEQN,Entropy_Day1:Entropy_Day7) %>% 
  inner_join(data_bert768,by="SEQN")

# BERT50 + entropy
data_bert50_entropy=data_entropy %>% 
  dplyr::select(SEQN,Entropy_Day1:Entropy_Day7) %>% 
  inner_join(data_bert50,by="SEQN")

# Cohere1024 + entropy
data_cohere1024_entropy=data_entropy %>% 
  dplyr::select(SEQN,Entropy_Day1:Entropy_Day7) %>% 
  inner_join(data_cohere1024,by="SEQN")

# Cohere50 + entropy
data_cohere50_entropy=data_entropy %>% 
  dplyr::select(SEQN,Entropy_Day1:Entropy_Day7) %>% 
  inner_join(data_cohere50,by="SEQN")


########################### AUC for ridge/lasso #########################
model_auc <- function(data_wide, sim){ 
  
  auc <- data.frame()
  
  for (i in 1:sim){
    
    set.seed(i)
    select_train = sample(nrow(data_wide), floor(nrow(data_wide)*0.8), replace = FALSE)
    train = data_wide[select_train,]
    test=data_wide[-select_train,]
    
    train_x=as.matrix(train[, !(names(train) %in% c("BMI", "SEQN"))])
    train_y <- train$BMI
    test_x <- as.matrix(test[, !(names(test) %in% c("BMI", "SEQN"))])
    test_y <- test$BMI  
    
    #############
    ### ridge
    #############
    
    cv.ridge <- cv.glmnet(train_x,train_y,alpha = 0,family = 'binomial',
                          standardize=TRUE,grouped=FALSE,nfolds = 10)
    coef_ridge=predict(cv.ridge, s = "lambda.min", type = "coefficients")
    
    y_predicted_pro_ridge <- predict(cv.ridge,s=cv.ridge$lambda.min,newx=test_x)
    
    y_predicted_ridge= exp(y_predicted_pro_ridge)/(1+exp(y_predicted_pro_ridge))
    y_predicted_ridge <- as.numeric(y_predicted_ridge)
    
    auc_ridge = auc(test_y, y_predicted_ridge)
    
    
    #############
    ### lasso
    #############
    
    # cv.lasso <- cv.glmnet(train_x, train_y, alpha = 1, family = 'binomial',
    #                       standardize = TRUE, grouped = FALSE, nfolds = 10)
    # # coef_lasso = predict(cv.lasso, s = "lambda.min", type = "coefficients")
    # 
    # y_predicted_pro_lasso <- predict(cv.lasso, s = cv.lasso$lambda.min, newx = test_x)
    # 
    # y_predicted_lasso = exp(y_predicted_pro_lasso) / (1 + exp(y_predicted_pro_lasso))
    # y_predicted_lasso <- as.numeric(y_predicted_lasso)
    # 
    # auc_lasso = auc(test_y, y_predicted_lasso)
    
    inter.auc = data.frame(auc_ridge = auc_ridge)
    auc = rbind(auc, inter.auc) 
    
  }  
  return(auc=auc)
}

auc_demo_only = model_auc(data_demo_only, sim = 20)
saveRDS(auc_demo_only, file = "E:/EntroLLM/auc_demo_only.rds")

auc_base <- model_auc(data_wide, sim = 20)
saveRDS(auc_base, file = "E:/EntroLLM/auc_base.rds")

auc_EntroGPT1536 <- model_auc(data_gpt1536_entropy, sim = 20)
saveRDS(auc_EntroGPT1536, file = "E:/EntroLLM/auc_EntroGPT1536.rds")

auc_EntroGPT50 <- model_auc(data_gpt50_entropy, sim = 20)
saveRDS(auc_EntroGPT50, file = "E:/EntroLLM/auc_EntroGPT50.rds")

auc_EntroBERT768 <- model_auc(data_bert768_entropy, sim = 20)
saveRDS(auc_EntroBERT768, file = "E:/EntroLLM/auc_EntroBERT768.rds")

auc_EntroBERT50 <- model_auc(data_bert50_entropy, sim = 20)
saveRDS(auc_EntroBERT50, file = "E:/EntroLLM/auc_EntroBERT50.rds")

auc_EntroCohere1024 <- model_auc(data_cohere1024_entropy, sim = 20)
saveRDS(auc_EntroCohere1024, file = "E:/EntroLLM/auc_EntroCohere1024.rds")

auc_EntroCohere50 <- model_auc(data_cohere50_entropy, sim = 20)
saveRDS(auc_EntroCohere50, file = "E:/EntroLLM/auc_EntroCohere50.rds")


###########################################################################

########################### AUC for Entropy-only Model #########################

auc_logistic <- function(data_wide, sim){ 
  
  auc <- data.frame()
  # roc <- list()
  
  for (i in 1:sim){
    
    set.seed(i)
    select_train = sample(nrow(data_wide), floor(nrow(data_wide)*0.8), replace = FALSE)
    train = data_wide[select_train,]
    test=data_wide[-select_train,]
    
    glm.fit <- glm(BMI ~ Gender + Age + Race +Education+Married+PIR+Entropy_Day1 + Entropy_Day2 + Entropy_Day3 + Entropy_Day4 + Entropy_Day5 + Entropy_Day6 + Entropy_Day7, 
                   data = train, 
                   family = 'binomial')
    
    y_predicted_pro = predict(glm.fit, newdata=test)
    y_predicted= exp(y_predicted_pro)/(1+exp(y_predicted_pro))
    y_predicted <- as.numeric(y_predicted)
    
    auc_logistic=auc(test$BMI, y_predicted)
    
    inter.auc = cbind(auc_logistic)
    auc = rbind(auc, inter.auc)
    
    
  }  
  return(auc=auc)
}

auc_entropy_only=auc_logistic(data_entropy,20)
saveRDS(auc_entropy_only, file = "E:/EntroLLM/auc_entropy_only.rds")

###########################################################################

########################### F1 score for ridge/lasso #########################

model_f1 <- function(data_wide, seed, alpha, interval) {
  
  # ridge: alpha=0
  # lasso: alpha=1
  
  set.seed(seed)
  select_train = sample(nrow(data_wide), floor(nrow(data_wide)*0.8), replace = FALSE)
  train = data_wide[select_train,]
  test=data_wide[-select_train,]
  
  train_x=as.matrix(train[, !(names(train) %in% c("BMI", "SEQN"))])
  train_y <- train$BMI
  test_x <- as.matrix(test[, !(names(test) %in% c("BMI", "SEQN"))])
  test_y <- test$BMI 
  
  cv.fit <- cv.glmnet(train_x,train_y,alpha = alpha,family = 'binomial',
                      standardize=TRUE,grouped=FALSE,nfolds = 10)
  # coef_ridge=predict(cv.ridge, s = "lambda.min", type = "coefficients")
  
  y_predicted_pro <- predict(cv.fit,s=cv.fit$lambda.min,newx=test_x)
  
  y_predicted= exp(y_predicted_pro)/(1+exp(y_predicted_pro))
  y_predicted <- as.numeric(y_predicted)
  
  roc=roc(test_y, y_predicted)
  
  # Initialize variables to store the best cutoff and the best F1 score
  best_cutoff <- 0
  best_f1 <- 0
  
  # Try different cutoff values to find the one that maximizes the F1 score
  for (cutoff in seq(0.5, 1, by=interval)) {
    y_hat <- ifelse(y_predicted >= cutoff, 1, 0)
    f1 <- F1_Score(y_hat,test_y)
    if (!is.na(f1) && f1 > best_f1) {
      best_f1 <- f1
      best_cutoff <- cutoff
    }
  }
  
  # calculate true positive, false positive, precision and recall based on maximum F1 score
  maxF1_y_hat <- ifelse(y_predicted >= best_cutoff, 1, 0)
  maxF1_confusion <- confusionMatrix(factor(maxF1_y_hat, levels=c(0, 1)), factor(test_y, levels=c(0, 1)))$table
  maxF1_precision <- maxF1_confusion[2, 2] / sum(maxF1_confusion[2, ])
  maxF1_recall <- maxF1_confusion[2, 2] / sum(maxF1_confusion[, 2])
  maxF1_TP = maxF1_confusion[2, 2]
  maxF1_FP = maxF1_confusion[2, 1]
  
  
  return(list(roc=roc,best_cutoff=best_cutoff, best_f1=best_f1,maxF1_precision=maxF1_precision,maxF1_recall=maxF1_recall,maxF1_TP=maxF1_TP,maxF1_FP=maxF1_FP))
}

F1_Score <- function(y_hat,y_true) {
  confusion <- confusionMatrix(factor(y_hat, levels=c(0, 1)), factor(y_true, levels=c(0, 1)))$table
  precision <- confusion[2, 2] / sum(confusion[2, ])
  recall <- confusion[2, 2] / sum(confusion[, 2])
  f1 <- 2 * (precision * recall) / (precision + recall)
  return(f1)
}


f1_demo_only <- model_f1(data_demo_only, seed=1, alpha=0, interval=0.01)
saveRDS(f1_demo_only, file = "E:/EntroLLM/f1_demo_only.rds")

f1_base <- model_f1(data_wide, seed=1, alpha=0, interval=0.01)
saveRDS(f1_base, file = "E:/EntroLLM/f1_base.rds")

f1_EntroGPT1536 <- model_f1(data_gpt1536_entropy, seed=14, alpha=0, interval=0.01) # seed=14
saveRDS(f1_EntroGPT1536, file = "E:/EntroLLM/f1_EntroGPT1536.rds")

f1_EntroGPT50 <- model_f1(data_gpt50_entropy, seed=1, alpha=0, interval=0.01)
saveRDS(f1_EntroGPT50, file = "E:/EntroLLM/f1_EntroGPT50.rds")

f1_EntroBERT768 <- model_f1(data_bert768_entropy, seed=1, alpha=0, interval=0.01)
saveRDS(f1_EntroBERT768, file = "E:/EntroLLM/f1_EntroBERT768.rds")

f1_EntroBERT50 <- model_f1(data_bert50_entropy, seed=1, alpha=0, interval=0.01)
saveRDS(f1_EntroBERT50, file = "E:/EntroLLM/f1_EntroBERT50.rds")

f1_EntroCohere1024 <- model_f1(data_cohere1024_entropy, seed=1, alpha=0, interval=0.01)
saveRDS(f1_EntroCohere1024, file = "E:/EntroLLM/f1_EntroCohere1024.rds")

f1_EntroCohere50 <- model_f1(data_cohere50_entropy, seed=1, alpha=0, interval=0.01)
saveRDS(f1_EntroCohere50, file = "E:/EntroLLM/f1_EntroCohere50.rds")
###########################################################################

########################### F1 score for Entropy-only Model #########################


model_f1_entropy <- function(data_wide, seed, interval) {
  
  set.seed(seed)
  select_train = sample(nrow(data_wide), floor(nrow(data_wide)*0.8), replace = FALSE)
  train = data_wide[select_train,]
  test=data_wide[-select_train,]
  
  glm.fit <- glm(BMI ~ Gender + Age + Race +Education+Married+PIR+Entropy_Day1 + Entropy_Day2 + Entropy_Day3 + Entropy_Day4 + Entropy_Day5 + Entropy_Day6 + Entropy_Day7, 
                 data = train, 
                 family = 'binomial')
  
  y_predicted_pro = predict(glm.fit, newdata=test)
  y_predicted= exp(y_predicted_pro)/(1+exp(y_predicted_pro))
  y_predicted <- as.numeric(y_predicted)
  
  roc=roc(test$BMI, y_predicted)
  
  # Initialize variables to store the best cutoff and the best F1 score
  best_cutoff <- 0
  best_f1 <- 0
  
  # Try different cutoff values to find the one that maximizes the F1 score
  for (cutoff in seq(0.5, 1, by=interval)) {
    y_hat <- ifelse(y_predicted >= cutoff, 1, 0)
    f1 <- F1_Score(y_hat,test$BMI)
    if (!is.na(f1) && f1 > best_f1) {
      best_f1 <- f1
      best_cutoff <- cutoff
    }
  }
  
  # calculate true positive, false positive, precision and recall based on maximum F1 score
  maxF1_y_hat <- ifelse(y_predicted >= best_cutoff, 1, 0)
  maxF1_confusion <- confusionMatrix(factor(maxF1_y_hat, levels=c(0, 1)), factor(test$BMI, levels=c(0, 1)))$table
  maxF1_precision <- maxF1_confusion[2, 2] / sum(maxF1_confusion[2, ])
  maxF1_recall <- maxF1_confusion[2, 2] / sum(maxF1_confusion[, 2])
  maxF1_TP = maxF1_confusion[2, 2]
  maxF1_FP = maxF1_confusion[2, 1]
  
  
  return(list(roc=roc,best_cutoff=best_cutoff, best_f1=best_f1,maxF1_precision=maxF1_precision,maxF1_recall=maxF1_recall,maxF1_TP=maxF1_TP,maxF1_FP=maxF1_FP))
}

F1_Score <- function(y_hat,y_true) {
  confusion <- confusionMatrix(factor(y_hat, levels=c(0, 1)), factor(y_true, levels=c(0, 1)))$table
  precision <- confusion[2, 2] / sum(confusion[2, ])
  recall <- confusion[2, 2] / sum(confusion[, 2])
  f1 <- 2 * (precision * recall) / (precision + recall)
  return(f1)
}

f1_entropy_only=model_f1_entropy(data_entropy,seed=1,interval = 0.01)
saveRDS(f1_entropy_only, file = "E:/EntroLLM/f1_entropy_only.rds")
###########################################################################

########################### Result #########################

# read in AUC results
auc_file_names <- list(
  auc_demo_only = "auc_demo_only.rds",
  auc_base = "auc_base.rds",
  auc_EntroGPT1536 = "auc_EntroGPT1536.rds",
  auc_EntroGPT50 = "auc_EntroGPT50.rds",
  auc_EntroBERT768 = "auc_EntroBERT768.rds",
  auc_EntroBERT50 = "auc_EntroBERT50.rds",
  auc_EntroCohere1024 = "auc_EntroCohere1024.rds",
  auc_EntroCohere50 = "auc_EntroCohere50.rds",
  auc_entropy_only = "auc_entropy_only.rds"
)

path <- "E:/EntroLLM/"

for (var_name in names(auc_file_names)) {
  file_path <- paste0(path, auc_file_names[[var_name]])
  assign(var_name, readRDS(file = file_path))
}

# read in F1 score results
f1_model_names <- list(
  f1_demo_only = "f1_demo_only.rds",
  f1_base = "f1_base.rds",
  f1_EntroGPT1536 = "f1_EntroGPT1536.rds",
  f1_EntroGPT50 = "f1_EntroGPT50.rds",
  f1_EntroBERT768 = "f1_EntroBERT768.rds",
  f1_EntroBERT50 = "f1_EntroBERT50.rds",
  f1_EntroCohere1024 = "f1_EntroCohere1024.rds",
  f1_EntroCohere50 = "f1_EntroCohere50.rds",
  f1_entropy_only = "f1_entropy_only.rds"
)

for (var_name in names(f1_model_names)) {
  file_path <- paste0(path, f1_model_names[[var_name]])
  assign(var_name, readRDS(file = file_path))
}


# AUC Boxplot 

AUC_ridge = cbind(auc_demo_only, auc_base, auc_EntroBERT768,auc_EntroBERT50,
                  auc_EntroCohere1024, auc_EntroCohere50,auc_entropy_only,auc_EntroGPT50,auc_EntroGPT1536)
colnames(AUC_ridge)=c("Demographic-only","Base","EntroBERT768","EntroBERT50",
                      "EntroCohere1024", "EntroCohere50","Entropy-only","EntroGPT50","EntroGPT1536")


AUC_ridge <- AUC_ridge %>%
  mutate(across(everything(), as.numeric)) %>% 
  pivot_longer(cols = everything(),  
               names_to = "Model",  
               values_to = "AUC")  

AUC_ridge$Model <- factor(AUC_ridge$Model, levels = c("Demographic-only","Base","EntroBERT768","EntroBERT50", 
                                                      "EntroCohere1024", "EntroCohere50","Entropy-only","EntroGPT50","EntroGPT1536"))

ggplot(AUC_ridge, aes(x = Model, y = AUC, fill = Model)) +
  geom_boxplot() +
  theme(
    legend.text = element_text(hjust = 0),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.spacing.x = unit(2, "cm"),
    panel.grid.major = element_line(colour = "grey80", size = 0.5),  
    panel.grid.minor = element_line(colour = "grey90", size = 0.25)  
  )  


# ROC plot.

f1_file_names <- list(
  f1_demo_only = "f1_demo_only",
  f1_base = "f1_base",
  f1_EntroGPT1536 = "f1_EntroGPT1536",
  f1_EntroGPT50 = "f1_EntroGPT50",
  f1_EntroBERT768 = "f1_EntroBERT768",
  f1_EntroBERT50 = "f1_EntroBERT50",
  f1_EntroCohere1024 = "f1_EntroCohere1024",
  f1_EntroCohere50 = "f1_EntroCohere50",
  f1_entropy_only = "f1_entropy_only"
)



roc <- function(Model, filename) {
  data.frame(
    specificity = Model[["roc"]][["specificities"]],
    sensitivity = Model[["roc"]][["sensitivities"]],
    Model = filename
  )
}

roc_list <- list()

for (var_name in f1_file_names) {

    model_data <- get(var_name)
    filename_prefix <- sub("^f1_", "", var_name)
    filename <- filename_prefix
    roc_list[[filename]] <- roc(model_data, filename_prefix)

}

new_model_names <- c("Demographic-only","Base", "EntroGPT1536", "EntroGPT50",
                     "EntroBERT768", "EntroBERT50", "EntroCohere1024", 
                     "EntroCohere50", "Entropy-only")

names(roc_list) <- new_model_names 

for (i in seq_along(roc_list)) {
  roc_list[[i]]$Model <- new_model_names[i]
}

roc_data <- do.call(rbind, roc_list)

roc_data$Model <- factor(roc_data$Model, levels = c("Demographic-only","Base","EntroBERT768","EntroBERT50","EntroCohere1024","EntroCohere50","Entropy-only","EntroGPT50","EntroGPT1536"))


ggplot(roc_data, aes(x = specificity, y = sensitivity, color = Model)) +
  geom_line() +
  theme( 
    legend.spacing.x = unit(2, "cm"),
    legend.justification = "left",   
    legend.box.just = "left",        
    legend.text = element_text(hjust = 0),
    panel.grid.major = element_blank(),  
    panel.grid.minor = element_blank()         
  ) +
  labs(x = "Specificity", y = "Sensitivity") 


# High-risk (HR) patients classification.

f1_file_names <- list(
  f1_demo_only = "f1_demo_only",
  f1_base = "f1_base",
  f1_EntroGPT1536 = "f1_EntroGPT1536",
  f1_EntroGPT50 = "f1_EntroGPT50",
  f1_EntroBERT768 = "f1_EntroBERT768",
  f1_EntroBERT50 = "f1_EntroBERT50",
  f1_EntroCohere1024 = "f1_EntroCohere1024",
  f1_EntroCohere50 = "f1_EntroCohere50",
  f1_entropy_only = "f1_entropy_only"
)


F1_data <- data.frame()


for (var_name in f1_file_names) {

    model <- get(var_name)
    filename_prefix <- sub("^f1_", "", var_name)
    temp_data <- data.frame(
      Model = filename_prefix,
      TruePositives = model$maxF1_TP,
      FalsePositives = model$maxF1_FP,
      Precision = model$maxF1_precision,
      Recall = model$maxF1_recall,
      MaxF1 = model$best_f1
    )
    F1_data <- rbind(F1_data, temp_data)

}


new_model_names <- c("Demographic-only","Base", "EntroGPT1536", "EntroGPT50",
                     "EntroBERT768", "EntroBERT50", "EntroCohere1024", 
                     "EntroCohere50", "Entropy-only")

F1_data$Model <- new_model_names

F1_data$Model <- factor(F1_data$Model, levels = c("Demographic-only","Base","EntroBERT768","EntroBERT50","EntroCohere1024","EntroCohere50","Entropy-only","EntroGPT50","EntroGPT1536"))


stacked_data <- F1_data %>%
  gather(key = "Type", value = "Count", TruePositives, FalsePositives)


ggplot(stacked_data, aes(fill = Type, y = Count, x = Model)) +
  geom_bar(position = "stack", stat = "identity") +
  geom_text(aes(label = Count), position = position_stack(vjust = 0.5)) +
  scale_fill_manual(values = c("TruePositives" = "#619CFF", "FalsePositives" = "#F8766D")) +
  labs(y = "Number of HR Patients",
       fill = "Type") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


line_data <- F1_data %>%
  gather(key = "Metric", value = "Score", Precision, Recall, MaxF1)


ggplot(line_data, aes(x = Model, y = Score, color = Metric, group = Metric)) +
  geom_line() +
  geom_point() +
  labs(y = "Scores",
       color = "Metric") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))









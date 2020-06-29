#Preparing Work Space
rm(list=ls()) #Clear all objects from environment
dev.off()     #Clear the graph window
cat("\014")   #Clear the console
####################################################
#               IST 707 DATA ANALYTIC              #
#                    Project                       #
#                HOTEL DATA SET                    #
#               DATA CLEANING FILE                 #
####################################################
library(tidyverse)
library(skimr)
library(caret)
library(rattle)
library(data.table)
library(cluster)    # clustering algorithms
library(factoextra) # clustering visualization
library(dendextend) # for comparing two dendrograms
library(tictoc) #to calculate time taken to run models
library(arules)
library(mltools)
library(kernlab)
library(ROCR)
library(C50)

set.seed(1)
#Extracting the dataset
raw_city_hotel <- read.csv("2/H2.csv", header = T, strip.white = T) %>% na.omit()

str(raw_city_hotel)

#Data Cleaning
city_hotel_df <- raw_city_hotel %>%
  mutate(IsCanceled = factor(IsCanceled),
         IsRepeatedGuest = factor(IsRepeatedGuest),
         RoomOfChoice = factor(case_when(
           as.character(AssignedRoomType) == as.character(ReservedRoomType) ~ 1,
           TRUE ~ 0
           ))) %>%
  select(-ArrivalDateMonth,
         -ArrivalDateYear,
         -ArrivalDateDayOfMonth,
         -ArrivalDateWeekNumber,
         -ReservationStatusDate,
         -ReservationStatus,
         -Country,
         -Agent,
         -AssignedRoomType,
         -ReservedRoomType,
         -Company)

#Converting data frame to data table
city_hotel <- as.data.table(city_hotel_df)

#one hot encoding of categorical columns
city_hotel_encoded <- one_hot(as.data.table(city_hotel[,-1]), dropUnusedLevels = T)
city_hotel_encoded$IsCanceled <- city_hotel$IsCanceled

#converting numerical columns to categorical columns
city_hotel_cat <- city_hotel %>% mutate_if(is.numeric, funs(discretize(., method="interval", breaks = 3)))

#Control object for all models
ctrl <- caret::trainControl(method = "cv",
                            number = 3,
                            selectionFunction = "oneSE")

#Naive Bayes
modelLookup('nb')
#BaseModel
tictoc::tic('nb_time_base')
features <- setdiff(names(city_hotel_cat), "IsCanceled")
nb_basemodel <- caret::train(city_hotel_cat[,features],
                         city_hotel_cat$IsCanceled,
                         method = "nb",
                         trControl = ctrl)
tictoc::toc()
confusionMatrix(nb_basemodel)

#tuning
nb_grid <- expand.grid(fL = c(0:4),
                       usekernel = F,
                       adjust = 1)

tictoc::tic('nb_time')
nb_model <- caret::train(city_hotel_cat[,features],
                         city_hotel_cat$IsCanceled,
                         method = "nb",
                         metric = 'Kappa',
                         trControl = ctrl,
                         tuneGrid = nb_grid)
tictoc::toc()
plot(nb_model)
nb_metrics <- confusionMatrix(nb_model)

#Desicion Trees
##rpart_base model
tictoc::tic('rpart base model')
rpart_basemodel <- caret::train(IsCanceled ~ .,
                            data = city_hotel,
                            method = "rpart",
                            trControl = ctrl,
                            tuneGrid = expand.grid(cp = 0.01))
tictoc::toc()
rpart_basemetrics <- confusionMatrix(rpart_basemodel)
#rpart tuning
rpart_grid <- expand.grid(cp = c(0.001,0.01,0.1,1))
tictoc::tic('rpart_time')
rpart_model <- caret::train(IsCanceled ~ .,
                            data = city_hotel,
                            method = "rpart",
                            metric = "Kappa",
                            trControl = ctrl,
                            tuneGrid = rpart_grid)
tictoc::toc()
rpart_metrics <- confusionMatrix(rpart_model)
plot(rpart_model)
rpart.plot::rpart.plot(rpart_model$finalModel, tweak = 2)

#C5.0
#base model
tictoc::tic("C5.0 base model")
c50_basemodel <- caret::train(IsCanceled ~ .,
                              data = city_hotel,
                              method = "C5.0",
                              trControl = ctrl,
                              tuneGrid = expand.grid(trials=1,winnow = F, model = 'tree'))
tictoc::toc()
c50_basemetrics <- confusionMatrix(c50_basemodel)

#tuning
modelLookup('C5.0')
c50_grid <- expand.grid(trials = c(1,10,20,30,40,50),
                        model = 'tree',
                        winnow = c(T,F))

tictoc::tic("C5.0")
c50_model <- caret::train(IsCanceled ~ .,
                          data = city_hotel,
                          method = "C5.0",
                          metric = "kappa",
                          trControl = ctrl,
                          tuneGrid = c50_grid)
tictoc::toc()
plot(c50_model)
c50_metrics <- confusionMatrix(c50_model)

#KNN
modelLookup('knn')

#Including the only best model in grid
#It takes long time to run, for grids of length>1
knn_model <- train(IsCanceled ~ ., 
                   data = city_hotel_encoded, 
                   method = "knn", 
                   trControl = ctrl,
                   preProcess = c("center","scale"),
                   tuneGrid = expand.grid(k=13) #c(5,7,9,11,13)
                   )


knn_metrics <- confusionMatrix(knn_model)
plot(knn_model)

#SVM
library(kernlab)
set.seed(123)
svm_data <- as.data.frame(city_hotel_encoded)
split <- createDataPartition(svm_data$IsCanceled, p=0.7, list = F)
training_svm <- svm_data[split,] 
testing_svm <- svm_data[-split,] 

svm_model <- ksvm(IsCanceled ~ ., 
                  data = training_svm)

svm_prediction <- predict(svm_model, testing_svm)


confusionMatrix(reference = testing_svm$IsCanceled, data = svm_prediction, mode = 'everything')

#Model using vanilladot
svm_model2 <- ksvm(IsCanceled ~ ., 
                  data = training_svm,
                  kernel = 'vanilladot')

svm_prediction2 <- predict(svm_model2, testing_svm)
confusionMatrix(reference = testing_svm$IsCanceled, data = svm_prediction2, mode = 'everything')

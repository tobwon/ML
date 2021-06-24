## ----setup, include=FALSE, warning=FALSE, message=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)
options(scipen = 100)


## -----------------------------------------------------------------------------------------------------------
#Target RMSE
target<-0.86490


## -----------------------------------------------------------------------------------------------------------
# RMSE formula
RMSE <- function(true_ratings, predicted_ratings){
    sqrt(mean((true_ratings - predicted_ratings)^2))
  }


## ----warning=FALSE, message=FALSE---------------------------------------------------------------------------
###########################################################
# Create edx set, validation set (final hold-out test set)#
###########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                            title = as.character(title),
                                            genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                            title = as.character(title),
                                            genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
      semi_join(edx, by = "movieId") %>%
      semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


## -----------------------------------------------------------------------------------------------------------
# Summary of "edx" data-set
summary(edx)


## -----------------------------------------------------------------------------------------------------------
# Display the first 5 rows of "edx"
head(edx)


## -----------------------------------------------------------------------------------------------------------
# Check the genres in "edx"
head(edx$genres)



## -----------------------------------------------------------------------------------------------------------
# Separate the genres in "edx"
edx_s_g<-edx%>%separate_rows(genres, sep = "\\|")

# Summarised genres in a table
edx_s_g%>%group_by(genres) %>%
  summarize(count = n(), .groups='drop') %>%
  arrange(desc(count))


## -----------------------------------------------------------------------------------------------------------
# Plot the total count of each rating
edx %>% ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.25, fill = "black")+
  xlab("Rating")+
  ylab("Number of Rating")


## -----------------------------------------------------------------------------------------------------------
# Display the top 10 movies with hightest number of rating 
edx %>% group_by(movieId, title) %>%
	summarize(count = n(), .groups='drop') %>%
	arrange(desc(count))


## -----------------------------------------------------------------------------------------------------------
# plot the number of times movies were rated
edx %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins=500, color = "black") +
  xlab("Number of Ratings per Movie")+
  ylab("Number of Movie")+
  scale_x_log10() + 
  ggtitle("Count of Rating per Movie")


## -----------------------------------------------------------------------------------------------------------
###################################
# Generate train_set and test_set #
###################################

test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

# Make sure userId and movieId in test set are also in train set
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

rm(test_index)


## -----------------------------------------------------------------------------------------------------------
##################################
# Model 1 Average of All Ratings #
##################################

mu<-mean(train_set$rating)
model1 <- mu

# RMSE calculation 
rmse1 <- RMSE(test_set$rating, model1)



## -----------------------------------------------------------------------------------------------------------
#####################################
# Model 2 Average with Movie Effect #
#####################################

b_i <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu), .groups='drop')

# Prediction
model2 <- mu + test_set %>% 
  left_join(b_i, by='movieId') %>%
  pull(b_i)

# RMSE calculation 
rmse2<-RMSE(test_set$rating, model2)



## -----------------------------------------------------------------------------------------------------------
###############################################
# Model 3 Average with Movie and User Effects #
###############################################

b_u <- train_set %>% 
  left_join(b_i, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i), .groups='drop')

# Prediction
model3 <- test_set %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# RMSE calculation 
rmse3 <- RMSE(test_set$rating, model3)



## -----------------------------------------------------------------------------------------------------------
######################################################
# Model 4 Average with Movie, User and Genre Effects #
######################################################

b_g <- train_set %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u), .groups='drop')

# Prediction
model4 <- test_set %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  left_join(b_g, by="genres") %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)

# RMSE calculation 
rmse4 <- RMSE(test_set$rating, model4)



## ----warning=FALSE, message=FALSE---------------------------------------------------------------------------
############################################################
# Model 5 Average with Movie, User, Genre and Time Effects #
############################################################

#Load lubridate to convert timestamp to date
if(!require(lubridate)) install.packages("data.table", repos = "http://cran.us.r-project.org")
library(lubridate)

b_t <- train_set %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  left_join(b_g, by='genres') %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week"))%>%
  group_by(date) %>%
  summarize(b_t = mean(rating - mu - b_i - b_u - b_g), .groups='drop')

# Prediction
model5 <- test_set %>%
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  left_join(b_g, by="genres") %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week"))%>%
  left_join(b_t, by='date')%>%
  mutate(pred = mu + b_i + b_u + b_g + b_t) %>%
  pull(pred)

# RMSE calculation 
rmse5 <- RMSE(test_set$rating, model5)


## -----------------------------------------------------------------------------------------------------------
###########################################################################
# Model 6 Average with Movie, User, Genre, Time and premiere Year Effects #
###########################################################################

b_y <- train_set %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  left_join(b_g, by='genres') %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week"))%>%
  left_join(b_t, by='date') %>%
  mutate(premiere = as.numeric(str_sub(title,-5,-2))) %>%
  group_by(premiere) %>%
  summarize(b_y = mean(rating - mu - b_i - b_u - b_g - b_t), .groups='drop')

# Prediction
model6 <- test_set %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  left_join(b_g, by="genres") %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week"))%>%
  left_join(b_t, by='date')%>%
  mutate(premiere = as.numeric(str_sub(title,-5,-2))) %>%
  left_join(b_y, by="premiere")%>%
  mutate(pred = mu + b_i + b_u + b_g + b_t + b_y) %>%
  pull(pred)

# RMSE calculation 
rmse6 <- RMSE(test_set$rating, model6)



## -----------------------------------------------------------------------------------------------------------
##########################################################################
# Model 7 Regularized Movie, User, Genre, Time and Premiere Year Effects #
##########################################################################

# Finding the optimum tuning value through cross validation
lambdas <- seq(4, 6, 0.25)

# For each lambda, find b_i, b_u, b_g, b_t and b_y followed by rating prediction
rmses1 <- sapply(lambdas, function(l){
  
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l), .groups='drop')
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i)/(n()+l), .groups='drop')
  
  b_g <- train_set %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu - b_i - b_u)/(n()+l), .groups='drop')
  
  b_t <- train_set %>% 
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    left_join(b_g, by='genres') %>%
    mutate(date = round_date(as_datetime(timestamp), unit = "week"))%>%
    group_by(date) %>%
    summarize(b_t = mean(rating - mu - b_i - b_u - b_g)/(n()+l), .groups='drop')
  
  b_y <- train_set %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    left_join(b_g, by='genres') %>%
    mutate(date = round_date(as_datetime(timestamp), unit = "week"))%>%
    left_join(b_t, by='date') %>%
    mutate(premiere = as.numeric(str_sub(title,-5,-2))) %>%
    group_by(premiere) %>%
    summarize(b_y = mean(rating - mu - b_i - b_u - b_g - b_t)/(n()+l), .groups='drop')

  predicted_ratings <- test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    mutate(date = round_date(as_datetime(timestamp), unit = "week"))%>%
    left_join(b_t, by='date')%>%
    mutate(premiere = as.numeric(str_sub(title,-5,-2))) %>%
    left_join(b_y, by="premiere")%>%
    mutate(pred = mu + b_i + b_u + b_g + b_t + b_y) %>%
    pull(pred)

    return(RMSE(test_set$rating, predicted_ratings))
})

lambda<-lambdas[which.min(rmses1)]

# Movie Effect
b_i <- train_set %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda), .groups='drop')

# User Effect
b_u <- train_set %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda), .groups='drop')

# Genre Effect
b_g <- train_set %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+lambda), .groups='drop')

# Time Effect
b_t <- train_set %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  left_join(b_g, by='genres') %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week"))%>%
  group_by(date) %>%
  summarize(b_t = mean(rating - mu - b_i - b_u - b_g)/(n()+lambda), .groups='drop')

# premiere Year Effect
b_y <- train_set %>%
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  left_join(b_g, by='genres') %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week"))%>%
  left_join(b_t, by='date') %>%
  mutate(premiere = as.numeric(str_sub(title,-5,-2))) %>%
  group_by(premiere) %>%
  summarize(b_y = mean(rating - mu - b_i - b_u - b_g - b_t)/(n()+lambda), .groups='drop')

# Prediction
model7 <- test_set %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week"))%>%
  left_join(b_t, by='date')%>%
  mutate(premiere = as.numeric(str_sub(title,-5,-2))) %>%
  left_join(b_y, by="premiere")%>%
  mutate(pred = mu + b_i + b_u + b_g + b_t + b_y) %>%
  pull(pred)

# RMSE calculation 
rmse7 <- RMSE(test_set$rating, model7)



## -----------------------------------------------------------------------------------------------------------
####################################################################################
# Model 8 Regularized Movie, User, Separated Genre, Time and Premiere Year Effects #
####################################################################################

# Separate the genres in "test_set" and "train_set"
test_set_s_g<-test_set%>%separate_rows(genres, sep = "\\|")
train_set_s_g<-train_set%>%separate_rows(genres, sep = "\\|")

# Finding the optimum tuning value through cross validation
lambdas2 <- seq(12, 14, 0.25)
 
# Calculate the average rating after separarted the genres 
mu2 <- mean(train_set_s_g$rating)

# For each lambda, find b_i, b_u, b_g, b_t and b_y followed by rating prediction
rmses2 <- sapply(lambdas2, function(l){
  
  b_i2 <- train_set_s_g %>% 
    group_by(movieId) %>%
    summarize(b_i2 = sum(rating - mu2)/(n()+l), .groups='drop')
  
  b_u2 <- train_set_s_g %>% 
    left_join(b_i2, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u2 = sum(rating - mu2 - b_i2)/(n()+l), .groups='drop')
  
  b_g2 <- train_set_s_g %>%
    left_join(b_i2, by="movieId") %>%
    left_join(b_u2, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g2 = sum(rating - mu2 - b_i2 - b_u2)/(n()+l), .groups='drop')
  
  b_t2 <- train_set_s_g %>% 
    left_join(b_i2, by='movieId') %>%
    left_join(b_u2, by='userId') %>%
    left_join(b_g2, by='genres') %>%
    mutate(date = round_date(as_datetime(timestamp), unit = "week"))%>%
    group_by(date) %>%
    summarize(b_t2 = mean(rating - mu2 - b_i2 - b_u2 - b_g2)/(n()+l), .groups='drop')
  
  b_y2 <- train_set_s_g %>%
    left_join(b_i2, by='movieId') %>%
    left_join(b_u2, by='userId') %>%
    left_join(b_g2, by='genres') %>%
    mutate(date = round_date(as_datetime(timestamp), unit = "week"))%>%
    left_join(b_t2, by='date') %>%
    mutate(premiere = as.numeric(str_sub(title,-5,-2))) %>%
    group_by(premiere) %>%
    summarize(b_y2 = mean(rating - mu2 - b_i2 - b_u2 - b_g2 - b_t2)/(n()+l), .groups='drop')

  predicted_ratings <- test_set_s_g %>%
    left_join(b_i2, by = "movieId") %>%
    left_join(b_u2, by = "userId") %>%
    left_join(b_g2, by = "genres") %>%
    mutate(date = round_date(as_datetime(timestamp), unit = "week"))%>%
    left_join(b_t2, by='date')%>%
    mutate(premiere = as.numeric(str_sub(title,-5,-2))) %>%
    left_join(b_y2, by="premiere")%>%
    mutate(pred = mu2 + b_i2 + b_u2 + b_g2 + b_t2 + b_y2) %>%
    pull(pred)

    return(RMSE(test_set_s_g$rating, predicted_ratings))
})

lambda2<-lambdas2[which.min(rmses2)]

# Movie Effect
b_i2 <- train_set_s_g %>% 
  group_by(movieId) %>%
  summarize(b_i2 = sum(rating - mu2)/(n()+lambda2), .groups='drop')

# User Effect
b_u2 <- train_set_s_g %>% 
  left_join(b_i2, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u2 = sum(rating - b_i2 - mu2)/(n()+lambda2), .groups='drop')

# Genre Effect
b_g2 <- train_set_s_g %>%
  left_join(b_i2, by="movieId") %>%
  left_join(b_u2, by="userId") %>%
  group_by(genres) %>%
  summarize(b_g2 = sum(rating - b_i2 - b_u2 - mu2)/(n()+lambda2), .groups='drop')

# Time Effect
b_t2 <- train_set_s_g %>% 
  left_join(b_i2, by='movieId') %>%
  left_join(b_u2, by='userId') %>%
  left_join(b_g2, by='genres') %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week"))%>%
  group_by(date) %>%
  summarize(b_t2 = mean(rating - mu2 - b_i2 - b_u2 - b_g2)/(n()+lambda2), .groups='drop')

# premiere Year Effect
b_y2 <- train_set_s_g %>%
  left_join(b_i2, by='movieId') %>%
  left_join(b_u2, by='userId') %>%
  left_join(b_g2, by='genres') %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week"))%>%
  left_join(b_t2, by='date') %>%
  mutate(premiere = as.numeric(str_sub(title,-5,-2))) %>%
  group_by(premiere) %>%
  summarize(b_y2 = mean(rating - mu2 - b_i2 - b_u2 - b_g2 - b_t2)/(n()+lambda2), .groups='drop')

# Prediction
model8 <- test_set_s_g %>% 
  left_join(b_i2, by = "movieId") %>%
  left_join(b_u2, by = "userId") %>%
  left_join(b_g2, by = "genres") %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week"))%>%
  left_join(b_t2, by='date')%>%
  mutate(premiere = as.numeric(str_sub(title,-5,-2))) %>%
  left_join(b_y2, by="premiere")%>%
  mutate(pred = mu2 + b_i2 + b_u2 + b_g2 + b_t2 + b_y2) %>%
  pull(pred)

# RMSE calculation 
rmse8 <- RMSE(test_set_s_g$rating, model8)



## -----------------------------------------------------------------------------------------------------------
##############################################################################
# Validation 1-Regularized Movie, User, Genre, Time and premiere Year Effects#
##############################################################################

# Calculate the average of the ratings in "edx" data-set
mu_val <- mean(edx$rating)

# Movie Effect
b_i_val <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i_val = sum(rating - mu_val)/(n()+lambda), .groups='drop')

# User Effect
b_u_val <- edx %>% 
  left_join(b_i_val, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u_val = sum(rating - b_i_val - mu_val)/(n()+lambda), .groups='drop')

# Genre Effect
b_g_val <- edx %>% 
  left_join(b_i_val, by="movieId") %>%
  left_join(b_u_val, by="userId") %>%
  group_by(genres) %>%
  summarize(b_g_val = sum(rating - b_i_val - b_u_val - mu_val)/(n()+lambda), .groups='drop')

# Time Effect
b_t_val <- edx %>%
  left_join(b_i_val, by='movieId') %>%
  left_join(b_u_val, by='userId') %>%
  left_join(b_g_val, by='genres') %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week"))%>%
  group_by(date) %>%
  summarize(b_t_val = mean(rating - mu_val - b_i_val - b_u_val - b_g_val)/(n()+lambda), .groups='drop')

# premiere Year Effect
b_y_val <- edx %>%
  left_join(b_i_val, by='movieId') %>%
  left_join(b_u_val, by='userId') %>%
  left_join(b_g_val, by='genres') %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week"))%>%
  left_join(b_t_val, by='date') %>%
  mutate(premiere = as.numeric(str_sub(title,-5,-2))) %>%
  group_by(premiere) %>%
  summarize(b_y_val = mean(rating - mu_val - b_i_val - b_u_val - b_g_val - b_t_val)/(n()+lambda), .groups='drop')

# Prediction
model_val1 <- validation %>% 
  left_join(b_i_val, by = "movieId") %>%
  left_join(b_u_val, by = "userId") %>%
  left_join(b_g_val, by = "genres") %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week"))%>%
  left_join(b_t_val, by='date')%>%
  mutate(premiere = as.numeric(str_sub(title,-5,-2))) %>%
  left_join(b_y_val, by="premiere")%>%
  mutate(pred = mu_val + b_i_val + b_u_val + b_g_val + b_t_val + b_y_val) %>%
  pull(pred)

# RMSE calculation 
rmse_val1 <- RMSE(validation$rating, model_val1)



## -----------------------------------------------------------------------------------------------------------
########################################################################################
# Validation 2-Regularized Movie, User, Separated Genre, Time and Premiere Year Effects#
########################################################################################

# Separate the genres in "validation" data-set
validation_s_g<-validation%>%separate_rows(genres, sep = "\\|")

# Calculate the average rating after separarted the genres 
mu_val2 <- mean(edx_s_g$rating)

# Movie Effect
b_i_val2 <- edx_s_g %>% 
  group_by(movieId) %>%
  summarize(b_i_val2 = sum(rating - mu_val2)/(n()+lambda2), .groups='drop')

# User Effect
b_u_val2 <- edx_s_g %>% 
  left_join(b_i_val2, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u_val2 = sum(rating - b_i_val2 - mu_val2)/(n()+lambda2), .groups='drop')

# Genre Effect
b_g_val2 <- edx_s_g %>% 
  left_join(b_i_val2, by="movieId") %>%
  left_join(b_u_val2, by="userId") %>%
  group_by(genres) %>%
  summarize(b_g_val2 = sum(rating - b_i_val2 - b_u_val2 - mu_val2)/(n()+lambda2), .groups='drop')

# Time Effect
b_t_val2 <- edx_s_g %>%
  left_join(b_i_val2, by='movieId') %>%
  left_join(b_u_val2, by='userId') %>%
  left_join(b_g_val2, by='genres') %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week"))%>%
  group_by(date) %>%
  summarize(b_t_val2 = mean(rating - mu_val2 - b_i_val2 - b_u_val2 - b_g_val2)/(n()+lambda2), .groups='drop')

# premiere Year Effect
b_y_val2 <- edx_s_g%>%
  left_join(b_i_val2, by='movieId') %>%
  left_join(b_u_val2, by='userId') %>%
  left_join(b_g_val2, by='genres') %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week"))%>%
  left_join(b_t_val2, by='date') %>%
  mutate(premiere = as.numeric(str_sub(title,-5,-2))) %>%
  group_by(premiere) %>%
  summarize(b_y_val2 = mean(rating - mu_val2 - b_i_val2 - b_u_val2 - b_g_val2 - b_t_val2)/(n()+lambda2), .groups='drop')

# Prediction
model_val2 <- validation_s_g %>% 
  left_join(b_i_val2, by = "movieId") %>%
  left_join(b_u_val2, by = "userId") %>%
  left_join(b_g_val2, by = "genres") %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week"))%>%
  left_join(b_t_val2, by='date')%>%
  mutate(premiere = as.numeric(str_sub(title,-5,-2))) %>%
  left_join(b_y_val2, by="premiere")%>%
  mutate(pred = mu_val2 + b_i_val2 + b_u_val2 + b_g_val2 + b_t_val2 + b_y_val2) %>%
  pull(pred)

# RMSE calculation 
rmse_val2 <- RMSE(validation_s_g$rating, model_val2)



## -----------------------------------------------------------------------------------------------------------
#################
# Summary table #
#################

# Prodcue summary table
summary <- tibble(Method = c("Target", "Model 1-Average of All Ratings", "Model 2-Average with Movie Effect", "Model 3-Average with Movie and User Effects", "Model 4-Average with Movie, User and Genre Effects", "Model 5-Average with Movie, User, Genre and Time Effects", "Model 6-Average with Movie, User, Genre, Time and Premiere Year Effects", "Model 7-Regularized Movie, User, Genre, Time and Premiere Year Effects", "Model 8-Regularized Movie, User, Separated Genre, Time and Premiere Year Effects", "Validation 1-Regularized Movie, User, Genre, Time and Premiere Year Effects", "Validation 2-Regularized Movie, User, Separated Genre, Time and Premiere Year Effects"), RMSE = c(target, rmse1, rmse2, rmse3, rmse4, rmse5, rmse6, rmse7, rmse8, rmse_val1, rmse_val2), "Difference to Model 1" = c(rmse1 - target, rmse1 - rmse1, rmse2 - rmse1, rmse3 - rmse1, rmse4 - rmse1, rmse5 - rmse1, rmse6 - rmse1, rmse7 - rmse1, rmse8 - rmse1, rmse_val1 - rmse1, rmse_val2 - rmse1), "Difference to Last Model" = c("NA", rmse1 - target, rmse2 - rmse1, rmse3 - rmse2, rmse4 - rmse3, rmse5 - rmse4, rmse6 - rmse5, rmse7 - rmse6, rmse8 - rmse7, rmse_val1 - rmse8, rmse_val2 - rmse_val1), "Difference to Target" = c(target-target, rmse1 - target, rmse2 - target, rmse3 - target, rmse4 - target, rmse5 - target, rmse6 - target, rmse7 - target, rmse8 - target, rmse_val1 - target, rmse_val2 - target))

summary



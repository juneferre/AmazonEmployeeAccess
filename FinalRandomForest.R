# ==============================================================================
# ========================== Classification Trees ==============================
# ==============================================================================
library(vroom)
library(tidymodels)
library(embed)
library(workflows)
library(kernlab)
library(themis)
library(lme4)
# -----------------------------------------------------------------------------
# Read in data sets
# -----------------------------------------------------------------------------
train <- vroom("/Users/juneferre/fall2025/stat348/Kaggle/amazon/train.csv") 
test <- vroom("/Users/juneferre/fall2025/stat348/Kaggle/amazon/test.csv")

train$ACTION <- as.factor(train$ACTION)
# -----------------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------------
my_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_predictors(), fn=factor) %>%
  step_lencode_mixed(all_factor_predictors(), outcome = vars(ACTION)) %>%
    step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) 

prepped <- prep(my_recipe, verbose = TRUE)
new_data <- bake(prepped, new_data = NULL)


# -----------------------------------------------------------------------------
# Define model
# -----------------------------------------------------------------------------
my_mod <- rand_forest(mtry  = 1, min_n = 10,trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# Workflow
wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod) %>%
  fit(data=train)


# Now you can predict on the test data
preds <- predict(wf, new_data = test, type = "prob")


# -----------------------------------------------------------------------------
# Kaggle Submission
# -----------------------------------------------------------------------------
final <- preds %>% select(.pred_1)
colnames(final)[1] <- "ACTION"

kaggle_submission <- final %>%
  bind_cols(test %>% select(id)) %>%
  select(id,ACTION)

# Write to CSV
vroom_write(kaggle_submission, file = "./ClassificationTrees_final1.csv", delim = ",")


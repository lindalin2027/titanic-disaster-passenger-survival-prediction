#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
})

message("===== Titanic R pipeline started =====")

# ---- helper to get script dir ----
get_script_path <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  path <- sub(file_arg, "", args[grep(file_arg, args)])
  if (length(path) == 0) {
    # fallback for interactive use
    return(normalizePath("src/R_code/main.R"))
  }
  normalizePath(path)
}

script_path  <- get_script_path()
script_dir   <- dirname(script_path)
project_root <- normalizePath(file.path(script_dir, "..", ".."))
data_dir     <- file.path(project_root, "src", "data")

train_path <- file.path(data_dir, "train.csv")
test_path  <- file.path(data_dir, "test.csv")

message("----- Loading Data -----")
if (!file.exists(train_path)) stop(paste("[ERROR] train file not found at", train_path))
if (!file.exists(test_path))  stop(paste("[ERROR] test file not found at", test_path))

train_df <- read_csv(train_path, show_col_types = FALSE)
test_df  <- read_csv(test_path, show_col_types = FALSE)

message(sprintf("[INFO] Training data loaded: %d rows × %d cols", nrow(train_df), ncol(train_df)))
message(sprintf("[INFO] Test data loaded: %d rows × %d cols", nrow(test_df), ncol(test_df)))

cat("\nFirst few rows of training data:\n")
print(head(train_df))
cat("\nFirst few rows of test data:\n")
print(head(test_df))

# ---------------- cleaning ----------------
message("----- Cleaning Data -----")
drop_cols <- c("Name", "Ticket", "Cabin")
train_df <- train_df[, setdiff(names(train_df), drop_cols)]
test_df  <- test_df[,  setdiff(names(test_df),  drop_cols)]

# Fill Age and Fare
if ("Age" %in% names(train_df)) {
  age_med <- median(train_df$Age, na.rm = TRUE)
  train_df$Age[is.na(train_df$Age)] <- age_med
  test_df$Age[is.na(test_df$Age)] <- age_med
}
if ("Fare" %in% names(test_df)) {
  fare_med <- median(train_df$Fare, na.rm = TRUE)
  test_df$Fare[is.na(test_df$Fare)] <- fare_med
}

# Encode Sex
train_df$Sex <- ifelse(train_df$Sex == "female", 1L, 0L)
test_df$Sex  <- ifelse(test_df$Sex  == "female", 1L, 0L)

# Family features
train_df <- train_df %>%
  mutate(FamilySize = SibSp + Parch + 1,
         IsAlone = ifelse(FamilySize == 1, 1L, 0L))
test_df <- test_df %>%
  mutate(FamilySize = SibSp + Parch + 1,
         IsAlone = ifelse(FamilySize == 1, 1L, 0L))

# Embarked cleanup
if ("Embarked" %in% names(train_df)) {
  mode_emb <- names(sort(table(train_df$Embarked), decreasing = TRUE))[1]
  train_df$Embarked[is.na(train_df$Embarked)] <- mode_emb
  test_df$Embarked[is.na(test_df$Embarked)] <- mode_emb
  train_df$Embarked <- factor(train_df$Embarked)
  test_df$Embarked  <- factor(test_df$Embarked, levels = levels(train_df$Embarked))
}

message("[INFO] After cleaning, preview of training data:")
print(head(train_df))

# ---------------- model ----------------
message("----- Training Model -----")
train_df$Survived <- as.integer(train_df$Survived)

model <- glm(
  Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + FamilySize + IsAlone,
  data = train_df,
  family = binomial(link = "logit")
)

summary_model <- summary(model)
message("[INFO] Model summary:")
print(summary_model$coefficients)

# Training accuracy
train_prob <- predict(model, newdata = train_df, type = "response")
train_pred <- ifelse(train_prob >= 0.5, 1L, 0L)
train_acc  <- mean(train_pred == train_df$Survived)
message(sprintf("[INFO] Training accuracy: %.4f", train_acc))

# ---------------- prediction ----------------
message("----- Predicting on Test Set -----")
test_passenger_ids <- test_df$PassengerId
pred_probs  <- predict(model, newdata = test_df, type = "response")
pred_labels <- ifelse(pred_probs >= 0.5, 1L, 0L)

out_df <- data.frame(
  PassengerId = test_passenger_ids,
  Survived = pred_labels
)

# Survival rate
survival_rate <- mean(out_df$Survived) * 100
message(sprintf("[INFO] Predicted survival rate: %.2f%%", survival_rate))

cat("\nFirst few predictions:\n")
print(head(out_df))

# ---------------- write output ----------------
out_path <- file.path(data_dir, "survival_predictions_r.csv")
dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
write_csv(out_df, out_path)
message(sprintf("[INFO] Saved predictions to %s", out_path))

message("===== Titanic R pipeline completed successfully =====")

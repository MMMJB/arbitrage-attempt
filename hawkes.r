# Install and load the emhawkes package
install.packages("emhawkes", repos = "https://cran.rstudio.com/")
library(emhawkes)

# --- Step 1: Load real data ---
# Each file format: timestamp spread_change

# Read up events
up_data <- read.table("up.txt", header = FALSE)
colnames(up_data) <- c("timestamp", "spread_change")
up_data$type <- 1   # "up" events = type 1

# Read down events
down_data <- read.table("down.txt", header = FALSE)
colnames(down_data) <- c("timestamp", "spread_change")
down_data$type <- 2 # "down" events = type 2

# Combine and sort by timestamp
all_data <- rbind(up_data, down_data)
all_data <- all_data[order(all_data$timestamp), ]

# --- Step 2: Prepare input vectors ---
# Shift UNIX timestamps to start at 0
timestamps <- all_data$timestamp - min(all_data$timestamp)

# Interarrival times
inter_arrival <- diff(c(0, timestamps))

# Event types (1 = up, 2 = down)
type <- all_data$type

# Spread change magnitudes
mark <- all_data$spread_change

# --- Step 3: Try without eta parameter first ---
# Create a simpler hspec object without eta (marks)
# spec_simple <- new("hspec",
#                    mu     = c(0.05, 0.05),
#                    alpha  = matrix(c(0.3, 0.1,
#                                      0.2, 0.4), nrow = 2, ncol = 2),
#                    beta   = matrix(c(1.0, 1.0,
#                                      1.0, 1.0), nrow = 2, ncol = 2),
#                    dimens = 2)

# --- Step 4: Fit the simple model first ---
# print("Fitting simple model without marks...")
# fit_simple <- hfit(
#   object = spec_simple,
#   inter_arrival = inter_arrival,
#   type = type,
#   method = "BFGS"
# )

# print("Simple model results:")
# print(coef(fit_simple))

# --- Step 5: Fit marked models ---

# Impact function with equal sensitivity for both dimensions
# impact_fixed <- function(param = NULL, mark, ...) {
#   eta_val <- 0.1  # Fixed sensitivity parameter
#   n_marks <- length(mark)
#   n_dims <- 2
  
#   # Create result matrix: 2 rows (dimensions) x n_marks columns
#   result <- matrix(nrow = n_dims, ncol = n_marks)
#   result[1, ] <- eta_val * mark  # Impact on dimension 1
#   result[2, ] <- eta_val * mark  # Impact on dimension 2
  
#   return(result)
# }

# print("Fitting marked model with equal sensitivity...")

# spec_marked_fixed <- new("hspec",
#                          mu     = c(0.05, 0.05),
#                          alpha  = matrix(c(0.3, 0.1,
#                                            0.2, 0.4), nrow = 2, ncol = 2),
#                          beta   = matrix(c(1.0, 1.0,
#                                            1.0, 1.0), nrow = 2, ncol = 2),
#                          impact = impact_fixed,
#                          dimens = 2)

# fit_marked_fixed <- hfit(
#   object = spec_marked_fixed,
#   inter_arrival = inter_arrival,
#   type = type,
#   mark = mark,
#   method = "BFGS"
# )

# print("Fixed marked model results:")
# print(coef(fit_marked_fixed))

# # Impact function with different sensitivities
# impact_different <- function(param = NULL, mark, ...) {
#   n_marks <- length(mark)
#   n_dims <- 2
  
#   eta1 <- 0.05  # Impact on dimension 1 (up events)
#   eta2 <- 0.08  # Impact on dimension 2 (down events)
  
#   result <- matrix(nrow = n_dims, ncol = n_marks)
#   result[1, ] <- eta1 * mark  # Different impact on dimension 1
#   result[2, ] <- eta2 * mark  # Different impact on dimension 2
  
#   return(result)
# }

print("Fitting marked model with different sensitivities...")

spec_marked_alt <- new("hspec",
                       mu     = c(0.05, 0.05),
                       alpha  = matrix(c(0.3, 0.1,
                                         0.2, 0.4), nrow = 2, ncol = 2),
                       beta   = matrix(c(1.0, 1.0,
                                         1.0, 1.0), nrow = 2, ncol = 2),
                       impact = impact_different,
                       dimens = 2)

fit_marked_alt <- hfit(
  object = spec_marked_alt,
  inter_arrival = inter_arrival,
  type = type,
  mark = mark,
  method = "BFGS"
)

print("Alternative marked model results:")
print(coef(fit_marked_alt))

# --- Step 6: Model summaries and comparison ---
# print("=== DETAILED SUMMARIES ===")

# print("Simple model summary:")
# summary(fit_simple)

# print("Fixed sensitivity marked model summary:")
# summary(fit_marked_fixed)

# print("Different sensitivity marked model summary:")
# summary(fit_marked_alt)

# print("=== MODEL COMPARISON ===")
# print("Simple model coefficients:")
# print(coef(fit_simple))

# print("Fixed sensitivity marked model coefficients:")
# print(coef(fit_marked_fixed))

# print("Different sensitivity marked model coefficients:")
# print(coef(fit_marked_alt))
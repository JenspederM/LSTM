
# Print Iteration ---------------------------------------------------------

print_iteration <- function(prediction, loss, currentIteration, totalIteration) {
  o <- prediction
  
  if(length(o) > 10) {
    ostring <- paste(c(head(sprintf("%6.3f", o), 5), "...", tail(sprintf("%6.3f", o), 5)), collapse = ", ")
  } else {
    ostring <- paste(sprintf("%6.3f", o), collapse = ", ")
  }
  
  cat(
    sprintf("Iteration %-5s of %-5s: %s = [ %s ] %s: %-.3f \n", 
            currentIteration, totalIteration, "Prediction", ostring, "Loss", loss[currentIteration])
  )
}

# Loss Layer --------------------------------------------------------------

ToyLossLayer <- R6Class(
  "ToyLossLayer",
  list(
    loss = function(pred, label) {
      (pred[1] - label) ^ 2
    },
    bottom_diff = function(pred, label) {
      diff <- pred * 0
      diff[1] <- 2 * (pred[1] - label)
      diff
    }
  )
)
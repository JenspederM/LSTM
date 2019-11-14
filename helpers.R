

print_early <- function(iteration, loss, y, pred) {
  cat(sprintf(
    "\n##### Early stopping on iteration %s with loss = %.6f #####\n",
    iteration,
    loss[iteration]
  ),
  sprintf(
    "\n%20s: [ %s ]\n",
    "Original Input",
    paste(sapply(y, function(idx)
      sprintf("%6.3f", idx)), collapse = ", ")
  ),
  sprintf(
    "\n%20s: [ %s ]",
    "Final Prediction",
    paste(sapply(1:length(y), function(idx)
      sprintf("%6.3f", pred[[idx]]$state$h[1])), collapse = ", ")
  ))
}

print_final <- function(iteration, loss, y, pred) {
  cat(sprintf(
    "\n##### Finished on iteration %s with loss = %.6f #####\n",
    iteration,
    loss[iteration]
  ),
  sprintf(
    "\n%20s: [ %s ]\n",
    "Original Input",
    paste(sapply(y, function(idx)
      sprintf("%6.3f", idx)), collapse = ", ")
  ),
  sprintf(
    "\n%20s: [ %s ]",
    "Final Prediction",
    paste(sapply(1:length(y), function(idx)
      sprintf("%6.3f", pred[[idx]]$state$h[1])), collapse = ", ")
  ))
}

print.LSTM <- function(x, ...) {
  i <- x$Input
  o <- x$Output
  
  
  if(length(i) > 10 || length(o) > 10) {
    istring <- paste(c(head(sprintf("%6.3f", i), 5), "...", tail(sprintf("%6.3f", i), 5)), collapse = ", ")
    ostring <- paste(c(head(sprintf("%6.3f", o), 5), "...", tail(sprintf("%6.3f", o), 5)), collapse = ", ")
  } else {
    istring <- paste(sprintf("%6.3f", i), collapse = ", ")
    ostring <- paste(sprintf("%6.3f", o), collapse = ", ")
  }
  
  cat(
    "Object Type: LSTM Network\n\n",
    sprintf("Finished on iteration %s of %s \n\n", x$FinishIterations, x$InputIterations),
    sprintf("%10s: [ %s ] \n", "Input", istring),
    sprintf("%10s: [ %s ] \n\n", "Output", ostring),
    sprintf("%10s = %.6f", "Final Loss", x$Loss[x$FinishIterations]),
    sep = ""
  )
}

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
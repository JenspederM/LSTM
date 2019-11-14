lstm_train <-
  function(y_list,
           cells = 100,
           memory = 50,
           learning_rate = 0.1,
           iterations = 100,
           early_stopping = TRUE,
           patience = 100,
           stop_tresh = 0.001,
           verbose = T,
           print_divider = 10,
           seed = 0) {
    # Set Seed
    set.seed(seed)
    # Assert that Y is list
    if(!is.list(y_list)) stop("Y must be a list")
    # Define LSTM Parameters
    lstm_param <- LstmParam$new(cells, memory)
    # Construct LSTM Network
    lstm_net <- LstmNetwork$new(lstm_param)
    # Construct Initial Prediction
    input_val_arr <- lapply(seq_along(y_list), function(x) runif(memory))
    # Calculate Index for Verbose
    if (verbose) print_idx <- as.integer(iterations / print_divider)
    # Construct Loss Vector
    loss <- vector("numeric", iterations)
    # Initiate Counter for Early Stopping
    if(early_stopping) stop_counter <- 0L
    
    # Train Network
    for (cur_iter in 1:iterations) {
      if (verbose &&
          cur_iter %% print_idx == 0)
        cat(sprintf("Iteration %-10s: ", cur_iter))
      
      ## Make Predictions
      for (ind in seq_along(y_list)) {
        lstm_net$x_list_add(input_val_arr[[ind]])
      }
      
      ## Print Prediction
      if (verbose && cur_iter %% print_idx == 0) {
        cat(sprintf("%s = [ %s ]",
                    "y_pred", paste(
                      sapply(1:length(y_list), function(ind)
                        sprintf("%6.3f", lstm_net$lstm_node_list[[ind]]$state$h[1])),
                      collapse = ", "
                    )))
      }
      
      # Calculate Loss
      loss[cur_iter] <- lstm_net$y_list_is(y_list, ToyLossLayer)
      
      # Print Loss
      if (verbose && cur_iter %% print_idx == 0) {
        cat(sprintf(" %s: %-.3f", "loss", loss[cur_iter]), "\n")
      }
      
      # Early Stopping
      if (early_stopping &&
          cur_iter != 1 &&
          abs(diff(loss))[cur_iter - 1] < stop_tresh) {
        stop_counter <- stop_counter + 1L
        if (stop_counter >= patience) {
          print_early(cur_iter, loss, y_list, lstm_net$lstm_node_list)
          break
        }
      }
      
      # Update Weights
      lstm_param$apply_diff(lr = learning_rate)
      
      # Clear Predictions
      lstm_net$x_list_clear()
      
      # Print Final Prediction
      if (cur_iter == iterations) {
        print_final(cur_iter, loss, y_list, lstm_net$lstm_node_list)
      }
    }
    
    
    structure(list(
      "Input" = y_list,
      "Output" = lapply(1:length(y_list), function(ind) lstm_net$lstm_node_list[[ind]]$state$h[1]),
      "Loss" = loss,
      "Network" = lstm_net,
      "InputIterations" = iterations,
      "FinishIterations" = cur_iter
    ),
    class = "LSTM"
    )
  }
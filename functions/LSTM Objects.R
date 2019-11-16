
# LSTM PARAMETERS ---------------------------------------------------------

LstmParam <- R6Class(
  "LstmParam",
  list(
    mem_cell_ct = 100,
    x_dim = 50,
    # Input Gate
    wi = matrix(runif(100 * 50, min = -0.1, max = 0.1), nrow = 100, ncol = 50),
    wi_diff = matrix(0.0, 100, 100 + 50),
    bi = runif(100, min = -0.1, max = 0.1),
    bi_diff = rep.int(0.0, 100),
    # Forget Gate
    wf = matrix(runif(100 * 50, min = -0.1, max = 0.1), nrow = 100, ncol = 50),
    wf_diff = matrix(0.0, 100, 100 + 50),
    bf = runif(100, min = -0.1, max = 0.1),
    bf_diff = rep.int(0.0, 100),
    # Control Gate
    wg = matrix(runif(100 * 50, min = -0.1, max = 0.1), nrow = 100, ncol = 50),
    wg_diff = matrix(0.0, 100, 100 + 50),
    bg = runif(100, min = -0.1, max = 0.1),
    bg_diff = rep.int(0.0, 100),
    # Output Gate
    wo = matrix(runif(100 * 50, min = -0.1, max = 0.1), nrow = 100, ncol = 50),
    wo_diff = matrix(0.0, 100, 100 + 50),
    bo = runif(100, min = -0.1, max = 0.1),
    bo_diff = rep.int(0.0, 100),
    
    # Defines Initialization Function
    initialize = function(mem_cell_ct, x_dim) {
      if (!missing(mem_cell_ct) && !missing(x_dim)) {
        self$mem_cell_ct <- mem_cell_ct
        self$x_dim <- x_dim
        concat_len <- x_dim + mem_cell_ct
        # Input Gate
        self$wi <- self$rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self$wi_diff <- matrix(0.0, mem_cell_ct, concat_len)
        self$bi <- self$rand_arr(-0.1, 0.1, mem_cell_ct)
        self$bi_diff <- rep.int(0.0, mem_cell_ct)
        # Forget Gate
        self$wf <- self$rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self$wf_diff <- matrix(0.0, mem_cell_ct, concat_len)
        self$bf <- self$rand_arr(-0.1, 0.1, mem_cell_ct)
        self$bf_diff <- rep.int(0.0, mem_cell_ct)
        # Control Gate
        self$wg <- self$rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self$wg_diff <- matrix(0.0, mem_cell_ct, concat_len)
        self$bg <- self$rand_arr(-0.1, 0.1, mem_cell_ct)
        self$bg_diff <- rep.int(0.0, mem_cell_ct)
        # Output Gate
        self$wo <- self$rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self$wo_diff <- matrix(0.0, mem_cell_ct, concat_len)
        self$bo <- self$rand_arr(-0.1, 0.1, mem_cell_ct)
        self$bo_diff <- rep.int(0.0, mem_cell_ct)
      }
    },
    
    rand_arr = function(a, b, ...) {
      set.seed(0)
      args <- list(...)
      if (length(args) == 1)
        return(runif(args[[1]], min = a, max = b))
      else if (length(args) == 2) {
        return(matrix(runif(args[[1]] * args[[2]], min = a, max = b), nrow = args[[1]], ncol = args[[2]]))
      } else {
        stop("length(Args) > 2")
      }
    },
    
    # Defines function to differentiate weights and bias
    apply_diff = function(lr = 1.0) {
      # Differentiate Weights
      self$wg <- self$wg - (lr * self$wg_diff)
      self$wi <- self$wi - (lr * self$wi_diff)
      self$wf <- self$wf - (lr * self$wf_diff)
      self$wo <- self$wo - (lr * self$wo_diff)
      
      # Differentiate Biases
      self$bg <- self$bg - (lr * self$bg_diff)
      self$bi <- self$bi - (lr * self$bi_diff)
      self$bf <- self$bf - (lr * self$bf_diff)
      self$bo <- self$bo - (lr * self$bo_diff)
      
      # Reset Differences
      self$wg_diff <- self$wg * 0.0
      self$wi_diff <- self$wi * 0.0
      self$wf_diff <- self$wf * 0.0
      self$wo_diff <- self$wo * 0.0
      self$bg_diff <- self$bg * 0.0
      self$bi_diff <- self$bi * 0.0
      self$bf_diff <- self$bf * 0.0
      self$bo_diff <- self$bo * 0.0
      invisible(self)
    }
  )
)


# LSTM STATE --------------------------------------------------------------

LstmState <- R6Class(
  "LstmState",
  list(
    g = rep.int(0.0, 100),
    i = rep.int(0.0, 100),
    f = rep.int(0.0, 100),
    o = rep.int(0.0, 100),
    s = rep.int(0.0, 100),
    h = rep.int(0.0, 100),
    bottom_diff_s = rep.int(0.0, 100),
    bottom_diff_h = rep.int(0.0, 100),
    
    
    initialize = function(mem_cell_ct) {
      if(!missing(mem_cell_ct)) {
        self$g <- rep.int(0.0, mem_cell_ct)
        self$i <- rep.int(0.0, mem_cell_ct)
        self$f <- rep.int(0.0, mem_cell_ct)
        self$o <- rep.int(0.0, mem_cell_ct)
        self$s <- rep.int(0.0, mem_cell_ct)
        self$h <- rep.int(0.0, mem_cell_ct)
        self$bottom_diff_s <- rep.int(0.0, mem_cell_ct)
        self$bottom_diff_h <- rep.int(0.0, mem_cell_ct)
      }
    }
  )
)


# LSTM NODE ---------------------------------------------------------------

LstmNode <- R6Class(
  "LstmNode",
  list(
    # store reference to parameters and to state
    state = LstmState$new(100),
    param = LstmParam$new(100, 50),
    # non-recurrent input concatenated with recurrent input
    xc = NULL,
    # previous states
    s_prev = NULL,
    h_prev = NULL,
    
    # Initialize Object
    initialize = function(lstm_param, lstm_state) {
      if (!missing(lstm_state) && !missing(lstm_param)) {
        self$state <- lstm_state
        self$param <- lstm_param
      }
    },
    
    sigmoid = function(x) {
      1 / (1 + exp(-x))
    },
    
    sigmoid_derivative = function(values) {
      values * (1 - values)
    },
    
    tanh_derivative = function(values) {
      (1 - values) ^ 2
    },
    
    
    #### LOCATE BOTTOM DATA
    bottom_data_is = function(x, s_prev = NULL, h_prev = NULL) {
      # if this is the first lstm node in the network
      if (is.null(s_prev))
        s_prev <- self$state$s * 0
      if (is.null(h_prev))
        h_prev <- self$state$h * 0
      
      # save data for use in backprop
      self$s_prev <- s_prev
      self$h_prev <- h_prev
      
      # concatenate x(t) and h(t-1)
      xc <- c(x, h_prev)
      
      self$state$g <- (tanh((self$param$wg %m% xc) + self$param$bg))
      self$state$i <- (self$sigmoid((self$param$wi %m% xc) + self$param$bi))
      self$state$f <- (self$sigmoid((self$param$wf %m% xc) + self$param$bf))
      self$state$o <- (self$sigmoid((self$param$wo %m% xc) + self$param$bo))
      self$state$s <- (self$state$g * self$state$i) + (s_prev * self$state$f)
      self$state$h <- self$state$s * self$state$o
      
      self$xc <- xc
      invisible(self)
    },
    
    
    #### CALCULATE TOP DIFF
    top_diff_is = function(top_diff_h, top_diff_s) {
      # notice that top_diff_s is carried along the constant error carousel
      ds <- self$state$o * top_diff_h + top_diff_s
      do <- self$state$s * top_diff_h
      di <- self$state$g * ds
      dg <- self$state$i * ds
      df <- self$s_prev * ds
      
      # diffs w.r.t. vector inside sigma / tanh function
      di_input <- self$sigmoid_derivative(self$state$i) * di
      df_input <- self$sigmoid_derivative(self$state$f) * df
      do_input <- self$sigmoid_derivative(self$state$o) * do
      dg_input <- self$tanh_derivative(self$state$g) * dg
      
      # diffs w.r.t. inputs
      self$param$wi_diff <- self$param$wi_diff + (di_input %op% self$xc)
      self$param$wf_diff <- self$param$wf_diff + (df_input %op% self$xc)
      self$param$wo_diff <- self$param$wo_diff + (do_input %op% self$xc)
      self$param$wg_diff <- self$param$wg_diff + (dg_input %op% self$xc)
      self$param$bi_diff <- self$param$bi_diff + di_input
      self$param$bf_diff <- self$param$bf_diff + df_input
      self$param$bo_diff <- self$param$bo_diff + do_input
      self$param$bg_diff <- self$param$bg_diff + dg_input
      
      # compute bottom diff
      dxc <- self$xc * 0
      dxc <- dxc + crossprod(self$param$wi, di_input)
      dxc <- dxc + crossprod(self$param$wf, df_input)
      dxc <- dxc + crossprod(self$param$wo, do_input)
      dxc <- dxc + crossprod(self$param$wg, dg_input)
      
      # save bottom diffs
      self$state$bottom_diff_s <- ds * self$state$f
      self$state$bottom_diff_h <- dxc[(self$param$x_dim + 1):length(dxc)]
      invisible(self)
    }
  )
)

# LSTM NETWORK ------------------------------------------------------------


LstmNetwork <- R6Class(
  "LstmNetwork",
  private = list(
    .input = NA,
    .output = NA,
    .loss = NA,
    .inputIteration = NA,
    .outputIteration = NA
  ),
  active = list(
    input = function(value) {
      if(missing(value)) {
        private$.input
      } else {
        private$.input <- value
      }
    },
    output = function(value) {
      if(missing(value)) {
        private$.output
      } else {
        private$.output <- value
      }
    },
    loss = function(value) {
      if(missing(value)) {
        private$.loss
      } else {
        private$.loss <- value
      }
    },
    inputIteration = function(value) {
      if(missing(value)) {
        private$.inputIteration
      } else {
        private$.inputIteration <- value
      }
    },
    outputIteration = function(value) {
      if(missing(value)) {
        private$.outputIteration
      } else {
        private$.outputIteration <- value
      }
    }
  ),
  public = list(
    lstm_param = LstmParam$new(),
    lstm_node_list = list(),
    # input sequence
    x_list = list(),
    
    initialize = function(lstm_param) {
      if(!missing(lstm_param)) {
        self$lstm_param <- lstm_param
      }
    },
    
    x_list_clear = function() {
      self$x_list <- list()
      invisible(self)
    },
    
    x_list_add = function(x) {
      self$x_list <- c(self$x_list, list(x))
      
      if (length(self$x_list) > length(self$lstm_node_list)) {
        lstm_state <- LstmState$new(self$lstm_param$mem_cell_ct)
        self$lstm_node_list <- c(self$lstm_node_list, list(LstmNode$new(self$lstm_param, lstm_state)))
      }
      
      x_length <- length(self$x_list)
      if (x_length == 1) {
        self$lstm_node_list[[x_length]]$bottom_data_is(x)
      } else {
        s_prev <- self$lstm_node_list[[x_length - 1]]$state$s
        h_prev <- self$lstm_node_list[[x_length - 1]]$state$h
        self$lstm_node_list[[x_length]]$bottom_data_is(x, s_prev, h_prev)
      }
      invisible(self)
    },
    
    y_list_is = function(y_list, loss_layer) {
      list_length <- length(self$x_list)
      if(!length(y_list) == list_length) stop("X and Y must be of equal length")
      # first node only gets diffs from label ...
      loss <- loss_layer$public_methods$loss(self$lstm_node_list[[list_length]]$state$h, y_list[[list_length]])
      diff_h <- loss_layer$public_methods$bottom_diff(self$lstm_node_list[[list_length]]$state$h, y_list[[list_length]])
      # here s is not affecting loss due to h(t+1), hence we set equal to zero
      diff_s <- vector("double", self$lstm_param$mem_cell_ct)
      self$lstm_node_list[[list_length]]$top_diff_is(diff_h, diff_s)
      ### ... following nodes also get diffs from next nodes, hence we add diffs to diff_h
      ### we also propagate error along constant error carousel using diff_s
      list_length <- list_length - 1
      for(i in list_length:1) {
        loss <- loss + loss_layer$public_methods$loss(self$lstm_node_list[[i]]$state$h, y_list[[i]])
        diff_h <- loss_layer$public_methods$bottom_diff(self$lstm_node_list[[i]]$state$h, y_list[[i]])
        diff_h <- diff_h + self$lstm_node_list[[i + 1]]$state$bottom_diff_h
        diff_s <- self$lstm_node_list[[i + 1]]$state$bottom_diff_s
        self$lstm_node_list[[i]]$top_diff_is(diff_h, diff_s)
      }
      
      return(loss)
    },
    
    print = function(...) {
      i <- private$.input
      o <- private$.output
      
      if(length(i) > 10 || length(o) > 10) {
        istring <- paste(c(head(sprintf("%6.3f", i), 5), "...", tail(sprintf("%6.3f", i), 5)), collapse = ", ")
        ostring <- paste(c(head(sprintf("%6.3f", o), 5), "...", tail(sprintf("%6.3f", o), 5)), collapse = ", ")
      } else {
        istring <- paste(sprintf("%6.3f", i), collapse = ", ")
        ostring <- paste(sprintf("%6.3f", o), collapse = ", ")
      }
      
      cat(
        "Object Type: LSTM Network\n\n",
        sprintf("Finished on iteration %s of %s \n\n", private$.outputIteration, private$.inputIteration),
        sprintf("%10s: [ %s ] \n", "Input", istring),
        sprintf("%10s: [ %s ] \n\n", "Output", ostring),
        sprintf("%10s = %.6f", "Final Loss", private$.loss[private$.outputIteration]),
        sep = ""
      )
    }
  )
)

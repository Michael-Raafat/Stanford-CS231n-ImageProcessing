import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = y.shape[0];
  C = W.shape[1];
  for i in range(N):
  	# Calculate estimations (1,C).
  	y_hat = np.matmul(X[i,:],W);
  	#prevent numeric instability.
  	y_hat -= np.max(y_hat);
  	y_hat_exp = np.exp(y_hat);
  	exp_sum = np.sum(y_hat_exp);
  	y_hat_prob = y_hat_exp / exp_sum;
  	for c in range(C):
  		act = 0;
  		if (c == y[i]):
  			loss += -np.log(y_hat_exp[c]/exp_sum);
  			act = 1;
  		dW[:,c] += X[i,:].T * (y_hat_prob[c] - act);


  loss /= N;
  dW /= N;
  dW += reg * 2 * W;
  loss += reg * np.sum(W * W);
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # y_hat (N,C).
  N = X.shape[0];
  C = W.shape[1];
  examples = range(N);
  y_hat = np.matmul(X,W);
  y_hat -= np.max(y_hat, axis = 1, keepdims = 1);
  y_hat_prob = np.exp(y_hat) / np.sum(np.exp(y_hat), axis = 1, keepdims = 1);
  loss = -np.sum(np.log(y_hat_prob[examples,y]));
  loss /= N;
  loss += reg * np.sum(W * W);
  y_hat_prob[examples, y] -= 1;
  dW = np.matmul(X.T, y_hat_prob);
  dW /= N;
  dW += reg * 2 * W;
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


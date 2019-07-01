from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class SoftmaxClassifier(BaseEstimator, ClassifierMixin):  
    """A softmax classifier"""

    def __init__(self, lr = 0.1, alpha = 100, n_epochs = 1000, eps = 1.0e-5,threshold = 1.0e-10 , regularization = True, early_stopping = True):
       
        """
            self.lr : the learning rate for weights update during gradient descent
            self.alpha: the regularization coefficient 
            self.n_epochs: the number of iterations
            self.eps: the threshold to keep probabilities in range [self.eps;1.-self.eps]
            self.regularization: Enables the regularization, help to prevent overfitting
            self.threshold: Used for early stopping, if the difference between losses during 
                            two consecutive epochs is lower than self.threshold, then we stop the algorithm
            self.early_stopping: enables early stopping to prevent overfitting
        """

        self.lr = lr 
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.eps = eps
        self.regularization = regularization
        self.threshold = threshold
        self.early_stopping = early_stopping
        


    """
        Public methods, can be called by the user
        To create a custom estimator in sklearn, we need to define the following methods:
        * fit
        * predict
        * predict_proba
        * fit_predict        
        * score
    """


    """
        In:
        X : the set of examples of shape nb_example * self.nb_features
        y: the target classes of shape nb_example *  1

        Do:
        Initialize model parameters: self.theta_
        Create X_bias i.e. add a column of 1. to X , for the bias term
        For each epoch
            compute the probabilities
            compute the loss
            compute the gradient
            update the weights
            store the loss
        Test for early stopping

        Out:
        self, in sklearn the fit method returns the object itself


    """

    def fit(self, X, y=None):
        X = np.array(X)
        
        prev_loss = np.inf
        self.losses_ = []

        self.nb_feature = X.shape[1]
        self.nb_classes = len(np.unique(y))

        col = np.ones((X.shape[0], 1))
        X_bias = np.hstack((X, col))
        self.theta_ = np.random.rand(X.shape[1] + 1, self.nb_classes)

        for epoch in range(self.n_epochs):

            logits = np.matmul(X_bias, self.theta_)
            probabilities = self._softmax(logits)
            
            loss = self._cost_function(probabilities, y)
            self.theta_ = self.theta_ - self.lr * self._get_gradient(X_bias, y, probabilities)
            
            self.losses_.append(loss)

            if self.early_stopping:
                if abs(loss - prev_loss) < self.threshold:
                    return self
            prev_loss = loss

        return self

    

   
    

    """
        In: 
        X without bias

        Do:
        Add bias term to X
        Compute the logits for X
        Compute the probabilities using softmax

        Out:
        Predicted probabilities
    """

    def predict_proba(self, X, y=None):
        try:
            getattr(self, "theta_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        col = np.ones((X.shape[0], 1))
        X_bias = np.hstack((X, col))

        logits = np.matmul(X_bias, self.theta_)
        return self._softmax(logits)

    """
        In: 
        X without bias

        Do:
        Add bias term to X
        Compute the logits for X
        Compute the probabilities using softmax
        Predict the classes

        Out:
        Predicted classes
    """

    
    def predict(self, X, y=None):
        try:
            getattr(self, "theta_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        pass

        col = np.ones((X.shape[0], 1))
        X_bias = np.hstack((X, col))

        logits = np.matmul(X_bias, self.theta_)
        return np.argmax(self._softmax(logits), axis = 1)

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X,y)

    """
        In : 
        X set of examples (without bias term)
        y the true labels

        Do:
            predict probabilities for X
            Compute the log loss without the regularization term

        Out:
        log loss between prediction and true labels

    """    

    def score(self, X, y=None):
        regularization_state = self.regularization
        self.regularization = False
        prob = self.predict_proba(X, y)
        loss = self._cost_function(prob, y)
        self.regularization = regularization_state  # restore initial state
        return loss
    

    """
        Private methods, their names begin with an underscore
    """

    """
        In :
        y without one hot encoding
        probabilities computed with softmax

        Do:
        One-hot encode y
        Ensure that probabilities are not equal to either 0. or 1. using self.eps
        Compute log_loss
        If self.regularization, compute l2 regularization term
        Ensure that probabilities are not equal to either 0. or 1. using self.eps

        Out:
        Probabilities
    """
    
    def _cost_function(self, probabilities, y):
        hot_y = self._one_hot(y)
        probabilities[probabilities < self.eps] = self.eps
        probabilities[probabilities > 1. - self.eps] = 1 - self.eps
        loss = -np.sum(np.multiply(hot_y, np.log(probabilities))) / hot_y.shape[0]
        regularizer = 0
        if self.regularization:
            regularizer = self.alpha * np.sum(np.square(self.theta_[:-1, :])) / hot_y.shape[0]
        return (loss + regularizer)

    
    """
        In :
        Target y: nb_examples * 1

        Do:
        One hot-encode y
        [1,1,2,3,1] --> [[1,0,0],
                         [1,0,0],
                         [0,1,0],
                         [0,0,1],
                         [1,0,0]]
        Out:
        y one-hot encoded
    """

    
    
    def _one_hot(self,y):
        nb_cats = self.nb_classes
        y_hot = np.zeros(shape=(len(y), nb_cats), dtype=int)
        dict = {}
        y_set = np.unique(y)
        i = 0
        for k in y_set:
            dict[k] = i
            i += 1
        for i in range(0, len(y)):
            y_hot[i, dict[y[i]]] = 1
        return y_hot

    """
        In :
        Logits: (self.nb_features +1) * self.nb_classes

        Do:
        Compute softmax on logits

        Out:
        Probabilities
    """
    
    def _softmax(self,z):
        e_sum = np.sum(np.exp(z), axis = 1)
        return np.exp(z) / e_sum[:, None]

    """
        In:
        X with bias
        y without one hot encoding
        probabilities resulting of the softmax step

        Do:
        One-hot encode y
        Compute gradients
        If self.regularization add l2 regularization term

        Out:
        Gradient

    """

    def _get_gradient(self, X, y, probas):
        hot_y = self._one_hot(y)
        gradient = np.dot(np.transpose(X), probas - hot_y) / X.shape[0]
        mytest = np.sum(gradient, axis=1)
        if self.regularization:
            row = np.zeros((1,self.theta_.shape[1]))
            theta = np.vstack((self.theta_[:-1,:], row))
            regularizer = 2 * self.alpha * theta / X.shape[0]
            gradient += regularizer
        return gradient
    

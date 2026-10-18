"""Multiclass logistic regression with explicit forward/loss/backward calculations.

Forward scores, loss, gradient and parameter updates are explicit NumPy code.
An adaptive gradient step uses backtracking to require a sufficient loss decrease.
Scikit-learn supplies estimator/pipeline interfaces and validation checks only.
"""
import warnings
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


def softmax(scores):
    shifted=scores-np.max(scores,axis=1,keepdims=True)
    numerator=np.exp(shifted)
    return numerator/numerator.sum(axis=1,keepdims=True)


def loss_gradient(parameters, X, encoded_y, sample_weights, C, n_classes):
    n_features=X.shape[1]
    W=parameters[:n_classes*n_features].reshape(n_classes,n_features)
    b=parameters[n_classes*n_features:]
    scores=np.asarray(X@W.T)+b
    shifted=scores-scores.max(axis=1,keepdims=True)
    log_normalizer=np.log(np.exp(shifted).sum(axis=1))
    negative_log_p=log_normalizer-shifted[np.arange(len(encoded_y)),encoded_y]
    S=sample_weights.sum()
    loss=np.dot(sample_weights,negative_log_p)/S+(W*W).sum()/(2*S*C)
    residual=softmax(scores)
    residual[np.arange(len(encoded_y)),encoded_y]-=1
    residual*=sample_weights[:,None]/S
    grad_W=np.asarray(X.T@residual).T+W/(S*C)
    grad_b=residual.sum(axis=0)
    return float(loss),np.concatenate([grad_W.ravel(),grad_b])


class ManualSoftmaxClassifier(ClassifierMixin,BaseEstimator):
    def __init__(self,C=1.0,class_weight='balanced',max_iter=2000,tol=1e-6):
        self.C=C;self.class_weight=class_weight;self.max_iter=max_iter;self.tol=tol

    def fit(self,X,y):
        X,y=check_X_y(X,y,accept_sparse='csr',dtype=np.float64)
        self.classes_,encoded=np.unique(y,return_inverse=True)
        self.n_features_in_=X.shape[1]
        K=len(self.classes_)
        if self.C<=0 or K<2:raise ValueError('Positive C and at least two classes required.')
        if self.class_weight=='balanced':
            counts=np.bincount(encoded,minlength=K)
            weights=len(y)/(K*counts[encoded])
        elif self.class_weight is None:weights=np.ones(len(y))
        else:raise ValueError('Supported class_weight values are balanced and None.')
        initial=np.zeros(K*(X.shape[1]+1))
        parameters=initial
        loss,gradient=loss_gradient(parameters,X,encoded,weights,self.C,K)
        step=1.0;self.loss_history_=[loss];success=False
        for iteration in range(self.max_iter):
            if np.max(np.abs(gradient)) <= self.tol:
                success=True;break
            # Follow the negative gradient. Halve an overlarge step until its
            # actual decrease is at least a small fraction of the linear one.
            for backtrack in range(60):
                proposal=parameters-step*gradient
                next_loss,next_gradient=loss_gradient(proposal,X,encoded,weights,self.C,K)
                if next_loss <= loss-1e-4*step*np.dot(gradient,gradient):break
                step*=0.5
            else:
                warnings.warn('Backtracking could not find a decreasing step.',ConvergenceWarning)
                break
            change=proposal-parameters
            gradient_change=next_gradient-gradient
            curvature=np.dot(change,gradient_change)
            # Recent parameter/gradient changes estimate inverse curvature:
            # use a larger step in a flat direction, then check it by loss.
            step=float(np.clip(np.dot(change,change)/curvature,1e-4,1000)) if curvature>0 else 1.0
            parameters=proposal;loss=next_loss;gradient=next_gradient
            self.loss_history_.append(loss)
        if not success and np.max(np.abs(gradient))<=self.tol:success=True
        if not success:warnings.warn('Manual gradient solver did not reach tolerance.',ConvergenceWarning)
        self.coef_=parameters[:K*X.shape[1]].reshape(K,X.shape[1])
        self.intercept_=parameters[K*X.shape[1]:]
        self.n_iter_=np.array([len(self.loss_history_)-1]);self.training_objective_=float(loss)
        self.gradient_max_=float(np.max(np.abs(gradient)))
        return self

    def decision_function(self,X):
        check_is_fitted(self)
        X=check_array(X,accept_sparse='csr',dtype=np.float64)
        return np.asarray(X@self.coef_.T)+self.intercept_

    def predict_proba(self,X):return softmax(self.decision_function(X))

    def predict(self,X):return self.classes_[np.argmax(self.decision_function(X),axis=1)]

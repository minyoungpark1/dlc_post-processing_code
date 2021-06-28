#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 12:22:07 2021

@author: minyoungpark
"""
import numpy as np

def compute_pseudo_R2(Y, Yhat, Ymean, distr=None, Yhat_null=0):
    eps = np.spacing(1)
    if distr is None:
        Modelr = np.sum(Y*np.log(eps+Yhat) - Yhat)
        Intercr = np.sum(Y*np.log(eps+Ymean) - Ymean)
        Sat_r = np.sum(Y*np.log(eps+Y) - Y)
    
        R2 = (1-(Sat_r-Modelr)/(Sat_r-Intercr))
    
    elif distr is 'logit':
        # Log likelihood of model under consideration
        L1 = 2*len(Y)*np.sum(Y*np.log((Yhat==0)+Yhat)/np.mean(Yhat) + 
                                (1-Y)*np.log((Yhat==1)+1-Yhat)/(1-np.mean(Yhat)));
      
        # Log likelihood of homogeneous model
        # b0_hat_null = mean(log((Y==0)+Y) - log((Y==1)+1-Y));
        
        # b0_hat_null = glmfit(ones(length(Y),1), Y, 'binomial', 'link', 'logit', 'constant', 'off');
        # Yhat_null = 1/(1+exp(-b0_hat_null));
        
        # Yhat_null = mean(Yhat);
      
        L0 = 2*len(Y)*np.sum(Y*np.log((Yhat_null==0)+Yhat_null)/np.mean(Yhat) + 
                             (1-Y)*np.log((Yhat_null==1)+1-Yhat_null)/(1-np.mean(Yhat)));
      
        # Log likelihood of saturated model
        Lsat = 2*len(Y)*np.sum(Y*np.log((Y==0)+Y)/np.mean(Y) + 
                               (1-Y)*np.log((Y==1)+1-Y)/(1-np.mean(Y)));
      
        # Note that saturated log likelihood is 0 for binomial distribution
        # R2 = (1-(Lsat - L1)./(Lsat-L0));
        R2 = 1 - L1/L0;
    
    elif distr is 'gamma':
        k = (np.mean(Y)**2/np.var(Y));
        theta = np.var(Y)/np.mean(Y);
        
        # Log likelihood of model under consideration
        L1 = (k-1)*np.sum(np.log(eps+Yhat)) - np.sum(Yhat)/theta;
      
        # Log likelihood of homogeneous model
        L0 = (k-1)*len(Y)*np.log(eps+Ymean) - len(Y)*Ymean/theta;
        
        # Log likelihood of saturated model
        Lsat = (k-1)*np.sum(np.log(eps+Y)) - np.sum(Y)/theta;
        R2 = (1-(Lsat - L1)/(Lsat-L0));
    
    return R2

def full_log_likelihood(w, X, y):
    score = np.dot(X, w).reshape(1, X.shape[0])
    return np.sum(-np.log(1 + np.exp(score))) + np.sum(y * score)

def null_log_likelihood(w, X, y):
    z = np.array([w if i == 0 else 0.0 for i, w in enumerate(w.reshape(1, X.shape[1])[0])]).reshape(X.shape[1], 1)
    score = np.dot(X, z).reshape(1, X.shape[0])
    return np.sum(-np.log(1 + np.exp(score))) + np.sum(y * score)

def mcfadden_rsquare(w, X, y):
    return 1.0 - (full_log_likelihood(w, X, y) / null_log_likelihood(w, X, y))

def mcfadden_adjusted_rsquare(w, X, y):
    k = float(X.shape[1])
    return 1.0 - ((full_log_likelihood(w, X, y) - k) / null_log_likelihood(w, X, y))
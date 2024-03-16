# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 22:56:58 2017

@author: jaehyuk
"""

import numpy as np
import pyfeng as pf
import scipy.stats as ss


def basket_check_args(spot, vol, corr_m, weights):
    '''
    This function simply checks that the size of the vector (matrix) are consistent
    '''
    n = spot.size
    assert( n == vol.size )
    '''
    Python assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常。断言可以在条件不满足程序运行的情况下直接返回错误，而不必等待程序运行后出现崩溃的情况
    '''
    assert( corr_m.shape == (n, n) )
    return None
    
def basket_price_mc_cv(
    strike, spot, vol, weights, texp, cor_m, 
    intr=0.0, divr=0.0, cp=1, n_samples=10000
):
    # price1 = MC based on BSM
    rand_st = np.random.get_state() # Store random state first
    price1 = basket_price_mc(
        strike, spot, vol, weights, texp, cor_m,
        intr, divr, cp, bsm = True, n_samples = n_samples)
    
    ''' 
    compute price2: mc price based on normal model
    make sure you use the same seed

    # Restore the state in order to generate the same state
    np.random.set_state(rand_st)  
    price2 = basket_price_mc(
        strike, spot, spot*vol, weights, texp, cor_m,
        intr, divr, cp, False, n_samples)
    '''

    np.random.set_state(rand_st)
    price2 = basket_price_mc(
        strike, spot, spot*vol, weights, texp, cor_m,
        intr, divr, cp, bsm = False, n_samples=n_samples)
    
    # print("price2",price2)

    ''' 
    compute price3: analytic price based on normal model
    
    price3 = basket_price_norm_analytic(
        strike, spot, vol, weights, texp, cor_m, intr, divr, cp)
    '''
    price3 = basket_price_norm_analytic(
        strike, spot, vol*spot, weights, texp, cor_m, intr, divr, cp)
    
    # print(price3)
    
    
    # return two prices: without and with CV
    return np.array([price1, price1 - (price2 - price3)])


def basket_price_mc(
    strike, spot, vol, weights, texp, cor_m,
    intr=0.0, divr=0.0, cp=1, bsm=True, n_samples = 100000
):
    basket_check_args(spot, vol, cor_m, weights)
    
    div_fac = np.exp(-texp*divr)
    disc_fac = np.exp(-texp*intr)
    forward = spot / disc_fac * div_fac

    cov_m = vol * cor_m * vol[:,None]
    chol_m = np.linalg.cholesky(cov_m)  # L matrix in slides

    n_assets = spot.size
    znorm_m = np.random.normal(size=(n_assets, n_samples))
    
    if( bsm ) :
        '''
        PUT the simulation of the geometric brownian motion below
        '''
        exponent_part = (np.diagonal(cov_m)*0.5*(-1)*texp)[:,None]+np.sqrt(texp)*chol_m @ znorm_m
        prices = forward[:,None] * np.exp(exponent_part)
    else:
        # bsm = False: normal model
        prices = forward[:,None] + np.sqrt(texp) * chol_m @ znorm_m
    
    price_weighted = weights @ prices
    
    price = np.mean( np.fmax(cp*(price_weighted - strike), 0) )
    return disc_fac * price


def basket_price_norm_analytic(
    strike, spot, vol, weights, 
    texp, cor_m, intr=0.0, divr=0.0, cp=1
):
    
    '''
    The analytic (exact) option price under the normal model
    
    1. compute the forward of the basket
    2. compute the normal volatility of basket
    3. plug in the forward and volatility to the normal price formula
    
    norm = pf.Norm(sigma, intr=intr, divr=divr)
    norm.price(strike, spot, texp, cp=cp)
    
    PUT YOUR CODE BELOW
    '''

    basket_check_args(spot, vol, cor_m, weights)
    
    div_fac = np.exp(-texp*divr)
    disc_fac = np.exp(-texp*intr)

    forward = spot / disc_fac * div_fac @ weights
    cov_m = vol * cor_m * vol[:,None]
    vol = np.sqrt((weights[None,:]@ cov_m @ weights[:,None])[0,0])
    

    '''
    norm = pf.Norm(sigma=vol,intr=intr, divr=divr)
    price = norm.price(strike, spot, texp, cp=cp)
    '''
    
    d = (forward - strike)/(vol*np.sqrt(texp))
    if cp == 1: 
        # call option
        price = (forward - strike)*ss.norm.cdf(d) + vol*np.sqrt(texp)*ss.norm.pdf(d)
        price = price * disc_fac
    else:
        price = (strike - forward)*ss.norm.cdf(-d) + vol*np.sqrt(texp)*ss.norm.pdf(d)
        price = price * disc_fac

    return price

import tensorflow as tf
import numpy as np
from utils import *
from model import *
import time

def dag_gnn(data, hid_dim = 20, h_tol = 1e-8, threshold = 0.3, lambda1 = 0.1, rho_max = 10e20, max_iter = 10e8, n_epochs = 20, test_freq = 5,verbose = False):
    '''
    Function that apply the DAG GNN method to estimate a DAG
    
    Inputs:
        data (numpy.matrix) : [n_samples, n_variables] samples matrix 
        hid_dim  (int) : list of dimensions for neural network hidden layer
        h_tol (float) : tolerance for constraint, exit condition 
        threshold (float) : threshold for W_est edge values
        lambda1 (float) : L1 regularization parameter
        rho_max (float) : max value for rho in augmented lagrangian
        max_iter (int) : max number of iterations
        n_epochs (int) : number of epochs
    Outputs:
        A_est (numpy.matrix): [n_variables, n_variables] estimated graph
    '''
    
    def update_optmizer(optimizer, rho):
        '''related LR to rho, whenever rho gets big, reduce LR proportionally'''
        MAX_LR = 1e-2
        MIN_LR = 1e-4
        
        new_lr = 1e-3/ (np.log10(rho) + 1e-10)
        lr = min(MAX_LR, max(MIN_LR, new_lr)) #if new_lr is inside limits, use it
        
        #update LR
        optimizer.lr = lr
        return optmizer, lr
    
    def train(optimizer):
        '''Model training'''
        
        #update optmizer
        optimizer, lr = update_optmizer(optimizer, rho)
        
        for epoch in range(n_epochs):
            for batch_id, batch_data in enumerate(train_loader):
                #passing through neural network
                en_outputs, logits, adjA, z, de_outputs = vae(batch_data)
                
                #eval loss and compute gradients
                with tf.GradientTape() as tape:
                    tape.watch(vae.trainable_variables)
                    #calculate loss
                    loss = vae._loss(adjA, logits, decoder_out, batch_data)
                gradients = tape.gradient(loss, vae.trainable_variables)
                optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
            
            #run model with test_data
            if epoch % test_freq == 0:
                start_time = time.time()
                mean = tf.keras.metrics.Mean()
                for batch_id, batch_data in enumarete(test_loader):
                    #passing through neural network
                    en_outputs, logits, adjA, z, de_outputs = vae(batch_data)
                    #calculate loss
                    mean(vae._loss(adjA, logits, decoder_out, batch_data))
            
                if verbose:
                    print("Finished epoch %d with mean test loss of %.3f and with %.3f seconds"
                          %(epoch, mean.result().numpy(), time.time() - start_time))
        if verbose:
            print("Finished optimization with constraint parameter rho.")
            
        return adjA
    
    
    ########################
    # Optimization process #
    ########################
        
    n_variables = data.shape[1]
    rho, alpha, h = 1., 0., np.Inf
    
    train_loader, test_loader = data_loader(data)
    
    #setup of neural networks
    new_adj = np.zeros((n_variables, n_variables))
    vae = DAG_GNN_VAE(new_adj, n_variables, hid_dim, n_variables)
    optimizer = tf.keras.optimizers.Adam(1e-3)
    
    for _ in range(int(max_iter)):
        h_new = None
        if verbose:
            print("Started %d iteration."%(max_iter))
        while rho < rho_max:
            A_est = train() 
            h_new = _h(A_est)
                
            #Update constraint parameter rho
            if h_new > 0.25 * h:
                rho = rho*10
            else:
                break
        
        if verbose:
            print("Finished optimization with parameter constraint alpha, final rho: %d"%(rho))
        #Ascent alpha
        h = h_new    
        alpha += rho * h
        
        #Verifyng constraint tolerance
        if h <= h_tol or rho >= rho_max:
            break
    
    #Applyng threshold
    A_est[A_est < A_threshold] = 0
    return A_est
        
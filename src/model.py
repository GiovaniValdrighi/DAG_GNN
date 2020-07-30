import tensorflow as tf

class Encoder(tf.keras.Model):
    '''
    Encoder class for DAG-GNN method

    Inputs:
    adjA (tensor [d, d]) : current estimated adjascency matrix
    ind_dim (int) : dimension of input layer
    hid_dim (int) : dimension of hidden layer
    out_dim (int) : dimension of output layer

    Outputs:
    out (tensor [batch, d]) : output of neural network
    ligs (tensor [d, d]) : product of (I - A^T @ out)
    adjA (tensor [d, d]) : current estimated adjascency matrix

    '''
    def __init__(self, adjA, in_dim, hid_dim, out_dim):
        super(Encoder, self).__init__()
        self.adjA = tf.Variable(initial_value = adjA, trainable = True)
        #self.Wa = tf.Variable(np.zeros(), trainable = True)

        self.fc1 = tf.keras.layers.Dense(hid_dim, activation= 'relu')
        self.fc2 = tf.keras.layers.Dense(out_dim)

    def call(self, inputs):
        '''Forward process of neural network'''
        #calculate I - A^T
        I_adjA = identity_transpose(self.adjA)
        hidden = self.fc1(inputs)
        outputs = self.fc2(hidden)
        logits = tf.matmul(I_adjA, outputs)
        return outputs, logits, self.adjA


class Decoder(tf.keras.Model):
    '''
    Decoder class for DAG-GNN method

    Inputs:
    ind_dim (int) : dimension of input layer
    out_dim (int) : dimension of output layer
    hid_dim (int) : dimension of hidden layer

    Outputs:
    '''
    def __init__(self, in_dim, hid_dim, out_dim):
        super(Decoder, self).__init__()
        self.fc1 = tf.keras.layers.Dense(hid_dim, activation = 'relu')
        self.fc2 = tf.keras.layers.Dense(out_dim)

    def call(self, z_inputs,  adjA):

        #calculate (I - A^T)^(-1)
        I_adjA = identity_transpose(adjA)
        z = tf.matmul(I_adjA, z_inputs)

        hidden = self.fc1(z)
        outputs = self.fc2(hidden)
        return z, outputs
    
class DAG_GNN_VAE(tf.keras.Model):
    '''
    Model class for DAG-GNN method training

    Inputs:
    adjA (tensor [d, d]) : current estimated adjascency matrix
    ind_dim (int) : dimension of input layer
    hid_dim (int) : dimension of hidden layer
    out_dim (int) : dimension of output layer

    Outputs:
    out (tensor [batch, d]) : output of neural network
    ligs (tensor [d, d]) : product of (I - A^T @ out)
    adjA (tensor [d, d]) : current estimated adjascency matrix

    '''
    def __init__(self, adjA, in_dim, hid_dim, out_dim):
        super(DAG_GNN_VAE, self).__init__()
        self.encoder = Encoder(adjA, in_dim, hid_dim, out_dim)
        self.decoder = Decoder(in_dim, hid_dim, out_dim)
    
    def call(self, inputs):
        en_outputs, logits, new_adjA = self.encoder(inputs)
        z, de_outputs = self.decoder(logits, new_adjA)
        return en_outputs, logits, new_adjA, z, de_outputs
        
    def _h(A):
        '''Calculate the constraint of A ensure that it's a DAG'''
        #(Yu et al. 2019 DAG-GNN)
        # h(w) = tr[(I + kA*A)^n_variables] - n_variables
        M = tf.eye(n_variables, num_columns = n_variables) + A/n_variables
        E = M
        for _ in range(n_variables - 2):
            E = tf.linalg.matmul(E, M)
        h = tf.math.reduce_sum(tf.transpose(E) * M) - n_variables
        return h
    
    def _loss(A, logits, X, Y):
        '''
        Function that evaluate the model loss
        loss = kl loss + nll loss + dag constraint + l1 reg + l2 reg
        '''
        # h constraint loss
        h = self._h(A)
        h_loss = 0.5 * rho * h * h + alpha * h
        
        #KL divergence
        kl_loss = tf.sum(tf.pow(logits, 2) / ( 2 * logits.shape[0]))
        
        #negative likelihood loss
        nll_loss = tf.sum(tf.pow(X - Y, 2) / (2 * n_variables))
        
        #L1 penalization
        l1_loss = lambda1 * tf.sum(tf.abs(A))
        
        #diagonal penalization
        diag_loss = 100 * tf.linalg.trace(A * A)
        
        loss = h_loss + kl_loss + nll_loss + l1_loss + diag_loss
        return loss
        
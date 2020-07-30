import tensorflow as tf

def identity_transpose(A):
    '''Calculate (I - A^T)'''
    return tf.eye(A.shape[0], A.shape[0]) - tf.transpose(A)

def identity_transpose_inverse(A):
    '''Calculate (I - A^T)^(-1)'''
    return tf.linalg.inv(identity_transpose(A))
#load data
from networkx import to_numpy_matrix
import numpy as np

def load_node_feature_matrix(X):
    order = sorted(list(X))
    return order

#gcn layer
#stack layer
def relu(x):
    return np.maximum(x,0) # sample activation function using ReLU(.) = max(0, .)

"""

Normalizing the feature representation
The feature representations can be normalized by node degree by transforming the adjacency matrix A by multiplying it with 
the inverse degree matrix D. Thus our simplified proporagation rule with weights looks like this:
f(X,A) = D^-1 * A * H(l) * W(l) or F(X,A) = D^(-0.5) * A * D^(-0.5) * H^(l) * W^(l))
Where H^(0) = X

f(X,A) has two inputs, as follow:
1. A of R^(N*N), the adjacency matrix of graph G, where N is the number of nodes in G
2. X of R^(N*D), the input node feature matrix, where D is the dimension of input node feature vectors

"""

#Layer of GCNs with relu activation function
def gcn_layer(A_hat, D_hat, X, W):
    #f(X,A) = D^-1 * A * H(l) * W(l) where D^-1 = D^-0.5 * D^-0.5 and H^(0)=X
    
    #print('A_hat shape:',A_hat.shape)
    #print('D_hat shape:', D_hat.shape)
    #print('X:', X.shape)
    #print('W shape:', W.shape)
   
    return relu(D_hat**-1 * A_hat * X * W )

#Graph Convolutional Networks(GCNs)
def gcn(X, graph):
    
    #load data 
    order = sorted(list(graph))
    A = to_numpy_matrix(graph, nodelist=order) #Return the graph adjacency matrix as a Numpy matrix
    I = np.eye(graph.number_of_nodes()) #numpy.eye() --> return a 2-D array with ones on the diagonal zeros elsewhere
    
    A_hat = A + I #is the adjacency matrix of the undirected graph G with added self-connection
    
    #applying propagation rules
    D_hat = np.array(np.sum(A_hat, axis=0))[0]
    D_hat = np.matrix(np.diag(D_hat))
   
    #determine random weights
    W_1 = np.random.normal(loc=0, scale=1, size=(X.shape[1], 4))
    
    
    
    
    W_2 = np.random.normal(loc=0, size=(W_1.shape[1], 2))
    
    #Hidden layer 1
    H_1 = gcn_layer(A_hat, D_hat, X, W_1) #H^(0)=X
    
    #Hidden layer 2
    H_2 = gcn_layer(A_hat, D_hat, H_1, W_2)

    output = H_2
    
    
    #Z=f(X,A)
    return output

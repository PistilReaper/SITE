import tensorflow as tf
import numpy as np
import time
import scipy.io as scio


np.random.seed(1234)
tf.set_random_seed(1234)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x, y, u, v, layers):
        
        X = np.concatenate([x, y], 1)
        
        self.lb = X.min(0)
        self.ub = X.max(0)
                
        self.X = X
        
        self.x = X[:,0:1]
        self.y = X[:,1:2]

        self.u = u
        self.v = v
        
        self.layers = layers
        
        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)        
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        self.v_tf = tf.placeholder(tf.float32, shape=[None, self.v.shape[1]])

        
        self.u_pred, self.v_pred = self.net_U(self.x_tf, self.y_tf)
        self.udiff_pred, self.vdiff_pred = self.net_diff(self.x_tf, self.y_tf)
        
        self.loss_u = tf.reduce_mean(tf.square(self.u_tf - self.u_pred))
        self.loss_v = tf.reduce_mean(tf.square(self.v_tf - self.v_pred))
                        
        self.loss = tf.log(self.loss_u  + self.loss_v) 
            
                    
                    
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 10000,
                                                                           'maxfun': 10000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 0.1 * np.finfo(float).eps})        
        
        self.optimizer_Adam = tf.contrib.opt.NadamOptimizer(learning_rate = 1e-4) 
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)                    
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
        
    def net_U(self, x, y):
        
        Y = self.neural_net(tf.concat([x,y], 1), self.weights, self.biases)
        
        u = Y[:, 0:1]
        v = Y[:, 1:2]
        
        return u, v
    
    def net_diff(self, x, y):
        u, v = self.net_U(x, y)          

        ## derivatives 
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]    

        return (u_x, u_y), (v_x, v_y)
        
    def callback(self, loss):
        print('Loss: %.3e.' % (loss))
    
    def train(self, nIter): 

        tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.u_tf: self.u, self.v_tf: self.v}
        
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()
            
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss],#
                                loss_callback = self.callback)
            
    
    def predict(self, x_star, y_star):
        
        tf_dict = {self.x_tf: x_star, self.y_tf: y_star}

        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        u_diff_star = self.sess.run(self.udiff_pred, tf_dict)
        v_diff_star = self.sess.run(self.vdiff_pred, tf_dict)
        
        return u_star, u_diff_star, v_star, v_diff_star
        
        
if __name__ == "__main__": 
      
    
    layers = [2, 60, 60, 60, 60, 60, 60, 60, 60, 2]
    

    train_data = np.loadtxt('data/TecGrid-Cavity_Kn0.005_nloop50000000.dat', delimiter=' ', skiprows = 3)
    x = train_data[:,0:1]
    y = train_data[:,1:2]
    u = train_data[:,3:4]
    v = train_data[:,4:5]
    iter_times = 20000
    
    model = PhysicsInformedNN(x, y, u, v, layers)
    model.train(iter_times)

    # Prediction
    u_pred, udiff_pred, v_pred, vdiff_pred = model.predict(x, y)

    u_x = udiff_pred[0]
    u_y = udiff_pred[1]
    v_x = vdiff_pred[0]
    v_y = vdiff_pred[1]

    # Error
    error_u = np.linalg.norm(u-u_pred,2)/np.linalg.norm(u,2)
    a =  (u-u_pred)/u
    abs_error_u = np.mean(abs(a))
    print('Error u: %e' % (error_u))
    print('Abs error u: %e' % (abs_error_u))

    error_v = np.linalg.norm(v-v_pred,2)/np.linalg.norm(v,2)
    b =  (v-v_pred)/v
    abs_error_v = np.mean(abs(b))
    print('Error v: %e' % (error_v))
    print('Abs error v: %e' % (abs_error_v))
            
    scio.savemat('data/gradients.mat', {'x':x,'y':y,'u':u,'v':v, 'u_x':u_x, 'u_y':u_y, 'v_x':v_x, 'v_y':v_y
                                                                })
    
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
class RBF(object):
    '''
    This implementation uses the gaussian kernel with scale parameter
    based on the selected supports. Other kernels are possible.
    The supports are found via orthogonal least squares. Random 
    initialization, k-means initialization and others are also possible.
    '''
    
    def __init__(self, x, y):
        '''
        x: n*k matrix, exogenous variables
        y: n vector, endogenous variable
        '''
        self.w = []
        self.x = x
        self.y = y
        n, k = self.x.shape
        m, = self.y.shape
        assert(n==m)
        
        er = np.zeros(n)
        for i, xi in enumerate(self.x):
            p = self.distance(xi.reshape((1,k)), self.x)
            er[i] = np.power(p@self.y, 2) / (p@p.T)
        self.w.append(np.argmax(er))
        self.er = er
    
        
    def add_factor(self):
        '''
        add a 'node' to the RBF 'network', ie add a support.
        Follows the principle of orthogonal least squares to
        find the next most informative support. 
        See S. Chen, C. F. N. Cowan, and P. M. Grant, `Orthogonal least squares learning algorithm for radial basis fuctions', IEEE transactions on neural networks vol 2 no 2, March 1991
        '''
        n, k = self.x.shape
        er = np.zeros(n)
        
        basis = self.distance(self.x[self.w,:], self.x)
        for i in range(1, len(self.w)):
            basis[i,:] = self.gram_schmidt(basis[i,:], basis[:i,:])
            
        for i, xi in enumerate(self.x):
            if i in self.w:
                continue
            w = self.gram_schmidt(self.distance(xi.reshape((1,k)), self.x), basis)
            
            er[i] = np.power(w@self.y, 2) / (w@w.T)
        self.w.append(np.argmax(er))
        
        return er
        
        
    def gram_schmidt(self, w, basis):
        '''
        Use the plain Gram-Schmidt procedure to obtain the part of w
        that is orthogonal to the (orthogonal) rows of basis.
        '''
        for pj in basis:
            a = (w@pj) / (pj@pj)
            w -= a*pj
        return w
        
        
    def distance(self, left, right):
        '''
        left: n1*n2 matrix
        right: n3*n2 matrix
        calculate kernelized distance between each pair of rows
        returns: n1*n3 matrix, with (i,j) the distance between left[i,:] and right[j,:]
        '''
        n1, n2 = left.shape
        n3, n4 = right.shape
        assert(n2==n4)
        
        sigma = self.sigma()
        
        d = left.reshape((n1,n2,1)) - right.T.reshape((1,n2,n3))
        return np.exp(-np.power(d,2).sum(1)/sigma)
    
    
    def sigma(self):
        '''
        the scale parameter of the distance measure
        '''
        supports = self.x[self.w,:]
        p,k = supports.shape
        if p <= 1:
            return 1
        dist = np.linalg.norm(supports.reshape(p,k,1) - supports.T.reshape(1,k,p), axis=1)
        return dist.sum() / (p-1) / (p-1) / 2
    
    
    def mse(self):
        '''mean squared error'''
        _, d, theta = self.fit()
        return np.power(self.y - d@theta, 2).mean()
        
    
    def fit(self):
        '''fits the network on the selected supports'''
        supports = self.x[self.w,:]
        distances = self.distance(self.x, supports)
        theta = np.linalg.lstsq(distances, self.y, rcond=None)[0]
        return supports, distances, theta
    
    
    def predict(self, z):
        '''predict the value of z'''
        supports, _, theta = self.fit()
        distance = self.distance(z, supports)
        return distance@theta
        
        

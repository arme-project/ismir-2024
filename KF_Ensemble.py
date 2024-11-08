import numpy as np
from scipy.linalg import block_diag

def cov2cor(V):
    '''Covariance to correlation matrix'''
    stdevs = np.sqrt(np.diag(V))
    V_cor = np.zeros(V.shape)
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            V_cor[i][j] = V[i][j]/(stdevs[i]*stdevs[j])
    
    return stdevs, V_cor

def cor2cov(stdevs, V_cor):
    '''Correlation to covariance matrix'''
    V = np.zeros(V_cor.shape)
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            V[i][j] = V_cor[i][j]*stdevs[i]*stdevs[j]
    
    if np.all(np.linalg.eigvals(V) > 0):
        print('Matrix is positive-definite!')
    else:
        print('Matrix is NOT positive-definite!')
    
    return V

def cov_mtx_ab(std, corr):
    '''
    Create covariance matrix for alphas and betas.
    Obs.: Only implemented to four performers.
    '''

    K = 4
    corr_block = np.array([
        [1, corr, corr],
        [corr, 1, corr],
        [corr, corr, 1]
    ])

    corr_mtx = block_diag(corr_block, corr_block, corr_block, corr_block)
    stdevs = std*np.ones(K*(K - 1))

    cov_mtx = cor2cov(stdevs, corr_mtx)

    return cov_mtx

class KalmanFilterEnsemble:
    '''
    Kalman Filter for ensemble synchronisation
    Following notation of:
    Giovanni Petris, Sonia Petrone & Patrizia Campagnoli (2009)
    'Dynamic Linear Models with R'
    with the small difference that time-indexes are 'n', not 't'.
    '''

    def __init__(self, y, cov_obs, cov_theta, k0, C0):

        #### NUMBER OF PERFORMERS ####
        self.K = y.shape[0]

        #### INPUT LENGTH ####
        self.N = y.shape[1]

        #### COLLECTION OF VARIABLES FROM INPUT ####
        self.y = y
        self.cov_obs = cov_obs
        self.cov_theta = cov_theta
        self.k0 = k0
        self.C0 = C0

        #### VARIABLE ALLOCATION ####
        # For 'theta_n | y_{1:n}' (filtering)
        self.k = []
        self.C = []
        self.KG = [] # Kalman Gain

        # For 'theta_n | y_{1:n - 1}' (filtering)
        self.a = []
        self.R = []

        # For 'y_n | y_{1:n - 1}' (filtering)
        self.f = []
        self.Q = []

        # Filtered time-series
        self.y_filt = []

        # For 'theta_n | y_{1:N}' (smoothing)
        self.s = []
        self.S = []

        # Noises
        self.V = [] # Observation noise
        self.W = [] # Transition noise

        # System matrices
        self.F = [] # Observation matrix
        self.G = [] # Transition matrix

        # Forecast error
        self.e = []

        #### INITIALISATION ####
        self.k.append(self.k0)
        self.C.append(self.C0)

        self.y_filt.append(self.y[:, 0])

        # Dummy initialisation on t = 0
        self.KG.append('NO')

        self.a.append('NO')
        self.R.append('NO')

        self.f.append('NO')
        self.Q.append('NO')

        self.F.append('NO')
        self.G.append('NO')

        self.V.append('NO')
        self.W.append('NO')

        self.e.append('NO')
    
    def filter(self):
        for n in range(1, self.N):
            # Update observation matrix
            # Will be constant in practice! Placeholder for future modifications on the code
            self.F.append(np.hstack([
                np.zeros((self.K, self.K)), np.identity(self.K), np.zeros((self.K, self.K*(self.K - 1))), np.zeros((self.K, self.K*(self.K - 1)))
            ]))
            
            # Update transition matrix
            A = []
            for i in range(self.K):
                A_i = []
                for j in range(self.K):
                    if j != i:
                        A_i.append(self.y[i, n - 1] - self.y[j, n - 1])
                A.append(A_i)
            G_A = -block_diag(*A) # Denoted by G_n^{T\beta}, G_n^{r\alpha}, and G_n^{r\beta} in paper
            
            self.G.append(np.block([
                [np.identity(self.K), np.zeros((self.K, self.K)), np.zeros((self.K, self.K*(self.K - 1))), G_A],
                [np.identity(self.K), np.zeros((self.K, self.K)), G_A, G_A],
                [np.zeros((self.K*(self.K - 1), self.K)), np.zeros((self.K*(self.K - 1), self.K)), np.identity(self.K*(self.K - 1)), np.zeros((self.K*(self.K - 1), self.K*(self.K - 1)))],
                [np.zeros((self.K*(self.K - 1), self.K)), np.zeros((self.K*(self.K - 1), self.K)), np.zeros((self.K*(self.K - 1), self.K*(self.K - 1))), np.identity(self.K*(self.K - 1))]
            ]))
            
            # Update noise covariances
            # Will be constant in practice! Placeholder for future modifications on the code
            self.V.append(self.cov_obs)
            self.W.append(self.cov_theta)
            
            # Predict one-step-ahead: theta_n | y_{1:n - 1}
            self.a.append(self.G[n].dot(self.k[n - 1]))
            self.R.append(self.G[n].dot(self.C[n - 1].dot(self.G[n].T)) + self.W[n])

            # Predict one-step-ahead: y_t | y_{1:n - 1}
            self.f.append(self.F[n].dot(self.a[n]))
            self.Q.append(self.F[n].dot(self.R[n].dot(self.F[n].T)) + self.V[n])

            # Update one-step-ahead prediction for theta_n: theta_n | y_{1:n}
            self.y_filt.append(self.f[n])
            e = self.y[:, n] - self.y_filt[n]
            
            self.KG.append(self.R[n].dot(self.F[n].T.dot(np.linalg.inv(self.Q[n]))))
            self.k.append(self.a[n] + self.KG[n].dot(e))
            self.C.append(self.R[n] - self.KG[n].dot(self.F[n].dot(self.R[n])))
                
        self.y_filt = np.array(self.y_filt).T
    
    def smooth(self):
        '''Kalman smoother. Requires that function filter() was run previously.'''

        ## Initialisation
        self.s.append(self.k[-1])
        self.S.append(self.C[-1])
        
        for n in range(self.N - 2, -1, -1):
            AUX = self.C[n].dot(self.G[n + 1].T.dot(np.linalg.inv(self.R[n + 1])))

            self.s.append(self.k[n] + AUX.dot(self.s[-1] - self.a[n + 1]))
            self.S.append(self.C[n] - AUX.dot((self.R[n + 1] - self.S[-1]).dot(AUX.T)))
        
        self.s = np.array(self.s).T[:, ::-1]
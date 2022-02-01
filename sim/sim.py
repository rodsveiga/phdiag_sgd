import numpy as np
from sklearn.preprocessing import normalize
from scipy.linalg import orth
import os
from simcy import cython_fit
np.random.seed(0)

class sim():
    def __init__(self, d, n, k):
        self.d = d
        self.n = n
        self.k = k

    def set_teacher(self, Wt= None, orthWt= False):
        self.Wt =  np.sqrt(self.d)*normalize(np.random.randn(self.k, self.d), axis=1, norm='l2') if Wt is None else Wt
        self.orthWt = orthWt
        if orthWt and Wt is None:
            self.Wt = np.sqrt(self.d)*orth(self.Wt.T,rcond=None).T
        return self.Wt


    def set_W0(self, W0= None, lin_coeff=None):
        self.W0 =  np.sqrt(self.d)*normalize(np.random.randn(self.n, self.d), axis=1, norm='l2') if W0 is None else W0
        self.lin_coeff = lin_coeff
        self.lin_coeffW0 = False

        if lin_coeff is not None and W0 is None:
            self.W0 = lin_coeff @ self.Wt
            self.lin_coeffW0 = True
            self.lin_coeff = lin_coeff

        return self.W0

    def fit(self, alpha_max, kappa, delta, eta0=1., norm= True, noise= 0., lazy= False, tol=1e-10, save_log= False, save_folder= 'results/d32/', save_key= ''):

        if (kappa + delta) > 0:
            print('GREEN region: kappa + delta > 0')
            x_scale = self.d**(1.+kappa+delta)
            alpha_scale = 'd%.4f' % (1.+kappa+delta)
            region = 'green'

        if (kappa + delta) == 0:
            print('BLUE line: kappa + delta = 0')
            x_scale = self.d**(1.+kappa+delta)
            alpha_scale = 'd%.4f' % (1.+kappa+delta)
            region = 'blue'

        if -delta  > kappa and -delta < (kappa+1.)/2:
            print('ORANGE region: kappa < -delta < (kappa+1)/2')
            x_scale = self.d**(1.+2*(kappa+delta))
            alpha_scale = 'd%.4f' % (1.+2*(kappa+delta))
            region = 'orange'

        eta_eff = eta0 / (self.d**(delta))

        print('alpha_scale: ',alpha_scale)
        if norm:
            print('Committees normalized with 1/n and 1/k')
        else:
            print('ATTENTION Committees are NOT normalized with 1/n and 1/k')

        if lazy:
            print('ATTENTION Lazy training regime')

        if self.orthWt:
            print('Wt orthogonal')
        if self.lin_coeffW0:
            print('W0 linear combination of Wt')

        if noise > 0:
            print('Teacher with additive output noise: y_t = f + sqrt(noise)*xi ; noise= %s' % '{:.0e}'.format(noise))
        else:
            print('Noiseless teacher')

        m_max = int(alpha_max*x_scale)

        log10_alpha_max = int(np.log10(alpha_max))
        savelist = self.savelog_list(log_x_max= log10_alpha_max, scale= x_scale)
        plotlist = self.xlog_scale(log_x_max= log10_alpha_max, scale= x_scale)
        W = np.copy(self.W0)

        dformat = '{:0'+str(int(round(np.log10(self.d))))+'}'
        print_d = dformat.format(self.d)

        if self.lin_coeffW0:
            if lazy:
                folder_id = '%s_d%s_n%s_k%s_kappa%.5f_delta%.5f_eta0%.5f_noise%s_norm%s_orthWt%s_lincombW0%s_lazy%s' % (region, print_d, '{:03}'.format(self.n), '{:03}'.format(self.k), kappa, delta, eta0, '{:.0e}'.format(noise), str(norm), str(self.orthWt), str(self.lin_coeffW0), str(lazy))
            else:
                folder_id = '%s_d%s_n%s_k%s_kappa%.5f_delta%.5f_eta0%.5f_noise%s_norm%s_orthWt%s_lincombW0%s' % (region, print_d, '{:03}'.format(self.n), '{:03}'.format(self.k),  kappa, delta, eta0, '{:.0e}'.format(noise), str(norm), str(self.orthWt), str(self.lin_coeffW0))
        else:
            if lazy:
                folder_id = '%s_d%s_n%s_k%s_kappa%.5f_delta%.5f_eta0%.5f_noise%s_norm%s_orthWt%s_lazy%s' % (region, print_d, '{:03}'.format(self.n), '{:03}'.format(self.k),  kappa, delta, eta0, '{:.0e}'.format(noise), str(norm), str(self.orthWt), str(lazy))
            else:
                folder_id = '%s_d%s_n%s_k%s_kappa%.5f_delta%.5f_eta0%.5f_noise%s_norm%s_orthWt%s' % (region, print_d, '{:03}'.format(self.n), '{:03}'.format(self.k),  kappa, delta, eta0, '{:.0e}'.format(noise), str(norm), str(self.orthWt))

        file_path_id = save_folder + folder_id
        isExist = os.path.exists(file_path_id)
        if not isExist:
            os.makedirs(file_path_id)

        if self.lin_coeffW0:
            id_ = 'W0_lincomb_Wt__linear_coeff.npy'
            file_path = file_path_id + '/' + id_
            np.save(file_path, self.lin_coeff)

        alphaf, egf, Wf, W0_ = cython_fit(m_max, norm, W, self.Wt, eta_eff, noise, x_scale, lazy, tol, alpha_max, plotlist, savelist, save_log, file_path_id, alpha_scale, save_key)

        return alphaf, egf, Wf, W0_



    def xlog_scale(self, log_x_max, scale, log_base=10):
        '''Logaritmic scale up to log_alpha_max'''

        bd_block = np.arange(0, log_base**2, log_base) + log_base
        bd_block = bd_block[0:-1]
        xlog = np.tile(bd_block, log_x_max)

        xlog[(log_base-1) : 2*(log_base-1)] = log_base*xlog[(log_base-1) : 2*(log_base-1)]

        for j in range(1, log_x_max - 1):
            xlog[(j+1)*(log_base-1) : (j+2)*(log_base-1)] = log_base*xlog[  j*(log_base-1) :  (j+1)*(log_base-1)  ]

        xlog = np.insert(xlog, 0,  np.arange(1,log_base), axis=0)
        xlog = np.insert(xlog, len(xlog),log_base**(log_x_max+1), axis=0)

        jlog = (xlog*scale).astype(int)

        return jlog


    def savelog_list(self, log_x_max, scale):
        '''Logaritmic scale up to log_alpha_max'''
        xlog = np.logspace(0, log_x_max, log_x_max+1, endpoint=True).astype(int)
        save_xlog = (xlog*scale).astype(int)
        return save_xlog

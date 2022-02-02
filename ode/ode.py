import numpy as np
from sklearn.preprocessing import normalize
import os
from odecy import odesolver
np.random.seed(0)

class ode():
    def __init__(self, d, n, k):
        self.d = d
        self.n = n
        self.k = k

    def set_M0(self, M0= None, d_init= None, orthWt= False):
        if M0 is None:
            self.Wt = np.random.randn(self.k, self.d)
            self.M0 =  self.W0 @ self.Wt.T / self.d
            # Flag informed False: generated here
            self.M0_inf = False
            return self.M0
        else:
            # Flag informed True: externally informed
            self.M0_inf = True
            self.d_init = d_init
            self.M0 = M0

    def set_P(self, P= None, d_init= None, orthWt= False):
        self.orthWt = orthWt
        if P is None:
            self.P = self.Wt @ self.Wt.T / self.d
            # Flag informed False: generated here
            self.P_inf = False
            return self.P
        else:
            # Flag informed True: externally informed
            self.P_inf = True
            self.d_init = d_init
            self.P = P


    def set_Q0(self, Q0= None, d_init= None, lin_coeff=False):
        self.lin_coeffW0 = lin_coeff
        if Q0 is None:
            self.W0 =  np.sqrt(self.d)*normalize(np.random.randn(self.n, self.d), axis=1, norm='l2')
            self.Q0 = self.W0 @ self.W0.T / self.d
            # Flag informed False: generated here
            self.Q0_inf = False
            return self.Q0
        else:
            # Flag informed True: externally informed
            self.Q0_inf = True
            self.d_init = d_init
            self.Q0 = Q0



    def fit(self, t_max, delta, kappa, gamma_0= None, norm= True, noise= 0., tol=1e-10, save_log= False, save_folder= 'results/d_kappa_del/', save_key= ''):

        M_dyn = True
        I3_on = True
        eta_inf = gamma_0
        alpha_max = t_max

        if eta_inf is None:
            eta = self.n
            eta_print = str(eta_inf)
        else:
            eta = eta_inf
            eta_print = '%.4f' % eta_inf


        if (kappa + delta) > 0:
            print('GREEN region: kappa + delta > 0')
            I4_on = False
            x_scale = self.d**(1.+kappa+delta)
            alpha_scale = 'd%.4f' % (1.+kappa+delta)
            region = 'green'

        if (kappa + delta) == 0:
            print('BLUE line: kappa + delta = 0')
            I4_on = True
            x_scale = self.d**(1.+kappa+delta)
            alpha_scale = 'd%.4f' % (1.+kappa+delta)
            region = 'blue'

        if -delta  > kappa and -delta < (kappa+1.)/2:
            print('ORANGE region: kappa < -delta < (kappa+1)/2')
            I4_on = True
            I3_on = False
            M_dyn = False
            x_scale = self.d**(1.+2*(kappa+delta))
            alpha_scale = 'd%.4f' % (1.+2*(kappa+delta))
            region = 'orange'

        print(' ')
        print('time scaling: ',alpha_scale)

        print('Initial conditions:')
        print('Q0 externally informed: %s' % str(self.Q0_inf))
        print('M0 externally informed: %s' % str(self.M0_inf))
        print('Teacher:')
        print('P externally informed: %s' % str(self.P_inf))
        if norm:
            print('Committees normalized with 1/n and 1/k')
        else:
            print('ATTENTION Committees are NOT normalized with 1/n and 1/k')


        if noise > 0:
            print('Teacher with additive output noise: y_t = f + sqrt(noise)*xi ; noise= %s' % '{:.0e}'.format(noise))
        else:
            print('Noiseless teacher')

        m_max = int(alpha_max*x_scale)


        log10_alpha_max = int(np.log10(alpha_max))
        savelist = self.savelog_list(log_x_max= log10_alpha_max, scale= x_scale)
        plotlist = self.xlog_scale(log_x_max= log10_alpha_max, scale= x_scale)

        dformat = '{:0'+str(int(round(np.log10(self.d))))+'}'
        print_d = dformat.format(self.d)

        if self.Q0_inf+self.M0_inf+self.P_inf > 0:
            if self.d_init is not None:
                d_initformat = '{:0'+str(int(round(np.log10(self.d_init))))+'}'
                print_dinit = d_initformat.format(self.d_init)

                folder_id = '%s_ddiscr%s_p%s_k%s_kappa%.5f_delta%.5f_gamma0_%s_noise%s_norm%s_init_inf_Q0_%s_M0_%s_P_%s_dinit%s_orthWt%s_lincombW0%s' % (region, print_d, '{:03}'.format(self.n), '{:03}'.format(self.k), kappa, delta, eta_print, '{:.0e}'.format(noise), str(norm), str(self.Q0_inf), str(self.M0_inf), str(self.P_inf), print_dinit, str(self.orthWt), str(self.lin_coeffW0))
            else:
                folder_id = '%s_ddiscr%s_p%s_k%s_kappa%.5f_delta%.5f_gamma0_%s_noise%s_norm%s_init_inf_Q0_%s_M0_%s_P_%s_orthWt%s_lincombW0%s' % (region, print_d, '{:03}'.format(self.n), '{:03}'.format(self.k), kappa, delta, eta_print, '{:.0e}'.format(noise), str(norm), str(self.Q0_inf), str(self.M0_inf), str(self.P_inf),  str(self.orthWt), str(self.lin_coeffW0))
        else:
            folder_id = '%s_ddiscr%s_n%s_p%s_kappa%.5f_delta%.5f_gamma0_%s_noise%s_norm%s_init_uninf' % (region, print_d, '{:03}'.format(self.n), '{:03}'.format(self.k), kappa, delta, eta_print, '{:.0e}'.format(noise), str(norm))

        file_path_id = save_folder + folder_id
        isExist = os.path.exists(file_path_id)
        if not isExist:
            os.makedirs(file_path_id)

        alphaf, egf = odesolver(m_max, norm, self.Q0, self.M0, self.P, self.d, eta, noise, x_scale, tol, alpha_max, plotlist, savelist, save_log, file_path_id, alpha_scale, save_key, I4_on, I3_on, M_dyn)

        return alphaf, egf



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

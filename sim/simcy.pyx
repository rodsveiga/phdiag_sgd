# cython: boundscheck = False
# cython: cdivision = True
# cython: wraparound = False
import cython
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, exp, erf, acos
from libc.stdlib cimport malloc, free
import time


cdef inline double square(double x): return x*x

cdef double f(double[:] loc_field):
    '''Model'''
    cdef double phi =0.
    cdef unsigned int j, L= loc_field.shape[0]
    for j in range(0,L):
        phi += erf( loc_field[j]/ sqrt(2) )
    return phi

cdef double f0(double[:] loc_field, loc_field0):
    '''Linearized model'''
    cdef double phi =0., pi= acos(-1)
    cdef unsigned int j, L= loc_field.shape[0]
    for j in range(0,L):
        phi += erf(loc_field0[j]/ sqrt(2)) + exp(-square(loc_field0[j])/2.)*sqrt(2./pi)*(loc_field[j]-loc_field0[j])
    return phi


cpdef cython_fit(long m_max, bint norm, double[:,:] W, double[:,:] Wt, double eta, double noise, double x_scale, bint lazy, double tol, double alpha_max, plotlist, savelist, save_log, file_path_id, alpha_scale, save_key):
    cdef unsigned int j, p, q, r, u, v, j1, j2, j3, k = Wt.shape[0], n = W.shape[0], d = W.shape[1], m = 100000
    cdef double alpha, eg_avj, yjt, yjs, fs_test, ns, nt, eta_, lz, eg_0= 0., eg_avj_old= 0., eg_, pi= acos(-1), noise_sqt = sqrt(noise)
    cdef double *arr = <double*>malloc(d * sizeof(double))
    cdef double[:] xj = <double[:d]>arr
    cdef double *arr2 = <double*>malloc(m * d * sizeof(double))
    cdef double[:,:] X_test = <double[:m,:d]>arr2
    cdef double *arr3 = <double*>malloc(n * sizeof(double))
    cdef double[:] h = <double[:n]>arr3
    cdef double *arr4 = <double*>malloc(k * sizeof(double))
    cdef double[:] ht = <double[:k]>arr4
    cdef double *arr5 = <double*>malloc(m * sizeof(double))
    cdef double[:] ft_test = <double[:m]>arr5
    cdef list eg_av_=[], alpha_plot_=[], WW0_ = []
    cdef str tol_string= False
    cdef double *arr6 = <double*>malloc(n * sizeof(double))
    cdef double[:] h0 = <double[:n]>arr6
    cdef double *arr7 = <double*>malloc(n * sizeof(double))
    cdef double[:] h0_test = <double[:n]>arr7
    cdef double *arr8 = <double*>malloc(n * n * sizeof(double))
    cdef double[:,:] WW0 = <double[:n,:n]>arr8

    '''Normalization of the committee'''
    if norm:
        ns = 1./n
        nt = 1./k
    else:
        ns = 1.
        nt = 1.

    '''Learning rate scale (from the updated eqs do not change)'''
    eta_= ns*(eta/sqrt(d))

    '''Lazy training'''
    lz = 1. if lazy else 0.

    '''Test set'''
    X_test = np.random.randn(m, d)
    ft_test[:] = 0.
    for r in range(0,m):
        ht[:] = 0.
        for u in range(0,k):
            for v in range(0,d):
                ht[u] += Wt[u,v]*X_test[r][v] / sqrt(d)
        ft_test[r] = f(ht)*nt

    W0 = np.copy(W)
    W0T = W0.T
    t0 = time.time()

    ''''Overlap with W0 (must be 1)'''
    if lazy:
        WW0[:,:] = 0.
        for j1 in range(0,n):
            for j2 in range(0,n):
                for j3 in range(0,d):
                    WW0[j1,j2] += W[j1,j3]*W0T[j3,j2] / d
        WW0_.append(np.array(WW0))

    '''Generalization error at initialization'''
    for r in range(0,m):
        h[:] = 0.
        h0_test[:] = 0.
        for u in range(0,n):
            for v in range(0,d):
                h[u] += W[u,v]*X_test[r][v] / sqrt(d)
                h0_test[u] += W0[u,v]*X_test[r][v] / sqrt(d)
        #####################################
        fs_test = f0(h,h0_test)*ns if lazy else f(h)*ns
        eg_0 += 0.5*square(ft_test[r]-fs_test)/m

    eg_av_.append(eg_0)
    alpha_plot_.append(0.)
    print('d= %d, k= %d, n= %d, eta= %.3f, noise= %s -- j= %d, alpha= %s, eg= %.10f, time= %.2f' % (d, k, n, eta, '{:.0e}'.format(noise), 0, '{:.0e}'.format(0), eg_0, 0) )
    '''-------------------------------------'''

    for j in range(1,m_max+1):
        alpha = j/x_scale
        '''Sampling'''
        xj = np.random.randn(d)
        '''Teacher'''
        #################################
        ht[:] = 0.
        for u in range(0,k):
            for v in range(0,d):
                ht[u] += Wt[u,v]*xj[v] / sqrt(d)
        ##################################
        yjt  = f(ht)*nt + noise_sqt*np.random.randn()

        '''Student'''
        #####################################
        h[:] = 0.
        h0[:] = 0.
        for u in range(0,n):
            for v in range(0,d):
                h[u] += W[u,v]*xj[v] / sqrt(d)
                h0[u] += W0[u,v]*xj[v] / sqrt(d)
        #####################################
        yjs = f0(h,h0)*ns if lazy else f(h)*ns
        '''Gradient'''
        for p in range(0,n):
            for q in range(0,d):
                W[p,q] += (yjt - yjs)*eta_*exp(-square((1.-lz)*h[p] + lz*h0[p])/2.)*sqrt(2./pi) *xj[q]

        plot_j = j in plotlist
        save_j = j in savelist

        if plot_j:
            '''Generalization error'''
            eg_ = 0.
            for r in range(0,m):
                h[:] = 0.
                for u in range(0,n):
                    for v in range(0,d):
                        h[u] += W[u,v]*X_test[r][v] / sqrt(d)
                #####################################
                fs_test = f0(h,h0_test)*ns if lazy else f(h)*ns
                eg_ += 0.5*square(ft_test[r]-fs_test)/m

            eg_avj = eg_
            eg_av_.append(eg_avj)
            alpha_plot_.append(alpha)

            t1 = time.time()
            print('d = %d, k = %d, n = %d, eta = %.3f, noise= %s -- j = %d, alpha = %s, eg = %.10f, time = %.2f' % (d, k, n, eta, '{:.0e}'.format(noise), j, '{:.0e}'.format(alpha), eg_avj,t1-t0) )
            t0 = t1

            if np.abs(eg_avj-eg_avj_old) < tol:
                tol_string = True
                break
            eg_avj_old = eg_avj

            if lazy:
                WW0[:,:] = 0.
                for j1 in range(0,n):
                    for j2 in range(0,n):
                        for j3 in range(0,d):
                            WW0[j1,j2] += W[j1,j3]*W0T[j3,j2] / d
                WW0_.append(np.array(WW0))

            if save_j and j < m_max:
                log_alpha_save = np.where(savelist == j)[0][0]
                if save_log:
                    if lazy:
                        id = 'LOGsave_alpscale_%s_alpha1e%d_%s_eg_Wt_Wf_W0_alpha_WW0.npz' % (alpha_scale, log_alpha_save, save_key)
                        dict_save = {'eg':np.array(eg_av_), 'Wt': np.array(Wt), 'Wf': np.array(W), 'W0': np.array(W0), 'alpha':np.array(alpha_plot_), 'WW0':np.array(WW0_)}
                        file_path = file_path_id + '/' + id
                        np.savez(file_path, **dict_save)
                    else:
                        id = 'LOGsave_alpscale_%s_alpha1e%d_%s_eg_Wt_Wf_W0_alpha.npz' % (alpha_scale, log_alpha_save, save_key)
                        dict_save = {'eg':np.array(eg_av_), 'Wt': np.array(Wt), 'Wf': np.array(W), 'W0': np.array(W0), 'alpha':np.array(alpha_plot_)}
                        file_path = file_path_id + '/' + id
                        np.savez(file_path, **dict_save)

    if tol_string:
        alpha_final = j/x_scale
        print('Terminating: | Delta(eg)| < %s (tol)' %  '{:.0e}'.format(tol) )
        if lazy:
            id = 'FINALsave_tol%s_alpscale_%s_alpha%s_%s_eg_Wt_Wf_W0_alpha_WW0.npz' % ('{:.0e}'.format(tol), alpha_scale, '{:.0e}'.format(alpha_final), save_key)
            dict_save = {'eg':np.array(eg_av_), 'Wt': np.array(Wt), 'Wf': np.array(W), 'W0': np.array(W0), 'alpha':np.array(alpha_plot_),'WW0':np.array(WW0_)}
        else:
            id = 'FINALsave_tol%s_alpscale_%s_alpha%s_%s_eg_Wt_Wf_W0_alpha.npz' % ('{:.0e}'.format(tol), alpha_scale, '{:.0e}'.format(alpha_final), save_key)
            dict_save = {'eg':np.array(eg_av_), 'Wt': np.array(Wt), 'Wf': np.array(W), 'W0': np.array(W0), 'alpha':np.array(alpha_plot_)}
    else:
        print('Terminating: alpha = %s (alpha_max)' %  '{:.0e}'.format(alpha_max) )
        if lazy:
            id = 'FINALsave_max_alpscale_%s_alpha1e%d_%s_eg_Wt_Wf_W0_alpha_WW0.npz' % (alpha_scale, np.log10(alpha_max), save_key)
            dict_save = {'eg':np.array(eg_av_), 'Wt': np.array(Wt), 'Wf': np.array(W), 'W0': np.array(W0), 'alpha':np.array(alpha_plot_),'WW0':np.array(WW0_)}
        else:
            id = 'FINALsave_max_alpscale_%s_alpha1e%d_%s_eg_Wt_Wf_W0_alpha.npz' % (alpha_scale, np.log10(alpha_max), save_key)
            dict_save = {'eg':np.array(eg_av_), 'Wt': np.array(Wt), 'Wf': np.array(W), 'W0': np.array(W0), 'alpha':np.array(alpha_plot_)}

    file_path = file_path_id + '/' + id
    np.savez(file_path, **dict_save)

    return np.array(alpha_plot_), np.array(eg_av_), np.array(W),  np.array(W0)

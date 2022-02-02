# cython: boundscheck = False
# cython: cdivision = True
# cython: wraparound = False
import cython
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, exp, erf, acos, asin
from libc.stdlib cimport malloc, free
import time

cdef inline double square(double x): return x*x

cpdef odesolver(long m_max, bint norm, double[:,:] Q, double[:,:] M, double[:,:] P, int d, double eta, double noise, double x_scale, double tol, double alpha_max, plotlist, savelist, save_log, file_path_id, alpha_scale, save_key, I4_on, I3_on, M_dyn):
    cdef unsigned int j, r, l, s, m, v, n = M.shape[0], k = M.shape[1]
    cdef double egt = 0., pi= acos(-1), dt = 1./x_scale, eg0old = 0., eg0, eg00, alpha, alpha_final, ns, nt
    cdef double C11, C12, C13, C14, C22, C23, C24, C33, C34, C44, L0, L1, L2, L3, L3_, I3, I4, J2
    cdef double *arr = <double*>malloc(n * n * sizeof(double))
    cdef double[:,:] dQ = <double[:n,:n]>arr
    cdef double *arr2 = <double*>malloc(n * k * sizeof(double))
    cdef double[:,:] dM = <double[:n,:k]>arr2
    cdef list alpha_= [], eg_=[]
    cdef str tol_string= False

    '''Normalization of the committee'''
    if norm:
        ns = 1./n
        nt = 1./k
    else:
        ns = 1.
        nt = 1.

    Q0 = np.copy(Q)
    M0 = np.copy(M)

    '''Teacher contribution to generalization error: constant'''
    for j in range(0,k):
        for l in range(0,k):
            egt += square(nt)*asin(P[j,l]/sqrt((1.+P[j,j])*(1.+P[l,l])))/pi

    '''Generalization error at initialization'''
    eg00 = egt
    # Student-student
    for j in range(0,n):
        for l in range(0,n):
            eg00 += square(ns)*asin(Q[j,l]/sqrt((1.+Q[j,j])*(1+Q[l,l])))/pi
    # Student-teacher
    for j in range(0,n):
        for l in range(0,k):
            eg00 -= ns*nt*2*asin(M[j,l]/sqrt((1.+Q[j,j])*(1+P[l,l])))/pi

    eg_.append(eg00)
    alpha_.append(0.)
    print('ddiscr= %d, k= %d, p= %d, gamma_0= %.4f, noise= %s  -- j= %d, t= %s, eg= %.10f, time= %.2f' % (d, k, n, eta,'{:.0e}'.format(noise), 0, '{:.0e}'.format(0), eg00, 0) )
    '''-------------------------------------'''
    t0 = time.time()

    '''- - - Integrating the ODEs - - - -'''
    for v in range(1,m_max+1):

        alpha = v/x_scale

        if I3_on:
            '''--- Integrating M ---'''
            for j in range(0,n):
                for r in range(0,k):
                    '''Fix the pair (j,r)'''
                    dM[j, r] = 0.
                    '''Sum over the student units'''
                    for l in range(0,n):
                        C11 = Q[j,j]
                        C12 = M[j,r]
                        C13 = Q[j,l]
                        C22 = P[r,r]
                        C23 = M[l,r]
                        C33 = Q[l,l]
                        L3 = (1+C11)*(1+C33)-square(C13)
                        I3 = 2*(C23*(1+C11)-C12*C13)/(pi*(1+C11)*sqrt(L3))
                        dM[j, r] -= dt*(eta*ns)*I3*ns
                    '''Sum over the teacher units'''
                    for s in range(0,k):
                        C11 = Q[j,j]
                        C12 = M[j,r]
                        C13 = M[j,s]
                        C22 = P[r,r]
                        C23 = P[r,s]
                        C33 = P[s,s]
                        L3 = (1+C11)*(1+C33)-square(C13)
                        I3 = 2*(C23*(1+C11)-C12*C13)/(pi*(1+C11)*sqrt(L3))
                        dM[j, r] += dt*(eta*ns)*I3*nt

        '''Integrating Q'''
        # Q is symmetric
        for j in range(0,n):
            for l in range(0,j+1):
                # Fix the pair (j,l)
                dQ[j,l] = 0.

                if I3_on:
                    ## I3 contribution
                    # Sum over the student units
                    for m in range(0,n):
                        ## Student-Student (jl)
                        C11 = Q[j,j]
                        C12 = Q[j,l]
                        C13 = Q[j,m]
                        C22 = Q[l,l]
                        C23 = Q[l,m]
                        C33 = Q[m,m]
                        L3 = (1+C11)*(1+C33)-square(C13)
                        I3 = 2*(C23*(1+C11)-C12*C13)/(pi*(1+C11)*sqrt(L3))
                        ## Student-Student (lj)
                        C11 = Q[l,l]
                        C12 = Q[l,j]
                        C13 = Q[l,m]
                        C22 = Q[j,j]
                        C23 = Q[j,m]
                        C33 = Q[m,m]
                        L3 = (1+C11)*(1+C33)-square(C13)
                        I3_ = 2*(C23*(1+C11)-C12*C13)/(pi*(1+C11)*sqrt(L3))

                        dQ[j,l] -=  dt*(eta*ns)*(I3+I3_)*ns

                    # Sum over the teacher units
                    for r in range(0,k):
                        ## Student-teacher (jl)
                        C11 = Q[j,j]
                        C12 = Q[j,l]
                        C13 = M[j,r]
                        C22 = Q[l,l]
                        C23 = M[l,r]
                        C33 = P[r,r]
                        L3 = (1+C11)*(1+C33)-square(C13)
                        I3 = 2*(C23*(1+C11)-C12*C13)/(pi*(1+C11)*sqrt(L3))
                        ## Student-teacher (lj)
                        C11 = Q[l,l]
                        C12 = Q[l,j]
                        C13 = M[l,r]
                        C22 = Q[j,j]
                        C23 = M[j,r]
                        C33 = P[r,r]
                        L3 = (1+C11)*(1+C33)-square(C13)
                        I3_ = 2*(C23*(1+C11)-C12*C13)/(pi*(1+C11)*sqrt(L3))

                        dQ[j,l] += dt*(eta*ns)*(I3+I3_)*nt

                ## I4 contribution
                if I4_on:
                    C11 = Q[j,j]
                    C12 = Q[j,l]
                    C22 = Q[l,l]
                    J2 = 2./(pi*sqrt(1+C11+C22+(C11*C22)-square(C12)))
                    dQ[j,l] += noise*dt*square(eta*ns)*J2

                    for p in range(0,n):
                        for q in range(0,n):
                            C11 = Q[j,j]
                            C12 = Q[j,l]
                            C13 = Q[j,p]
                            C14 = Q[j,q]
                            C22 = Q[l,l]
                            C23 = Q[l,p]
                            C24 = Q[l,q]
                            C33 = Q[p,p]
                            C34 = Q[p,q]
                            C44 = Q[q,q]
                            L4 = (1+C11)*(1+C22) - square(C12)
                            L0 = L4*C34 - C23*C24*(1+C11) - C13*C14*(1+C22) + C12*C13*C24 + C12*C14*C23
                            L1 = L4*(1+C33) - square(C23)*(1+C11) - square(C13)*(1+C22) + 2*C12*C13*C23
                            L2 = L4*(1+C44) - square(C24)*(1+C11) - square(C14)*(1+C22) + 2*C12*C14*C24
                            I4 = (4./(square(pi)*sqrt(L4)))*asin(L0/sqrt(L1*L2))
                            dQ[j,l] += dt*square(eta*ns)*I4*square(ns)

                    for r in range(0,k):
                        for s in range(0,k):
                            C11 = Q[j,j]
                            C12 = Q[j,l]
                            C13 = M[j,r]
                            C14 = M[j,s]
                            C22 = Q[l,l]
                            C23 = M[l,r]
                            C24 = M[l,s]
                            C33 = P[r,r]
                            C34 = P[r,s]
                            C44 = P[s,s]
                            L4 = (1+C11)*(1+C22) - square(C12)
                            L0 = L4*C34-C23*C24*(1+C11) - C13*C14*(1+C22) + C12*C13*C24 + C12*C14*C23
                            L1 = L4*(1+C33) - square(C23)*(1+C11)-square(C13)*(1+C22)+2*C12*C13*C23
                            L2 = L4*(1+C44) - square(C24)*(1+C11)-square(C14)*(1+C22)+2*C12*C14*C24
                            I4 = (4./(square(pi)*sqrt(L4)))*asin(L0/sqrt(L1*L2))
                            dQ[j,l] += dt*square(eta*ns)*I4*square(nt)


                    for p in range(0,n):
                        for r in range(0,k):
                            C11 = Q[j,j]
                            C12 = Q[j,l]
                            C13 = Q[j,p]
                            C14 = M[j,r]
                            C22 = Q[l,l]
                            C23 = Q[l,p]
                            C24 = M[l,r]
                            C33 = Q[p,p]
                            C34 = M[p,r]
                            C44 = P[r,r]
                            L4 = (1+C11)*(1+C22) - square(C12)
                            L0 = L4*C34-C23*C24*(1+C11) - C13*C14*(1+C22) + C12*C13*C24 + C12*C14*C23
                            L1 = L4*(1+C33) - square(C23)*(1+C11)-square(C13)*(1+C22)+2*C12*C13*C23
                            L2 = L4*(1+C44) - square(C24)*(1+C11)-square(C14)*(1+C22)+2*C12*C14*C24
                            I4 = (4./(square(pi)*sqrt(L4)))*asin(L0/sqrt(L1*L2))
                            dQ[j,l] -= 2*dt*square(eta*ns)*I4*(ns*nt)


        # Update Q
        for j in range(0,n):
            for l in range(0,j+1):
                Q[j,l] += dQ[j,l]
                if j != l:
                    Q[l,j] = Q[j,l]

        if M_dyn:
            # Update M
            for j in range(0,n):
                for r in range(0,k):
                    M[j,r] += dM[j,r]

        plot_v = v in plotlist
        save_v = v in savelist
        if plot_v:
            '''Generalization error'''
            eg0 = egt
            for j in range(0,n):
                for l in range(0,n):
                    eg0 += square(ns)*asin(Q[j,l]/sqrt((1.+Q[j,j])*(1+Q[l,l])))/pi
            # Student-teacher
            for j in range(0,n):
                for l in range(0,k):
                    eg0 -= 2*ns*nt*asin(M[j,l]/sqrt((1.+Q[j,j])*(1+P[l,l])))/pi

            eg_.append(eg0)
            alpha_.append(alpha)

            t1 = time.time()
            print('ddiscr= %d, k= %d, p= %d, gamma_0= %.4f, noise= %s  -- j= %d, t= %s, eg= %.10f, time= %.2f' % (d, k, n, eta,'{:.0e}'.format(noise), v, '{:.0e}'.format(alpha), eg0, t1-t0) )
            #print('Q= ', np.array(Q))
            #print('M= ', np.array(M))
            t0 = time.time()

            if np.abs(eg0 - eg0old) < tol:
                tol_string = True
                break
            eg0old = eg0


            if save_v and v < m_max:
                log_alpha_save = np.where(savelist == v)[0][0]
                if save_log:
                    id = 'LOGsave_alpscale_%s_alpha1e%d_%s_eg_Qf_Mf_Q0_M0_P_alpha.npz' % (alpha_scale, log_alpha_save, save_key)
                    dict_save = {'eg':np.array(eg_), 'Qf': np.array(Q), 'Mf': np.array(M), 'Q0': np.array(Q0), 'M0': np.array(M0), 'P': np.array(P), 'alpha':np.array(alpha_)}
                    file_path = file_path_id + '/' + id
                    np.savez(file_path, **dict_save)


    if tol_string:
        alpha_final = j/x_scale
        print('Terminating: | Delta(eg)| < %s (tol)' %  '{:.0e}'.format(tol) )
        id = 'FINALsave_tol%s_alpscale_%s_alpha%s_%s_eg_Qf_Mf_Q0_M0_P_alpha.npz' % ('{:.0e}'.format(tol), alpha_scale, '{:.0e}'.format(alpha_final), save_key)
    else:
        print('Terminating: alpha = %s (alpha_max)' %  '{:.0e}'.format(alpha_max) )
        id = 'FINALsave_max_alpscale_%s_alpha1e%d_%s_eg_Qf_Mf_Q0_M0_P_alpha.npz' % (alpha_scale, np.log10(alpha_max), save_key)

    dict_save = {'eg':np.array(eg_), 'Qf': np.array(Q), 'Mf': np.array(M), 'Q0': np.array(Q0), 'M0': np.array(M0), 'P': np.array(P), 'alpha':np.array(alpha_)}
    file_path = file_path_id + '/' + id
    np.savez(file_path, **dict_save)

    return np.array(alpha_), np.array(eg_)

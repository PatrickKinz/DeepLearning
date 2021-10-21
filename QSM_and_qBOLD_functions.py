import numpy as np

def f_hyper(x):
    '''
    Write hypergeometric function as taylor order 10 for beginning and as x-1 for larger numbers
    Exakt equation: hypergeom(-0.5,[0.75,1.25],-9/16*x.^2)-1
    (Intersection>x)*taylor + (x>=Intersection)*(x-1)
    taylor = - (81*x^8)/10890880 + (27*x^6)/80080 - (3*x^4)/280 + (3*x^2)/10
    Intersection at approx x = 3.72395
    '''
    Intersection = 3.72395
    a = (Intersection>x)*( -81./10890880*pow(x,8) +27./80080*pow(x,6) -3./280*pow(x,4) +3./10*pow(x,2) )
    b = (x>=Intersection)*(x-1)
    return a + b

def f_qBOLD(S0, R2, Y, nu, chi_nb, t):
    output = np.zeros((len(S0),len(t)))
    TE = 40/1000
    Hct = 0.357
    # Blood Hb volume fraction
    psi_Hb = Hct*0.34/1.335
    # Susceptibility of oxyhemoglobin in ppm
    chi_oHb = -0.813
    # Susceptibility of plasma in ppm
    chi_p = -0.0377
    # Susceptibility of fully oxygenated blood in ppm
    chi_ba = psi_Hb*chi_oHb + (1-psi_Hb)*chi_p
    #print('chi_ba', chi_ba)
    #CF = gamma *B0
    gamma = 267.513 #MHz/T
    B0 = 3 #T
    delta_chi0 = 4*np.pi*0.273 #in ppm
    delta_chi0*Hct
    dw = 1./3 * gamma * B0* (Hct * delta_chi0 * (1-Y) + chi_ba - chi_nb )
    #print('dw', dw, dw.shape)
    #FID
    for i in range(6):
        output[:,i] = S0 * np.exp(-R2*t[i] -nu * f_hyper(dw *     t[i] ) )
    #SE rise
    for i in range(6,13):
        output[:,i] = S0 * np.exp(-R2*t[i] -nu * f_hyper(dw * (TE-t[i]) ) )
    #SE fall
    for i in range(13,len(t)):
        output[:,i] = S0 * np.exp(-R2*t[i] -nu * f_hyper(dw * (t[i]-TE) ) )
    return output

def f_QSM(Y, nu, chi_nb ):
    Hct = 0.357
    SaO2 = 0.98
    # Ratio of deoxygenated and total blood volume
    alpha = 0.77;
    # Susceptibility difference between dHb and Hb in ppm
    delta_chi_Hb = 12.522;
    # Blood Hb volume fraction
    psi_Hb = Hct*0.34/1.335
    # Susceptibility of oxyhemoglobin in ppm
    chi_oHb = -0.813
    # Susceptibility of plasma in ppm
    chi_p = -0.0377
    # Susceptibility of fully oxygenated blood in ppm
    chi_ba = psi_Hb*chi_oHb + (1-psi_Hb)*chi_p

    a = (chi_ba/alpha +psi_Hb*delta_chi_Hb * ((1-(1-alpha)*SaO2)/alpha - Y) )*nu
    b = (1 - nu/alpha) * chi_nb
    return np.array([a + b]).T #why transpose

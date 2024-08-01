import numpy as np
#import tensorflow as tf

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

def f_qBOLD(S0, R2, Y, nu, chi_nb, t,TE = 40./1000 ):
    output = np.zeros((len(S0),len(t)))
    #TE = 40/1000
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

def f_qBOLD_GRE(S0, R2, Y, nu, chi_nb, t ):
    output = np.zeros((len(S0),len(t)))
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
    for i in range(len(t)):
        output[:,i] = S0 * np.exp(-R2*t[i] -nu * f_hyper(dw *     t[i] ) )
    return output

def f_qBOLD_GRE_1value(S0, R2, Y, nu, chi_nb, t ):
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
    output = S0 * np.exp(-R2*t -nu * f_hyper(dw *     t ) )
    return output

def f_qBOLD_GRE_3D(S0, R2, Y, nu, chi_nb, t ):
    output = np.zeros((S0.shape[0],S0.shape[1],S0.shape[2],len(t)))
    print('output shape in f_qBOLD_GRE_3D')
    print(output.shape)
    for x in range(output.shape[0]):
        for y in range(output.shape[1]):
            output[x,y,:,:]= f_qBOLD_GRE(S0[x,y,:],R2[x,y,:],Y[x,y,:],nu[x,y,:],chi_nb[x,y,:],t)
    return output

def f_qBOLD_GESSE_1value(S0, R2, Y, nu, chi_nb, t,TE = 47./1000 ):
    output = np.zeros((len(t),1))
    #TE = 47./1000   #tenth echo
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
    #SE rise
    for i in range(0,10):
        output[i,0] = S0 * np.exp(-R2*t[i] -nu * f_hyper(dw * (TE-t[i]) ) )
    #SE fall
    for i in range(10,len(t)):
        output[i,0] = S0 * np.exp(-R2*t[i] -nu * f_hyper(dw * (t[i]-TE) ) )
    return output

def f_qBOLD_GESSE(S0, R2, Y, nu, chi_nb, t,TE = 47./1000 ):
    output = np.zeros((len(S0),len(t)))
    #TE = 47./1000   #tenth echo
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
    #SE rise
    for i in range(0,10):
        output[:,i] = S0 * np.exp(-R2*t[i] -nu * f_hyper(dw * (TE-t[i]) ) )
    #SE fall
    for i in range(10,len(t)):
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

def f_nu(Y,chi_nb,QSM):
    Hct = 0.357
    SaO2 = 0.98
    alpha = 0.77;              # Ratio of deoxygenated and total blood volume
    delta_chi_Hb = 12.522;     # Susceptibility difference between dHb and Hb in ppm
    psi_Hb = Hct*0.34/1.335    # Blood Hb volume fraction
    chi_oHb = -0.813           # Susceptibility of oxyhemoglobin in ppm
    chi_p = -0.0377            # Susceptibility of plasma in ppm
    chi_ba = psi_Hb*chi_oHb + (1-psi_Hb)*chi_p # Susceptibility of fully oxygenated blood in ppm

    nenner = (chi_ba-chi_nb)/alpha + psi_Hb*delta_chi_Hb * ((1-(1-alpha)*SaO2)/alpha - Y)
    nu = (QSM - chi_nb) / nenner
    return nu

def f_hyper_tensor(x):
    '''
    Write hypergeometric function as taylor order 10 for beginning and as x-1 for larger numbers
    Exakt equation: hypergeom(-0.5,[0.75,1.25],-9/16*x.^2)-1
    (Intersection>x)*taylor + (x>=Intersection)*(x-1)
    taylor = - (81*x^8)/10890880 + (27*x^6)/80080 - (3*x^4)/280 + (3*x^2)/10
    Intersection at approx x = 3.72395
    '''
    Intersection = tf.constant(3.72395,dtype=tf.float32)
    a = -81./10890880*tf.math.pow(x,8) +27./80080*tf.math.pow(x,6) -3./280*tf.math.pow(x,4) +3./10*tf.math.pow(x,2)
    b = x-1
    return tf.where(tf.math.greater(Intersection,x),a, b)

def test_f_hyper_tensor():
    t=tf.constant([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,40,50,60,70], dtype=tf.float32)/10
    out = f_hyper_tensor(t)
    plt.plot(t.numpy(),out.numpy(),'o-')

#test_f_hyper_tensor()

def f_qBOLD_tensor(tensor):
    a = tensor[0]
    b = tensor[1]
    c = tensor[2]
    d = tensor[3]
    e = tensor[4]
    S0 = a   #S0     = 1000 + 200 * randn(N).T
    S0 = tf.expand_dims(S0,-1)
    R2 = (30-1) * b + 1
    R2 = tf.expand_dims(R2,-1)
    SaO2 = 0.98
    Y  = (SaO2 - 0.01) * c + 0.01
    Y = tf.expand_dims(Y,-1)
    nu = (0.1 - 0.001) * d + 0.001
    nu = tf.expand_dims(nu,-1)
    chi_nb = ( 0.1-(-0.1) ) * e - 0.1
    chi_nb = tf.expand_dims(chi_nb,-1)
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
    #CF = gamma *B0
    gamma = 267.513 #MHz/T
    B0 = 3 #T
    delta_chi0 = 4*np.pi*0.273 #in ppm
    dw = 1./3 * gamma * B0* (Hct * delta_chi0 * (1-Y) + chi_ba - chi_nb )

    n_elements=a.shape[0]

    TE=40./1000
    t_FID=tf.constant([3,6,9,12,15,18], dtype=tf.float32)/1000
    t_FID=tf.expand_dims(t_FID,0)
    output_FID = S0 * tf.math.exp(-R2*t_FID - nu*f_hyper_tensor(dw*t_FID))
    t_Echo_rise=tf.constant([21,24,27,30,33,36,39], dtype=tf.float32)/1000
    t_Echo_rise=tf.expand_dims(t_Echo_rise,0)
    output_Echo_rise = S0 * tf.math.exp(-R2*t_Echo_rise - nu*f_hyper_tensor(dw*(TE-t_Echo_rise)))
    t_Echo_fall=tf.constant([42,45,48], dtype=tf.float32)/1000
    t_Echo_fall=tf.expand_dims(t_Echo_fall,0)
    output_Echo_fall = S0 * tf.math.exp(-R2*t_Echo_fall - nu*f_hyper_tensor(dw*(t_Echo_fall-TE)))
    return tf.clip_by_value(tf.concat([output_FID,output_Echo_rise,output_Echo_fall],axis=-1),0,2)


def f_QSM_tensor(tensor):
    c = tensor[0]
    d = tensor[1]
    e = tensor[2]

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

    Y  = (SaO2 - 0.01) * c + 0.01
    nu = (0.1 - 0.001) * d + 0.001
    chi_nb = ( 0.1-(-0.1) ) * e - 0.1

    Summand1 = (chi_ba/alpha +psi_Hb*delta_chi_Hb * ((1-(1-alpha)*SaO2)/alpha - Y) )*nu
    Summand2 = (1 - nu/alpha) * chi_nb

    return tf.clip_by_value(Summand1+Summand2,-1,1) #np.array version is np.array([a+b]).T, maybe transpose here too

def f_nu_tensor(c,e,QSM):
    Hct = 0.357
    SaO2 = 0.98
    alpha = 0.77;              # Ratio of deoxygenated and total blood volume
    delta_chi_Hb = 12.522;     # Susceptibility difference between dHb and Hb in ppm
    psi_Hb = Hct*0.34/1.335    # Blood Hb volume fraction
    chi_oHb = -0.813           # Susceptibility of oxyhemoglobin in ppm
    chi_p = -0.0377            # Susceptibility of plasma in ppm
    chi_ba = psi_Hb*chi_oHb + (1-psi_Hb)*chi_p # Susceptibility of fully oxygenated blood in ppm

    Y= (SaO2 - 0.01) * c + 0.01
    chi_nb = ( 0.1-(-0.1) ) * e - 0.1

    nenner = (chi_ba-chi_nb)/alpha + psi_Hb*delta_chi_Hb * ((1-(1-alpha)*SaO2)/alpha - Y)
    nu = tf.math.divide_no_nan( QSM - chi_nb, nenner)
    d = (nu-0.001)/(0.1-0.001)
    return tf.clip_by_value(d,0,1)

def f_nu_tensor_2(tensor):
    c = tensor[0]
    e = tensor[1]
    QSM=tensor[2]
    Hct = 0.357
    SaO2 = 0.98
    alpha = 0.77;              # Ratio of deoxygenated and total blood volume
    delta_chi_Hb = 12.522;     # Susceptibility difference between dHb and Hb in ppm
    psi_Hb = Hct*0.34/1.335    # Blood Hb volume fraction
    chi_oHb = -0.813           # Susceptibility of oxyhemoglobin in ppm
    chi_p = -0.0377            # Susceptibility of plasma in ppm
    chi_ba = psi_Hb*chi_oHb + (1-psi_Hb)*chi_p # Susceptibility of fully oxygenated blood in ppm

    Y= (SaO2 - 0.01) * c + 0.01
    chi_nb = ( 0.1-(-0.1) ) * e - 0.1

    nenner = (chi_ba-chi_nb)/alpha + psi_Hb*delta_chi_Hb * ((1-(1-alpha)*SaO2)/alpha - Y)
    nu = tf.math.divide_no_nan( QSM - chi_nb, nenner)
    d = (nu-0.001)/(0.1-0.001)
    return tf.clip_by_value(d,0,1)


def grid_search_wrapper(input_tensor):
    #map function over n_batch
    output=tf.map_fn(grid_search_nu_Y_tensor,input_tensor,fn_output_signature=tf.float32,parallel_iterations=10)
    #output=tf.ensure_shape(output,[None,900,20,20,5+16+1])
    return output

def grid_search_nu_Y_tensor(input_tensor):
    n_voxel=input_tensor[0].shape[0]
    a     = tf.ensure_shape(input_tensor[0],[n_voxel])                                                    # shape (n_voxel)
    b     = tf.ensure_shape(input_tensor[1],[n_voxel])                                                    # shape (n_voxel)
    c     = tf.ensure_shape(input_tensor[2],[n_voxel])                                                  # shape (n_voxel)
    d     = tf.ensure_shape(input_tensor[3],[n_voxel])                                                   # shape (n_voxel)
    e     = tf.ensure_shape(input_tensor[4],[n_voxel])                                                  # shape (n_voxel)
    qBOLD = tf.ensure_shape(input_tensor[5],[n_voxel,16])
    QSM   = tf.ensure_shape(input_tensor[6],[n_voxel])

    n_grid_c = 20 #Y
    n_grid_d = 20 #nu
    #a                                                # shape n_voxel
    #tf.expand_dims(a,-1)                             #shape (n_voxel,1)
    #tf.repeat(tf.expand_dims(a,-1),n_grid,axis-1)    #shape (n_voxel,n_grid)
    #tf.expand_dims(tf.repeat(tf.expand_dims(a,-1),n_grid,axis-1),-1)    #shape (n_voxel,n_grid,1)
    a_plane = tf.expand_dims(tf.expand_dims(a,-1),-1)*tf.ones([a.shape[0],n_grid_c,n_grid_d])  #shape (n_voxel,n_grid,n_grid)
    b_plane = tf.expand_dims(tf.expand_dims(b,-1),-1)*tf.ones([b.shape[0],n_grid_c,n_grid_d])

    c_start = tf.clip_by_value( tf.expand_dims(c - 0.3,-1)*tf.ones([1,n_grid_d]),0.01,1-0.3 )                      # shape (n_voxel,n_grid)
    c_stop  = tf.clip_by_value( tf.expand_dims(c + 0.3,-1)*tf.ones([1,n_grid_d]),0+0.3,1 )                       # shape (n_voxel,n_grid)
    c_plane = tf.linspace(c_start,c_stop,n_grid_c,axis=-2)                        # shape (n_voxel,n_grid,n_grid)          Y varied along first n_grid

    #d_calc = tf.debugging.check_numerics(f_nu_tensor(c,e,QSM),message='d_calc numerics problem')
    d_calc = f_nu_tensor(c,e,QSM)
    d_start = tf.clip_by_value( tf.expand_dims(d_calc - 0.3,-1)*tf.ones([1,n_grid_d]),0.01,1-0.3 )                      # shape (n_voxel,n_grid)
    d_stop  = tf.clip_by_value( tf.expand_dims(d_calc + 0.3,-1)*tf.ones([1,n_grid_d]),0+0.3,1 )                       # shape (n_voxel,n_grid)
    d_plane = tf.linspace(d_start,d_stop,n_grid_d,axis=-1)                        # shape (n_voxel,n_grid,n_grid)          nu varied along second n_grid

    e_plane = tf.expand_dims(tf.expand_dims(e,-1),-1)*tf.ones([a.shape[0],n_grid_c,n_grid_d])

    a_plane=tf.reshape(a_plane,[n_voxel*n_grid_c*n_grid_d])
    b_plane=tf.reshape(b_plane,[n_voxel*n_grid_c*n_grid_d])
    c_plane=tf.reshape(c_plane,[n_voxel*n_grid_c*n_grid_d])
    d_plane=tf.reshape(d_plane,[n_voxel*n_grid_c*n_grid_d])
    e_plane=tf.reshape(e_plane,[n_voxel*n_grid_c*n_grid_d])

    #qBOLD_calculated = tf.debugging.check_numerics(f_qBOLD_tensor([a_plane,b_plane,c_plane,d_plane,e_plane]), message='qBOLD calculated numerics problem')                        # shape (n_voxel,n_grid,n_grid,n_echoes)
    qBOLD_calculated = f_qBOLD_tensor([a_plane,b_plane,c_plane,d_plane,e_plane])                        # shape (n_voxel,n_grid,n_grid,n_echoes)
    qBOLD_residuals = qBOLD_calculated - tf.repeat(qBOLD,n_grid_c*n_grid_d,axis=0)                        # shape (n_voxel,n_grid,n_grid,n_echoes)
    qBOLD_residuals =tf.reshape(qBOLD_residuals,[n_voxel,n_grid_c,n_grid_d,16])

    #QSM_calculated =  tf.debugging.check_numerics(f_QSM_tensor([c_plane,d_plane,e_plane]) , message='QSM calculated numerics problem')                             # shape (n_voxel,n_grid,n_grid)
    QSM_calculated = f_QSM_tensor([c_plane,d_plane,e_plane])                              # shape (n_voxel,n_grid,n_grid)
    QSM_residuals = QSM_calculated - tf.repeat(QSM,n_grid_c*n_grid_d,axis=0)    # shape (n_voxel,n_grid,n_grid,1)
    QSM_residuals =tf.reshape(QSM_residuals,[n_voxel,n_grid_c,n_grid_d,1])

    a_plane=tf.reshape(a_plane,[n_voxel,n_grid_c,n_grid_d,1])
    b_plane=tf.reshape(b_plane,[n_voxel,n_grid_c,n_grid_d,1])
    c_plane=tf.reshape(c_plane,[n_voxel,n_grid_c,n_grid_d,1])
    d_plane=tf.reshape(d_plane,[n_voxel,n_grid_c,n_grid_d,1])
    e_plane=tf.reshape(e_plane,[n_voxel,n_grid_c,n_grid_d,1])


    output = tf.concat([a_plane,
                        b_plane,
                        c_plane,
                        d_plane,
                        e_plane,
                        qBOLD_residuals,
                        QSM_residuals],
                        axis=-1)                                                 # shape (n_batch,n_voxel,n_grid,n_grid,5+n_echoes+1)
    #output=tf.ensure_shape(output,[n_voxel,n_grid_c,n_grid_d,5+16+1])
    return output


def grid_search_nu_Y_tensor_no_wrapper(input_tensor):
    #n_batch=10
    n_voxel=input_tensor[0].shape[1]
    a     = tf.reshape(input_tensor[0],[-1])                                                    # shape (n_voxel)
    b     = tf.reshape(input_tensor[1],[-1])                                                    # shape (n_voxel)
    c     = tf.reshape(input_tensor[2],[-1])                                                  # shape (n_voxel)
    d     = tf.reshape(input_tensor[3],[-1])                                                   # shape (n_voxel)
    e     = tf.reshape(input_tensor[4],[-1])                                                  # shape (n_voxel)
    qBOLD = tf.reshape(input_tensor[5],[-1,16])
    QSM   = tf.reshape(input_tensor[6],[-1])

    n_grid_c = 20 #Y
    n_grid_d = 20 #nu
    #a                                                # shape n_voxel
    #tf.expand_dims(a,-1)                             #shape (n_voxel,1)
    #tf.repeat(tf.expand_dims(a,-1),n_grid,axis-1)    #shape (n_voxel,n_grid)
    #tf.expand_dims(tf.repeat(tf.expand_dims(a,-1),n_grid,axis-1),-1)    #shape (n_voxel,n_grid,1)
    a_plane = tf.expand_dims(tf.expand_dims(a,-1),-1)*tf.ones([1,n_grid_c,n_grid_d])  #shape (n_voxel,n_grid,n_grid)
    #a_plane = tf.ensure_shape(a_plane,[n_batch*n_voxel,n_grid_c,n_grid_d])
    b_plane = tf.expand_dims(tf.expand_dims(b,-1),-1)*tf.ones([1,n_grid_c,n_grid_d])

    c_start = tf.clip_by_value( tf.expand_dims(c - 0.3,-1)*tf.ones([1,n_grid_d]),0.01,1-0.3 )                      # shape (n_voxel,n_grid)
    c_stop  = tf.clip_by_value( tf.expand_dims(c + 0.3,-1)*tf.ones([1,n_grid_d]),0+0.3,1 )                       # shape (n_voxel,n_grid)
    c_plane = tf.linspace(c_start,c_stop,n_grid_c,axis=-2)                        # shape (n_voxel,n_grid,n_grid)          Y varied along first n_grid
    #c_plane =tf.ensure_shape(c_plane,[n_batch*n_voxel,n_grid_c,n_grid_d])

    d_calc = f_nu_tensor(c,e,QSM)
    d_start = tf.clip_by_value( tf.expand_dims(d_calc - 0.3,-1)*tf.ones([1,n_grid_d]),0.01,1-0.3 )                      # shape (n_voxel,n_grid)
    d_stop  = tf.clip_by_value( tf.expand_dims(d_calc + 0.3,-1)*tf.ones([1,n_grid_d]),0+0.3,1 )                       # shape (n_voxel,n_grid)
    d_plane = tf.linspace(d_start,d_stop,n_grid_d,axis=-1)                        # shape (n_voxel,n_grid,n_grid)          nu varied along second n_grid
    #d_plane =tf.ensure_shape(c_plane,[n_batch*n_voxel,n_grid_c,n_grid_d])

    e_plane = tf.expand_dims(tf.expand_dims(e,-1),-1)*tf.ones([1,n_grid_c,n_grid_d])

    a_plane=tf.reshape(a_plane,[-1])
    b_plane=tf.reshape(b_plane,[-1])
    c_plane=tf.reshape(c_plane,[-1])
    d_plane=tf.reshape(d_plane,[-1])
    e_plane=tf.reshape(e_plane,[-1])

    qBOLD_calculated = f_qBOLD_tensor([a_plane,b_plane,c_plane,d_plane,e_plane])                        # shape (n_voxel,n_grid,n_grid,n_echoes)
    #qBOLD_calculated = tf.ensure_shape(qBOLD_calculated,[n_batch*n_voxel*n_grid_c*n_grid_d,16])
    qBOLD_residuals = qBOLD_calculated - tf.repeat(qBOLD,n_grid_c*n_grid_d,axis=0)                        # shape (n_voxel,n_grid,n_grid,n_echoes)
    qBOLD_residuals =tf.reshape(qBOLD_residuals,[-1,n_voxel,n_grid_c,n_grid_d,16])

    QSM_calculated = f_QSM_tensor([c_plane,d_plane,e_plane])                            # shape (n_voxel,n_grid,n_grid)
    #QSM_calculated = tf.ensure_shape(QSM_calculated,[n_batch*n_voxel*n_grid_c*n_grid_d])
    QSM_residuals = QSM_calculated - tf.repeat(QSM,n_grid_c*n_grid_d,axis=0)    # shape (n_voxel,n_grid,n_grid,1)
    QSM_residuals =tf.reshape(QSM_residuals,[-1,n_voxel,n_grid_c,n_grid_d,1])

    a_plane=tf.reshape(a_plane,[-1,n_voxel,n_grid_c,n_grid_d,1])
    b_plane=tf.reshape(b_plane,[-1,n_voxel,n_grid_c,n_grid_d,1])
    c_plane=tf.reshape(c_plane,[-1,n_voxel,n_grid_c,n_grid_d,1])
    d_plane=tf.reshape(d_plane,[-1,n_voxel,n_grid_c,n_grid_d,1])
    e_plane=tf.reshape(e_plane,[-1,n_voxel,n_grid_c,n_grid_d,1])


    output = tf.concat([a_plane,
                        b_plane,
                        c_plane,
                        d_plane,
                        e_plane,
                        qBOLD_residuals,
                        QSM_residuals],
                        axis=-1)                                                 # shape (n_batch,n_voxel,n_grid,n_grid,5+n_echoes+1)
    #output=tf.ensure_shape(output,[n_batch,n_voxel,n_grid_c,n_grid_d,5+16+1])
    return output

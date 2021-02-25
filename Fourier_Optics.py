import time
t = time.time()
###########################################################################################################################################
###########################################################################################################################################
import numpy as np

if __name__ == '__main__':
    print("")
else:
    print("Loading Constants")

taskName = "ACSinObject"

numberOfThreads = -1  # number of threads used in the simulation with -1 means using all threads, -2 means using all threads minus 1.

simulatingSpaceSize = 400 # total simulating space in nm
simulatingSpaceTotalStep = 1 + 2**12 # total simulating steps

constant_type = "IBM AC"
object_type = "Step phase contrast"

if constant_type == 'basic':
##################### Basic setting ####################
    U_a = 18e3      # eV    Accelerating Voltage
    U_o = 10        # ev    Electron Voltage

    C_c = 59.8e-3   # m     1st order, 1st degree Chromatic Aberration Coefficient   

    C_3c = 0        # m     3rd order, 1st degree Chromatic aberration coefficient
    C_cc = 0        # m     1st order, 2nd degree Chromatic aberration coefficient

    C_3 = 55.8e-3   # m     3rd order Spherical Aberration Coefficient
    delta_E = 0.5

    M_L = 0.653         #       Lateral Magnification

##################### Basic setting ####################
if constant_type == "IBM AC":
##################### AC Constants##
    
    U_a = 15.01e3  # eV  Accelerating Voltage
    U_o = 10  # eV  Electron Voltage
    C_c = 0  # m   Chromatic Aberration Coefficient
    C_3c = -67.4
    C_cc = 27.9
    C_3 = 0  # m   Spherical Aberration Coefficient
    C_5 = 92.8

    alpha_ap = 7.37E-3  # rad Acceptance Angle of the Contrast Aperture
    alpha_ill = 0.1E-3  # rad Illumination Divergence Angle

    delta_E = 0.25  # eV  Energy Spread
    M_L = 0.653  # Lateral Magnification
    
##################### AC Constants######################


# ##################### NAC Constants######################
if constant_type =='IBM NAC':
    U_a = 15.01e3  # eV  Accelerating Voltage
    U_o = 10  # eV  Electron Voltage
    C_c = -0.075  # m   Chromatic Aberration Coefficient
    C_3c = -59.37
    C_cc = 23.09
    C_3 = 0.345  # m   Spherical Aberration Coefficient
    C_5 = 39.4

    alpha_ap = 2.34E-3  # rad Acceptance Aangle of the Contrast Aperture
    alpha_ill = 0.1E-3  # rad Illumination Divergence Angle
    
    delta_E = 0.25  # eV  Energy Spread
    M_L = 0.653  # Lateral Magnification
# ##################### NAC Constants######################

if constant_type == "2017":
    U_a = 18e3
    U_o = 10.75
    C_c = -0.0987
    C_3c = 0
    C_cc = 0
    C_3 = 0.211
    C_5 = 0
    
    alpha_ap = 6e-3 #original 6e-3
    alpha_ill = 0.11e-3 #p.3
    
    delta_E = 0.75      #p.3
    M_L = 1

if constant_type == 'IBM AC (05-01-2021)':
    U_a = 15e3
    U_o = 10
    
    C_3 = 0
    C_5 = 77.2
    C_c = 0
    C_3c = -41
    C_cc = 16
    
    alpha_ap = 7.1e-3
    alpha_ill = 0.06e-3
    
    delta_E = 0.25
    M_L = 1

# print(delta_z_series)

if constant_type == 'Trial':
    U_a = 15e3
    U_o = 10
    
    C_3 = 0
    C_5 = 77.2
    C_c = 0
    C_3c = -41
    C_cc = 16
    
    alpha_ap = 6.4e-3
    alpha_ill = 0.06e-3
    
    delta_E = 0.25
    M_L = 1
    
###################################################################
delta_fc = C_c * (delta_E / U_a)
delta_f3c = C_3c * (delta_E / U_a)
delta_fcc = C_cc * (delta_E / U_a) ** 2

lamda = 6.6262e-34 / np.sqrt(2 * 1.6022e-19 * 9.1095e-31 * U_a)
lamda_o = 6.6262e-34 / np.sqrt(2 * 1.6022e-19 * 9.1095e-31 * U_o)

q_max = alpha_ap / lamda
q_ill = alpha_ill / lamda

#############convert defocus_current into delta_z#############
# defocus_current_series = np.linspace(-7, 7, 31)  # mA
# delta_zo_series = defocus_current_series * 5.23 * 10 ** -6
# delta_z_series = delta_zo_series * 3.2
#############convert defocus_current into delta_z#############

##################directly use delta_z#############
# delta_z_series = np.array([0])
delta_z_series = np.array([100e-6])
##################directly use delta_z#############

##################delta_z_star#################
# delta_z_star = [0]
# delta_z_series = np.multiply(delta_z_star,(C_3*lamda)**(1/2))

##################delta_z_star#################
print("Constants Loaded")
###########################################################################################################################################
###########################################################################################################################################

from datetime import datetime
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
startTimeStamp = datetime.now().strftime('%Y%m%d_%H%M%S')

object_wavelength = 900e-9

print('Creating Sample')

def create_simulated_space():
    simulated_space = np.linspace(-simulatingSpaceSize/2, simulatingSpaceSize/2, simulatingSpaceTotalStep)
    return simulated_space

simulated_space = create_simulated_space()

if object_type == 'Step phase contrast':
    K = 1
    h = np.zeros_like(simulated_space)
    for counter, element in enumerate(simulated_space):
        if element > 0:
            h[counter] = 1
    phase_shift = K * h * np.pi
    amp = np.ones_like(simulated_space)
    
if object_type == 'Sin phase':
    K = 90
    h = simulated_space
    phase_shift = K*np.pi*np.sin(h/(simulatingSpaceSize/2)*2*np.pi)
    amp = np.ones_like(h)
    
        
if object_type == 'Sin amplitude':
    h = simulated_space
    amp = np.sin(h/(simulatingSpaceSize/2)*2*np.pi)
    phase_shift = np.zeros_like(h)    
        
if object_type == 'Step amplitude contrast':
    K = 1
    h = np.ones_like(simulated_space)
    for counter, element in enumerate(simulated_space):
        if element > 0:
            h[counter] = 1/np.sqrt(2)
            
    phase_shift = np.zeros_like(simulated_space)
    amp = K*h

if object_type == 'Slope':
    K = 1
    h = np.zeros_like(simulated_space)
    pos = 200
    for counter, element in enumerate(simulated_space):
        if element >= pos:
            h[counter] = 1
        if element >= -pos and element < pos:
            h[counter] = (element+pos)/(2*pos)
    phase_shift = K * h * np.pi
    amp = np.ones_like(simulated_space)
    
if object_type == 'Custom':
    K = 1
    h = np.zeros_like(simulated_space)
    pos = 50
    for counter, element in enumerate(simulated_space):
        if element >= pos:
            h[counter] = 0.5
        if element >= -pos and element < pos:
            h[counter] = 0
        if element < -pos:
            h[counter] = 0.5
    phase_shift = K * h * np.pi
    amp = np.ones_like(simulated_space)

obj = np.multiply(amp, np.exp(1j * phase_shift))
reversed_obj = obj[::-1] # the object is reversed in the laens, this line is to reversed the object for simplier calculation.

print('Sample created.')
print('Simulating')

object_space = (create_simulated_space()+simulatingSpaceSize/simulatingSpaceTotalStep)
image_space = object_space#/M_L
    
def FO1D(z, zCounter):
    R_o = np.exp(1j * 2 * np.pi * (
            C_3 * lamda ** 3 * (Q ** 4 - QQ ** 4) / 4 + C_5 * lamda ** 5 * (Q ** 6 - QQ ** 6) / 6 - z * lamda * (
            Q ** 2 - QQ ** 2) / 2))
    E_s = np.exp(-np.pi ** 2 * q_ill ** 2 * (
            C_3 * lamda ** 3 * (Q ** 3 - QQ ** 3) + C_5 * lamda ** 5 * (Q ** 5 - QQ ** 5) - z * lamda * (
            Q - QQ)) ** 2 / (4 * np.log(2)))

    AR = np.multiply(np.multiply(np.multiply(A, R_o), E_s), E_ct)

    for i in range(len(q)):
        for j in range(i + 1, len(q)):
            matrixI[:, zCounter] = matrixI[:, zCounter] + 2 * (
                    AR[j][i] * np.exp(1j * 2 * np.pi * (Q[j][i] - QQ[j][i]) * simulated_space)).real
        

    matrixI[:, zCounter] = matrixI[:, zCounter] + np.trace(AR) * np.ones_like(simulated_space)

    return matrixI

simulated_space = create_simulated_space()
simulated_space *= 1e-9
wave_obj = reversed_obj

objectFileName = "FO1DObjectWave_" + taskName + "_" + startTimeStamp + ".npy"

F_wave_obj = np.fft.fftshift(np.fft.fft(wave_obj, simulatingSpaceTotalStep) * (1 / simulatingSpaceTotalStep))

# n_max = np.floor(q_max / (1 / object_wavelength))
q = 1 / (simulated_space[1] - simulated_space[0]) * np.arange(0, simulatingSpaceTotalStep, 1) / (simulatingSpaceTotalStep)
q = q - (np.max(q) - np.min(q)) / 2

a = np.sum(np.abs(q) <= q_max)

if len(q) > a:
    q = q[int(np.ceil(simulatingSpaceTotalStep / 2 + 1 - (a - 1) / 2)):int(np.floor(simulatingSpaceTotalStep / 2 + 1 + (a + 1) / 2))]
    F_wave_obj = F_wave_obj[
                 int(np.ceil(simulatingSpaceTotalStep / 2 + 1 - (a - 1) / 2)):int(np.floor(simulatingSpaceTotalStep / 2 + 1 + (a + 1) / 2))]

Q, QQ = np.meshgrid(q, q)
F_wave_obj_q, F_wave_obj_qq = np.meshgrid(F_wave_obj, np.conj(F_wave_obj))

A = np.multiply(F_wave_obj_q, F_wave_obj_qq)
E_cc = (1 - 1j * np.pi * delta_fcc * lamda * (Q ** 2 - QQ ** 2) / (4 * np.log(2))) ** (-0.5)
E_ct = E_cc * np.exp(-np.pi ** 2 * (delta_fc * lamda * (Q ** 2 - QQ ** 2) + 1 / 2 * delta_f3c * lamda ** 3 * (
        Q ** 4 - QQ ** 4)) ** 2 * E_cc ** 2 / (16 * np.log(2)))

matrixI = np.zeros((len(simulated_space), len(delta_z_series)), dtype=complex)

# print("Task:", taskName)
# print("Total Task:", len(delta_z_series))
# print("Total Parallel Steps:", np.ceil(len(delta_z_series) / (multiprocessing.cpu_count() + numberOfThreads + 1)))

with Parallel(n_jobs=numberOfThreads, verbose=50, max_nbytes="50M") as parallel:
    parallelResult = parallel(delayed(FO1D)(z, zCounter) for zCounter, z in enumerate(delta_z_series))

for mat in parallelResult:
    matrixI = matrixI + mat

matrixI = np.abs(matrixI)

print('Simuation finished.')

def plotting():
    print('Plotting result.')

    object_space = (create_simulated_space()+simulatingSpaceSize/simulatingSpaceTotalStep)
    image_space = object_space/M_L



    def object_plot():
        plt.figure(figsize=(5,8))
        plt.subplot(211)

        plt.plot(object_space, amp, label = 'Amplitude', linestyle=(8,(5,12)))
        plt.plot(object_space, phase_shift, label = 'Phase', linestyle=(0,(5,12)))
    
        
        plt.title(object_type)
        
        # plt.xlim(-30,30)
        plt.xlim(-100,100)
        
        plt.xlabel('Position (nm)')
        plt.ylabel('Amplitude ' + r'$(\frac{W}{m^2})$' + '\n Phase ' + '(rad)')
        
        # plt.yticks([0, 1, np.pi/2, 2, np.pi])
        
        plt.legend(loc=0)
        plt.subplot(212)
        
        plt.tight_layout()
    
    object_plot()
    
    linestyle = ['-',(0,(5,12)),(8,(5,12)),(0,(5,12)),(8,(5,12))]
    # linestyle = ['-','-','-']
    
    color_list = ['black','red','blue','green','purple']
    for i in range(0,len(delta_z_series)):
        # plt.plot(image_space,matrixI[:,i],color=color_list[i],linestyle=linestyle[i]) #parallel program for defocus
        plt.plot(image_space,matrixI,color=color_list[i],linestyle=linestyle[i])

    
    plt.title(object_type + ' with \n' + constant_type + ' constant')
    

    plt.xlabel('Position (nm)')
    plt.ylabel('Intensity ' + r'$(\frac{W}{m^2})$')

    # plt.text(0,-0.5,txt,ha='center')

    #dz_star_list = []
    #for z_star in delta_z_series:    
    #    dz_star_list.append(r'$\Delta z^*$ = ' + str(z_star))    
    #plt.legend(dz_star_list,loc='upper right')

    # plt.xlim(-30,30)
    plt.xlim(-100,100)
    plt.ylim(0,2)
    
    # plt.yticks([0,0.5,1,1.5])

    # plt.xlim(-1,1)
    # plt.ylim(1.325,1.35)    
    
    txt = 'Resolution of q = ' + str((q[1]-q[0])*1e-9) + ' /nm'
    plt.text(0, -0.4, txt, ha='center')
    
    plt.tight_layout()    

####################plot the result###############

# print(matrixI[0][1])

####################save the plot###############

    resultPlotName = constant_type + object_type + startTimeStamp +'.png'
# print("Saving result to:",resultPlotName)
    plt.savefig(resultPlotName)
    

####################save the plot###############

    plt.show()
    print('Result plotted.')
    print("Plot saved.")
    
plotting()
print('Programme ended.')

print('Program started, loading libraries.')
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import pytz
from joblib import Parallel, delayed
import multiprocessing
from numba import jit
print('Libraries imported.')
########################################################################################################################################

main = True #True #False
isDirect_Constant = True
Constant_type = 'IBM NAC'


########################################################################################################################################
print('Time setting.')
fmt = '%H:%M:%S'
timezone = 'Asia/Hong_Kong'
start_time = time.time()
timeZonePytz = pytz.timezone(timezone)

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def printStatus(counter, done=False):
        if counter != 0:
            elapsedTime = ((time.time() - start_time) / 60)
            progress = (counter / len(loopList)) * 100
            totalTime = elapsedTime / (progress / 100)
            timeLeft = totalTime - elapsedTime
            nowDT = datetime.now(timeZonePytz)
            currentHKTime = nowDT.strftime(fmt)
            if done:
                print("-Total Time: %.2f Minutes -" % elapsedTime)
            else:
                print("-ID:" + str(counter) + "---Elapsed Time: %.2f / %.2f min---" % (elapsedTime, totalTime)
                          + "Time Left: %.2f  min---" % timeLeft + "%.2f" % progress + "%-- Time: " + currentHKTime)
########################################################################################################################################
                




########################################################################################################################################
def create2DSimulatedSpace():   #   in m
    space = [[[0,0] for i in range(simulatingSpaceTotalStep)] for j in range(simulatingSpaceTotalStep)]
    x = np.linspace(-simulatingSpaceSize,simulatingSpaceSize,simulatingSpaceTotalStep)
    y = x
    for y_count in range(simulatingSpaceTotalStep):
        for x_count in range(simulatingSpaceTotalStep):
            space[y_count][x_count][0],space[y_count][x_count][1] = x[x_count],y[y_count]
    return space

def create2DSimulatedObject():
    space = create2DSimulatedSpace()
    
    def Phase_0pi0():
        amp = np.ones((simulatingSpaceTotalStep,simulatingSpaceTotalStep))
        phase_shift = np.zeros_like(amp)
        half_width = 3e-9
        for y_count in range(simulatingSpaceTotalStep):
            for x_count in range(simulatingSpaceTotalStep):
                if space[y_count][x_count][0]>-half_width and space[y_count][x_count][0]<half_width:
                    if space[y_count][x_count][1]>-half_width and space[y_count][x_count][1]<half_width:
                        phase_shift[y_count][x_count] = np.pi
        return amp,phase_shift
    
    def Amp_sq05_1_sq05():
        amp = 1/np.sqrt(2)*np.ones((simulatingSpaceTotalStep,simulatingSpaceTotalStep))
        phase_shift = np.zeros_like(amp)
        half_width = 2.5e-9
        for y_count in range(simulatingSpaceTotalStep):
            for x_count in range(simulatingSpaceTotalStep):
                if space[y_count][x_count][0]>-half_width and space[y_count][x_count][0]<half_width:
                    if space[y_count][x_count][1]>-half_width and space[y_count][x_count][1]<half_width:
                        amp[y_count][x_count] = 1
        return amp,phase_shift
    
    def Amp_101():
        amp = np.ones((simulatingSpaceTotalStep,simulatingSpaceTotalStep))
        phase_shift = np.zeros_like(amp)
        
        for y_count in range(simulatingSpaceTotalStep):
            for x_count in range(simulatingSpaceTotalStep):
                if space[y_count][x_count][0]>-5e-9 and space[y_count][x_count][0]<5e-9:
                    if space[y_count][x_count][1]>-5e-9 and space[y_count][x_count][1]<5e-9:
                        amp[y_count][x_count] = 0
        return amp,phase_shift
   
    def Amp_circle():
        amp = 1/np.sqrt(2)*np.ones((simulatingSpaceTotalStep,simulatingSpaceTotalStep))
        phase_shift = np.zeros_like(amp)
        radius = 2.5e-9
        for y_count in range(simulatingSpaceTotalStep):
            for x_count in range(simulatingSpaceTotalStep):
                x = space[y_count][x_count][0]
                y = space[y_count][x_count][1]
                r2 = x**2+y**2
                if r2 < radius**2:
                    amp[y_count][x_count] = 1
        return amp,phase_shift
    
    def Phase_circle():
        amp = np.ones((simulatingSpaceTotalStep,simulatingSpaceTotalStep))
        phase_shift = np.zeros_like(amp)
        radius = 3e-9
        for y_count in range(simulatingSpaceTotalStep):
            for x_count in range(simulatingSpaceTotalStep):
                x = space[y_count][x_count][0]
                y = space[y_count][x_count][1]
                r2 = x**2+y**2
                if r2 < radius**2:
                    phase_shift[y_count][x_count] = np.pi
        return amp,phase_shift
    
    
    amp, phase_shift = Amp_sq05_1_sq05()
    obj = np.multiply(amp, np.exp(1j * phase_shift))
    
    obj_dict =  dict(amp=amp,obj=obj,phase_shift=phase_shift,space=space)
    return obj_dict
########################################################################################################################################




########################################################################################################################################

print('Loading Constant')
numberOfThreads = -1 # number of threads used in the simulation with -1 means using all threads, -2 means using all threads minus 1.

simulatingSpaceSize = 25 * 1e-9  # total simulating space in m
simulatingSpaceTotalStep = 501  # total simulating steps

U_a = 15.01e3  # eV  Accelerating Voltage
U_o = 10  # eV  Electron Voltage
L = 2.24e-3

delta_E = 0.25  # eV  Energy Spread
C_3m = 0.119    # experiment value of Elmitec instruments in Tony's thesis
C_cm = -0.707   # experiment value of Elmitec instruments in Tony's thesis
C_5m = 0        # The value is not mention in Tony's thesis, it is the magnetic part.
C_3cm = 0       # The value is not mention in Tony's thesis, it is the magnetic part.
C_ccm = 0       # The value is not mention in Tony's thesis, it is the magnetic part.



if isDirect_Constant:
    if Constant_type == 'IBM AC':
        C_3 = 0 #   NAC 0.345   AC 0
        C_c = 0 #   NAC -0.075  AC 0
        C_5 = 92.8  #   NAC 39.4    AC 92.8
        C_3c = -67.4    #   NAC -59.37  AC -67.4
        C_cc = 27.9     #   NAC 23.09   AC 27.9
        alpha_ap = 7.37e-3
        
    if Constant_type == 'IBM NAC':
        C_3 = 0.345
        C_c = -0.075
        C_5 = 39.4
        C_3c = -59.37
        C_cc = 23.09
        alpha_ap = 2.34e-3
else:
    C_3 = L * ( U_a / U_o ) ** 0.5 + C_3m  # m   Spherical Aberration Coefficient
    C_c = -L * ( U_a / U_o ) ** 0.5 + C_cm
    C_5 = 0.25 * L * ( U_a / U_o ) ** 1.5 + C_5m
    C_3c = -0.5 * L * ( U_a / U_o )**1.5 +C_3cm             # 0
    C_cc = -0.25 * L * ( U_a / U_o ) ** 1.5 + C_ccm


alpha_ill = 0.1E-3  # rad Illumination Divergence Angle



M_L = 0.653  # Lateral Magnification

###################################################################

lamda = 6.6262e-34 / np.sqrt(2 * 1.6022e-19 * 9.1095e-31 * U_a)
lamda_o = 6.6262e-34 / np.sqrt(2 * 1.6022e-19 * 9.1095e-31 * U_o)

Sch_z = (0.5*C_3*lamda)**0.5
#defocus = 0  # mA
#delta_zo = -1 * defocus * 5.23 * 10 ** -6  # m
delta_z = 2e-6  # delta_zo * 3.2
# delta_z = z  
q_max = alpha_ap / lamda
q_ill = alpha_ill / lamda

delta_fc = C_c * (delta_E / U_a)
delta_f3c = C_3c * (delta_E / U_a)
delta_fcc = C_cc * (delta_E / U_a) ** 2

RoConstant0 = 1j * 2 * np.pi
RoConstant1 = 1 / 4 * C_3 * lamda ** 3
RoConstant2 = 1 / 6 * C_5 * lamda ** 5
RoConstant3 = -1 / 2 * delta_z * lamda

EsConstant0 = -np.pi ** 2 / 4 / np.log(2) * q_ill ** 2
EsConstant1 = C_3 * lamda ** 3
EsConstant2 = C_5 * lamda ** 5
EsConstant3 = - delta_z * lamda

EccConstant0 = 1j * np.pi / 4 / np.log(2) * delta_fcc * lamda

EctConstant0 = -np.pi ** 2 / 16 / np.log(2)
EctConstant1 = delta_fc * lamda
EctConstant2 = 1 / 2 * delta_f3c * lamda ** 3



print('Constant Loaded.')
########################################################################################################################################





########################################################################################################################################
if main:
    print("Generating Object.")
    Objdict = create2DSimulatedObject()
    Object = Objdict['obj']
    
    sampleStepSize = Objdict['space'][1][0][1]-Objdict['space'][0][0][1]
    print("Object Generated.")
    ########################################################################################################################################
    
    
    
    
    
    ########################################################################################################################################
    
    print("Start Calculation...")
    ObjectFT = np.fft.fftshift(np.fft.fft2(Object) / simulatingSpaceTotalStep ** 2)
    
    qSpaceCoor = 1 / sampleStepSize / (simulatingSpaceTotalStep - 1) * np.arange(simulatingSpaceTotalStep)
    qSpaceCoor = qSpaceCoor - (np.amax(qSpaceCoor) - np.amin(qSpaceCoor)) / 2  # adjust qSpaceCoor center
    
    qSpaceXX, qSpaceYY = np.meshgrid(qSpaceCoor, qSpaceCoor)
    
    # setup aperture function
    apertureMask = qSpaceXX ** 2 + qSpaceYY ** 2 <= q_max ** 2
    aperture = np.zeros_like(qSpaceYY)
    aperture[apertureMask] = 1
    
    # apply aperture function
    maskedWaveObjectFT = ObjectFT[aperture == 1]
    
    maskedQSpaceXX = qSpaceXX[aperture == 1]
    maskedQSpaceYY = qSpaceYY[aperture == 1]
    
    sampleCoorRealSpaceXX, sampleCoorRealSpaceYY = np.mgrid[-simulatingSpaceSize:simulatingSpaceSize:simulatingSpaceTotalStep * 1j,
                                                       -simulatingSpaceSize:simulatingSpaceSize:simulatingSpaceTotalStep * 1j]
    
    @jit(nopython=True, cache=True)
    def outerForLoop(counter_i):
        returnMatrix = np.zeros_like(sampleCoorRealSpaceXX, dtype=np.complex128)
    
        qq_i = maskedQSpaceXX[counter_i] + 1j * maskedQSpaceYY[counter_i]
        abs_qq_i = np.absolute(qq_i)
    
        abs_qq_i_2 = abs_qq_i ** 2
        abs_qq_i_4 = abs_qq_i_2 ** 2
        abs_qq_i_6 = abs_qq_i_2 ** 3
    
        for counter_j in range(len(maskedQSpaceYY)):
    
            if counter_i >= counter_j:
    
                qq_j = maskedQSpaceXX[counter_j] + 1j * maskedQSpaceYY[counter_j]
                abs_qq_j = np.absolute(qq_j)
    
                abs_qq_j_2 = abs_qq_j ** 2
                abs_qq_j_4 = abs_qq_j_2 ** 2
                abs_qq_j_6 = abs_qq_j_2 ** 3
                R_o = np.exp(RoConstant0 *
                             (RoConstant1 * (abs_qq_i_4 - abs_qq_j_4)
                              + RoConstant2 * (abs_qq_i_6 - abs_qq_j_6)
                              + RoConstant3 * (abs_qq_i_2 - abs_qq_j_2))
                             )
    
                E_s = np.exp(EsConstant0 *
                             np.abs(EsConstant1 * (qq_i * abs_qq_i_2 - qq_j * abs_qq_j_2)
                                    + EsConstant2 * (qq_i * abs_qq_i_4 - qq_j * abs_qq_j_4)
                                    + EsConstant3 * (qq_i - qq_j)) ** 2
                             )
    
                E_cc = np.sqrt(1 - EccConstant0 * (abs_qq_i_2 - abs_qq_j_2))
    
                E_ct_exponent = EctConstant0 * (EctConstant1 * (abs_qq_i_2 - abs_qq_j_2)
                                                + EctConstant2 * (abs_qq_i_4 - abs_qq_j_4)) ** 2
    
                E_ct = E_cc * np.exp(E_ct_exponent * E_cc ** 2)
    
                EXP_exponent = 2j * np.pi * (
                        (qq_i - qq_j).real * sampleCoorRealSpaceXX + (qq_i - qq_j).imag * sampleCoorRealSpaceYY)
    
                EXP = np.exp(EXP_exponent)
                returnMatrix = returnMatrix + R_o * E_s * E_ct * maskedWaveObjectFT[counter_i] * np.conj(
                    maskedWaveObjectFT[counter_j]) * EXP
                if counter_i > counter_j:
    
                    R_o_sym = 1 / R_o
                    E_s_sym = E_s
                    E_cc_sym = np.sqrt(1 - EccConstant0 * (abs_qq_j_2 - abs_qq_i_2))
                    E_ct_sym = E_cc_sym * np.exp(E_ct_exponent * E_cc_sym ** 2)
    
                    EXP_sym = EXP.real - 1j * EXP.imag
    
    
                    returnMatrix = returnMatrix + R_o_sym * E_s_sym * E_ct_sym * maskedWaveObjectFT[
                        counter_j] * np.conj(
                        maskedWaveObjectFT[counter_i]) * EXP_sym
            else:
                break
    
        return returnMatrix
    
    def ijSymmetry(counter_i):
    
        if counter_i == int(totalOuterLoopCall / 2):
            returnMatrix = outerForLoop(counter_i)
        else:
            returnMatrix1 = outerForLoop(counter_i)
            returnMatrix2 = outerForLoop(totalOuterLoopCall - counter_i - 1)
            returnMatrix = returnMatrix1+returnMatrix2
    
        return returnMatrix
    
    num_cores = multiprocessing.cpu_count()
    totalOuterLoopCall = len(maskedQSpaceXX)
    loopList = list(range(totalOuterLoopCall))[:int(totalOuterLoopCall / 2) + 1]
    
    breakProcess = list(chunks(loopList, num_cores * 2))
    numberOfChunk = int(len(breakProcess))
    print("Total Process: ", len(loopList))
    
    print("Number of Thread: " + str(num_cores))
    print("Number of Chunk: " + str(numberOfChunk))
    
    print("Start multiprocessing")
    
    processTemp = np.zeros_like(sampleCoorRealSpaceXX)
    
    with Parallel(n_jobs=numberOfThreads, verbose=50) as parallel:
        for process in breakProcess:
    
            multicoreResults = parallel(delayed(ijSymmetry)(counter_i) for counter_i in process)
            tempArray = np.array(multicoreResults)
            tempArray = np.sum(tempArray, axis=0)
            processTemp = processTemp + tempArray
            printStatus(process[-1])
    
    print("End multiprocessing")
    matrixI = np.fft.fftshift(processTemp)
    matrixI = np.absolute(matrixI)
    print("start saving matrix")
    
    matrixI = matrixI.T
    
    
    print("End Calculation")
    printStatus(100, done=True)
    print("----------------------------------------------------")
########################################################################################################################################





########################################################################################################################################
def plot2DArray(plot_data, vmin = None, vmax=None):
    plt.imshow(plot_data.T, 
               extent=[-simulatingSpaceSize*1e9,simulatingSpaceSize*1e9,-simulatingSpaceSize*1e9,simulatingSpaceSize*1e9],
               interpolation='nearest',cmap='jet',  #np.flipud(plot_data), cmap='jet'
               origin='lower',vmin=vmin, vmax=vmax) #vmin=0, vmax=2,
     
    if title == 'Phase shift':
        plt.clim(0,2*np.pi)
    else:
        plt.clim(0,1.5)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Position x (nm)")
    plt.ylabel("Position y (nm)")
    
if main:
    print("Plotting")
    
    plt.figure(figsize=(16,4))
    
    plt.subplot(131)
    title='Intensity'
    plot2DArray(matrixI)
    
    plt.subplot(132)
    title='Amplitude'
    plot2DArray(Objdict['amp'])
    
    plt.subplot(133)
    title='Phase shift'
    plot2DArray(Objdict['phase_shift'])
    
    plt.tight_layout()
    
    def savefig():
        print('Saving figure...')
        plt.savefig(str(time.time()) +'.png')
        print("Figure saved.")
    savefig()
    
    plt.show()



    np.save('Intensity',matrixI)
    np.save('amp',Objdict['amp'])
    np.save('phase_shift',Objdict['phase_shift'])
print("Program End.")

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
import matplotlib.dates as mdates
import datetime as dt

from dataManagement import data, dateForPlot, pandas
from constants import constants

# from lmfit import Parameters, minimize

'''
Abbiamo implementato il modello SEIRD, una variazione del classico modello SIR.
dS/dt = - (beta/N) * S *I
dE/dT =  (beta/N) * S *I - alpha * E
dI/dt = alpha * E - 1/T * I 
dR/dT =  (1 - f)/T * I
dD/dt = f/T * I 

#TODO: $\alpha$, cioè il tasso con il quale un individuo infetto lascia lo stato di incubazione, può essere assunto costante

beta = infection rate
alpha = incubation leave rate
T = average infectious period
gamma = 1/T
epsilon = fraction of all removed individuals who die
'''

"Popolazione della Lombardia"
N = constants.N

"""
In questa fase decidiamo come procedere con il modello: 
    - totalDays: sono i giorni totali di cui possediamo i dati osservati
A questo punto si suddividono i giorni totali in due parti per dividere le stime:
    - daysFirstIteration sono i giorni che predisponiamo per la prima fase del processo di ottimizzazione in cui facciamo una prima stima dei parametri
    - daysIteration sono i giorni che decidiamo di prendere in considerazione per il processo di ottimizzazione discretizzato
    """

totalDays = constants.totalDays
daysFirstIteration = constants.daysFirstIteration
daysIteration = constants.daysIteration

"""
   Definisco il deltaT iniziale

"""
initDeltaT = constants.initDeltaT
#TODO: cosa è numberOfIteration?
numberOfIteration = 0


"""
Inizializzo i vettori in cui andrò a memorizzare tutti i parametri nei vari intervalli di tempo deltaT
"""


betaEstimated = []
alphaEstimated = []
epsilonEstimated = []
gammaEstimated = []
roEstimated = []

betaNewEstimated = []

firstDayIteration = []
lastDayIteration = []


# def betaFunction(t, ro, iteration):
#     #t->array dei giorni su cui si prendono i dati (e.g. [init,fin,step])
#     #ro->
#     boundarybeta = 0
#     if ((t >= (lastDayIteration[iteration]-1))):
#         boundarybeta= 1

#     tk = firstDayIteration[iteration]
#     """
#     Scegliere se usare beta razionale od esponenziale
#        Se si cambia, bisogna cambiare anche i vincoli di RO
#     """
#     #razionale
#     result = betaEstimated[iteration] * (1 - ro * (t - tk) / t)
#     #esponenziale
#     #result = betaEstimated[iteration] * np.e**(-ro * (t - tk))

#     if(boundarybeta):
#         if (len(betaEstimated)-1==iteration):
#             betaEstimated.append(result)
#         else:
#             betaEstimated[iteration+1] = result
#     else:
#         betaNewEstimated.append(result)
#     return result


# definisco il modello SEIRD
def odeModel(z,t,susc, prob_symp, tau, cont_mat, delta_E, gamma):
    #INPUT-OUTPUT
    #z -> Initial data
    #t -> Timespan
    #susc -> susceptibility
    #prob_symp -> probability of developing symptoms
    #tau -> decreased infectivity for asymptomatics
    #cont_mat -> contact matrix
    #delta_E -> exit rate from latency period
    #gamma ->removal rate
    
    # S, E, I, A, R = z
    S = z[0:6]
    E = z[6:12]
    I = z[12:18]
    A = z[18:24]
    R = z[24:30]
    #TODO good syntax for susc, tau and prob_symp scalars
    dSdt = -susc * (cont_mat @ (I + tau * A))
    dEdt = susc * (cont_mat @ (I + tau * A)) - delta_E * E
    dIdt = prob_symp * delta_E * E - gamma * I
    dAdt = (1 - prob_symp) * delta_E * E - gamma * A
    dRdt = gamma * I + gamma * A
    result = np.array([dSdt, dEdt, dIdt, dAdt, dRdt])

    #Result is reshaped so that it's a vector
    return result.reshape(z.shape[0],)

# def fodeModel(z, t, ro, alpha, gamma, epsilon, iteration):
#     S, E, I, R, D = z

#     dSdt = -betaFunction(t, ro, iteration) * S * I / N
#     dEdt = betaFunction(t, ro, iteration) * S * I / N - alpha * E
#     dIdt = alpha * E - gamma * I
#     dRdt = gamma * I * (1 - epsilon)
#     dDdt = gamma * I * epsilon

#     return [dSdt, dEdt, dIdt, dRdt, dDdt]


# deifnisco il solver di EQ differeziali: utilizzo la funzione odeint che si basa su algotitmo LSODA (a passo variabile)
def odeSolver(t, initial_conditions, susc, prob_symp, tau, cont_mat, delta_E, gamma):
    initE, initI, initA, initR = initial_conditions
    initS = N - initE - initI - initA -  initR
    # susc = params['susc']
    # prob_symp = params['prob_symp']
    # tau = params['tau']
    # cont_mat = params['cont_mat']
    # delta_E = params['delta_E']
    # gamma = params['gamma']
    y0 = np.array([initS, initE, initI, initA, initR])
    dimx = y0.shape[0]
    dimy = y0.shape[1]
    y0 = y0.reshape(dimx*dimy,)
    
    res = odeint(odeModel, y0, t, args=(susc, prob_symp, tau, cont_mat, delta_E, gamma))
    return res


# def fodeSolver(t, initial_conditions, params, iteration):
#     initE, initI, initR, initD = initial_conditions
#     initS = N - initE - initI - initR - initD
#     ro = params['ro']
#     alpha = params['alpha']
#     gamma = params['gamma']
#     epsilon = params['epsilon']
#     res = odeint(fodeModel, [initS, initE, initI, initR, initD], t, args=(ro, alpha, gamma, epsilon, iteration))
#     return res


# # Definisco la funzione "error" deve essere minimizzata. Questa funzione contiene la differenza tra il valore calcolato dal modello ed i dati effettivi
# def error(params, initial_conditions, tspan, data, timek, timek_1):
#     sol = odeSolver(tspan, initial_conditions, params)
#     return (sol[:, 2:5] - data[timek:timek_1]).ravel()


# def errorRO(params, initial_conditions, tspan, data, iteration):
#     sol = fodeSolver(tspan, initial_conditions, params, iteration)
#     return (sol[:, 2:5] - data[firstDayIteration[iteration]:lastDayIteration[iteration]]).ravel()



# def deltaTCalibration(infected, iteration):
#     firstDay = firstDayIteration[iteration]
#     lastDay = lastDayIteration[iteration]

#     totalInfectedObserved = sum(data[firstDay:lastDay+1,0])
#     totalInfectedModel = sum(infected[firstDay:lastDay+1])

#     intervalTime = lastDay - firstDay

#     rateInfectedModel = totalInfectedModel/intervalTime
#     rateInfectedObserved = totalInfectedObserved/intervalTime

#     if (abs(rateInfectedModel - rateInfectedObserved) >= constants.ERROR_RANGE_MIN_INTERVAL):
#         deltaT = constants.MIN_INTERVAL

#     elif(abs(rateInfectedModel - rateInfectedObserved) <= constants.ERROR_RANGE_MAX_INTERVAL):
#         deltaT = constants.MAX_INTERVAL

#     else:
#         deltaT = constants.MEDIUM_INTERVAL


#     firstDayIteration.append(lastDay)
#     lastDayIteration.append(lastDay + deltaT)



# def prevision(daysPrevision):
#     tspan = np.arange(totalDays, totalDays + daysPrevision, 1)
#     exposed = data[lastDayIteration[numberOfIteration]-1, 0] * 10
#     initial_conditions = [exposed, data[lastDayIteration[numberOfIteration]-1, 0], data[lastDayIteration[numberOfIteration]-1, 1], data[lastDayIteration[numberOfIteration]-1, 2]]

#     previsionParameters = Parameters()

#     previsionParameters.add('ro', roEstimated[roEstimated.__len__() - 1])
#     #previsionParameters.add('beta', betaNewEstimated[betaNewEstimated.__len__() - 1])
#     previsionParameters.add('alpha', alphaEstimated[alphaEstimated.__len__() - 1])
#     previsionParameters.add('gamma', gammaEstimated[gammaEstimated.__len__() - 1], min=0.04, max=0.05)
#     previsionParameters.add('epsilon', epsilonEstimated[epsilonEstimated.__len__() - 1])

#     sol = fodeSolver(tspan, initial_conditions, previsionParameters, numberOfIteration)
#     return sol

if __name__ == "__main__":
    # Prendo i valori iniziali degli infetti, dei recovered e dei deceduti direttamente dal database della protezione civile
    "Setting dei valori iniziali, data è la matrice contenente i valori osservati, per approfondimenti si veda in dataManagement.py"
    initI = constants.InitI
    # Individui in stato E scelti arbitrariamente come 10x degli individui in stato I
    initE = constants.InitE
    initA = constants.InitA
    initR = constants.InitR
    # initD = data[constants.firstDay, 2]

    "Setting dei parametri"
    T = constants.initT
    gamma = 1 / T
    delta_E = constants.Delta_E
    tau = constants.Tau
    susc = constants.Susc
    cont_mat = constants.Cont_mat
    prob_symp = constants.Prob_symp
    gamma = constants.Gamma
    rho = constants.Rho


    initial_conditions = [initE, initI, initA, initR]

    # Creo un vettore con tempo di 30 giorni
    tspan = np.arange(constants.firstDay,30,1)
    
    
    # parametersToOptimize = Parameters()
    # # TODO Perché beta aveva min=0 e max=1?
    # # parametersToOptimize.add('beta', beta, min=0, max=1)
    # parametersToOptimize.add('beta', beta)
    # parametersToOptimize.add('alpha', alpha)
    # # TODO Perché gamma ha min=0.04 e max=0.05?
    # parametersToOptimize.add('gamma', gamma, min=0.04, max=0.05)
    # parametersToOptimize.add('epsilon', epsilon)
    

    # "Avvio la prima stima di parametri sul primo range del tempo (da 0 a daysFirstIteration)"
    # result = minimize(error, parametersToOptimize, args=(initial_conditions, tspan, data, constants.firstDay, constants.firstDay + daysFirstIteration))

    # beta0 = result.params['beta'].value
    # alpha0 = result.params['alpha'].value
    # epsilon0 = result.params['epsilon'].value
    # gamma0 = result.params['gamma'].value
    # ro0 = constants.initRO

    # "Salvo i parametri nelle mie liste"
    # betaEstimated.append(beta0)
    # alphaEstimated.append(alpha0)
    # epsilonEstimated.append(epsilon0)
    # gammaEstimated.append(gamma0)
    # roEstimated.append(ro0)

    # parametersOptimized = Parameters();
    # parametersOptimized.add('beta', betaEstimated[0], min=0, max=1)
    # parametersOptimized.add('alpha', alphaEstimated[0])
    # parametersOptimized.add('gamma', gammaEstimated[0], min=0.04, max=0.05)
    # parametersOptimized.add('epsilon', epsilonEstimated[0])

    "Calcolo le soluzioni del sistema con i parametri stimati nel primo intervallo da 0 a daysFirstIteration"
    # parameters = Parameters()
    # parameters.add('susc',susc)
    # parameters.add('prob_symp',prob_symp)
    # parameters.add('tau',tau)
    # parameters.add('cont_mat',cont_mat)
    # parameters.add('delta_E', delta_E)
    # parameters.add('gamma',gamma)
    tspan = np.arange(0,10)
    result = odeSolver(tspan, initial_conditions, susc, prob_symp, tau, cont_mat, delta_E, gamma)

    "Grafico del risultato per la popolazione infetta con età 70+"
    plt.plot(tspan,result[:,17])
    # indexInit = totalDays - daysIteration - 1

    # "Vettori in cui salvo le soluzioni (Infetti, Guariti e Morti, Esposti)"
    # totalModelInfected = []
    # totalModelRecovered = []
    # totalModelDeath = []
    # totalModelExposed = []

    # "Memorizzo i primi valori (da 0 a daysFirstIteration) delle soluzioni di Infetti, Guariti, Morti, Esposti"
    # totalModelInfected[0:daysFirstIteration] = model_init[:, 2]
    # totalModelRecovered[0:daysFirstIteration] = model_init[:, 3]
    # totalModelDeath[0:daysFirstIteration] = model_init[:, 4]
    # totalModelExposed[0:daysFirstIteration] = model_init[:, 1]

    # firstDayIteration.append(constants.firstDay + daysFirstIteration)
    # lastDayIteration.append(constants.firstDay + daysFirstIteration+initDeltaT)

    # finishedIteration = 1
    # k=0
    # "Definisco la mia k-esima iterata"
    # while(finishedIteration):
    #     """
    #     Definisco gli estremi del mio intervallo
    #           <-deltaT->
    #     -----|-----------|----------------
    #         timek        timek_1
    #     """
    #     timek = firstDayIteration[k]
    #     timek_1 = lastDayIteration[k]


    #     tspank = np.arange(timek, timek_1, 1)

    #     tspank_model = np.arange(timek, timek_1, 1)

    #     "Aggiorno gli esposti alla k_esima iterazione"
    #     exposed_k = data[timek, 0]*10

    #     "Aggiorno le condizioni iniziali considerando i veri dati osservati"
    #     initial_conditions_k = [exposed_k, data[timek, 0], data[timek, 1], data[timek, 2]]

    #     #parametersToOptimize.add('ro', roEstimated[k], min=0)
    #     parametersToOptimize.add('ro', roEstimated[k], min=0, max=1)
    #     parametersToOptimize.add('alpha', alphaEstimated[k])
    #     parametersToOptimize.add('gamma', gammaEstimated[k], min=0.04, max=0.05)
    #     parametersToOptimize.add('epsilon', epsilonEstimated[k])

    #     "Stimo i parametri alla k_esima iterazione con le condizioni iniziali aggiornate"
    #     resultForcedIteration = minimize(errorRO, parametersToOptimize,
    #                                      args=(initial_conditions_k, tspank, data, k))

    #     rok = resultForcedIteration.params['ro'].value
    #     alphak = resultForcedIteration.params['alpha'].value
    #     epsilonk = resultForcedIteration.params['epsilon'].value
    #     gammak = resultForcedIteration.params['gamma'].value

    #     roEstimated.append(rok)
    #     alphaEstimated.append(alphak)
    #     epsilonEstimated.append(epsilonk)
    #     gammaEstimated.append(gammak)

    #     #parametersOptimized.add('ro', roEstimated[k + 1], min=0)
    #     parametersOptimized.add('ro', roEstimated[k + 1], min=0, max=1)
    #     parametersOptimized.add('alpha', alphaEstimated[k + 1])
    #     parametersOptimized.add('gamma', gammaEstimated[k + 1], min=0.04, max=0.05)
    #     parametersOptimized.add('epsilon', epsilonEstimated[k + 1])

    #     "Calcolo il modello con i parametri stimati"
    #     modelfk = fodeSolver(tspank_model, initial_conditions_k, parametersOptimized, k)

    #     "Salvaggio dei dati relativi alla finestra temporale pari a deltaT (da timeK a timeK_1"
    #     totalModelInfected[timek:timek_1] = modelfk[:, 2]
    #     totalModelRecovered[timek:timek_1] = modelfk[:, 3]
    #     totalModelDeath[timek:timek_1] = modelfk[:, 4]
    #     totalModelExposed[timek:timek_1] \
    #         = modelfk[:, 1]

    #     deltaTCalibration(totalModelInfected, k)

    #     if(lastDayIteration[k+1]>totalDays):
    #         finishedIteration = 0
    #         numberOfIteration = k
    #     else:
    #         k=k+1



    # #Perform my Prevision

    # daysPrevision = constants.daysPrevision
    # myprevision = prevision(daysPrevision)

    # totalModelInfected[lastDayIteration[k]:lastDayIteration[k]+daysPrevision] = myprevision[:, 2]



    # datapoints = lastDayIteration[k] + daysPrevision


    # #Convert DataTime to String in order to Plot the data
    # lastDay = dateForPlot[lastDayIteration[numberOfIteration]-1]

    # dayPrevision = lastDay + dt.timedelta(days=daysPrevision+1)
    # days = mdates.drange(dateForPlot[constants.firstDay], dayPrevision, dt.timedelta(days=1))

    # daysArangeToPrint = mdates.drange(lastDay, dayPrevision, dt.timedelta(days=1))
    # daysArangeToPrint = mdates.drange(lastDay, dayPrevision, dt.timedelta(days=1))
    # # print(daysArangeToPrint)

    # lenghtInfected = len(data[:,0])

    # columns = ['data', 'osservati', 'stimati']
    # df_comparison = pandas.DataFrame(columns=columns);

    # for i in range(0, daysPrevision):
    #     if(lastDayIteration[numberOfIteration] + i < lenghtInfected):
    #         df2 = pandas.DataFrame({"data": mdates.num2date(daysArangeToPrint[i]),
    #                             "osservati": data[lastDayIteration[numberOfIteration]-1 + i, 0],
    #                             "stimati":  totalModelInfected[lastDayIteration[numberOfIteration] + i - constants.firstDay]}, index=[0])
    #         df_comparison = df_comparison.append(df2, ignore_index = True)

    #         print(str(mdates.num2date(daysArangeToPrint[i])) + "Forecasted " + str(totalModelInfected[lastDayIteration[numberOfIteration] + i - constants.firstDay]) +
    #              " Obseved " + str(data[lastDayIteration[numberOfIteration]-1 + i, 0]) + "\n")

    # df_comparison.to_csv("comparison.csv")


    # R0 = betaNewEstimated * T

    # #plt.plot(epsilonEstimated)
    # #plt.plot(alphaEstimated)
    # #plt.plot(betaEstimated)
    # #plt.plot(betaNewEstimated)
    # #plt.plot(R0)


    # "Plot dei valori calcolati con il modello"

    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    # plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=16))
    # plt.plot(days, totalModelInfected[:], label="Infected (Model and Predicted)")

    # plt.plot(dateForPlot[constants.firstDay:lastDayIteration[k]], data[constants.firstDay:lastDayIteration[k], 0], label="Infected(Observed)")


    # #plt.plot(dateForPlot[constants.firstDay:lastDayIteration[numberOfIteration]], totalModelExposed[:], label="Esposti (Model)")
    # #plt.plot(dateForPlot[constants.firstDay:lastDayIteration[numberOfIteration]], totalModelRecovered[:], label="Recovered (Model)")
    # #plt.plot(dateForPlot[constants.firstDay:lastDayIteration[numberOfIteration]], totalModelDeath[:], label="Death(Model)")

    # "Plot dei valori osservati"
    # #print(totalModelExposed)

    # #plt.plot(dateForPlot[0:lastDayIteration[k]], data[0:lastDayIteration[k], 1], label="Recovered (Observed)")
    # #plt.plot(dateForPlot[0:lastDayIteration[k]], data[0:lastDayIteration[k], 2], label="Death (Observed)")

    # # plt.plot(betaEstimated)

    # plt.gcf().autofmt_xdate()

    # plt.legend()
    # plt.show()

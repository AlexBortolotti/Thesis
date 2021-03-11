from dataManagement import data
from dataManagement import contact_data
import numpy as np
from scipy.sparse import csr_matrix

def matrix_converter(k_fine, coarse_bds, fine_bds, pyramid):
    #Function that transforms Prem's contact matrix (fine_bds) into a contact matrix with Istat sierlogical study's age classes (coarse_bds)
    aggregator = np.zeros(fine_bds.size - 1.) #This matrix stores where each class in finer structure is in coarser structure
    non_aggregator = np.arange(aggregator.size)
    for i in range(fine_bds.size-1):
        aggregator[i]=(np.where(coarse_bds >= fine_bds[i+1])[0])[0]-1.;
    
    pyramid = pyramid/np.sum(pyramid) #Normalize to give proportions
    agg_pop_pyramid = (csr_matrix((pyramid, (aggregator, non_aggregator)))).sum(1) #Sparse matrix defined here just splits pyramid into rows corresponding to coarse boundaries, then summing each row gives aggregated pyramid
    rel_weights = pyramid * agg_pop_pyramid[aggregator]
    
    # Now define contact matrix with age classes from sierological survey
    pop_weight_matrix = csr_matrix((rel_weights, (aggregator, non_aggregator)))
    pop_no_weight = csr_matrix((np.ones(aggregator.size), (aggregator, non_aggregator)))
    k_coarse = pop_weight_matrix @ k_fine.transpose() @ pop_no_weight
    
    return k_coarse
    
    


class constants :    
    
    """
    Parametri Iniziali Pietro
    """
    # initRO = 5
    initT = 20
    """
    Parametri Iniziali Alex
    """
    
    #Inizio modello il 1 Settembre 20
    firstDay = 190
    sieroDay = 142
    
    #Exit rate from latent state
    Delta_E = 0.52
    #Reduced infectivity factors by age vector for asymptomatic individuals
    # Tau = [np.zeros(8)]
    Tau = 0
    # #Susceptibility by age vector
    # Susc= []
    #Contact matrix by Prem
    # Cont_mat = contact_data
    Cont_mat = np.array([[1,1,1,1,1,1], [1,1,1,1,1,1], [1,1,1,1,1,1], [1,1,1,1,1,1], [1,1,1,1,1,1], [1,1,1,1,1,1]])
    #Probability of developing symptoms by FBK estimates
    Prob_symp = 0.3102
    # #Probability of developing symptoms by age vector
    # Prob_symp = []
    #Removal rate
    Gamma = 1/initT
    #Scaling factor from Hilton and Keeling work
    Rho = 0.1433
    #Susceptibility constants
    Susc = Rho / Prob_symp;
    #Initial data constants
    #ISS data 1 Sept
    InitA = (data[firstDay,0]*0.628)*np.array([13.1, 14.8, 20.2, 19.9, 13.3, 18.7])
    InitI = (data[firstDay,0]*0.372)*np.array([13.1, 14.8, 20.2, 19.9, 13.3, 18.7])
    #Estimation of new infections after latent period estimate by Gatto et al
    InitE = ((data[firstDay + 5, 0] + data[firstDay + 4, 0] - 2 * data[firstDay,0])/2)*np.array([13.1, 14.8, 20.2, 19.9, 13.3, 18.7])
    #InitR = sierological prevalence at 15th July + recovered from 15 Jul to 1 Sept + dead at 1 Sept
    InitR = (754331 + (data[firstDay,1] - data[sieroDay,1]) + data[firstDay,2])*np.array([13.1, 14.8, 20.2, 19.9, 13.3, 18.7])

    #Popolazione dell'Emilia Romagna
    # N = 4459000

    #Popolazione Lombardia
    N = 10027602 * np.array([15.75, 17.75, 21.25, 15.9, 11.9, 17.45])
                            
    """
    In questa fase decidiamo come procedere con il modello: 
        - totalDays: sono i giorni totali di cui possediamo i dati osservati
    A questo punto suddivido i giorni totali in due parti per dividere le stime: 
        - daysIteration sono i giorni che decidiamo di prendere in considerazione per il processo di ottimizzazione discretizzato
        - daysFirstIteration sono i giorni che predisponiamo per la prima fase del processo di ottimizzazione in cui facciamo una prima stima dei parametri
    """
    totalDays = len(data)
    # totalDays = 60

    daysFirstIteration = 30

    daysIteration = totalDays - firstDay - daysFirstIteration

    #daysFirstIteration = firstDay + totalDays - daysIteration

    """
       Definisco il deltaT iniziale su cui fare la mia prima calibrazione con la prima iterazione

    """
    initDeltaT = 10

    """
        Definitisco la lunghezza dei miei intervalli per le calibrazioni

    """
    MIN_INTERVAL = 7
    MEDIUM_INTERVAL = 10
    MAX_INTERVAL = 15
    #TODO $\Delta T$ vale tra 7 e 15, ma nel paper di Zama-Piccolomini $\Delta T$ ha una variabilità diversa. Si può correggere?

    """
        Definisco le mie tolleranze di errore su cui poi modificare l'ampiezza degli intervalli

    """
    ERROR_RANGE_MIN_INTERVAL = 600
    ERROR_RANGE_MAX_INTERVAL = 300


    daysPrevision = 30


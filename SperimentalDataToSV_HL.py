import numpy as np #np serve per richiamare/abbreviare numpy
import scipy
import pylab
from numpy import *
import os
from datetime import datetime
from scipy.optimize import minimize
path=os.path.dirname(os.path.abspath(__file__)) #autorileva cartella in cui si trova il file
print ("This script calculates the Sfwitf-Voce approximation starting from sperimental engineering data")
print ("To run the script is needed a .txt file with tensile test engineering data: elongation in % and load in MPa.")
print ("The file and the script must be in the same folder and the data in the file must be placed in two coloumns (along. and load) without any symbols.")
print ("The script will create the following set of data: True Curve, Plastic True Curve, Swift Approx., Voce Approx., S-V  approx. at 100% of deformation and the parameters for each approximation")
start=input("Press Enter to start.")
FileS=input("Insert the name of raw data file without extension: ")
Mat=input("Enter material name: ")
For=input("Enter supplier name: ")
today=datetime.now() #ottengo data e ora correnti per salvataggio file
td=today.strftime("%Y_%m_%d_%H_%M")
Sper=np.array(genfromtxt(path+"/"+FileS+".txt")) #importo dati da file txt
EngS=Sper[:,1] #scorporo dati di sigma
EngEpc=Sper[:,0] #scorporo dati di epsilon
EngE=EngEpc/100 # trasformo dati di allungamento NON in percentuale
PlotEng=input("Want to plot imported curve? (y,n) ") #plotto con def % per non confondere
if PlotEng=="y":
    pylab.plot(EngEpc,EngS)
    pylab.title("Eng.Curve")
    pylab.show()
NeckPos=1+EngS.argmax() #trovo la posizione nell'array del valore massimo di sigma
SNeck=EngS[0:NeckPos]# taglio l'array EngS al necking
ENeck=EngE[0:NeckPos]# taglio l'array EngE al necking
TrueE=np.log(ENeck+1) #Trasformo in curva true
TrueS=SNeck*(1+ENeck) #Trasformo in curva true
np.savetxt(path+"/"+td+"_"+Mat+"_"+For+"_TrueCurve.txt", np.c_[TrueE,TrueS])
PlotTC=input("Want to plot True curve? (y,n) ")
if PlotTC=="y":
    pylab.plot(TrueE,TrueS)
    pylab.title("True Curve")
    pylab.show()
Yield=float(input('Enter Yiled strength value ')) #chiedo in input il valore di snervamento da cui far partire la curva plastica
def find_nearest (TrueS, Yield): # trovo il valore all'interno della curva più vicino allo snervamento indicato
    idx=(np.abs(TrueS - Yield)).argmin()
    return TrueS[idx]
YPos=1+(np.abs(TrueS - Yield)).argmin() #Trovo la posizione in cui è lo snervamento
PlasS=TrueS[YPos:]
PlasE=(TrueE[YPos:]-TrueE[YPos])
np.savetxt(path+"/"+td+"_"+Mat+"_"+For+"_TruePlasticCurve.txt", np.c_[PlasE,PlasS])
PlotTPC=input("Want to plot Plastic True Curve? (y,n) ")
if PlotTPC=="y":
    pylab.plot(PlasE,PlasS)
    pylab.title("True Plastic Curve")
    pylab.show()

#Eseguo approssimazione SV
ep=PlasE
epm=float(np.amax(ep))#trovo il max di ep
sp=PlasS
ys=float(np.amin(sp)) #trovo il min di sp (snervamento)

#Calcolo app Voce
def err(x):
    Q, B = x
    vo = ys+(Q*(1-(np.exp(-B*ep))))

    return np.sum((sp-vo)**2)

def con(x):
    Q, B = x
    return Q-(((Q*(B+1)*(np.exp(-B*epm))))-ys)

guess = (1, 1)
cons={'type':'eq', 'fun': con}
res = minimize(err, guess ,method='SLSQP',constraints=cons)
Q, B = res.x
vo = ys+(Q*(1-(np.exp(-B*ep))))
np.savetxt(path+"/"+td+"_"+Mat+"_"+For+"_Voce.txt", np.c_[ep,vo])
np.savetxt(path+"/"+td+"_"+Mat+"_"+For+"_Voce_QeBeta_Param.txt", np.c_[Q,B])

# Calcolo app Swift
def err(x):
    ei = x
    sw = (ys/(ei**(ei+epm)))*((ep+ei)**(ei+epm))
    return np.sum((sp-sw)**2)

guesssw = (0.00001)
res = minimize(err, guesssw ,method='Nelder-Mead')
ei = res.x
sw = (ys/(ei**(ei+epm)))*((ep+ei)**(ei+epm))
np.savetxt(path+"/"+td+"_"+Mat+"_"+For+"_Swift.txt", np.c_[ep,sw])
np.savetxt(path+"/"+td+"_"+Mat+"_"+For+"_Swift_e0_Param.txt", ei)

# Calcolo app Swift-Voce
for con in range (1,100):
    alfa=float(input("Enter a value for alpha to calculate the S-V approx.: "))
    sv=(1-alfa)*vo+(alfa*sw)
    ep1=np.arange(0,0.08,0.002)
    ep2=np.arange(0.08,1.02,0.02)
    ep100=np.concatenate([ep1,ep2])
    sv=(1-alfa)*(ys+(Q*(1-(np.exp(-B*ep100)))))+(alfa*((ys/(ei**(ei+epm)))*((ep100+ei)**(ei+epm))))
    pylab.plot(ep,sp, label='Sperimental Data')
    pylab.plot(ep100,sv, label='SwiftVoce')
    pylab.title("Swift-Voce")
    pylab.legend(loc='lower right')
    pylab.show()
    ok=input("Want to recalculate with another value of alpha? (y,n) ")
    if ok=="n":
        break
np.savetxt(path+"/"+td+"_"+Mat+"_"+For+"_SwiftVoce_def100"+"_Alfa_"+str(alfa)+".txt", np.c_[ep100,sv])
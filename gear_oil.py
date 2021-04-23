import csv
import datetime
import os
import numpy as np
import random
import math
import pandas as pd
import matplotlib.pyplot as plt

from htm.bindings.sdr import SDR, Metrics
from htm.encoders.rdse import RDSE, RDSE_Parameters
from htm.encoders import *
from htm.encoders.date import DateEncoder
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.algorithms import TemporalMemory
from htm.algorithms.anomaly_likelihood import AnomalyLikelihood
from htm.bindings.algorithms import Predictor

# Get the current directory and create variable for file name
cwd = os.getcwd()
input_file_path = cwd+'/gear_oil.csv'

#dictionary of parameters for the encoder
#in this case we have 2 encoders: the temperature values and the time
parameters = {
 'enc': {
      "value" :
         {'resolution': 0.88, 'size': 700, 'sparsity': 0.02},
      "time": 
         {'timeOfDay': (30, 1), 'weekend': 21}
 },
 'predictor': {'sdrc_alpha': 0.1},
 'sp': {'boostStrength': 3.0,
        'columnCount': 1638,
        'localAreaDensity': 0.04395604395604396,
        'potentialPct': 0.85,
        'synPermActiveInc': 0.04,
        'synPermConnected': 0.13999999999999999,
        'synPermInactiveDec': 0.006},
 'tm': {'activationThreshold': 17,
        'cellsPerColumn': 13,
        'initialPerm': 0.21,
        'maxSegmentsPerCell': 128,
        'maxSynapsesPerSegment': 64,
        'minThreshold': 10,
        'newSynapseCount': 32,
        'permanenceDec': 0.1,
        'permanenceInc': 0.1},
 'anomaly': {
   'likelihood': 
       {
        'probationaryPct': 0.1,
        'reestimationPeriod': 100}
 }
}
       
# Read the data into an array
records = []
with open(input_file_path, 'r') as fin:
    reader = csv.reader(fin)
    headers = next(reader)
    next(reader)
    next(reader)
    for record in reader:
        records.append(record)
        
inps = []
for record in records:
    inps.append(record[1])

# Convert the data into a pandas dataframe
df = pd.DataFrame(inps, columns=['oil_temp'])

#Create Datetime encoder
dateEncoder = DateEncoder(
    timeOfDay = parameters['enc']['time']['timeOfDay'],
    weekend = parameters["enc"]["time"]["weekend"])

#Create the encoder for the temp. values
scalarEncoderParams = RDSE_Parameters() # RDSE -> Random Distributed Scalar Encoder
scalarEncoderParams.size       = parameters["enc"]["value"]["size"]
scalarEncoderParams.sparsity   = parameters["enc"]["value"]["sparsity"]
scalarEncoderParams.resolution = parameters["enc"]["value"]["resolution"]
scalarEncoder = RDSE(scalarEncoderParams)

encodingWidth = (dateEncoder.size + scalarEncoder.size)
enc_info = Metrics( [encodingWidth], 999999999) #performance metrics

# Create the spatial pooler
spParams = parameters['sp']

sp = SpatialPooler(
    inputDimensions = (encodingWidth,),
    columnDimensions = (spParams["columnCount"],),
    potentialPct = spParams["potentialPct"],
    potentialRadius = encodingWidth,
    globalInhibition = True,
    localAreaDensity = spParams["localAreaDensity"],
    synPermInactiveDec = spParams["synPermInactiveDec"],
    synPermActiveInc = spParams["synPermActiveInc"],
    synPermConnected = spParams["synPermConnected"],
    boostStrength = spParams["boostStrength"],
    wrapAround = True
)
sp_info = Metrics(sp.getColumnDimensions(), 999999999)

tmParams = parameters['tm']
tm = TemporalMemory(
    columnDimensions = (spParams["columnCount"],),
    cellsPerColumn = tmParams["cellsPerColumn"],
    activationThreshold = tmParams["activationThreshold"],
    initialPermanence = tmParams["initialPerm"],
    connectedPermanence = spParams["synPermConnected"],
    minThreshold = tmParams["minThreshold"],
    maxNewSynapseCount = tmParams["newSynapseCount"],
    permanenceIncrement = tmParams["permanenceInc"],
    permanenceDecrement = tmParams["permanenceDec"],
    predictedSegmentDecrement = 0.0,
    maxSegmentsPerCell = tmParams["maxSegmentsPerCell"],
    maxSynapsesPerSegment = tmParams["maxSynapsesPerSegment"]
)
tm_info = Metrics( [tm.numberOfCells()], 999999999)

# setup the anomaly and likelihood thresholds
anParams = parameters['anomaly']['likelihood']
probationaryPeriod = int(math.floor(float(anParams['probationaryPct'])*len(records)))
learningPeriod = int(math.floor(probationaryPeriod / 2.0))
anomaly_history = AnomalyLikelihood(learningPeriod = learningPeriod,
       estimationSamples = probationaryPeriod - learningPeriod,
       reestimationPeriod = anParams['reestimationPeriod'])
predictor = Predictor(steps=[1,5], alpha=parameters['predictor']['sdrc_alpha'])
predictor_resolution = 1

#dateString = datetime.datetime.fromisoformat(records[200][0]).strftime('%m/%d/%y %H:%M')
#print(dateString)

# Trial learning loop
predictor.reset()
predictor = Predictor(steps=[1,5], alpha=parameters['predictor']['sdrc_alpha'])
predictor_resolution = 1
inputs = []
anomaly = []
anomalyProb = []
predictions = {1: [], 5:[]}

for count, record in enumerate(records):
    dateString = datetime.datetime.fromisoformat(record[0]).strftime('%m/%d/%y %H:%M')
    dateString = datetime.datetime.strptime(dateString, '%m/%d/%y %H:%M')
    consumption = float(record[1])
    inputs.append(consumption)
    dateBits = dateEncoder.encode(dateString)
    consumptionBits = scalarEncoder.encode(consumption)
    encoding = SDR(encodingWidth).concatenate([consumptionBits, dateBits])
    enc_info.addData(encoding)
    activeColumns = SDR(sp.getColumnDimensions())
    sp.compute(encoding, True, activeColumns)
    tm_info.addData(tm.getActiveCells().flatten())
    tm.compute(activeColumns, learn=True)
    predictor.learn(count, tm.getActiveCells(), int(round(consumption/predictor_resolution)))

# Original Training Loop
inputs = []
anomaly = []
anomalyProb = []
predictions = {1: [], 5:[]}
predictor.reset()
for count, record in enumerate(records): #iterating through the listified data
    dateString = datetime.datetime.fromisoformat(record[0]).strftime('%m/%d/%y %H:%M') #Convert date and time values from string to datetime format
    dateString = datetime.datetime.strptime(dateString, '%m/%d/%y %H:%M')
    consumption = float(record[1])
    inputs.append(consumption)
    
    #creates SDR's for all input values using the encoder
    dateBits = dateEncoder.encode(dateString)
    consumptionBits = scalarEncoder.encode(consumption)
    
    # Combine all SDR's into a bigger SDR for spatial pooling
    encoding = SDR(encodingWidth).concatenate([consumptionBits, dateBits])
    enc_info.addData(encoding)
    
    # SDR created from active columns in above SDR
    activeColumns = SDR(sp.getColumnDimensions())
    
    # Enter input data into spatial pooler
    sp.compute(encoding, True, activeColumns)
    tm_info.addData(tm.getActiveCells().flatten())
    tm.compute(activeColumns, learn=True)

    pdf = predictor.infer( tm.getActiveCells() )
    for n in (1,5):
        if pdf[n]:
            predictions[n].append( np.argmax( pdf[n] ) * predictor_resolution )
        else:
            predictions[n].append(float('nan'))
    
    anomalyLikelihood = anomaly_history.anomalyProbability( consumption, tm.anomaly )
    anomaly.append(tm.anomaly)
    anomalyProb.append(anomalyLikelihood)
    predictor.learn(count, tm.getActiveCells(), int(consumption/predictor_resolution))

old_inputs = inputs

print("Encoded Input", enc_info)
print("")
print("Spatial Pooler Mini-Columns", sp_info)
print(str(sp))
print("")
print("Temporal Memory Cells", tm_info)
print(str(tm))
print("")

for n_steps, pred_list in predictions.items():
    for x in range(n_steps):
        pred_list.insert(0, float('nan'))
        pred_list.pop()
        
for n in predictions:
    print(n)
    
acc_lists = {1: [], 5: []}
accuracy = {1: 0, 5: 0}
accuracy_samples = {1: 0, 5: 0}
for idx, inp in enumerate(inputs):
    for n in predictions:
        val = predictions[n][idx]
        if not math.isnan(val):
            accuracy[n] += (inp - val) ** 2
            accuracy_samples[n] += 1
    for n in sorted(predictions):
        try:
            accuracy[n] = (accuracy[n] / accuracy_samples[n]) ** .5
        except Exception as e:
            print(e)
            print("Predictive Error (root-mean-squared): ", n, "steps ahead:", accuracy[n])
        acc_lists[n].append(accuracy[n])
        
    #print('Anomaly Mean: ', np.mean(anomaly))
    #print('Anomaly Std: ', np.std(anomaly))
    #print()
    
powers = []
temp = inps
for t in temp:
    powers.append(float(t))
    
df = pd.DataFrame({'time_1': acc_lists[1], 'time_5': acc_lists[5], 'actual': inputs})

import matplotlib.pyplot as plt
plt.subplot(2,1,1)
plt.title("Predictions")
plt.xlabel("Time")
plt.ylabel("Power Consumption")
plt.plot(np.arange(len(inputs)), inputs, 'red',
       np.arange(len(inputs)), predictions[1], 'blue',
       np.arange(len(inputs)), predictions[5], 'green',)
plt.legend(labels=('Input', '1 Step Prediction, Shifted 1 step', '5 Step Prediction, Shifted 5 steps'))
plt.subplot(2,1,2)
plt.title("Anomaly Score")
plt.xlabel("Time")
plt.ylabel("Power Consumption")
inputs = np.array(inputs) / max(inputs)
plt.plot(np.arange(len(inputs)), inputs, 'red',
       np.arange(len(inputs)), anomaly, 'blue',)
plt.legend(labels=('Input', 'Anomaly Score'))
plt.show()

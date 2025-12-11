This folder contains the predictions recorded for the maintenance schemes using the Nov 25 method, where:
- Whenever adaptation is required at the next hdist, instead of predicting with each MCS/USI/MCS+USI one by one, 
we predict for all MCS/USI/MCS+USI and choose the one that gives the highest mean reliability that meets the reliability requirement.
- The outcome of predictions for all MCS/USI/MCS+USI are also recorded for analysis, with the chosen MCS/USI/MCS+USI indicated.
- The adaptation is made to stop at D_max (obtained from ground truth), and the initial MCS/USI/MCS+USI is determined from highest possible MCS & lowest possible USI (based on ground truth).
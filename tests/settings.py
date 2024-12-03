
#CH options
CH_PICKS = [True, 'mag', 'grad']

#ICA options
ICA_TRAIN = [True, False]
ICA_EOG = [True, False]
ICA_ECG = [True, False]
ICA_THRESH = [.4, .6, .8]
ICA_RESAMPLE = [100, 200]
ICA_NCOMPS = [10, 20]

#Maxwell options
MW_DESTINATION = [None, (0, 0, .4)]

#Source options
SOURCE = ['surface', 'volume']
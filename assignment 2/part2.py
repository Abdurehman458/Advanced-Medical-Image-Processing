from __future__ import print_function

import logging
import os
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
import nrrd
import radiomics
from radiomics import featureextractor
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

dtype= ["/valid/","/train/","/test/"]
nrrd_dir = "./nrrd_ds"
base = "./dataset"

def create_nrrd_ds():
    print("\n","!!!!!!!!! creating nrrd dataset!!!!!!!!!","\n")
    nrrrd="./nrrd_ds"
    Path(nrrrd).mkdir(parents=True, exist_ok=True)
    for split in dtype:
        for directory in tqdm(sorted(os.listdir(base+split))):
            # print(directory)
            img = np.load(base+split+directory+"/img.npy")
            # print(base+split+directory+"/img.npy")
            label = np.load(base+split+directory+"/label.npy")
            seg = np.load(base+split+directory+"/seg.npy")
            seg_clip = np.clip(seg,0,1)
            # img = np.concatenate((img, np.expand_dims(seg_clip,0)),axis=0)
            seg_concat = np.expand_dims(seg_clip,0)
            for i in range(img.shape[0]-1):
                seg_concat = np.concatenate((seg_concat,np.expand_dims(seg_clip,0)),axis=0)
            filename = directory
            nrrd_dir = nrrrd+split+directory 
            Path(nrrd_dir).mkdir(parents=True, exist_ok=True)
            nrrd.write(nrrd_dir+'/image.nrrd', img)
            nrrd.write(nrrd_dir+'/seg.nrrd', seg_concat)
            nrrd.write(nrrd_dir+'/label.nrrd',label)


def get_csv():
    for split in dtype:
        data1 = pd.DataFrame(columns=["Image","Mask","L"])

        for iter,directory in tqdm(enumerate(sorted(os.listdir(nrrd_dir+split)))):
            path =nrrd_dir+split+directory 
            img_path = os.path.join(path,"image.nrrd")
            seg_path = os.path.join(path,"seg.nrrd")
            lab_path = os.path.join(path,"label.nrrd")
            label, header = nrrd.read(lab_path)
            label = label.tolist() 
            test = label[0]

         
            #creating a Dataframe of image "Name" and "label"
            data1.loc[iter,"Image"]=img_path
            data1.loc[iter,"Mask"]=seg_path
            data1.loc[iter,"L"]=test
            # data1.to_csv('data1.csv', index=False)


        # datac = data1.append(data, ignore_index=False)
        if split == "/valid/":
            data1.to_csv('valid_cases.csv', index=False)
        if split == "/train/":
            data1.to_csv('train_cases.csv', index=False)
        if split == "/test/":
            data1.to_csv('test_cases.csv', index=False)


def radiomics_feat(input_csv=None, outputfile=None):
  outPath = r''

  inputCSV = os.path.join(outPath, input_csv)
  outputFilepath = os.path.join(outPath, outputfile)
  progress_filename = os.path.join(outPath, 'pyrad_log.txt')
  params = os.path.join(outPath, 'exampleSettings', 'Params.yaml')

  # Configure logging
  rLogger = logging.getLogger('radiomics')

  # Set logging level
  # rLogger.setLevel(logging.INFO)  # Not needed, default log level of logger is INFO

  # Create handler for writing to log file
  handler = logging.FileHandler(filename=progress_filename, mode='w')
  handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
  rLogger.addHandler(handler)

  # Initialize logging for batch log messages
  logger = rLogger.getChild('batch')

  # Set verbosity level for output to stderr (default level = WARNING)
  radiomics.setVerbosity(logging.INFO)

  logger.info('pyradiomics version: %s', radiomics.__version__)
  logger.info('Loading CSV')

  # ####### Up to this point, this script is equal to the 'regular' batchprocessing script ########

  try:
    # Use pandas to read and transpose ('.T') the input data
    # The transposition is needed so that each column represents one test case. This is easier for itertion over
    # the input cases
    flists = pd.read_csv(inputCSV).T
  except Exception:
    logger.error('CSV READ FAILED', exc_info=True)
    exit(-1)

  logger.info('Loading Done')
  logger.info('Patients: %d', len(flists.columns))

  if os.path.isfile(params):
    extractor = featureextractor.RadiomicsFeatureExtractor(params)
  else:  # Parameter file not found, use hardcoded settings instead
    settings = {}
    settings['binWidth'] = 25
    settings['resampledPixelSpacing'] = None  # [3,3,3]
    settings['interpolator'] = sitk.sitkBSpline
    settings['enableCExtensions'] = True

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    # extractor.enableInputImages(wavelet= {'level': 2})

  logger.info('Enabled input images types: %s', extractor.enabledImagetypes)
  logger.info('Enabled features: %s', extractor.enabledFeatures)
  logger.info('Current settings: %s', extractor.settings)

  # Instantiate a pandas data frame to hold the results of all patients
  results = pd.DataFrame()

  for entry in flists:  # Loop over all columns (i.e. the test cases)
    logger.info("(%d/%d) Processing Patient (Image: %s, Mask: %s)",
                entry + 1,
                len(flists),
                flists[entry]['Image'],
                flists[entry]['Mask'])

    imageFilepath = flists[entry]['Image']
    maskFilepath = flists[entry]['Mask']
    label = flists[entry].get('Label', None)

    if str(label).isdigit():
      label = int(label)
    else:
      label = None

    if (imageFilepath is not None) and (maskFilepath is not None):
      featureVector = flists[entry]  # This is a pandas Series
      featureVector['Image'] = os.path.basename(imageFilepath)
      featureVector['Mask'] = os.path.basename(maskFilepath)

      try:
        # PyRadiomics returns the result as an ordered dictionary, which can be easily converted to a pandas Series
        # The keys in the dictionary will be used as the index (labels for the rows), with the values of the features
        # as the values in the rows.
        result = pd.Series(extractor.execute(imageFilepath, maskFilepath, label))
        featureVector = featureVector.append(result)
      except Exception:
        logger.error('FEATURE EXTRACTION FAILED:', exc_info=True)

      # To add the calculated features for this case to our data frame, the series must have a name (which will be the
      # name of the column.
      featureVector.name = entry
      # By specifying an 'outer' join, all calculated features are added to the data frame, including those not
      # calculated for previous cases. This also ensures we don't end up with an empty frame, as for the first patient
      # it is 'joined' with the empty data frame.
      results = results.join(featureVector, how='outer')  # If feature extraction failed, results will be all NaN
  

  results = results.T
  results.drop(results.iloc[:, 3:21], inplace = True, axis = 1)
  results.drop(results.iloc[:, 5:7], inplace = True, axis = 1)

  logger.info('Extraction complete, writing CSV')
  # .T transposes the data frame, so that each line will represent one patient, with the extracted features as columns
  results.to_csv(outputFilepath, index=False, na_rep='NaN')
  logger.info('CSV writing complete')

if __name__ == '__main__':
    create_nrrd_ds()
    get_csv()
    radiomics_feat(input_csv='valid_cases.csv',outputfile='valid_features.csv')
    radiomics_feat(input_csv='train_cases.csv',outputfile='train_features.csv')  
    radiomics_feat(input_csv='test_cases.csv',outputfile='test_features.csv')     

    train_df = pd.read_csv("train_features.csv")
    test_df = pd.read_csv("test_features.csv")
    valid_df = pd.read_csv("valid_features.csv")

    # print(test_df['label'].value_counts())

    xtrain = train_df.drop(['Image','Mask','L'],axis=1)
    ytrain = train_df.L

    xtest = test_df.drop(['Image','Mask','L'],axis=1)
    ytest = test_df.L

    xval = valid_df.drop(['Image','Mask','L'],axis=1)
    yval = valid_df.L


######### Logistic Regression ########
    lreg = LogisticRegression()
    lreg.fit(xtrain, ytrain)
    y_pred = lreg.predict(xtest)

    accuracy = metrics.accuracy_score(ytest, y_pred)
    accuracy_percentage = 100 * accuracy
    lr_f1 = f1_score(ytest, y_pred)
    print("Logistic Regression Acc:",accuracy_percentage,"F1 Score:",lr_f1)
    plot_confusion_matrix(lreg, xtest, ytest)  

######### MLP #########
    mlp = MLPClassifier(hidden_layer_sizes=(32,32,32), activation='relu', solver='adam', max_iter=500)
    mlp.fit(xtrain.values, ytrain.values)
    predict_test = mlp.predict(xtest.values)

    accuracy = metrics.accuracy_score(ytest.values, predict_test)
    accuracy_percentage = 100 * accuracy
    nn_f1 = f1_score(ytest, predict_test)   
    print("Neural Network Acc:",accuracy_percentage,"F1 Score:",nn_f1)
    plot_confusion_matrix(mlp, xtest, ytest)  

######### Random Forest #########
    rf = RandomForestClassifier(n_estimators=50, oob_score=True, random_state=123456)
    rf.fit(xtrain, ytrain)
    predicted = rf.predict(xtest)

    accuracy = metrics.accuracy_score(ytest, predicted)
    accuracy_percentage = 100 * accuracy
    rf_f1 = f1_score(ytest, predicted)  
    print("Random Forest Acc:",accuracy_percentage,"F1 Score:",rf_f1)
    plot_confusion_matrix(rf, xtest, ytest)  
    # plt.show()
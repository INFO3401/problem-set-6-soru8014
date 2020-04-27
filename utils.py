import pandas as pd

import math
import numpy as np
import scipy.stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Graphics Libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Machine Learning Libraries
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif

# Import data handling libraries
import datetime as dt
import operator

def helloWorld():
  print("Hello, World!")

def loadAndCleanData(filename):
    data = pd.read_csv(filename)
    data = data.fillna(0)
    #print(data)
    return data

def computeProbability(feature, bin, data):
    # Count the number of datapoints in the bin
    count = 0.0

    for i,datapoint in data.iterrows():
        # See if the data is in the right bin
        if datapoint[feature] >= bin[0] and datapoint[feature] < bin[1]:
            count += 1

    # Count the total number of datapoints
    totalData = len(data)

    # Divide the number of people in the bin by the total number of people
    probability = count / totalData

    # Return the result
    return probability

def computeConfidenceInterval(data):
      # Confidence intervals
      npArray = 1.0 * np.array(data)
      stdErr = scipy.stats.sem(npArray)
      n = len(data)
      return stdErr * scipy.stats.t.ppf((1+.95)/2.0, n - 1)

def getEffectSize(d1,d2):
    m1 = d1.mean()
    m2 = d2.mean()
    s1 = d1.std()
    s2 = d2.std()

    return (m1 - m2) / math.sqrt((math.pow(s1, 3) + math.pow(s2, 3)) / 2.0)

def runTTest(d1,d2):
    return scipy.stats.ttest_ind(d1,d2)

# pip install statsmodels
# vars is a string with our independent and dependent variables
# " dvs ~ ivs"
def runANOVA(dataframe, vars):
    model = ols(vars, data=dataframe).fit()
    aov_table = sm.stats.anova_lm(model, typ=2)
    return aov_table

# Plot a timeline of my data
def plotTimeline(data, time_col, val_col):
    sns.lineplot(data=data, x=time_col, y=val_col)
    plt.show()

# Plot a timeline of my data broken down by each category (cat_col)
def plotMultipleTimelines(data, time_col, val_col, cat_col):
    sns.lineplot(data=data, x=time_col, y=val_col, hue=cat_col)
    plt.show()

# Run a linear regression over the data. Models an equation
# as y = mx + b and returns the list [m, b].
def runTemporalLinearRegression(data, x, y):
    # Format our data for sklean by reshaping from columns to np arrays
    x_col = data[x].map(dt.datetime.toordinal).values.reshape(-1,1)
    y_col = data[y].values.reshape(-1, 1)

    # Run the regression using an sklearn regression object
    regr = LinearRegression()
    regr.fit(x_col, y_col)

    # Compute the R2 score and print it. Good scores are close to 1
    y_hat = regr.predict(x_col)
    fitScore = r2_score(y_col, y_hat)
    print("Linear Regression Fit: " + str(fitScore))

    # Plot linear regression against data. This will let us visually judge whether
    # or not our model is any good. With small data, a high R2 doesn't always mean
    # a good model: we can use our intuition as well.
    plt.scatter(data[x], y_col, color='lightblue')
    plt.plot(data[x], y_hat, color='red', linewidth=2)
    plt.show()

    # y = mx + b
    # Return m and b
    return [regr.coef_[0][0], regr.intercept_[0]]


# Define a logistic function that we can use to model logistic data without
# requiring classification.
def logistic(x, x0, m, b):
    y = 1.0 / (1.0 + np.exp(-m*(x - x0) + b))
    return (y)

# Define a logistic modeling regression. Use this regression for modeling the
# data rather than a classification. Note that your y value must be between
# 0 and 1 for this function to work correctly.
def runTemporalLogisticRegression(data, x, y):
    # Process the data
    x_col = data[x].map(dt.datetime.toordinal)
    y_col = data[y]

    # Give the curve a crappy fit to start with
    # In this case, we'll start with x0 as the median and define a straight
    # line between 0 and 1. The curve_fit function will adjust the line
    # to minimize the residuals.
    p0 = [np.median(x_col), 1, min(y_col)]
    params, pcov = curve_fit(logistic, x_col, y_col, p0)

    # Show the fit with the actual data in blue and the model in red. Note that
    # m = params[1] and b = params[2].
    plt.scatter(data[x], y_col, color='lightblue')
    plt.plot(data[x], logistic(x_col, params[0], params[1], params[2]), color='red', linewidth=2)
    plt.show()

    # Compute the fit using R2
    # Recall that the function is 1 - (sum of squares residuals / sum of squares total)
    residuals = y_col - logistic(x_col, params[0], params[1], params[2])
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_col - np.mean(y_col))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print("Logistic Regression Fit: " + str(r_squared))

    return params
    
def runPCA(df): 
    # Standardize our features to a unit distribution
    # Each feature gets mapped to mean = 0, stddev = 1
    target_features = df.select_dtypes(include="number")
    x = StandardScaler().fit_transform(target_features)
    
    # Run PCA on the standardized data
    pca = PCA(n_components=2)
    components = pca.fit_transform(x)
    
    # Merge data with original dataframe
    newDf = pd.DataFrame(data=components, columns=["Component 1", "Component 2"])
    return pd.concat([newDf, df], axis=1)

def runKNN(df, x, y, k): 
    # Let's assume that we're using numeric features to predict a categorical label
    X = x.values
    Y = y.values.reshape(-1, 1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1, stratify=Y)
    
    # Build a kNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    
    # Compute the quality of our predictions
    score = knn.score(X_test, Y_test)
    print("Predicts " + y.name + " with " + str(score) + " accuracy")
    print("F1 score is " + str(f1_score(Y_test, knn.predict(X_test), labels=y.unique(), average='weighted')))
    print("Chance is: " + str(1.0 / len(y.unique())))
    
    return knn
    
def runKMeans(df, k):
    # Prep the data by only choosing the numeric features in the data
    X = df.select_dtypes(include="number")
    
    # Run the k-Means algorithm
    kmeans = KMeans(n_clusters=k)

    # Train the model on the features we've selected
    kmeans.fit(X)
    
    # Add the data to our dataframe in a column called 'Cluster'. 
    # Cluster will give a numeric label corresponding to the index of the closest mean
    df["Cluster"] = pd.Series(kmeans.predict(X), index=df.index)
    
    # Return the model
    return kmeans

# Enable feature selection by computing the mutual information (how much 
# do we think each feature in x will tell us about y) for a set of features (x). 
# Note that this returns a sorted dictionary of features and their corresponding
# scores with higher scores predicting more useful features
def computeInformationGain(x, y): 
    # Create a dictionary matching each column in x to a 
    # corresponding mutual information value
    feature_list = dict(zip(x.columns, mutual_info_classif(x, y, discrete_features=True)))
    
    # Sort the list from smallest to largest
    feature_list = sorted(feature_list.items(), key=operator.itemgetter(1))
    return feature_list

# Create a forward selection wrapper for feature selection. 
# best_df is the dataframe with the best feature set we've seen 
# so far. Score is the score of that dataframe. Note that you can 
# turn off the print statement to clean up your console outputs. 
def forwardSelectionKNN(best_df, x, y, k, score):
    # Build a k-NN classifier
    knn = KNeighborsClassifier(n_neighbors = k)
    Y = y.values.reshape(-1, 1)
    best_feature = ""
    
    # Determine the best feature to add in this round of forward selection
    # We can do that by testing each feature individually
    for feature in x.columns: 
        # If the feature is already in my dataset, skip it
        if feature in best_df.columns:
            continue
            
        # Create a new dataframe that's the copy of the prior best
        # configuration. This let's us test features one at a time
        # by adding that feature to a clean copy of the dataframe
        new_df = best_df.copy()
        new_df[feature] = x[feature]
        
        # Train and test my k-NN with the new feature set
        # Note that this is just your standard train-test code for k-NN
        X_train, X_test, Y_train, Y_test = train_test_split(new_df.values, Y, test_size=0.2, random_state=1, stratify=Y)
        model = knn.fit(X_train, Y_train)
        new_score = f1_score(Y_test, knn.predict(X_test), labels=y.unique(), average='weighted')
        
        # Compare the new values to the current best configuration (score). 
        # If it's better, update the score and bookmark that feature
        if new_score > score: 
            print("Adding feature " + feature + " with score " + str(new_score))
            best_feature = feature
            score = new_score
            
    # If no feature improved our prediction score, stop looking. 
    if best_feature == "":
        return best_df
    
    # If a feature improved our prediction score, add it and 
    # see if adding anything else helps. 
    else: 
        best_df[best_feature] = x[best_feature]
        print("iterating with: ")
        print(best_df.columns)
        return forwardSelectionKNN(best_df, x, y, k, score)
    
    
    
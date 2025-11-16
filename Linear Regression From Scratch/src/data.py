import numpy as np

class LoadData:

    """
    Note: This class expects the data has already be converted to numeical equivalents.

    The LoadData Class loads the data from a csv (for now) which is specified as a path, while
    calling the class, and performs standard data preparation tasks such as loading it to a
    variable, checking for nulls and unsupported filetipes. It can also split the data into the
    training and validation sets. It has the capapbility to standardize values and add bias term
    to the data itself, making calculations easier. it can also describe the data provided.

    *args

    filepath (str) : This is the filepath to your file either in the local directory or anywhere
                    on the system.

    Use variable = LoadData('..path../dataset.csv').load_data() to load the csv and store
                    it in the variable.
    """

    def __init__(self,filepath:str):
        
        """
        Initializeing the filepath and making sure the csv can be loaded.
        Returns ValueError in case of failure to load.
        """

        try:
            self.filepath = filepath
            self.df = None
        except:
            raise TypeError("Please provide the path to a valid .csv file.")
    
    def load_csv(self):
        import pandas as pd

        """
        This method is to be called while assigning the class to a variable. This performs a final
        check and returns the pandas.DataFrame if no error is occured, otherwise it raises a
        ValueError letting the user know the problem in a descriptive message.
        """
        try:
            self.df = pd.read_csv(self.filepath)
            return self
        except:
            raise ValueError("It seems the path is incorrect or does not contain a valid .csv file.\nPlease check your value and retry.")
            
    def train_val_split(self,randomize=True,perc=0.2):

        """
        This method performs a split, splitting the data into training and
        validation sets.

        *args:

            randomize (bool) [Default: True] : When set to true, it performs a shuffle
            before the split.
            perc (float/np.float64) [Default: 0.2] : Specifies the percentage of the dataset
            to be held back for validation.

        Returns:

            X_train, X_val, y_train, y_val
        """

        if isinstance(perc,(float,np.float64)) and perc>0.01 and perc<0.99:


            if randomize:
                self.df = self.df.sample(frac=1,random_state=42)
                self.df = self.df.reset_index(drop=True)
            
            X = self.df.iloc[:,0:len(self.df.columns)-1]
            y = self.df.iloc[:,-1]

            n_rows = self.df.shape[0]

            n_train = int((1 - perc) * n_rows)

            X_train = np.array(X.iloc[0:n_train,],dtype=float)
            X_val = np.array(X.iloc[n_train:,],dtype=float)
            y_train = np.array(y.iloc[0:n_train],dtype=float)
            y_val = np.array(y.iloc[n_train:],dtype=float)

            return X_train, X_val, y_train, y_val
        
        else:
            raise ValueError("[Perc] is the percentage of training data you would like to keep aside.\nPlease provide a value between 0.01 and 0.99")
        

    def standardize(self,X_train:np.ndarray,X_val:np.ndarray):

        """
        This method helps to perform standardization on the dataset, to make
        sure that the values are spread evenly with a mean of 0 and standard
        deviation of 1.

        *args

            X_train (np.ndarray) : The training dataset
            X_val (np.ndarray) : The validation dataset

        Returns:

            Standardized versions of the training and testing datasets
            X_train, X_val
        """

        mean = np.mean(X_train,axis=0)
        std = np.std(X_train,axis=0)
        std[std == 0] = 1

        X_train = (X_train - mean)/std
        X_val = (X_val - mean)/std

        return X_train, X_val
    
    def add_bias_term(self,X_train:np.ndarray,X_val:np.ndarray):

        """
        This method adds the coefficients of the bias term as 1 to the dataset in
        the 1st column so that the weight vector can contain the bias term as the 1st
        vector and separate operations will not needed to be performed.

        *args

            X_train (np.ndarray) : The training dataset
            X_val (np.ndarray) : The validation dataset

        Returns:

            X_train, X_val
        """

        X_train = np.c_[np.ones(X_train.shape[0]).reshape(X_train.shape[0],1),X_train]
        X_val = np.c_[np.ones(X_val.shape[0]).reshape(X_val.shape[0],1),X_val]

        return X_train, X_val
    
    def describe(self):

        """
        This method provides the descriptive statistics of the dataset.
        Same as pandas.DataFrame.describe()

        Returns:
            pandas.DataFrame
        """

        return self.filepath.describe()
    




# libraries
import numpy as np
import pandas as pd
from kymatio.numpy import Scattering1D
import logging
import sys

def transform_data(data, J, Q, verbose = 0):
    """transform_data 

    Used to create wavelet transform coefficient features for sound data

    Args:
        data (dict): training/test with file name (str) as key and sound data (nd.array) as value
        J (int): Wavelets per octave
        Q (int): Number of octaves
        verbose (int, optional): Level of logging desired (1 for DEBUG). Defaults to 0.

    returns:
        DataFrame: Name to who sound wave belongs with wavelet coefficient features
    """    

    # Adjust logging level based on verbosity
    if verbose == 0:
        logging.getLogger().setLevel(logging.CRITICAL)
    elif verbose == 1:
        logging.getLogger().setLevel(logging.INFO)

    wavelet_coefs = {}
    completion_index = 0


    for k,v in data.items():
    
        # normalising sound wave
        v = v / np.max(np.abs(v))

        # defining scattering object

        # defining length of sound wave
        T = v.shape[0]
        
        logging.info(f"defining scattering object for {k}...")
        scattering = Scattering1D(J, T, Q)
        meta = scattering.meta()

        # compute scattering transform
        logging.info(f"computing scattering transform coefs for {k}...")
        Sv = scattering(v)

        # obtaining order of coefficients
        order0 = np.where(meta['order'] == 0)
        order1 = np.where(meta['order'] == 1)
        order2 = np.where(meta['order'] == 2)

        logging.info(f"shape of 0th order for {k} is {Sv[order0].shape}")
        logging.info(f"shape of 1st order for {k} is {Sv[order1].shape}")
        logging.info(f"shape of 2nd order for {k} is {Sv[order2].shape}")

        # Averaging over time domain for each order of wavelets
        # reshaping to be a row vector
        pooled_coef0 = np.mean(Sv[order0], axis=1).reshape(1, -1)
        pooled_coef1 = np.mean(Sv[order1], axis=1).reshape(1, -1)
        pooled_coef2 = np.mean(Sv[order2], axis=1).reshape(1, -1)

        logging.info(f"shape of 0th order for {k} pooled is {pooled_coef0.shape}")
        logging.info(f"shape of 1st order for {k} pooled is {pooled_coef1.shape}")
        logging.info(f"shape of 2nd order for {k} pooled is {pooled_coef2.shape}")

        # horizontally stacking the vectors
        features = np.hstack((pooled_coef0, pooled_coef1, pooled_coef2))
        logging.info(f"shape of features is {features.shape}")

        # adding info to wavelet_coefs dict
        wavelet_coefs[k] = features.flatten()

        completion_index += 1




        sys.stdout.write(f"\rtransformation is {completion_index / len(data):.2%} complete")
        sys.stdout.flush()

    print("Returning data frame...")
    df = pd.DataFrame(wavelet_coefs)
    df = df.T.reset_index().rename(columns={"index": "name"})

    # obtaining only names in name column
    df["name"] = df["name"].apply(lambda x:x.split("_")[1])

    return df
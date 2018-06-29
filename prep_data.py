import numpy as np
import pandas as pd
from scipy import signal, io
from scipy.fftpack import fft, fftfreq, fftshift
from typing import Union, Optional


# %%
def butter_bandpass(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.lfilter(b, a, data)
    return y


def load_synced_data(loc: str, start: Optional[int] = 0) \
        -> pd.DataFrame:
    """
    Loads the data from specified location into a labeled pandas dataset

    :param start: first index to use for the dataset
    :param loc: location of file on disk
    :return:
    """
    data = io.loadmat(loc)['alldata'][start:]
    columns = ['time',
               'x_trc', 'y_trc', 'z_trc',
               'x_acc', 'y_acc', 'z_acc',
               'x_mag', 'y_mag', 'z_mag']
    return pd.DataFrame(data, columns=columns)


def make_target(all_data: pd.DataFrame) -> pd.DataFrame:
    """
    uses x,y and z locations from the tracker to create a velocity vector for use as
    the target for machine learning, also returns angles.

    :param all_data: Pandas dataframe containing both the tracker and vesper data
    :return: target: Pandas dataframe with the velocity vector and angles calculated based on the traacker
    """
    trc_loc = all_data[['x_trc', 'y_trc', 'z_trc']]
    trc_vel = trc_loc.diff()
    target = make_vec(trc_vel.x_trc, trc_vel.y_trc, trc_vel.z_trc)
    return target


def make_vec(x: Union[np.array, pd.DataFrame, pd.Series, list],
             y: Union[np.array, pd.DataFrame, pd.Series, list],
             z: Optional[Union[np.array, pd.DataFrame, pd.Series, list]] = None) \
        -> Union[pd.DataFrame, pd.Series]:
    """
    Calculates magnitude and angles(s) of vector based on readings from 2 or 3 dimensions

    :param x: measurements in one of 3 or 2 dimensions
    :param y: measurements in one of 3 or 2 dimensions
    :param z: measurements in one of 3 or 2 dimensions
    :return: combined dataframe of vector derived from all measurements: contains magnitude and angles
    """
    if z is None:
        magnitude = np.linalg.norm([x, y], axis=0)
        angle_xy = np.arctan(x / y)
        data = pd.DataFrame({'magnitude': magnitude,
                             'angle_xy': angle_xy})
    else:
        magnitude = np.linalg.norm([x, y, z], axis=0)
        angle_xy = np.arctan(x / y)
        angle_zxy = np.arctan(magnitude / z)
        data = pd.DataFrame({'magnitude': magnitude,
                             'angle_xy': angle_xy,
                             'angle_zxy': angle_zxy})
    return data


def make_features(all_data: pd.DataFrame) -> pd.DataFrame:
    """
    Generates features for use as input to machine learning algorithm

    :param all_data: Pandas dataframe containing both the tracker and vesper data
    :return:
    A pandas dataframe containing labeled features calculated from the vesper data
    """
    vesper_data = pd.DataFrame(butter_bandpass(
        all_data[['x_acc', 'y_acc', 'z_acc',
                  'x_mag', 'y_mag', 'z_mag']].T,
        1, 5, 50).T,
                               columns=['x_acc', 'y_acc', 'z_acc',
                                        'x_mag', 'y_mag', 'z_mag'])

    # Derive vectors from x,y,z data
    acc_vec = make_vec(*vesper_data[['x_acc', 'y_acc', 'z_acc']].values.T)
    mag_vec = make_vec(*vesper_data[['x_mag', 'y_mag', 'z_mag']].values.T)

    # Derive velocity from acceleration
    vel = all_data[['x_acc', 'y_acc', 'z_acc']].cumsum()
    vel_vec = make_vec(*vel.values.T)  # Derive velocity vector

    # Derive angular velocity from magnetometer reading
    angvel = mag_vec[['angle_xy', 'angle_zxy']].diff()
    angvel_vec = make_vec(*angvel.values.T)  # Derive angular velocity vector

    features = pd.DataFrame(
        {'x_acc': vesper_data.x_acc,
         'y_acc': vesper_data.y_acc,
         'z_acc': vesper_data.z_acc,
         'acc_vec_magnitude': acc_vec.magnitude,
         'acc_vec_angle_xy': acc_vec.angle_xy,
         'acc_vec_angle_zxy': acc_vec.angle_zxy,
         'x_vel': vel.x_acc,
         'y_vel': vel.y_acc,
         'z_vel': vel.z_acc,
         'vel_vec_magnitude': vel_vec.magnitude,
         'vel_vec_angle_xy': vel_vec.angle_xy,
         'vel_vec_angle_zxy': vel_vec.angle_zxy,
         'x_mag': vesper_data.x_mag,
         'y_mag': vesper_data.y_mag,
         'z_mag': vesper_data.z_mag,
         'mag_vec_magnitude': mag_vec.magnitude,
         'mag_vec_angle_xy': mag_vec.angle_xy,
         'mag_vec_angle_zxy': mag_vec.angle_zxy,
         'angvel_xy': angvel.angle_xy,
         'angvel_zxy': angvel.angle_zxy,
         'angvel_vec_magnitude': angvel_vec.magnitude,
         'angvel_vec_angle_xy': angvel_vec.angle_xy}
    )
    return features

def standardize(arr: Union[np.array, pd.Series, pd.DataFrame]) \
                -> object:
    """
    Z-Standardization of data

    :param arr: an array of any size
    :return:
    z-scored array
    """
    means = arr.mean()
    stds = arr.std(ddof=0)
    standardized_features = (arr - means) / stds
    return standardized_features
# %%
if __name__ == '__main__':
    data_loc = 'alldata.mat'
    start_ind = 800
    real_data = load_synced_data(data_loc, start_ind)
    real_target = make_target(real_data)
    real_features = make_features(real_data)
    real_std_features = standardize(real_features)

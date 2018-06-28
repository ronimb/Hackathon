import numpy as np
import pandas as pd
from scipy import signal, io
import pickle
import matplotlib.pyplot as plt
from typing import Union, Optional


# %%
class Data:
    pass


def load_synced_data(loc: str, start: Optional[int] = 0) \
        -> pd.DataFrame:
    """

    :param start:
    :param loc:
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

    :param all_data:
    :return:
    """
    trc_loc = all_data[['x_trc', 'y_trc', 'z_trc']]
    trc_vel = trc_loc.diff()
    target = make_vec(trc_vel.x_trc, trc_vel.y_trc, trc_vel.z_trc)
    return target


def make_vec(x: Union[np.array, pd.DataFrame, pd.Series, list],
             y: Union[np.array, pd.DataFrame, pd.Series, list],
             z: Optional[Union[np.array, pd.DataFrame, pd.Series, list]] = None) \
        -> Union[pd.DataFrame, pd.Series]:
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

    :param all_data:
    :return
    """
    vesper_data = pd.DataFrame(signal.savgol_filter(
        all_data[['x_acc', 'y_acc', 'z_acc',
          'x_mag', 'y_mag', 'z_mag']].T,
        window_length=31, polyorder=1).T,
                               columns=['x_acc', 'y_acc', 'z_acc',
          'x_mag', 'y_mag', 'z_mag'])

    # Derive vectors from x,y,z data
    acc_vec = make_vec(*vesper_data[['x_acc', 'y_acc', 'z_acc']].values.T)
    mag_vec = make_vec(*vesper_data[['x_mag', 'y_mag', 'z_mag']].values.T)

    # Derive velocity from acceleration
    vel = all_data[['x_acc', 'y_acc', 'z_acc']].cumsum()
    vel_vec = make_vec(*vel.values.T) # Derive velocity vector

    # Derive angular velocity from magnetometer reading
    angvel = mag_vec[['angle_xy', 'angle_zxy']].diff()
    angvel_vec = make_vec(*angvel.values.T) # Derive angular velocity vector

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

def standardize(features: Union[list, np.array, pd.Series, pd.DataFrame]) \
    -> object:
    """

    :param features:
    :return:
    """
    means = features.mean()
    stds = features.std(ddof=0)
    standardized_features = (features - means) / stds
    return standardized_features
# %%
if __name__ == '__main__':
    data_loc = r"E:\data\18-06-26_data_for_hackton\alldata.mat"
    start_ind = 800
    data = load_synced_data(data_loc, start_ind)
    target = make_target(data)
    features = make_features(data)
    std_features = standardize(make_features(data))
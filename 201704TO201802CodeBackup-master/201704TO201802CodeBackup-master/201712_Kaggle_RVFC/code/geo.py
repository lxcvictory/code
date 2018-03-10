#coding:utf-8
import math
import numpy as np

def cart2rho(x, y):
    rho = np.sqrt(x**2 + y**2)
    return rho


def cart2phi(x, y):
    phi = np.arctan2(y, x)
    return phi


def rotation_x(row, alpha):
    x = row['latitude']
    y = row['longitude']
    return x*math.cos(alpha) + y*math.sin(alpha)


def rotation_y(row, alpha):
    x = row['latitude']
    y = row['longitude']
    return y*math.cos(alpha) - x*math.sin(alpha)


def add_rotation(degrees, df):
    namex = "rot" + str(degrees) + "_X"
    namey = "rot" + str(degrees) + "_Y"

    df['num_' + namex] = df.apply(lambda row: rotation_x(row, math.pi/(180/degrees)), axis=1)
    df['num_' + namey] = df.apply(lambda row: rotation_y(row, math.pi/(180/degrees)), axis=1)

    return df

def operate_on_coordinates(tr_df, te_df):
    for df in [tr_df, te_df]:
        # polar coordinates system # Tokyo 35.6895° N, 139.6917° E
        df["num_rho_tokyo"] = df.apply(lambda x: cart2rho(x["latitude"] - 139.6917, x["longitude"]-35.6895), axis=1)
        df["num_phi_tokyo"] = df.apply(lambda x: cart2phi(x["latitude"] - 139.6917, x["longitude"]-35.6895), axis=1)
        #df["num_rho_osaka"] = df.apply(lambda x: cart2rho(x["latitude"] - 135.5022, x["longitude"]-34.6937), axis=1)
        #df["num_phi_osaka"] = df.apply(lambda x: cart2phi(x["latitude"] - 135.5022, x["longitude"]-34.6937), axis=1)
        #rotations
        for angle in [15,30,45,60]:
            df = add_rotation(angle, df)

    return tr_df, te_df


### 作法如下，套入你的train/test
# train, test = operate_on_coordinates(train, test) 即可
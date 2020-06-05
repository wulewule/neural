# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, csv
import tensorflow as tf
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
from sklearn.base import BaseEstimator,TransformerMixin,RegressorMixin,clone
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline,make_pipeline
from scipy.stats import skew
from sklearn.decomposition import PCA,KernelPCA
from sklearn.model_selection import cross_val_score,GridSearchCV,KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,ExtraTreesRegressor
from sklearn.svm import SVR,LinearSVR
from sklearn.linear_model import ElasticNet,SGDRegressor,BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor

'''
标签编码，对不连续的数字或者文本进行编号，转换成连续的数值型变量

如果使用TransformerMixin作为基类，则自动实现fit_transform()函数，fit_transform() <==> fit().transform()
如果添加BaseEstimator作为基类（注意此时__init__函数不能接受 ∗args *args∗args 和 ∗∗kwargs **kwargs∗∗kwargs），还可以使用两个额外的方法（get_params()和set_params()）
'''
class labelenc(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        lab = LabelEncoder()
        X['YearBuilt'] = lab.fit_transform(X['YearBuilt'])
        X['YearRemodAdd'] = lab.fit_transform(X['YearRemodAdd'])
        X['GarageYrBlt'] = lab.fit_transform(X['GarageYrBlt'])
        return X

'''
对数字特征进行数据平滑处理，增添虚拟变量
'''
class skew_dummies(BaseEstimator, TransformerMixin):
    def __init__(self, skew=0.5):
        self.skew = skew

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_numeric = X.select_dtypes(exclude=["object"]) # 剔除文字类型特征，pandas中存储字符串用object
        skewness = X_numeric.apply(lambda x:skew(x)) # 计算数字类型特征数据峰度
        skewness_feature = skewness[abs(skewness) >= self.skew].index # 提取峰度大于阈值的特征index
        X[skewness_feature] = np.log1p(X[skewness_feature]) # 对这些峰度较大的特征做数据平滑处理，使其更符合高斯分布
        X = pd.get_dummies(X) # 增添虚拟变量，本质上就是把所有特征转为数值变量，也就是增添one-hot特征
        return X

'''
特征组合
'''
class add_feature(BaseEstimator, TransformerMixin):
    def __init__(self, additional=1):
        self.additional = additional

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.additional==1:
            X["TotalHouse"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
            X["TotalArea"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"]

        else:
            X["TotalHouse"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
            X["TotalArea"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"]

            X["+_TotalHouse_OverallQual"] = X["TotalHouse"] * X["OverallQual"]
            X["+_GrLivArea_OverallQual"] = X["GrLivArea"] * X["OverallQual"]
            X["+_oMSZoning_TotalHouse"] = X["oMSZoning"] * X["TotalHouse"]
            X["+_oMSZoning_OverallQual"] = X["oMSZoning"] + X["OverallQual"]
            X["+_oMSZoning_YearBuilt"] = X["oMSZoning"] + X["YearBuilt"]
            X["+_oNeighborhood_TotalHouse"] = X["oNeighborhood"] * X["TotalHouse"]
            X["+_oNeighborhood_OverallQual"] = X["oNeighborhood"] + X["OverallQual"]
            X["+_oNeighborhood_YearBuilt"] = X["oNeighborhood"] + X["YearBuilt"]
            X["+_BsmtFinSF1_OverallQual"] = X["BsmtFinSF1"] * X["OverallQual"]

            X["-_oFunctional_TotalHouse"] = X["oFunctional"] * X["TotalHouse"]
            X["-_oFunctional_OverallQual"] = X["oFunctional"] + X["OverallQual"]
            X["-_LotArea_OverallQual"] = X["LotArea"] * X["OverallQual"]
            X["-_TotalHouse_LotArea"] = X["TotalHouse"] + X["LotArea"]
            X["-_oCondition1_TotalHouse"] = X["oCondition1"] * X["TotalHouse"]
            X["-_oCondition1_OverallQual"] = X["oCondition1"] + X["OverallQual"]


            X["Bsmt"] = X["BsmtFinSF1"] + X["BsmtFinSF2"] + X["BsmtUnfSF"]
            X["Rooms"] = X["FullBath"] + X["TotRmsAbvGrd"]
            X["PorchArea"] = X["OpenPorchSF"] + X["EnclosedPorch"] + X["3SsnPorch"] + X["ScreenPorch"]
            X["TotalPlace"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"] + X["OpenPorchSF"] + X["EnclosedPorch"] + X["3SsnPorch"] + X["ScreenPorch"]
        return X

def map_values(full):
    full["oMSSubClass"] = full.MSSubClass.map({'180':1,
                                        '30':2, '45':2,
                    '190':3, '50':3, '90':3,
                    '85':4, '40':4, '160':4,
                    '70':5, '20':5, '75':5, '80':5, '150':5,
                    '120': 6, '60':6})

    full["oMSZoning"] = full.MSZoning.map({'C (all)':1, 'RH':2, 'RM':2, 'RL':3, 'FV':4})

    full["oNeighborhood"] = full.Neighborhood.map({'MeadowV':1,
                       'IDOTRR':2, 'BrDale':2,
                       'OldTown':3, 'Edwards':3, 'BrkSide':3,
                       'Sawyer':4, 'Blueste':4, 'SWISU':4, 'NAmes':4,
                       'NPkVill':5, 'Mitchel':5,
                       'SawyerW':6, 'Gilbert':6, 'NWAmes':6,
                       'Blmngtn':7, 'CollgCr':7, 'ClearCr':7, 'Crawfor':7,
                       'Veenker':8, 'Somerst':8, 'Timber':8,
                       'StoneBr':9,
                       'NoRidge':10, 'NridgHt':10})

    full["oCondition1"] = full.Condition1.map({'Artery':1,
                     'Feedr':2, 'RRAe':2,
                     'Norm':3, 'RRAn':3,
                     'PosN':4, 'RRNe':4,
                     'PosA':5 ,'RRNn':5})

    full["oBldgType"] = full.BldgType.map({'2fmCon':1, 'Duplex':1, 'Twnhs':1, '1Fam':2, 'TwnhsE':2})

    full["oHouseStyle"] = full.HouseStyle.map({'1.5Unf':1,
                     '1.5Fin':2, '2.5Unf':2, 'SFoyer':2,
                     '1Story':3, 'SLvl':3,
                     '2Story':4, '2.5Fin':4})

    full["oExterior1st"] = full.Exterior1st.map({'BrkComm':1,
                      'AsphShn':2, 'CBlock':2, 'AsbShng':2,
                      'WdShing':3, 'Wd Sdng':3, 'MetalSd':3, 'Stucco':3, 'HdBoard':3,
                      'BrkFace':4, 'Plywood':4,
                      'VinylSd':5,
                      'CemntBd':6,
                      'Stone':7, 'ImStucc':7})

    full["oMasVnrType"] = full.MasVnrType.map({'BrkCmn':1, 'None':1, 'BrkFace':2, 'Stone':3})

    full["oExterQual"] = full.ExterQual.map({'Fa':1, 'TA':2, 'Gd':3, 'Ex':4})

    full["oFoundation"] = full.Foundation.map({'Slab':1,
                     'BrkTil':2, 'CBlock':2, 'Stone':2,
                     'Wood':3, 'PConc':4})

    full["oBsmtQual"] = full.BsmtQual.map({'Fa':2, 'None':1, 'TA':3, 'Gd':4, 'Ex':5})

    full["oBsmtExposure"] = full.BsmtExposure.map({'None':1, 'No':2, 'Av':3, 'Mn':3, 'Gd':4})

    full["oHeating"] = full.Heating.map({'Floor':1, 'Grav':1, 'Wall':2, 'OthW':3, 'GasW':4, 'GasA':5})

    full["oHeatingQC"] = full.HeatingQC.map({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})

    full["oKitchenQual"] = full.KitchenQual.map({'Fa':1, 'TA':2, 'Gd':3, 'Ex':4})

    full["oFunctional"] = full.Functional.map({'Maj2':1, 'Maj1':2, 'Min1':2, 'Min2':2, 'Mod':2, 'Sev':2, 'Typ':3})

    full["oFireplaceQu"] = full.FireplaceQu.map({'None':1, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})

    full["oGarageType"] = full.GarageType.map({'CarPort':1, 'None':1,
                                               'Detchd':2,
                                               '2Types':3, 'Basment':3,
                                               'Attchd':4, 'BuiltIn':5})

    full["oGarageFinish"] = full.GarageFinish.map({'None':1, 'Unf':2, 'RFn':3, 'Fin':4})

    full["oPavedDrive"] = full.PavedDrive.map({'N':1, 'P':2, 'Y':3})

    full["oSaleType"] = full.SaleType.map({'COD':1, 'ConLD':1, 'ConLI':1, 'ConLw':1, 'Oth':1, 'WD':1,
                   'CWD':2, 'Con':3, 'New':3})

    full["oSaleCondition"] = full.SaleCondition.map({'AdjLand':1, 'Abnorml':2, 'Alloca':2, 'Family':2, 'Normal':3, 'Partial':4})
    return full

def csv_write(data):
    f = open('result.csv', 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["Id", "SalePrice"])
    for i in range(len(data)):
        csv_writer.writerow([str(i+1461), data[i][0]])

def train():
    '''
    数据读取
    '''
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    train.drop(train[(train.GrLivArea>4000) & (train.SalePrice<300000)].index, inplace=True) # 剔除不现实的数据
    data = pd.concat([train, test], ignore_index=True) # 同时清洗训练数据和测试数据
    data.drop(['Id'], axis=1, inplace=True)
    print('origion data shape:', data.shape)
    
    '''
    数据清洗（填补NaN）
    '''
    # # 查看出现NaN的特征，并按出现次数排序
    # aa = data.isnull().sum()
    # aa = aa[aa>0].sort_values(ascending=False)

    # 用None填补，特征多为抽象
    N_list = ["PoolQC" , "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageQual", "GarageCond", "GarageFinish", 
              "GarageYrBlt", "GarageType","BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1", "MasVnrType"]
    for n in N_list:
        data[n].fillna("None", inplace=True)
    
    # 用0填补，特征多为数字
    Z_list=["MasVnrArea", "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "BsmtFinSF2", "BsmtFinSF1", "GarageArea"]
    for z in Z_list:
        data[z].fillna(0, inplace=True)

    # 其它特征用众数填补，[0]为取多个众数的第一个
    M_list = ["MSZoning", "BsmtHalfBath", "BsmtFullBath", "Utilities", "Functional", "Electrical", "KitchenQual", "SaleType","Exterior1st", "Exterior2nd"]
    for m in M_list:
        data[m].fillna(data[m].mode()[0], inplace=True)

    # 按照数字大小，十个为一组将特征划分，目的是防止下面取中位数后仍为NaN
    data['LotAreaCut'] = pd.qcut(data['LotArea'], 10)
    
    # 按照LotAreaCut和Neighborhood分组后的中位数进行填补NaN
    data['LotFrontage'] = data.groupby(['LotAreaCut', 'Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    data['LotFrontage'] = data.groupby(['LotAreaCut'])['LotFrontage'].transform(lambda x:x.fillna(x.median()))
 
    '''
    特征工程
    '''
    # 将一部分离散的数字特征转为字符串
    NumStr = ["MSSubClass", "BsmtFullBath", "BsmtHalfBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "MoSold", 
              "YrSold", "YearBuilt", "YearRemodAdd", "LowQualFinSF", "GarageYrBlt"]
    for col in NumStr:
        data[col] = data[col].astype(str)
    
    # 手动分类，用于处理不能直接用LabelEncoder的特征
    data.groupby(['MSSubClass'])[['SalePrice']].agg(['mean','median','count'])
    data = map_values(data)
    print('after map_value:', data.shape)
    
    # 删掉无作用的两个特征
    data.drop("LotAreaCut", axis=1, inplace=True)
    data.drop(['SalePrice'], axis=1, inplace=True)
    
    # 封装处理步骤，并处理数据 (1.标准化标签 2.数据平滑处理，增填虚拟变量)
    pipe = Pipeline([('labenc', labelenc()), ('skew_dummies', skew_dummies(skew=1))])
    copy = data.copy()
    data_pipe = pipe.fit_transform(copy)
    print('after pipe1:', data_pipe.shape)
    
    '''
    数据标准化
    '''
    scaler = RobustScaler() # 标准化：mean=0 std=1
    n_train = train.shape[0]
    x_train = data_pipe[:n_train] # 截取训练数据 
    x_test = data_pipe[n_train:] # 截取测试数据
    y_train = train.SalePrice # 获取训练标签，即房价

    x_train_scaled = scaler.fit(x_train).transform(x_train) # 根据训练数据获取标准化模型，并以此标准化训练数据
    x_test_scaled = scaler.transform(x_test) # 依照上面的标准化模型来标准化测试数据，不需要重新fit
    y_log = np.log(train.SalePrice)
    print('after normalizion', x_train_scaled.shape)

    '''
    绘制特征重要性，便于特征组合
    '''
    lasso = Lasso(alpha=0.001)
    lasso.fit(x_train_scaled, y_train)
    FI_lasso = pd.DataFrame({"Feature Importance":lasso.coef_}, index=data_pipe.columns)
    # print(FI_lasso.sort_values(['Feature Importance'], ascending=False))
    FI_lasso[FI_lasso["Feature Importance"]!=0].sort_values("Feature Importance").plot(kind='barh', figsize=(15, 25))
    plt.xticks(rotation=90)
    # plt.savefig('config.png')
    # plt.show()

    # 封装处理步骤，并处理数据 (1.标准化标签 2.特征组合 3.数据平滑处理，增填虚拟变量)
    # 据作者解释，两个pipeline的作用是两者间进行比对，证明第二个比第一个好
    pipe = Pipeline([
    ('labenc', labelenc()),
    ('add_feature', add_feature(additional=2)),
    ('skew_dummies', skew_dummies(skew=1))])

    data_pipe = pipe.fit_transform(data)
    print('after pipe2:', data_pipe.shape)
    x_train = data_pipe[:n_train]
    x_test = data_pipe[n_train:]
    y_train = train.SalePrice
    x_train_scaled = scaler.fit(x_train).transform(x_train)
    y_log = np.log(train.SalePrice)
    label = np.reshape(y_log.values, (-1, 1))
    x_test_scaled = scaler.transform(x_test)
    
    '''
    PCA
    '''
    pca = PCA(n_components=410)
    x_train_scaled = pca.fit_transform(x_train_scaled)
    x_test_scaled = pca.transform(x_test_scaled)
    print('after PCA:', x_train_scaled.shape, x_test_scaled.shape)
    
    '''
    训练部分
    '''
    train_x_data = x_train_scaled[:1200]
    test_x_data = x_train_scaled[1200:]
    train_y_data = label[:1200]
    test_y_data = label[1200:]
    
    # 网络搭建
    input_size = 410
    num_classes = 1
    weight_decay = 0.05 # 权重衰减系数
    hidden_units_size = 2*input_size + 1
    X = tf.placeholder(tf.float32, shape = [None, input_size], name='x')
    Y = tf.placeholder(tf.float32, shape = [None, num_classes], name='y')

    W1 = tf.get_variable("W1", shape=[input_size, hidden_units_size], initializer=tf.contrib.layers.xavier_initializer())
    B1 = tf.Variable(tf.constant (0.1), [hidden_units_size])
    W2 = tf.get_variable("W2", shape=[hidden_units_size, num_classes], initializer=tf.contrib.layers.xavier_initializer())
    B2 = tf.Variable(tf.constant (0.1), [num_classes])

    hidden_opt = tf.matmul(X, W1) + B1  # 输入层到隐藏层正向传播
    hidden_opt = tf.nn.relu(hidden_opt)  # 激活函数，用于计算节点输出值
    final_opt = tf.matmul(hidden_opt, W2) + B2  # 隐藏层到输出层正向传播
    tf.add_to_collection('pred_network', final_opt)

    loss = tf.reduce_mean(tf.losses.mean_squared_error(Y, final_opt))
    l2_loss = weight_decay * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])
    tf.summary.scalar('l2_loss', l2_loss)
    loss = loss + l2_loss

    train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    NEPOCH = 2000
    for i in range(NEPOCH):
        train_loss = sess.run([final_opt, train_step], feed_dict={X: train_x_data, Y: train_y_data})
        test_loss = sess.run(final_opt, feed_dict={X: test_x_data})
        if i%100 == 0:
            print('step:', i, 'train_mean_squared_error:', mean_squared_error(train_loss[0], train_y_data), 'test_mean_squared_error:', mean_squared_error(test_loss, test_y_data))
        if mean_squared_error(test_loss, test_y_data) < 0.01:
            break
    
    result = sess.run(final_opt, feed_dict={X: x_test_scaled})
    result = [np.exp(x) for x in result]
    csv_write(result)

train()

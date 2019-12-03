# OS:windows
# python3
# coding=utf-8
import os,re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def files_path(dir_name):
    """
    Reading all files' path in directory specified

    return:
    	files_path -- list of python
    
    """
    files_path = []
    for dirpath, _, filesname in os.walk(dir_name):
        for filename in filesname:
        	files_path += [dirpath + "/" + filename]
    return files_path


def data_filter(files_path, features, model):
    columns_specified = []
    for feature in features:
    	columns_specified += ["smart_{0}_raw".format(feature)]
    columns_specified = ["date", "model", "failure"] + columns_specified
    if not os.path.exists("data_preprocess/"):
    	os.makedirs("data_preprocess/")
    for path in files_path:
	t = re.split(r'/', path)
        datadf = pd.read_csv(path)
        data_model = datadf[datadf.model == model]
        data_model = data_model.set_index('serial_number')
        data_model = data_model[columns_specified]
        data_model.to_csv('data_preprocess/%s' % t[-1])


def creat_dataset(nday, features):
    filespath = files_path('data_preprocess/')
    columns_specified = []
    for feature in features:
    	columns_specified += ["smart_{0}_raw".format(feature)]
    columns_specified = ["serial_number", "date", "model", "failure"] + columns_specified
    sample_data = pd.DataFrame(columns=columns_specified)
    for i, path in enumerate(filespath[::-1]):
        df = pd.read_csv(path)

        Negative = df[df['failure'] == 1]
        Positive = df[df['failure'] == 0].sample(n=Negative.shape[0])
        sample_data = pd.concat([sample_data, Positive, Negative], axis=0, join='outer', sort=False)

        if i < len(filespath)-nday:
            for j in range(nday-1):
                df_next = pd.read_csv(filespath[i+j+1])
                for s_num in np.array(Negative['serial_number']):
                    df_next.loc[df_next.serial_number == s_num, 'failure'] = 1
                    Negative_next = df_next[df_next.serial_number == s_num]
                    Positive_next = df[df['failure'] == 0].sample(n=Negative_next.shape[0])
                    sample_data = pd.concat([sample_data, Negative_next, Positive_next],
                                            axis=0, join='outer', sort=False)
    sample_data.to_csv('dataset.csv')


def eval_knn(features, k, kfold=10, path_dataset="dataset.csv"):
    """
    Evaluating KNN model

    para:
        features -- input space features,list of python
        k -- hyperparameter of KNN,int
        kfold -- k-fold cross-validation,int

	return:
		res -- instance of EvalIndex,it includes four attribute:precission,recall,accuracy,f1_score
    """
    features_specified, roc_auc, precision, recall, acc, f1 = [[] for _ in range(6)]
    for feature in features:
        features_specified += ["smart_{0}_raw".format(feature)]
    df = pd.read_csv(path_dataset)
    df = df.fillna(0)
    
    X = df[features_specified]
    y = df['failure'].values
    # preprocessing for X
    X = np.log(X+1)

    # k-fold cross-validation
    for i in range(1, kfold+1):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=i)
	    
	    # KNN
        knn = KNN(n_neighbors=k, algorithm="auto", weights="distance")
	    # X with shape (m,n) y with shape (m,)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        # cm = confusion_matrix(y_test, y_pred)
        precision += [precision_score(y_test, y_pred, average='binary')]
        recall += [recall_score(y_test, y_pred, average='binary')]
        acc += [accuracy_score(y_test, y_pred)]
        f1 += [f1_score(y_test, y_pred, average='binary')]
	    
	    # ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        roc_auc += [auc(fpr, tpr)]
    roc_auc = sum(roc_auc) / kfold
    plt.title('Receiver Operating Characteristic(k=%d)'%k)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig("ROC.png", dpi=300)
    # plt.show()
    EvalIndex = namedtuple("EvalIndex", ["precision","recall","accuracy","f1_score"])
    res = EvalIndex(sum(precision)/kfold, sum(recall)/kfold, sum(acc)/kfold, sum(f1)/kfold)
    print("k={0}\nprecision:{1}\nrecall:{2}\naccuracy:{3}\nf1_score:{4}".format(k,res.precision,res.recall,res.accuracy,res.f1_score))
    return res


def main():
    # 1.特征选择，型号选择
    features = [5, 9, 187, 188, 193, 194, 197, 198, 241, 242]
    model = "ST4000DM000"
    # 2.数据过滤:过滤出指定特征和型号
    #filespath = files_path("data/")
    #data_filter(filespath, features, model)
    # 3.创建数据集：a.取raw数据b.故障回溯采正样本c.平衡数据集采负样本d.规范化输入
    #creat_dataset(nday=10, features=features)
    # 4.使用KNN模型预测硬盘故障并评估效果
    eval_knn(features=features,k=2)


if __name__ == "__main__":
    main()

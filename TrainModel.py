import pandas as pd
import os
import jieba
import re
import joblib
import shutil
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC 
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression

'''读取所需要用于训练的数据'''
def read_needed_data(path, filename):
    read_path = os.path.join(path, filename)
    df_file = pd.read_excel(read_path)
    # print(df_file.head())
    return df_file

'''导入自定义用户字典'''
def import_userdict(path, filename):
    read_path = os.path.join(path, filename)
    jieba.load_userdict(read_path)

'''对数据进行清洗,获得所需内容'''
def words_clean(word):
    word = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9,，。?!\s]+", '', word)
    return " ".join(jieba.lcut(word))

'''对数据进行空值处理以及分词'''
def split_words(df_file):
    comment = df_file
    comment = comment.fillna(value="missing")
    comment["cutted_comments"] = comment["comment2"].apply(words_clean)
    comment = comment[~comment["cutted_comments"].isin(["missing", ""])]
    # print(comment.head())
    # print(comment.shape)
    x=comment[["cutted_comments"]]
    y=comment.level
    # print(comment.head())
    # print(y.shape)
    # print(x.shape)
    return x,y

'''展示分割后的训练集和测试集'''
def test_split_train_test_datas(x_train, x_test, y_train, y_test):
    print('训练集：'+str(x_train.shape)+' '+str(y_train.shape))
    print('测试集：'+str(x_test.shape)+' '+str(y_test.shape))

'''获取用户字典'''
def get_custom_stopwords(path, filename):
    fpath = os.path.join(path, filename)
    with open(fpath, encoding="utf-8") as f:
        stopwords=f.read()
    stopwords_list=stopwords.split('\n')
    return stopwords_list

'''使用df特征进一步处理数据'''
def df_further_processing(x_train, stopwords, max_df=0.8, min_df=4):
    vect_df=CountVectorizer(max_df=max_df,min_df=min_df,stop_words=stopwords)
    term_matrix_df = pd.DataFrame(vect_df.fit_transform(x_train.cutted_comments).toarray(), columns=vect_df.get_feature_names_out())
    print('df方法进一步处理后的特征数量:\t'+str(term_matrix_df.shape))
    return vect_df, term_matrix_df

'''使用tfidf特征进一步处理数据'''
def tfidf_further_processing(x_train, stopwords, max_df=0.8, min_df=4):
    # max_df=0.8  # 在超过这一比例的文档中出现的关键词（过于平凡），去除掉。
    # min_df=4    # 在低于这一数量的文档中出现的关键词（过于独特），去除掉。
    vect_tfidf = TfidfVectorizer(max_df = max_df,\
                     min_df = min_df,\
                     token_pattern = r"(?u)\b[^\d\W]\w+\b",ngram_range=(1 ,1),stop_words=stopwords)
    term_matrix_tfidf2 = pd.DataFrame(vect_tfidf.fit_transform(x_train.cutted_comments).toarray(), columns=vect_tfidf.get_feature_names_out())
    print('tfidf方法进一步处理后的特征数量:\t'+str(term_matrix_tfidf2.shape))
    return vect_tfidf


'''训练数据'''
def Multi_Classify_Feature(type_feature, vect_feature, x_train, y_train, x_test, y_test):
    if "MultinomialNB" == type_feature:
        nb = MultinomialNB()
        pipe = make_pipeline(vect_feature, nb)
    elif "RandomForest" == type_feature:
        n_estimator_params = range(1, 100,5)
        for n_estimator in n_estimator_params:
            rf = RandomForestClassifier(n_estimators=n_estimator, n_jobs=-1, verbose=True)
        pipe = make_pipeline(vect_feature, rf)
    elif "SVM" == type_feature:
        LSVC = LinearSVC(max_iter=10000, dual="auto")
        pipe = make_pipeline(vect_feature, LSVC)
    elif "LogisticRegressionCV" == type_feature:
        lrvc = LogisticRegressionCV(Cs=[0.0001,0.005,0.001,0.05,0.01,0.1,0.5,1,10],scoring='accuracy',random_state=42,solver='saga',max_iter=10000,penalty='l2')
        pipe=make_pipeline(vect_feature,lrvc)
    elif "LogisticRegression" == type_feature:
        lr=LogisticRegression(solver='lbfgs',max_iter=10000,n_jobs=-1,verbose=True)
        pipe=make_pipeline(vect_feature,lr)
    else:
        print(f"{type_feature} Error!!!")
        print("type_feature Only \nMultinomialNB | RandomForest | \nSVM | LogisticRegressionCV | LogisticRegression")
        exit(1)
    clf = pipe.fit(x_train["cutted_comments"], y_train)
    y_pre = pipe.predict(x_test["cutted_comments"])
    cross_result = cross_val_score(pipe, x_train["cutted_comments"], y_train, cv=5, scoring="accuracy").mean()
    return y_pre, y_test, cross_result, clf

'''展示训练数据后的模型准确度和得分'''
def test_accuracy(type_feature, y_test, y_pre, cross_result):
    print(f'*************************{type_feature}*************************')
    print('交叉验证的准确率：'+str(cross_result))
    print(f"{type_feature}准确率测试")
    accuracy = metrics.accuracy_score(y_test, y_pre)
    precise = metrics.precision_score(y_test, y_pre, average="micro")
    recall = metrics.recall_score(y_test, y_pre, average="weighted")
    # f1_score: 评估分类模型的性能,它同时考虑了模型的精确率和召回率
    f1_score = metrics.f1_score(y_pre, y_test, average="micro")
    print("准确率: {0:.3f}".format(accuracy))
    print("精确率: {0:.3f}".format(precise))
    print("召回率: {0:.3f}".format(recall))
    print("f1_score: {0:.3f}".format(f1_score))
    print("每个类别的精确率和召回率: ")
    print(classification_report(y_test, y_pre))
    matrix = str(metrics.confusion_matrix(y_test, y_pre))
    print(f"混淆矩阵:\n {matrix}")

'''创建目标路径,若存在先删除再创建'''
def make_path(path_save):
    if os.path.exists(path_save):
        return
                                        # 判断文件夹是否存在
        # isexist=input("\n文件已存在,是否删除?(y/n): ")
        # if isexist == "y":
            # shutil.rmtree(path_save)    # 删除文件夹
        # else:
            # return
    os.mkdir(path_save)                 # 创建文件夹

'''保存模型'''
def save_clf(clf, path_clf="./", clfname="clf.pkl"):
    make_path(path_clf)
    save_path = os.path.join(path_clf, clfname)
    joblib.dump(clf, save_path)

'''加载模型'''
def load_clf(path_clf="./", clfname="clf.pkl"):
    load_path = os.path.join(path_clf, clfname)
    clf = joblib.load(load_path)
    return clf

'''加载模型对数据做预测'''
def predict_clf(path_clf="./", clfname="clf.pkl", word="test"):
    clf = load_clf(path_clf, clfname)
    word = [words_clean(word)]
    # print(word)
    ZeroOrOne = clf.predict(word)[0]
    prob = clf.predict_proba(word)[:]
    prob = sorted(prob)
    # print(prob)
    # print(ZeroOrOne)
    prob = prob[0]
    return ZeroOrOne, prob

def menu():
    print("==============train_model steps================")
    print("初始化(导入用户字典、导入停用词、读取所需数据、分词、拆分数据集)")
    print("进行特征降维")
    print("\t分类方法:MultinomialNB | RandomForest | SVM | LogisticRegressionCV | LogisticRegression")
    print("\t特征选择:df | tfidf")
    print("保存模型")
    print("模型预测")


def Main_train_model(filepath, filename, usrdict, stopwordsname):
    menu()
    '''=========初始化(导入用户字典、导入停用词、读取所需数据、分词、拆分数据集)======='''
    import_userdict(filepath, usrdict)
    stopwords = get_custom_stopwords(filepath, stopwordsname)
    df_file = read_needed_data(filepath, filename)
    x, y = split_words(df_file)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state=1)
    test_split_train_test_datas(x_train, x_test, y_train, y_test)
    print("\tMultinomialNB | RandomForest | SVM | LogisticRegressionCV | LogisticRegression")
    type_classifiy = input("input the type of classification: (default is RandomForest,press <Enter>) ")
    if not type_classifiy:
        type_classifiy = "RandomForest"
    print("\t\ndf | tfidf")
    type_feature = input("input the type of feature:(default is tfidf,press <Enter>) ")
    '''=======================特征降维=========================='''
    if type_feature == "tfidf" or not type_feature:
        type_feature = "tfidf"
        vect_tfidf = tfidf_further_processing(x_train, stopwords)
        vect_feature = vect_tfidf
    if type_feature == "df":
        vect_df, term_matrix_df = df_further_processing(x_train, stopwords, max_df=0.8, min_df=4)
        vect_feature = vect_df
    '''========================================================'''

    y_pre, y_test, cross_result, clf = Multi_Classify_Feature(type_classifiy, vect_feature, x_train, y_train, x_test, y_test)
    test_accuracy(f"{type_classifiy}_{type_feature}", y_test, y_pre, cross_result)

    '''============================保存模型=============================='''
    try:
        save_clf(path_clf="./Model/", clfname=f"clf_{type_classifiy}_{type_feature}.pkl" ,clf=clf)
    except Exception as e:
        print("Error: ", e)


def apply_model():
    print("\tMultinomialNB | RandomForest | SVM | LogisticRegressionCV | LogisticRegression")
    type_classifiy = input("input the type of classification: (default is RandomForest,press <Enter>) ")
    if not type_classifiy:
        type_classifiy = "RandomForest"
    print("\t\ndf | tfidf")
    type_feature = input("input the type of feature:(default is tfidf,press <Enter>) ")
    if not type_feature:
        type_feature = "tfidf"
    '''=======================模型预测=========================='''
    try:
        # 你真的是太棒了
        word = input("input the word you want to predict (you can use 你真的是太棒了): ")
        ZeroOrOne, prob = predict_clf(path_clf="./Model/", clfname=f"clf_{type_classifiy}_{type_feature}.pkl", word=word)
        print(f"为积极评论的概率为{prob[1]}")
        print(f"为消极评论的概率为{prob[0]}")
        print("\nFinal Result: ")
        if ZeroOrOne==1:
            print("积极评论,概率为%f"%(prob[1]))
        else:
            print("消极评论,概率为%f"%(prob[0]))
    except Exception as e:
        print("Error: ", e)

def train():
    filepath = "./goal_dir"
    filename = "MotionAnalysisData.xlsx"
    usrdict = "userdict.txt"
    stopwordsname = "stopwords.dict"
    Main_train_model(filepath, filename, usrdict, stopwordsname)
 

if __name__ == '__main__':
    print("Press n or N to quit")
    while True:
        starttime = input("Do you want to start Training model now? (y/n) (default is y,press <Enter>): ")
        if starttime == "n" or starttime == "N":
            break
        if starttime == "y" or starttime == "Y":
            train()
            break
        if not starttime:
            train()
            break
        else:
            print("wrong and input again!!!")
    while True:
        starttime = input("Do you want to start to apply or test model now? (y/n) (default is y,press <Enter>): ")
        if starttime == "n" or starttime == "N":
            break
        if starttime == "y" or starttime == "Y":
            apply_model()
            break
        if not starttime:
            apply_model()
            break
        else:
            print("wrong and input again!!!")
    
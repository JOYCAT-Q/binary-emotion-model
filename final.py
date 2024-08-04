from DataGet import get
from DataClean import clean
from TrainModel import train
from TrainModel import apply_model
from DataVision import vision
import os

if __name__ == '__main__':
    # 一、获取数据
    #     数据来源于豆瓣top电影排行榜评论以及对应的标签
    # 二、数据清洗
    #     对爬取的数据进行清洗，进行特征降维、去除重复以及空值，并保存至excel表中，方便后续使用
    # 三、模型训练
    #     读取保存的数据，并对齐正向与负向数据，使其数量一致，使用随机森林以及TF-IDF特征对数据集进行训练，并保存后续使用
    #     （同时提供
    #         MultinomialNB | RandomForest | SVM | LogisticRegressionCV | LogisticRegression    分类方法
    #         df | tfidf    特征
    #     供选择）
    # 四、模型性能评估
    #     使用交叉验证准确率，准确率、精确率、召回率、f1_score对模型进行评估
    # 五、加载模型进行应用
    #     对训练好的模型加载使用
    envir = input("是否需要补全环境?(y/n) (default is n, press <Enter>): ")
    if not envir:
        print("\n默认不补全环境")
    else:
        os.system("pip install -r packages.txt -i https://pypi.tuna.tsinghua.edu.cn/simple")
    starttime = input("Do you want to start now? (y/n) (default is y,press <Enter>): ")
    if starttime == "n" or starttime == "N":
        print("Thanks for using")
        exit(0)
    if not starttime:
        # 获取数据(若不需获取数据而使用已爬取好的数据,请自行注释掉)
        get()
        clean()
        # 模型性能评估在模型训练过程中给出
        train()
        # 数据可视化,包括词云图等,默认注释掉
        # vision()
        apply_model()
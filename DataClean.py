import zhconv
import os
from opencc import OpenCC
import pandas as pd
import json
import re

'''将所有获取的txt文本文件融合'''
def txt_merge(path, saveFile):
    ori_dirs = os.listdir(path)
    for dir in ori_dirs:
        newDir = os.path.join(path, dir)
        if not os.path.isfile(newDir):
            txt_merge(newDir, saveFile)
        elif os.path.splitext(newDir)[1]==".txt":
            copy_file(newDir, saveFile)
        else:
            print(f"{newDir} \t既不是txt,也不是文档")


'''将一个txt文件内容写入另一个文件(可增加正则化以及繁简转换)'''
def copy_file(path, saveFile):
    with open(path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            saveFile.write(line)
            # saveFile.write(t2s(line))


'''实现txt文件融合的主函数'''
def main_merge_txt(path, filename):
    path_file = os.path.join(path, filename)
    saveFile = open(path_file, "w", encoding="utf-8")
    txt_merge(path, saveFile)
    saveFile.close()
    print("TXT聚合结束!!!!")

'''删除目标文件夹下的已存在文件'''
def remove_file_error(save_path, file):
    if os.path.exists(os.path.join(save_path,file)):
        try:
            os.remove(os.path.join(save_path,file))
            print('Deleted file:', os.path.join(save_path,file)) 
        except OSError as e:
            print('Deleted file Error:', os.path.join(save_path,file))
            print('Error:', e.strerror) 
            return False
    else:
        print("NO such file like: {}{}".format(save_path, file))

'''繁简转换,生成一个新文件存放内容'''
def CopyAndConvert2ZHtw(ori_path, filename, convreted_path, converted_filename):
    file_path = os.path.join(ori_path, filename)
    goal_path = os.path.join(convreted_path, converted_filename)
    with open(file_path, 'r', encoding="utf-8") as ori_f:
        contents = ori_f.read()
        with open(goal_path, 'w', encoding="utf-8") as final_f:
            final_f.write(zhconv.convert(contents, "zh-hans"))
    print("繁简转换完成!!!!!")


def t2s(text):
    """
    繁体转简体
    """
    output_text = OpenCC("t2s").convert(text)
    return output_text

'''处理合并过后的数据,转换成DataFrame对象'''
def process_merge_data(path, filename):
    rows = []
    goal_path = os.path.join(path, filename)
    with open(goal_path, "r", encoding="utf-8") as f:
        contents = f.readlines()
    for line in contents:
        rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    return df


'''展示处理成DataFrame对象后的数据情况'''
def test_processed_data(df):
    # df = process_merge_data(path, filename)
    print("数据维度情况: ============================")
    print(df.shape)
    print("每列存在空值情况: ========================")
    print(df.isnull().sum())
    print("列名: ===================================")
    print(df.columns.values)
    print("前五行数据: ==============================")
    print(df.head())

'''将合并后的数据按标签分开重新合并'''
def split_Xls(path, filename):
    df = process_merge_data(path, filename)
    df1 = df[~df["url"].isnull()].dropna(axis=1, how="all")
    df2 = df[~df["comment"].isnull()].dropna(axis=1, how="all")
    df3 = df[~df["type"].isnull()].dropna(axis=1, how="all")
    df_merge1 = pd.merge(df1, df2, on="id")
    df_merge2 = pd.merge(df_merge1, df3, on="id")
    # print("每列存在空值情况: ========================")
    # print(df_merge2.isnull().sum())
    return df_merge2

'''处理重新合并后的数据,去除内容长度为0的行'''
def process_split_Xls(path, filename):
    df = split_Xls(path, filename)
    df["author_len"] = df["author"].apply(lambda x: len(x))
    df["comment_len"] = df["comment"].apply(lambda x:len(x))
    # print(df[df["comment_len"]==0])
    # print(df[df["author_len"]==0])
    # print(df.shape)
    df = df[~df["author_len"].isin([0])]
    df = df[~df["comment_len"].isin([0])]
    # print(df[df["comment_len"]==0])
    # print(df[df["author_len"]==0])
    # print(df.shape)
    print("去除author, comment为0的行 结束!!!")
    return df

'''去除重复行'''
def remove_same_lines(df):
    #提取出list为string
    df["author2"]=df["author"].apply(lambda x:x[0])
    df["comment2"]=df["comment"].apply(lambda x:x[0])
    if df.duplicated(subset=["comment2", "title_x"], keep=False).any():
        print("存在重复数据!!!")
        df = df.drop_duplicates(subset=["comment2", "title_x"], keep="last")
    else:
        print("不存在重复数据!!!")
    # print(df.shape)
    return df

'''去除应用于统计内容长度以及多余的列'''
def remove_extra_columns(df):
    df.drop(["author","author_len","comment","comment_len",'title_y'],axis=1,inplace=True)
    print(df.shape)
    return df

'''将DataFrame对象保存至excel中'''
def Save2Excel(path_df, filename, df):
    saveFile = os.path.join(path_df, filename)
    df = df.fillna(value="missing")
    df["comment2"] = df["comment2"].apply(words_clean)
    df = df[~df["comment2"].isin(["missing", ""])]
    # print(saveFile)
    if df is None:
        print("df参数为空")
        return False
    df.to_excel(saveFile, index=False)


'''制作自定义的用户以及标签字典'''
def get_DIY_dict(df):
    info = df[["title_x", "actors", "directors"]]
    titles = list(info["title_x"].unique())
    titles = [title for title in titles if len(title)>=1]
    directors = list(info["directors"].unique())
    rows = df.shape[0]
    actors =[]
    for i in range(rows):
        actors.extend(info.iloc[i, 1])
    actors = set(actors)

    return actors, directors, titles


'''将获取到的用户以及标签数据保存至dict中'''
def Save2Dict(path_file, filename1, filename2, actors, directors, titles):
    
    saveFile1 = os.path.join(path_file, filename1)
    saveFile2 = os.path.join(path_file, filename2)
    f1 = open(saveFile1, "w", encoding="utf-8")
    f2 = open(saveFile2, "w", encoding="utf-8")
    for i in actors:
        if not i.strip():
            continue
        f1.write(i+ " nr\n")
        f2.write(i+ " actor\n")
    for i in directors:
        if not i.strip():
            continue
        f1.write(i+ " nr\n")
        f2.write(i+ " director\n")
    for i in titles:
        if not i.strip():
            continue
        f1.write(i+ " n\n")
        f2.write(i+ " movie\n")
    f1.close()
    f2.close()

'''展示标签重新分类前后对比'''
def get_result_classifiy(df_before, df_after):
    print('*******************替换前各类得分数量*******************************')
    print(str(df_before["score"].value_counts()))
    print('*******************替换后正负评价数量*******************************')
    print(str(df_after['level'].value_counts()))

'''对标签重新分类'''
def reclassifiy_tags(df):
    df_after = df[["score", "comment2"]]
    pd.set_option("mode.chained_assignment", None)
    df_after["level"] = df_after["score"].apply(lambda x:0 if x=="较差" else 0 if x=="很差" else 1)
    df_after.drop(["score"],axis=1,inplace=True)
    df_after = df_after[["level", "comment2"]]
    return df_after

'''平衡不同标签的数据,方便训练模型'''
def balance_classifiy(df):
    tmp = df[df['level']==0]
    scale = (len(df[df['level']==1])/len(df[df['level']==0]))
    add=tmp[:(len(df[df['level']==1])-len(df[df['level']==0])*int(scale))]
    scale_round = round(scale, 0)
    if scale > scale_round:
        pass
    else:
        scale_round = scale_round - 1
    for _ in range(int(scale_round-1)):
        df = pd.concat([df, tmp], ignore_index=True)
    
    df = pd.concat([df, add], ignore_index=True)
    return df

'''获取平衡数据后的结果'''
def get_balance_classifiy(df):
    print('*******************数据平衡后正负评价数量*******************************')
    print(str(df['level'].value_counts()))

'''对数据清洗,得到所需内容'''
def words_clean(word):
    # word = re.sub(r"[*=+&#@~]+", '', word)
    word = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9,，。?!\s]+", '', word)
    # print(word)
    return word

def Main_data_clean(original_filename, original_path, converted_filename, goal_path, dict_filename1, dict_filename2):
    remove_file_error(original_path, original_filename)
    remove_file_error(goal_path, converted_filename)
    main_merge_txt(original_path, original_filename)
    CopyAndConvert2ZHtw(ori_path=original_path, filename=original_filename, \
                        convreted_path=goal_path, converted_filename=converted_filename)
    df = process_merge_data(goal_path, converted_filename)
    test_processed_data(df)
    df = process_split_Xls(goal_path, converted_filename)
    test_processed_data(df)
    df = remove_same_lines(df)
    df = remove_extra_columns(df)
    actors, directors, titles = get_DIY_dict(df)
    Save2Dict(goal_path, dict_filename1, dict_filename2, actors, directors, titles)
    Save2Excel(goal_path, "finaldatas.xlsx", df)
    df_after = reclassifiy_tags(df)
    get_result_classifiy(df, df_after)
    df = balance_classifiy(df_after)
    get_balance_classifiy(df)
    Save2Excel(goal_path, "MotionAnalysisData.xlsx", df)
    
def clean():
    original_filename = "AllData.txt"
    original_path = "./Data_Get/"
    goal_path = "./goal_dir/"
    converted_filename = "convertData.txt"
    dict_filename1="userdict.txt"
    dict_filename2="tagdict.txt"


    Main_data_clean(original_filename, original_path, converted_filename, goal_path, dict_filename1, dict_filename2)



if __name__ == '__main__':
    print("Press n or N to quit")
    while True:
        starttime = input("Do you want to start datas cleaning now? (y/n) (default is y,press <Enter>): ")
        if starttime == "n" or starttime == "N":
            break
        if starttime == "y" or starttime == "Y":
            clean()
            break
        if not starttime:
            clean()
            break
        else:
            print("wrong and input again!!!")
    



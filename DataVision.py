import os
import shutil
import jieba
import pandas as pd
import numpy as np
from PIL import Image
from wordcloud import *
from jieba.analyse import *
import matplotlib.pyplot as plt
from wordcloud import WordCloud,ImageColorGenerator

'''获取所需要可视化数据'''
def read_needed_data(path, filename):
    read_path = os.path.join(path, filename)
    df_file = pd.read_excel(read_path)
    return df_file

'''创建目标文件夹路径'''
def make_path(path_save):
    if os.path.exists(path_save):       # 判断文件夹是否存在
        print(f"已存在  {path_save}   文件夹")
        shutil.rmtree(path_save)        # 删除文件夹
        print(f"正在删除  {path_save}   文件夹")
    os.mkdir(path_save)                 # 创建文件夹


'''对所需数据进行截取,去重操作'''
def process_needed_files(df_file):
    print("正在获取所需数据 ~~~~")
    word = df_file[["tag", "comment2"]]
    finaltag = word["tag"].drop_duplicates().values.tolist()
    return finaltag, word

'''将所有评论数据按类别导出,方便后续按类别可视化'''
def export_comments(goalpath, word, finaltag):
    make_path(goalpath)
    word["comment2"].to_csv(f"{goalpath}/allcomments.txt", encoding="utf-8", index=False)
    for i in range(len(finaltag)):
        print(f"正在导出    {finaltag[i]}   评论数据  ~~~~")
        tag_comment = word[word["tag"]==finaltag[i]]
        tag_comment["comment2"].to_csv(f"{goalpath}/{finaltag[i]}.txt", encoding="utf-8", index=False)


'''获取常用停用词'''
def get_custom_stopwords(path, filename):
    fpath = os.path.join(path, filename)
    with open(fpath, encoding="utf-8") as f:
        stopwords=f.read()
    stopwords_list=stopwords.split('\n')
    return stopwords_list

'''导入数据清洗得到的自定义用户词典'''
def import_userdict(path, userdict):
    read_path = os.path.join(path, userdict)
    jieba.load_userdict(read_path)

'''生成词云'''
def genarate_wordcloud(text, stopwords):
    file_in = open(text, 'r',encoding='utf-8')
    content = file_in.read()
    content = content.replace('电影', "")
    content = content.replace('故事', "")
    content = content.replace('剧情', "")
    content = content.replace('影片', "")
    content = content.replace('本片', "")

    allow_pos=('n','nr') # allowPOS是选择提取的词性，n是名词
    tags = jieba.analyse.extract_tags(content, topK=1000, withWeight=False,allowPOS= allow_pos)
    wl = ' '.join(tags)

    image= Image.open('./goal_dir/bg.jpg')
    graph = np.array(image)
    wc = WordCloud(background_color = None, mode='RGBA',  # 设置背景颜色
                   max_words = 1000,                      # 设置最大显示的字数
                   prefer_horizontal=1,                   # 设置横向文字比例，1为全横
                   font_path = r"./goal_dir/wryh.ttf", 
                   # 设置中文字体，使得词云可以显示（词云默认字体是“DroidSansMono.ttf字体库”，不支持中文）
                   max_font_size = 100,min_font_size= 50,  # 设置字体最大值/最小值
                   random_state = 60,                      # 设置有多少种随机生成状态，即有多少种配色方案
                   width=160,height=90,                    # 设置画布高度、宽度
                   scale=1,                                #  值越高，越清晰，速度越慢
                   mask=graph,                             # 设置词云形状
                   stopwords=stopwords)

    myword = wc.generate(wl)
    image_color = ImageColorGenerator(graph)
    plt.imshow(myword.recolor(color_func=image_color))
    wc.to_file(text.split(".txt")[0]+'.png')


'''遍历文件夹,获取分类的评论数据'''
def eachFile(filepath, stopwords):
    print("正在遍历文件夹   {}  …………".format(filepath))
    pathDir = os.listdir(filepath)          # 获取当前路径下的文件名，返回List
    for s in pathDir:
        newDir=os.path.join(filepath,s)     # 将文件命加入到当前文件路径后面
        if os.path.isfile(newDir) :         # 如果是文件
            if os.path.splitext(newDir)[1]==".txt":  # 判断是否是txt
                print(f"正在生成    {os.path.splitext(newDir)[0]}   词云  ~~~~")
                # print(newDir)
                genarate_wordcloud(newDir, stopwords)
        else:
            eachFile(newDir, stopwords)     # 如果不是文件，递归这个文件夹的路径
    print(f'词云生成完成,请前往 {filepath}  查看!!!')

'''主函数,对以上函数按顺序调用'''
def Mainfunction(path, goalpath, filename, usrdict, dictspath, stopwordsname):
    # path:             所需数据路径 
    # goalpath:         目标文件存放路径
    # filename:         所需文件名称
    # usrdict:          用户字典名称
    # dictspath:        字典所在路径
    # stopwordsname:    停用词文件名称

    # 导入用户词典
    import_userdict(dictspath, usrdict)
    print("数据可视化Starting   ~~~~")
    # 读取所需数据
    df_file = read_needed_data(path, filename)
    # 处理数据
    finaltag, word = process_needed_files(df_file)
    # print(word["tag"].value_counts())
    # 导出评论数据,方便后续可视化
    export_comments(goalpath, word, finaltag)
    # 获取停用词
    stopwords = get_custom_stopwords(dictspath, stopwordsname)
    # 遍历文件,绘制词云
    eachFile(goalpath, stopwords)
    print("\n数据可视化完成!!!")

def vision():
    path = "./goal_dir/"
    goalpath = "./Data_Vision/"
    filename = "finaldatas.xlsx"
    usrdict = "userdict.txt"
    stopwordsname = "stopwords.dict"
    Mainfunction(path, goalpath, filename, usrdict, path, stopwordsname)

if __name__ == "__main__":
    print("Press n or N to quit")
    while True:
        starttime = input("Do you want to start datas Vision now? (y/n) (default is y,press <Enter>): ")
        if starttime == "n" or starttime == "N":
            break
        if starttime == "y" or starttime == "Y":
            vision()
            break
        if not starttime:
            vision()
            break
        else:
            print("wrong and input again!!!")
    
    
    
    
    
    
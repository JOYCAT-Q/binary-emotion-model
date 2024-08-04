import requests
import json
from bs4 import BeautifulSoup
import os
import time
import random


'''创建目标文件存放文件夹'''
def make_dirs(types_id):
    for key in types_id.keys():
        save_path = "./Data_Get/{}/".format(key)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(os.path.join(save_path, "comments/")):
            os.makedirs(os.path.join(save_path, "comments/"))
        if not os.path.exists(os.path.join(save_path, "movie_details/")):
            os.makedirs(os.path.join(save_path, "movie_details/"))

'''随机睡眠,减少触发反爬机制'''
def random_sleep(start=1, end=5):
    time_int = random.randint(start, end)
    time.sleep(time_int)

'''删除目标路径下已存在的目标文件'''
def remove_file_error(key, file):
    save_path = "./data/{}/".format(key)
    if os.path.exists(os.path.join(save_path,file)):
        try:
            os.remove(os.path.join(save_path,file))
            print('Deleted file:', os.path.join(save_path,file)) 
        except OSError as e:
            print('Deleted file Error:', os.path.join(save_path,file))
            print('Error:', e.strerror) 
            return False

'''获取每个类别的前20部电影'''
def get_top20_info(types_id, headers):
    for key, n in types_id.items():
        url = "https://movie.douban.com/j/chart/top_list?type="+str(n[0])+"&interval_id=100%3A90&action=&start=0&limit=20"
        random_sleep()
        respose_data = requests.get(url, headers = headers)
        if respose_data.status_code != 200:
            print(f"{key}_link lost")
            print("请刷新页面(60s内)!!!")
            time.sleep(60)
            respose_data = requests.get(url, headers = headers)
            if respose_data.status_code == 200:
                continue
            else:
                print("未刷新页面,即将退出!!!!")
                return False
        else:
            print(f"{key}_movies_toplist_info正在获取…………")

        if respose_data.text:
            json_data = respose_data.json()
        else:
            print("NO respose_data")
            continue


        id = []
        title =[]
        save_path = "./data/{}/".format(key)
        
        remove_file_error(key, file=(key + "movies_info" + '.txt'))

        if json_data and not os.path.exists(os.path.join(save_path,(key + "movies_info" + '.txt'))):
            file_movies = open(os.path.join(save_path,(key + "movies_info" + '.txt')),'w',encoding = 'utf-8')
            for i in range(len(json_data)):
                # print(json_data[i])
                # print(json_data[i]["rating"])
                data = {
                    'title':json_data[i]['title'],
                    'rate':json_data[i]['rating'][0],
                    'url':json_data[i]['url'],
                    'id':json_data[i]['id'],
                    'tag':"{}".format(key)
                }
                id.append(data['id'])
                title.append(data['title'])
                file_movies.write(json.dumps(data,ensure_ascii = False)  + "\n")
            file_movies.close()

'''获取每个类别对应的ID和Title'''
def get_id_title(key):
    id_title = []
    save_path = "./data/{}/".format(key)
    if os.path.exists(os.path.join(save_path,(key + "movies_info" + '.txt'))):
        with open(os.path.join(save_path,(key + "movies_info" + '.txt')),'r',encoding = 'utf-8') as f:
            movies_info = f.readlines()
            for info in movies_info:
                info = json.loads(info)
                id_title.append(((info["id"], info["title"])))
            return id_title
    else:
        print("no {}_movies_info".format(key))
        return False

'''获取每个电影的详细信息'''
def get_movies_details(types_id, headers):
    for key, n in types_id.items():
        save_path = "./data/{}/".format(key)
        id_title = get_id_title(key)
        for d, title in id_title:
            url ='https://movie.douban.com/subject/'+ str(d) + '/comments?start=0&limit=20&sort=new_score&status=P'
            random_sleep(start=2, end=6)
            wb_data = requests.get(url, headers = headers)
            if  wb_data.status_code != 200:
                print(f"{title}_movie_detail_link lost")
                print("请刷新页面(60s内)!!!")
                time.sleep(60)
                wb_data = requests.get(url, headers = headers)
                if  wb_data.status_code == 200:
                    continue
                else:
                    print("未刷新页面,即将退出!!!!")
                    return False
            else:
                print(f"{title}__movie_detail正在获取…………")

            soup = BeautifulSoup(wb_data.text,"lxml")

            titles = soup.select("h1")
            directors = soup.select("span > p:nth-child(1) > a")
            actors1 = soup.select("span > p:nth-child(2) >a ")
            actors = [i.get_text() for i in actors1]
            types = soup.select("span > p:nth-child(3)")
    
            movie_detail = {
                            "id": str(d),
                            "title":titles[0].get_text()[:-3],
                            "actors":actors,
                            "directors":directors[0].get_text(),
                            "type":types[0].get_text().split(":")[-1].strip()
                        }
            filename = movie_detail['title']
            remove_file_error(key, ("movie_details/" + filename + "__movie_detail " + '.txt'))
            with open(os.path.join(save_path,("movie_details/" + filename + "__movie_detail " + '.txt')),'w',encoding = 'utf-8') as f:
                # print("test")
                f.write(json.dumps(movie_detail,ensure_ascii = False)  + "\n")

'''获取每个电影的评论数据'''
def get_movies_comments(types_id, headers):
    for key, n in types_id.items():
        save_path = "./data/{}/".format(key)
        id_title = get_id_title(key)
        for (m, title) in id_title:
            f = open( os.path.join(save_path,("comments/" + str(title) + '.txt')),'w',encoding = 'utf-8')
            urls = ['https://movie.douban.com/subject/'+ str(m) + '/comments?start=' +str(n) + '&limit=20&sort=new_score&status=P' for n in range(0,2000,20)]
            for url in urls: 
                random_sleep(2, 5)
                wb_data = requests.get(url,headers = headers)
                # print(wb_data)
                if wb_data.status_code != 200:
                    print(f"{title}_comments_link lost")
                    print("请刷新页面(60s内)!!!")
                    time.sleep(60)
                    wb_data = requests.get(url,headers = headers)
                    if wb_data.status_code == 200:
                        continue
                    else:
                        print("未刷新页面,即将退出!!!!")
                        return False
                else:
                    print(f"{title}_comments正在获取…………")
                soup = BeautifulSoup(wb_data.text,"lxml")

                authors = soup.select('span.comment-info > a')
                votes = soup.select("span.comment-vote > span")
                times = soup.select('span.comment-time')
                comments = soup.select('span.short')
                scores = soup.select("span.rating")

                for author,vote,time_,comment,score in zip (authors,votes,times,comments,scores):
                    data1 = {
                        "id":str(m),
                        "author":list(author.stripped_strings),
                        "vote":vote.get_text(),
                        "times":time_.get_text().strip(),
                        "comment":list(comment.stripped_strings),
                        "score":score.get("title")
                    }   
           
                    f.write(json.dumps(data1,ensure_ascii = False)  + "\n")
            f.close()

def Main_get_data(types_id, headers):
    print("开始获取数据***********")
    # 创建目标文件存放文件夹
    make_dirs(types_id)
    # 获取不同类型电影各自前20部电影
    get_top20_info(types_id, headers)
    # 获取前20部电影各自详细信息
    get_movies_details(types_id, headers)
    # 获取每个电影的评论数据
    get_movies_comments(types_id, headers)

def get():
    types_id = {
            '悬疑':[10, "%E6%82%AC%E7%96%91"],
            '爱情':[13, "%E7%88%B1%E6%83%85"],
            '科幻':[17, "%E7%A7%91%E5%B9%BB"], 
            '恐怖':[20, "%E6%81%90%E6%80%96"], 
            '动作':[5, "%E5%8A%A8%E4%BD%9C"]
            }
    headers = {
        'Accept':'*/*',
        'Accept-Language':'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
        'Connection':'keep-alive',
        'Cookie':'ll="118187"; bid=9WBQHF80cj0; _pk_id.100001.4cf6=4c39d68d266ff238.1698151637.; __yadk_uid=2K6o8RsBi8jsyPP1WWxKtvT68oSV2XwQ; _vwo_uuid_v2=D8D71351800994190EAC7ADF32B91E9EC|3d0d7462bd8b99ae592362bf4823a522; _pk_ref.100001.4cf6=%5B%22%22%2C%22%22%2C1698321700%2C%22https%3A%2F%2Fwww.bing.com%2F%22%5D; _pk_ses.100001.4cf6=1; __utma=30149280.1301981671.1698151637.1698317979.1698321700.8; __utmz=30149280.1698321700.8.3.utmcsr=bing|utmccn=(organic)|utmcmd=organic|utmctr=(not%20provided); __utma=223695111.1298746994.1698151637.1698317979.1698321700.8; __utmb=223695111.0.10.1698321700; __utmz=223695111.1698321700.8.3.utmcsr=bing|utmccn=(organic)|utmcmd=organic|utmctr=(not%20provided); dbcl2="275531034:549XQ75H+LA"; push_noty_num=0; push_doumail_num=0; __utmv=30149280.27553; __utmb=30149280.4.10.1698321700; ck=HveI; __utmc=30149280; __utmc=223695111; frodotk_db="14dceca49566e34f3e96c5edc034a9ed"',
        'Dnt':'1',
        'Host':'movie.douban.com',
        'Sec-Ch-Ua':'"Chromium";v="118", "Microsoft Edge";v="118", "Not=A?Brand";v="99"',
        'Sec-Ch-Ua-Mobile':'?0',
        'Sec-Ch-Ua-Platform':"Windows",
        'Sec-Fetch-Dest':'empty',
        'Sec-Fetch-Mode':'cors',
        'Sec-Fetch-Site':'same-origin',
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36 Edg/118.0.2088.61',
        'X-Requested-With':'XMLHttpRequest'
        }
    


    Main_get_data(types_id, headers)



    
if __name__ == "__main__":
    print("Press n or N to quit")
    while True:
        starttime = input("Do you want to start get movies datas now? (y/n) (default is y,press <Enter>): ")
        if starttime == "n" or starttime == "N":
            break
        if starttime == "y" or starttime == "Y":
            get()
            break
        if not starttime:
            get()
            break
        else:
            print("wrong and input again!!!")

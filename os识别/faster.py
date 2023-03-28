# @author lsg
# @date 2023/3/13
# @file faster.py

import requests
from requests.sessions import Session
import time
from threading import Thread,local
from queue import Queue


# 1.打开url文件，读取
def loadUrl(fileName):
    f = open(fileName, encoding='utf-8')
    urlList = []
    for line in f:
        urlList.append(line.strip())
    f.close()
    return urlList


url_list = loadUrl(r"D:\edusrc\sqlurl\url.txt")
q = Queue(maxsize=0)            #Use a queue to store all URLs
for url in url_list:
    q.put(url)
thread_local = local()          #The thread_local will hold a Session object

def get_session() -> Session:
    if not hasattr(thread_local,'session'):
        thread_local.session = requests.Session() # Create a new Session if not exists
    return thread_local.session

headers = {
    'Connection': 'close'
}
def download_link() -> None:
    '''download link worker, get URL from queue until no url left in the queue'''
    session = get_session()
    while True:
        url = q.get()
        with session.get(url,headers=headers) as response:
            print(f'Read {len(response.content)} from {url}')
        q.task_done()          # tell the queue, this url downloading work is done

def download_all(urls) -> None:
    '''Start 10 threads, each thread as a wrapper of downloader'''
    thread_num = 10
    for i in range(thread_num):
        t_worker = Thread(target=download_link)
        t_worker.start()
    q.join()                   # main thread wait until all url finished downloading

print("start work")
start = time.time()
download_all(url_list)
end = time.time()
print(f'download {len(url_list)} links in {end - start} seconds')


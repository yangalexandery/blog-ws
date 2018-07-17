import time
import math
import json
import requests
import shutil
from PIL import Image

subreddit = "catsstandingup"
save_dir = "images"

tot_images = 10000

image_urls = set()
month = 0
while len(image_urls) < tot_images:
    url = "http://api.pushshift.io/reddit/search/submission/?subreddit=%s&size=500&before=%d" \
          % (subreddit, math.floor(time.time() - 24 * 60 * 60 * 30 * month))
    r = requests.get(url)
    data = r.json()['data']
    for submission in data:
        if submission['url'][-4:] == '.jpg' and len(image_urls) < tot_images:
            image_urls.add(submission['url'])
    print(len(image_urls))
    month += 1

f = open("image_urls.txt", "w+")
for url in image_urls:
    f.write(url + '\n')
#for url in image_urls:
#    r = requests.get(url, stream=True)
#    with open('%s/img_%05d.jpg' % (save_dir, img_counter), 'wb') as out_file:
#	pass
# todo: check for not 130x60 (image deleted)

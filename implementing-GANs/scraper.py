import time
import math
import json
import requests
import shutil
from PIL import Image, ImageOps

subreddit = "catsstandingup"
save_dir = "images"

tot_images = 10000

image_urls = set()
month = 0
desired_size = 128
img_counter = 0

while img_counter < tot_images:
    url = "http://api.pushshift.io/reddit/search/submission/?subreddit=%s&size=500&before=%d" \
          % (subreddit, math.floor(time.time() - 24 * 60 * 60 * 30 * month))
    r = requests.get(url)
    data = r.json()['data']
    for submission in data:
        if (submission['domain'] == 'i.redd.it' or submission['domain'] == 'i.imgur.com') and submission['url'][-4:] == '.jpg' and len(image_urls) < tot_images:
            image_url = submission['url']
            r = requests.get(image_url, stream=True)
            image = Image.open(r.raw)
            try:
                if image.size[0] != 130 or image.size[1] != 60 and image_url not in image_urls and img_counter < tot_images:
                    # print(image.size, image_url)
                    ratio = desired_size / max(image.size)
                    new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                    image = image.resize(new_size, Image.ANTIALIAS)

                    im = Image.new("RGB", (desired_size, desired_size))
                    im.paste(image, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))
                    im.save('%s/img_%05d.png' % (save_dir, img_counter), 'PNG')
                    image_urls.add(submission['url'])
                    img_counter += 1
                    if img_counter % 100 == 99:
                        print("# of images: ", img_counter + 1)
            except:
                pass
    month += 1

f = open('image_urls.txt', 'w+')
for url in image_urls:
    f.write(url + '\n')
# for url in image_urls:
#     r = requests.get(url, stream=True)
#     image = Image.open(r.raw)
#     print(image.size, url)
#     ratio = desired_size / max(image.size)
#     new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
#     image = image.resize(new_size, Image.ANTIALIAS)

#     im = Image.new("RGB", (desired_size, desired_size))
#     im.paste(image, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))
#     im.save('%s/img_%05d.png' % (save_dir, img_counter), 'PNG')
#     img_counter += 1


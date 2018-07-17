import praw
import json
import requests
import shutil
import time

### subreddit to be scraped
sub_name = "catsstandingup"
save_dir = "images"

### total number of images to be scraped
tot_images = 1123

config = json.load(open('config.json'))

reddit = praw.Reddit(client_id=config['client_id'],
                     client_secret=config['secret'],
                     password=config['password'],
                     user_agent=config['user_agent'],
                     username=config['username'])

print(reddit.user.me())

subreddit = reddit.subreddit(sub_name)
image_urls = set()
for submission in subreddit.hot(limit=1000):
    # print(submission.url)
    if submission.url[-4:] == '.jpg':
        image_urls.add(submission.url)
        if len(image_urls) % 100 == 0:
            print("# of images: ", len(image_urls))

for submission in subreddit.top(limit=1000):
    # print(submission.url)
    if submission.url[-4:] == '.jpg':
        image_urls.add(submission.url)
        if len(image_urls) % 100 == 0:
            print("# of images: ", len(image_urls))

for submission in subreddit.search("cat"):
    # print(submission.url)
    if submission.url[-4:] == '.jpg':
        image_urls.add(submission.url)
        if len(image_urls) % 100 == 0:
            print("# of images: ", len(image_urls))

print("random selection")
while len(image_urls) < tot_images:
    submission = subreddit.random()
    if submission.url[-4:] == '.jpg':
        image_urls.add(submission.url)
        #if len(image_urls) % 100 == 0:
        print("# of images: ", len(image_urls))

print("downloading images")
img_counter = 0
for url in image_urls:
    r = requests.get(url, stream=True)
    with open('%s/img_%05d.jpg' % (save_dir, img_counter), 'wb') as out_file:
        shutil.copyfileobj(r.raw, out_file)
    img_counter += 1
    del r


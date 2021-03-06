import os
from pixivpy3 import AppPixivAPI


def download():
    pixiv = AppPixivAPI()
    if len(os.listdir('./data/')) < 350:
        ranking = pixiv.illust_ranking('day')
        while len(os.listdir(os.path.curdir + '/data/')) < 350:
            print("Not Enough Data, Downloading, now %i images", len(os.listdir(os.path.curdir + '/data/')))
            for image in ranking.illusts:
                pixiv.download(image.image_urls.large, './data/')
            next_qs = pixiv.parse_qs(ranking.next_url)
            ranking = pixiv.illust_ranking(**next_qs)


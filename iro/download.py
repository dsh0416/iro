import os
from pixivpy3 import AppPixivAPI


def download():
    pixiv = AppPixivAPI()
    if len(os.listdir(os.path.curdir + '/data/')) < 350:
        ranking = pixiv.illust_ranking('day')
        while len(os.listdir(os.path.curdir + '/data/')) < 350:
            print("Not Enough Data, Downloading, now %i images", len(os.listdir(os.path.curdir + '/data/')) - 2)
            for image in ranking.illusts:
                # print(" p1 [%s] %s" % (image.title, image.image_urls.large))
                pixiv.download(image.image_urls.large, path=os.path.curdir + '/data/')
            next_qs = pixiv.parse_qs(ranking.next_url)
            ranking = pixiv.illust_ranking(**next_qs)


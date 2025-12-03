import scrapetube
import re
import requests

videos = scrapetube.get_channel(channel_username = "Poondaedalin") #channel ID

for i,video in enumerate(videos):

    tit = video['title']['runs'][0]['text']
    if not re.findall(r"YNO Project|Yume 2kki", tit):
        continue

    print(i)
    vid = video['videoId']
    vc = re.findall(r"([\d,]+) views",video['viewCountText']['simpleText'])[0]
    vc = vc.replace(",","")

    try:
        tit_num = re.findall(r"(?:YNO Project|Yume 2kki) \((\d+)\)", tit)[0]
    except Exception as e:
        tit_num = "NAN"

    with open(f"thumbnails/{vc}-YNO{tit_num}.png", "wb") as file:
        im = requests.get(f"https://img.youtube.com/vi/{vid}/0.jpg")
        file.write(im.content)
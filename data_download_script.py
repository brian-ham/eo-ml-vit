import requests
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd
import numpy as np

class GoogleDownloader:
    def __init__(self, access_token):
        self.access_token = access_token
        self.url = "https://maps.googleapis.com/maps/api/staticmap?center={},{}&zoom={}&size=640x640&maptype=satellite&key={}"

    def download(self, lat, long, zoom):
        res = requests.get(self.url.format(lat, long, zoom, self.access_token))
        # server needs to make image available, takes a few seconds
        if res.status_code == 403:
            return "RETRY"
        assert res.status_code < 400, print(
            f"Error - failed to download {lat}, {long}, {zoom}"
        )
        image = plt.imread(BytesIO(res.content))
        return image

key=""
loader = GoogleDownloader(key)

dhs_data = pd.read_csv("dhs_clusters_2014.csv")

images = []
for i, row in dhs_data.iterrows():
    image = loader.download(row["lat"], row["lon"], 16)
    if type(image) == str and image == "RETRY":
        print("Retrying")
        image = loader.download(row["lat"], row["lon"], 18)
    images.append(image)
    if i % 100 == 0:
        # print that we're onthe ith row
        print("On row:", i)
        print("Len of images array:", len(images))

np.savez('images_2014-now.npz', *images)
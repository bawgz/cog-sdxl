sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
sudo chmod +x /usr/local/bin/cog

cog build -t stable-dripfusion-2
docker exec stable-dripfusion-2 python3 ./script/download_weights.py

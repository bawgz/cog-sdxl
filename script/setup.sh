sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
sudo chmod +x /usr/local/bin/cog

wget -c https://replicate.delivery/pbxt/K7ku1HCBJMUchwXERHHSMi4Vkm3W75Qox5Rt5nKG7kGYmgkf/trained_model.tar -O - | tar -xz

sudo cog build -t stable-dripfusion-2

sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
sudo chmod +x /usr/local/bin/cog

mkdir trained-model-luk
mkdir trained-model-tok
sudo wget -c https://replicate.delivery/pbxt/K8l70F8kIPrIy6GDcoMok2k2C7EJSeWL3kQ4V52LKhsBqhe8/trained_model_luk.tar -O - | tar -C trained-model-luk -xz
sudo wget -c https://replicate.delivery/pbxt/K7ku1HCBJMUchwXERHHSMi4Vkm3W75Qox5Rt5nKG7kGYmgkf/trained_model.tar -O - | tar -C trained-model-tok -xz

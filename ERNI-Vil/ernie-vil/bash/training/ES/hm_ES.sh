#bash -x ./env.sh

bash /content/vilio/ernie-vil/bash/training/ES/hm_ES36.sh

bash /content/vilio/ernie-vil/bash/training/ES/hm_ESVCR36.sh

bash /content/vilio/ernie-vil/bash/training/ES/hm_ESV50.sh

bash /content/vilio/ernie-vil/bash/training/ES/hm_ES72.sh

bash /content/vilio/ernie-vil/bash/training/ES/hm_ESVCR72.sh

# Simple Average
python utils/ens.py --enspath /content/vilio/ernie-vil/data/hm/ --enstype sa --exp ES365072
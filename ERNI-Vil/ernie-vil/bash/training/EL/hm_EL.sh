#bash -x /content/vilio/ernie-vil/env.sh

bash /content/vilio/ernie-vil/bash/training/EL/hm_EL36.sh

bash /content/vilio/ernie-vil/bash/training/EL/hm_ELVCR36.sh

bash /content/vilio/ernie-vil/bash/training/EL/hm_ELV50.sh

bash /content/vilio/ernie-vil/bash/training/EL/hm_EL72.sh

bash /content/vilio/ernie-vil/bash/training/EL/hm_ELVCR72.sh

# Simple Average
python /content/vilio/ernie-vil/utils/ens.py --enspath /content/vilio/ernie-vil/data/hm/ --enstype sa --exp EL365072
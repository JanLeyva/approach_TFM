#bash -x /content/vilio/ernie-vil/env.sh

### ATT 36, Normal

mv /content/vilio/ernie-vil/data/hm/hm_vgattr10100.tsv /content/vilio/ernie-vil/data/hm/HM_gt_img.tsv
mv /content/vilio/ernie-vil/data/hm/hm_vgattr3636.tsv /content/vilio/ernie-vil/data/hm/HM_img.tsv

bash run_finetuning.sh hm conf/hm/model_conf_hm \
/content/vilio/ernie-vil/data/ernielargevcr/vocab.txt \
/content/vilio/ernie-vil/data/ernielargevcr/ernie_vil.large.json \
/content/vilio/ernie-vil/data/ernielargevcr/params \
train \
2500

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
/content/vilio/ernie-vil/data/ernielargevcr/vocab.txt \
/content/vilio/ernie-vil/data/ernielargevcr/ernie_vil.large.json \
/content/vilio/ernie-vil/output_hm/step_2500train \
/content/vilio/ernie-vil/data/log \
dev_seen ELVCR36 False

##################### TRAINDEV

bash run_finetuning.sh hm conf/hm/model_conf_hm \
/content/vilio/ernie-vil/data/ernielargevcr/vocab.txt \
/content/vilio/ernie-vil/data/ernielargevcr/ernie_vil.large.json \
/content/vilio/ernie-vil/data/ernielargevcr/params \
traindev \
2500

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
/content/vilio/ernie-vil/data/ernielargevcr/vocab.txt \
/content/vilio/ernie-vil/data/ernielargevcr/ernie_vil.large.json \
/content/vilio/ernie-vil/output_hm/step_2500traindev \
/content/vilio/ernie-vil/data/log \
test_seen ELVCR36 False

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
/content/vilio/ernie-vil/data/ernielargevcr/vocab.txt \
/content/vilio/ernie-vil/data/ernielargevcr/ernie_vil.large.json \
/content/vilio/ernie-vil/output_hm/step_2500traindev \
/content/vilio/ernie-vil/data/log \
test_unseen ELVCR36 False
#bash -x /content/vilio/ernie-vil/env.sh

### VGATT 50

mv /content/vilio/ernie-vil/data/hm/hm_vg10100.tsv /content/vilio/ernie-vil/data/hm/HM_gt_img.tsv
mv /content/vilio/ernie-vil/data/hm/hm_vg5050.tsv /content/vilio/ernie-vil/data/hm/HM_img.tsv

bash run_finetuning.sh hm conf/hm/model_conf_hm \
/content/vilio/ernie-vil/data/erniesmall/vocab.txt \
/content/vilio/ernie-vil/data/erniesmall/ernie_vil_config.base.json \
/content/vilio/ernie-vil/data/erniesmall/params \
train \
2500

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
/content/vilio/ernie-vil/data/erniesmall/vocab.txt \
/content/vilio/ernie-vil/data/erniesmall/ernie_vil_config.base.json \
/content/vilio/ernie-vil/output_hm/step_2500train \
/content/vilio/ernie-vil/data/log \
dev_seen ESV50 False

### TRAINDEV

bash run_finetuning.sh hm conf/hm/model_conf_hm \
/content/vilio/ernie-vil/data/erniesmall/vocab.txt \
/content/vilio/ernie-vil/data/erniesmall/ernie_vil_config.base.json \
/content/vilio/ernie-vil/data/erniesmall/params \
traindev \
2500

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
/content/vilio/ernie-vil/data/erniesmall/vocab.txt \
/content/vilio/ernie-vil/data/erniesmall/ernie_vil_config.base.json \
/content/vilio/ernie-vil/output_hm/step_2500traindev \
/content/vilio/ernie-vil/data/log \
test_seen ESV50 False

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
/content/vilio/ernie-vil/data/erniesmall/vocab.txt \
/content/vilio/ernie-vil/data/erniesmall/ernie_vil_config.base.json \
/content/vilio/ernie-vil/output_hm/step_2500traindev \
/content/vilio/ernie-vil/data/log \
test_unseen ESV50 False
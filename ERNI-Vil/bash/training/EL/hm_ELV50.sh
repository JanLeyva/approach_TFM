#bash -x /content/vilio/ernie-vil/env.sh

### VGATTR 50

mv /content/vilio/ernie-vil/data/hm/hm_vg10100.tsv /content/vilio/ernie-vil/data/hm/HM_gt_img.tsv
mv /content/vilio/ernie-vil/data/hm/hm_vg5050.tsv /content/vilio/ernie-vil/data/hm/HM_img.tsv

bash run_finetuning.sh hm conf/hm/model_conf_hm \
/content/vilio/ernie-vil/data/ernielarge/vocab.txt \
/content/vilio/ernie-vil/data/ernielarge/ernie_vil.large.json \
/content/vilio/ernie-vil/data/ernielarge/params \
train \
2500

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
/content/vilio/ernie-vil/data/ernielarge/vocab.txt \
/content/vilio/ernie-vil/data/ernielarge/ernie_vil.large.json \
/content/vilio/ernie-vil/output_hm/step_2500train \
/content/vilio/ernie-vil/data/log \
dev_seen ELV50 False

# Save Space

#rm -r /content/vilio/ernie-vil/data/hm/img

##################### TRAINDEV


bash run_finetuning.sh hm conf/hm/model_conf_hm \
/content/vilio/ernie-vil/data/ernielarge/vocab.txt \
/content/vilio/ernie-vil/data/ernielarge/ernie_vil.large.json \
/content/vilio/ernie-vil/data/ernielarge/params \
traindev \
2500

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
/content/vilio/ernie-vil/data/ernielarge/vocab.txt \
/content/vilio/ernie-vil/data/ernielarge/ernie_vil.large.json \
/content/vilio/ernie-vil/output_hm/step_2500traindev \
/content/vilio/ernie-vil/data/log \
test_seen ELV50 False

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
/content/vilio/ernie-vil/data/ernielarge/vocab.txt \
/content/vilio/ernie-vil/data/ernielarge/ernie_vil.large.json \
/content/vilio/ernie-vil/output_hm/step_2500traindev \
/content/vilio/ernie-vil/data/log \
test_unseen ELV50 False

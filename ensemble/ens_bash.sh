#bash -x ./env.sh
bash python ens_v4.py --enspath ./0507Results/ \
                        --fileFairface annotations_fairface.json \
                        --enstype sa \
                        --exp ens0507 \
                        --meme_anno_path ./annotations \
                        --racism_rule True
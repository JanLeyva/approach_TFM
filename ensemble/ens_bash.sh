#bash -x ./env.sh
bash python ens.py --enspath ./ \
                        --fileFairface annotations_fairface.json \
                        --enstype sa \
                        --exp ens0507 \
                        --meme_anno_path ./annotations \
                        --racism_rule False
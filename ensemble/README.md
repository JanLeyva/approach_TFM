# Ensemble

The ensemble will be done in two parts:
- First ensemble type choosed.
- Second a racism classifier, the racism classifier is based on a *heuristic* where use the FairFace features and text memes in order to classify if a meme is racist or not.

The scrip `ens.py` can be run as the following example:

```
python ens_v4.py --enspath ./0507Results/ \
                        --fileFairface annotations_fairface.json \
                        --enstype sa \
                        --exp ens0507 \
                        --meme_anno_path ./annotations \
                        --racism_rule True
```

- enspath: must store the results (.csv) of the models.
- fileFairface: where you store the fairface features
- enstype: ensamble type, there are different ensamble enable:
	- sa: simple average
	- ra: rank average
	- optimizer: optimization the weight for each prediction using simplex algorithm based on [repository](https://github.com/chrisstroemel/Simple). (minimum 3 files to optimizate).
	- exp: experiment name, will create a folder with the output of the ensamble.
	- meme_anno_path: where the HM annotations (.jsonl) are stored.
	- racism_rule: (boolean) a heuristic where detect if the meme contains the words (or similar) 
	['crime', 'hang', 'rob', 'steal', 'jail', 'prison', 'slave', 'apes', 'criminal', 'gorilla'] and if the fairface race is black.



	    if args.enstype == "loop":
        main(args.enspath)
    elif args.enstype == "best_ens":
        choose_opt(args.enspath)
    elif args.enstype == "ens_ens":
        ens_ens(args.enspath)
    elif args.enstype == "sa":
        sa_wrapper(args.enspath)
    elif args.enstype == "optimizer":
        optimization(args.enspath)
    elif args.enstype == "rank_avg":
        rank_average_method(args.enspath)


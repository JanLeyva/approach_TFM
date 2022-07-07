# Ensemble

First of all, we must run the `sort_files.py` scrip in order to sort the values as the original files. For some reason when mmf models predict in validation set it reorder the prediction output. To solve that we will run the scrip as following example and set `showmetrics True` in case that we want that the individual metrics were showed.

```
python sort_files.py --annotations_file ./annotations/ \
	 --path ./results/ \
	 --output_path ./result_files/ \
	 --showmetrics True
```

The ensemble will be done in two parts:
- First ensemble type choosed.
- Second a racism heuristic, the racism classifier is based on a *heuristic* where use the FairFace features and text memes in order to classify if a meme is racist or not.

The scrip `ens.py` can be run as the following example:

```
python ens.py --enspath ./results/ \
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
	- best_ens: run all the ensemble type and return which one reach the highest AUROC in test set.
	- loop: loops through Averaging, Power Averaging, Rank Averaging, Optimization to find the best ensemble.
	- ens_ens: ensemble the ensembles outputs.
- exp: experiment name, will create a folder with the output of the ensamble.
- meme_anno_path: where the HM annotations (.jsonl) are stored.
- racism_rule: (boolean) a heuristic where detect if the meme contains the words (or similar) 
	['crime', 'hang', 'rob', 'steal', 'jail', 'prison', 'slave', 'apes', 'criminal', 'gorilla'] and if the fairface race is black.


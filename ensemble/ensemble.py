import glob
import shutil



def merge(dfs):
    return sum([df.proba.values for df in dfs]) / len(dfs)


def get_mean_predict(out_path):

    csv_list = glob.glob(visualBERT_)
    csv_list += glob.glob(visualBERTCoco_)
    csv_list += glob.glob(vilBERT_)
    csv_list += glob.glob(ernie_vil_)

    print(f"Found {len(csv_list)} csv eval result!")

    ensem_list = []
    All = False
    for csv_file in csv_list:
        # print(csv_file)
        if not All:
            yn = input(f"Include {csv_file} to ensemble? (y/n/all)")
        else:
            yn = 'y'
        yn = yn.strip().lower()
        if yn == 'all':
            All = True
        
        if yn == 'y' or All:
            ensem_list.append(csv_file)
            # dir_name = os.path.basename(os.path.dirname(csv_file))
            # shutil.copy(
            #     csv_file,
            #     os.path.join(
            #         gather_dir,
            #         f"{dir_name}_{os.path.basename(csv_file)}"
            #     )
            # )
    assert len(ensem_list) >= 2, f'You must select at least two file to ensemble, only {len(ensem_list)} is picked'
    
    base = pd.read_csv(ensem_list[0])
    print(len(ensem_list))
    ensem_list = [pd.read_csv(c) for c in ensem_list]
    base.proba = merge(ensem_list)



    # rasicm_idx = rasicm_det(
    #     os.path.join(root_dir, 'data/hateful_memes/test_unseen.jsonl'),
    #     os.path.join(root_dir, 'data/hateful_memes/box_annos.race.json'),
    #     os.path.join(root_dir, 'data/hateful_memes/img_clean'),
    # )
    # for i in rasicm_idx:
    #     base.at[int(base.index[base['id']==i].values), 'proba'] = 1.0

    base.to_csv(out_path, index=False)
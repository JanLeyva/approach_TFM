import os
import pandas as pd
output_path= '/content/results'
dst_path='/content/files'
results=[]
for j, folder in enumerate(os.listdir(output_path)):
  for k, path in enumerate(os.listdir(os.path.join(output_path, folder))):
    if path.startswith("hateful_memes"):
      for l, path_ in enumerate(os.listdir(os.path.join(output_path, folder, path, 'reports'))):
        file_=os.listdir(os.path.join(output_path))
        if 'test' in path_:
          # print('test: {}'.format(path_))
          test=pd.read_csv(os.path.join(output_path, folder, path, 'reports', path_))
          if (test.shape) == (2000, 3):
            print("unseen")
            pred=set_label(test_unseen, pd.merge(test_unseen_order, test))
            results.append({
                            'set': "test_unseen_"+file_[j],
                            'auc': roc_auc_score(test_unseen['label'], pred['proba']),
                            'acc': accuracy_score(test_unseen['label'], pred['label'])
                            })
            print(results)
            shutil.copyfile(os.path.join(output_path, folder, path, 'reports', path_), 
                            os.path.join(dst_path, file_[j]+"_test_unseen.csv"))
          elif (test.shape) == (1000, 3):
            print("seen")
            pred=set_label(test_seen, pd.merge(test_seen_order, test))
            results.append({
                            'set': "test_seen_"+file_[j],
                            'auc': roc_auc_score(test_seen['label'], pred['proba']),
                            'acc': accuracy_score(test_seen['label'], pred['label'])
                            })
            print(results)
            shutil.copyfile(os.path.join(output_path, folder, path, 'reports', path_), 
                            os.path.join(dst_path, file_[j]+"_test_seen.csv"))
          else:
            print("Assertion, wrong shape file.")


        if 'dev' in path_:
          # print('test: {}'.format(path_))
          dev=pd.read_csv(os.path.join(output_path, folder, path, 'reports', path_))
          if (dev.shape) == (540, 3):
            print("unseen")
            pred=set_label(dev_unseen, pd.merge(dev_unseen_order, dev))
            results.append({
                            'set': "dev_unseen"+file_[j],
                            'auc': roc_auc_score(dev_unseen['label'], pred['proba']),
                            'acc': accuracy_score(dev_unseen['label'], pred['label'])
                            })
            print(results)
            shutil.copyfile(os.path.join(output_path, folder, path, 'reports', path_), 
                            os.path.join(dst_path, file_[j]+"_dev_unseen.csv"))
          elif (dev.shape) == (500, 3):
            print("seen")
            pred=set_label(dev_seen, pd.merge(dev_seen_order, dev))
            results.append({
                            'set': "dev_seen_"+file_[j],
                            'auc': roc_auc_score(dev_seen['label'], pred['proba']),
                            'acc': accuracy_score(dev_seen['label'], pred['label'])
                            })
            print(results)
            shutil.copyfile(os.path.join(output_path, folder, path, 'reports', path_), 
                            os.path.join(dst_path, file_[j]+"_dev_seen.csv"))
          else:
            print("Assertion, wrong shape file.")
          
results=pd.DataFrame(results)
results.to_csv("results.csv", index=False)
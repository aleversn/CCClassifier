# %%
import os
from tqdm import tqdm

def process_data(dir, save_dir):
    cls_list = os.listdir(dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    print('{}\n'.format(cls_list))
    for idx, _cls in enumerate(cls_list):
        print('{}\n'.format(_cls))
        data_list = []
        files = os.listdir(os.path.join(dir, _cls))
        for file in tqdm(files):
            with open(os.path.join(dir, _cls, file), encoding='utf-8', mode='r') as f:
                content = f.read()
            content = content.replace('\n', '||')
            content = content.replace('\t', ' ')
            data_list.append('{}\t{}'.format(idx, content))
        with open(os.path.join(save_dir, _cls + '.csv'), mode='w+') as f:
            f.write("")
        with open(os.path.join(save_dir, _cls + '.csv'), mode='a+') as f:
            for item in data_list:
                f.write('{}\n'.format(item))
    with open(os.path.join(save_dir, 'tags_list.csv'), mode='w+') as f:
        result = ''
        for idx, _cls in enumerate(cls_list):
            result += '{}\t{}\n'.format(idx, _cls)
        f.write(result)
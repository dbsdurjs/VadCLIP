import numpy as np
import torch, os
import torch.utils.data as data
import pandas as pd
import utils.tools as tools

class UCFDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, file_path_cap: str, test_mode: bool, label_map: dict, normal: bool = False, using_caption: bool = False):
        self.df = pd.read_csv(file_path)
        self.df_cap = pd.read_csv(file_path_cap)
        self.clip_dim = clip_dim    # 256
        self.test_mode = test_mode
        
        # lable_map = {'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson', 'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion', 'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery', 'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing', 'Vandalism': 'vandalism'}
        self.label_map = label_map  
        self.normal = normal
        self.using_caption = using_caption
        print(f'captioning 사용여부 : {self.using_caption}')
        if normal == True and test_mode == False:
            self.df = self.df.loc[self.df['label'] == 'Normal']
            self.df_cap = self.df_cap.loc[self.df_cap['label'] == 'Normal']
            
            self.df = self.df.reset_index()
            self.df_cap = self.df_cap.reset_index()
        elif test_mode == False:
            self.df = self.df.loc[self.df['label'] != 'Normal']
            self.df_cap = self.df_cap.loc[self.df_cap['label'] != 'Normal']

            self.df = self.df.reset_index()
            self.df_cap = self.df_cap.reset_index()
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        clip_path = self.df.loc[index]['path']
        clip_feature = np.load(clip_path)

        if self.using_caption:
            base_file = os.path.basename(clip_path)
            base_video_name = base_file.split('__')[0]

            matching_rows = self.df_cap[self.df_cap['path'].str.contains(base_video_name) & 
                                    self.df_cap['path'].str.endswith(base_video_name + ".npy")]
            if matching_rows.empty:
                raise KeyError(f"No matching clip_cap_feature for video {base_video_name}")
            
            cap_path = matching_rows.iloc[0]['path']
            clip_cap_feature = np.load(cap_path)

            if clip_feature.shape[0] < clip_cap_feature.shape[0]:
                pad_frames = clip_cap_feature.shape[0] - clip_feature.shape[0]
                clip_feature = np.pad(clip_feature, ((0, pad_frames), (0, 0)), mode='constant', constant_values=0)
            elif clip_cap_feature.shape[0] < clip_feature.shape[0]:
                pad_frames = clip_feature.shape[0] - clip_cap_feature.shape[0]
                clip_cap_feature = np.pad(clip_cap_feature, ((0, pad_frames), (0, 0)), mode='constant', constant_values=0)

            clip_feature = np.concatenate([clip_feature, clip_cap_feature], axis=1)


        if self.test_mode == False:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        clip_label = self.df.loc[index]['label']
        return clip_feature, clip_label, clip_length

class XDDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        clip_feature = np.load(self.df.loc[index]['path'])
        if self.test_mode == False:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        clip_label = self.df.loc[index]['label']
        return clip_feature, clip_label, clip_length
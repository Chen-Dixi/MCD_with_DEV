import random
from torch.utils.data import DataLoader

class PairedDataLoader(object):
    def __init__(self, data_loader_A, data_loader_B, max_dataset_size, flip):
        self.data_loader_A = data_loader_A
        self.data_loader_B = data_loader_B
        self.stop_A = False
        self.stop_B = False
        self.max_dataset_size = max_dataset_size
        self.flip = flip

    def __iter__(self): #有iter
        self.stop_A = False
        self.stop_B = False
        self.data_loader_A_iter = iter(self.data_loader_A)
        self.data_loader_B_iter = iter(self.data_loader_B)
        self.iter = 0
        return self #__iter__要返回self. 

    def __next__(self):
        A, A_paths = None, None
        B, B_paths = None, None
        try:
            A, A_paths = next(self.data_loader_A_iter)
        except StopIteration:
            if A is None or A_paths is None:
                self.stop_A = True
                self.data_loader_A_iter = iter(self.data_loader_A)
                A, A_paths = next(self.data_loader_A_iter)

        try:
            B, B_paths = next(self.data_loader_B_iter)
        except StopIteration:
            if B is None or B_paths is None:
                self.stop_B = True
                self.data_loader_B_iter = iter(self.data_loader_B)
                B, B_paths = next(self.data_loader_B_iter)

        if (self.stop_A and self.stop_B) or self.iter > self.max_dataset_size:
            self.stop_A = False
            self.stop_B = False
            # 这里不用设置iter=0。
            raise StopIteration()
        else:
            self.iter += 1
            # 这里不用关系
            if self.flip and random.random() < 0.5:
                idx = [i for i in range(A.size(3) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A = A.index_select(3, idx)
                B = B.index_select(3, idx)
            return {'s': A, 's_target': A_paths,
                    't': B, 't_target': B_paths}
    def __len__(self):
        return min(max(len(self.data_loader_A), len(self.data_loader_B)), self.max_dataset_size)

# 主要用在Domain Adaptation上面
class PairedDatasetHelper(object):
    def __init__(self, dataset_A,dataset_B,batch_size,shuffle=True,num_workers=0):
        #normalize = transforms.Normalize(mean=mean_im,std=std_im)
        self.max_dataset_size = float("inf")
        data_loader_A = DataLoader(
            dataset_A,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers)
        data_loader_B = DataLoader(
            dataset_B,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers)
        self.dataset_A = dataset_A
        self.dataset_B = dataset_B
        flip = False
        self.paired_dataLoader = PairedDataLoader(data_loader_A, data_loader_B, self.max_dataset_size, flip)

    def name(self):
        return 'UnalignedDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return min(max(len(self.dataset_A), len(self.dataset_B)), self.max_dataset_size)

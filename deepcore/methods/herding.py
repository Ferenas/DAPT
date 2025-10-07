from .earlytrain import EarlyTrain
import torch
import numpy as np
from .methods_utils import euclidean_dist
from ..nets.nets_utils import MyDataParallel


class Herding(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200,
                 specific_model="ResNet18", balance: bool = False, metric="euclidean", **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs=epochs, specific_model=specific_model, **kwargs)

        if metric == "euclidean":
            self.metric = euclidean_dist
        elif callable(metric):
            self.metric = metric
        else:
            self.metric = euclidean_dist
            self.run = lambda: self.finish_run()

            def _construct_matrix(index=None):
                data_loader = torch.utils.data.DataLoader(
                    self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
                    batch_size=self.n_train if index is None else len(index), num_workers=self.args.workers)
                inputs, _ = next(iter(data_loader))
                return inputs.flatten(1).requires_grad_(False).to(self.args.device)

            self.construct_matrix = _construct_matrix

        self.balance = balance
        self.select_bs = self.args.DATASET.SELECTION_BATCH_SIZE

    def num_classes_mismatch(self):
        raise ValueError("num_classes of pretrain dataset does not match that of the training dataset.")

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        pass

    #Initial achievement, may not optimal
    def mixing_feature(self,img_fea,text_fea,lam=0.5):
        # return img_fea
        return lam*img_fea + (1-lam)*text_fea

    def construct_matrix(self, index=None):
        self.model.eval()
        self.model.no_grad = True
        with torch.no_grad():
            # with self.model.embedding_recorder:
                sample_num = self.n_train if index is None else len(index)
                matrix = torch.zeros([sample_num, self.emb_dim], requires_grad=False).cuda()
                data_loader = self.select_dm(self.dst_train,index,is_train=False)
                for i, batch in enumerate(data_loader):
                    image,label = batch['img'].cuda(),batch['label'].cuda()
                    img_f,text_f,_ = self.model(image, label, record=True)
                    final_embed = self.mixing_feature(img_f,text_f)  #Using the mixed image_feature and text_feature
                    matrix[i * self.select_bs:min((i + 1) * self.select_bs, sample_num)] = final_embed

        self.model.no_grad = False
        self.model.train()
        return matrix

    def before_run(self):
        self.emb_dim = self.model.image_encoder.output_dim

    def herding(self, matrix, budget: int, index=None):

        sample_num = matrix.shape[0]

        if budget < 0:
            raise ValueError("Illegal budget size.")
        elif budget > sample_num:
            budget = sample_num

        indices = np.arange(sample_num)
        with torch.no_grad():
            mu = torch.mean(matrix, dim=0)
            select_result = np.zeros(sample_num, dtype=bool)

            for i in range(budget):
                if i % self.args.TRAIN.PRINT_FREQ == 0:
                    print("| Selecting [%3d/%3d]" % (i + 1, budget))
                dist = self.metric(((i + 1) * mu - torch.sum(matrix[select_result], dim=0)).view(1, -1),
                                   matrix[~select_result])
                p = torch.argmax(dist).item()
                p = indices[~select_result][p]
                select_result[p] = True
        if index is None:
            index = indices
        return index[select_result]

    def finish_run(self):
        if isinstance(self.model, MyDataParallel):
            self.model = self.model.module

        if self.balance:
            selection_result = np.array([], dtype=np.int32)
            for c in range(self.num_classes):
                class_index = np.arange(self.n_train)[self.dst_train_label == c]
                selection_result = np.append(selection_result, self.herding(self.construct_matrix(class_index),
                        budget=round(self.fraction * len(class_index)), index=class_index))
        else:
            selection_result = self.herding(self.construct_matrix(), budget=self.coreset_size)
        return {"indices": selection_result}

    def select(self, **kwargs):
        selection_result = self.run()
        return selection_result



from .earlytrain import EarlyTrain
from .methods_utils.euclidean import euclidean_dist_pair_np
from .methods_utils.cossim import cossim_pair_np
import numpy as np
import torch
from tqdm import tqdm
from .. import nets
from copy import deepcopy
from torchvision import transforms


class Cal(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200, specific_model=None,
                 balance=False, metric="euclidean", neighbors: int = 10, pretrain_model: str = "ResNet18", **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs, specific_model, **kwargs)

        self.balance = balance

        assert neighbors > 0 and neighbors < 100
        self.neighbors = neighbors

        if metric == "euclidean":
            self.metric = euclidean_dist_pair_np
        elif metric == "cossim":
            self.metric = lambda a, b: -1. * cossim_pair_np(a, b)
        elif callable(metric):
            self.metric = metric
        else:
            self.metric = euclidean_dist_pair_np

        self.pretrain_model = pretrain_model

    def num_classes_mismatch(self):
        raise ValueError("num_classes of pretrain dataset does not match that of the training dataset.")

    #Initial achievement, may not optimal
    def mixing_feature(self,img_fea,text_fea,lam=0.5):
        # return img_fea
        return lam*img_fea + (1-lam)*text_fea

    def find_knn(self):
        """
        Find k-nearest-neighbor data points with the pretrained embedding model
        :return: knn matrix
        """

        # Initialize pretrained model
        # model = nets.__dict__[self.pretrain_model](channel=self.args.channel, num_classes=self.args.num_classes,
        #                                            im_size=(224, 224), record_embedding=True, no_grad=True,
        #                                            pretrained=True).to(self.args.device)
        self.model.eval()
        probs = []
        # # Resize dst_train to 224*224
        # if self.args.im_size[0] != 224 or self.args.im_size[1] != 224:
        #     dst_train = deepcopy(self.dst_train)
        #     dst_train.transform = transforms.Compose([dst_train.transform, transforms.Resize(224)])
        # else:
        #     dst_train = self.dst_train

        # Calculate the distance matrix and return knn results
        if self.balance:
            knn = []
            for c in tqdm(range(self.num_classes)):
                print(f'Start processing class {c}/{self.num_classes}')
                class_index = np.arange(self.n_train)[self.dst_train_label == c]

                # Start recording embedding vectors
                #                batch_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dst_train, class_index),
                #                                            batch_size=self.args.selection_batch,
                #                                            num_workers=self.args.workers)
                embdeddings = []
                c_probs = np.zeros([len(class_index), self.num_classes])
                data_loader = self.select_dm(self.dst_train, class_index, is_train=False)
                for i, batch in enumerate(data_loader):
                    image, label = batch['img'].cuda(), batch['label'].cuda()
                    img_f, text_f,logit = self.model(image, label, record=True)
                    final_feature = self.mixing_feature(img_f,text_f)
                    embdeddings.append(final_feature.cpu().numpy())
                    c_probs[i * self.args.DATASET.SELECTION_BATCH_SIZE:(i + 1) * self.args.DATASET.SELECTION_BATCH_SIZE] = \
                    torch.softmax(logit, dim=1).detach().cpu()

                embdeddings = np.concatenate(embdeddings, axis=0)
                probs.append(c_probs)
                knn.append(np.argsort(self.metric(embdeddings), axis=1)[:, 1:(self.neighbors + 1)])
            self.probs = np.concatenate(probs,axis=0)
            return knn
        else:
            # Start recording embedding vectors
            embdeddings = []
            batch_loader = self.select_dm(self.dst_train, None, is_train=False)
            print(f'Start processing all class')
            for i, batch in enumerate(tqdm(batch_loader)):
                image, label = batch['img'].cuda(), batch['label'].cuda()
                img_f, text_f,logit = self.model(image, label, record=True)
                final_feature = self.mixing_feature(img_f, text_f)
                embdeddings.append(final_feature.cpu().numpy())
                probs[i * self.args.DATASET.SELECTION_BATCH_SIZE:(i + 1) * self.args.DATASET.SELECTION_BATCH_SIZE] = \
                    torch.softmax(logit, dim=1).detach().cpu()
            embdeddings = np.concatenate(embdeddings, axis=0)
            self.probs = np.concatenate(probs, axis=0)
            return np.argsort(self.metric(embdeddings), axis=1)[:, 1:(self.neighbors + 1)]

    def calc_kl(self, knn, index=None):
        self.model.eval()
        self.model.no_grad = True
        sample_num = self.n_train if index is None else len(index)
        # probs = np.zeros([sample_num, self.num_classes])
        #
        # batch_loader = torch.utils.data.DataLoader(
        #     self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
        #     batch_size=self.args.selection_batch, num_workers=self.args.workers)
        # batch_num = len(batch_loader)
        #
        # for i, (inputs, _) in enumerate(batch_loader):
        #     probs[i * self.args.selection_batch:(i + 1) * self.args.selection_batch] = torch.nn.functional.softmax(
        #         self.model(inputs.to(self.args.device)), dim=1).detach().cpu()
        probs = self.probs[index]
        s = np.zeros(sample_num)
        for i in range(0, sample_num, self.args.DATASET.SELECTION_BATCH_SIZE):

            print("| Caculating KL-divergence for batch [%3d/%3d] with batchsize [%3d]" % (i, sample_num, self.args.DATASET.SELECTION_BATCH_SIZE))
            aa = np.expand_dims(probs[i:(i + self.args.DATASET.SELECTION_BATCH_SIZE)], 1).repeat(self.neighbors, 1)
            bb = probs[knn[i:(i + self.args.DATASET.SELECTION_BATCH_SIZE)], :]
            s[i:(i + self.args.DATASET.SELECTION_BATCH_SIZE)] = np.mean(
                np.sum(0.5 * aa * np.log(aa / bb) + 0.5 * bb * np.log(bb / aa), axis=2), axis=1)
        self.model.no_grad = False
        return s

    def finish_run(self):
        scores=[]
        if self.balance:
            selection_result = np.array([], dtype=np.int32)
            for c, knn in zip(range(self.num_classes), self.knn):
                class_index = np.arange(self.n_train)[self.dst_train_label == c]
                scores.append(self.calc_kl(knn, class_index))
                selection_result = np.append(selection_result, class_index[np.argsort(
                    #self.calc_kl(knn, class_index))[::1][:round(self.fraction * len(class_index))]])
                    scores[-1])[::1][:round(self.fraction * len(class_index))]])
        else:
            selection_result = np.argsort(self.calc_kl(self.knn))[::1][:self.coreset_size]
        return {"indices": selection_result, "scores":scores}

    def select(self, **kwargs):
        self.knn = self.find_knn()
        selection_result = self.run()
        return selection_result
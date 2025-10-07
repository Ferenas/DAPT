from .earlytrain import EarlyTrain
import torch, time
import numpy as np
from ..nets.nets_utils import MyDataParallel
from tqdm import tqdm

class GraNd(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200, repeat=1,
                 specific_model=None, balance=False, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs, specific_model,**kwargs)
        self.epochs = epochs
        self.n_train = len(self.dst_train)
        self.coreset_size = round(self.n_train * fraction)
        self.specific_model = specific_model
        self.repeat = repeat

        self.balance = balance

    # def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
    #     if batch_idx % self.args.print_freq == 0:
    #         print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
    #             epoch, self.epochs, batch_idx + 1, (self.n_train // batch_size) + 1, loss.item()))

    def before_run(self):
        if isinstance(self.model, MyDataParallel):
            self.model = self.model.module

    def calc_gradient(self, index=None):
        '''
        Calculate gradients matrix on current network for specified training dataset.
        '''
        self.model.eval()
        data_loader = self.select_dm(self.dst_train, index, is_train=False)
        # Initialize a matrix to save gradients.
        # (on cpu)
        gradients = []

        for i, batch in enumerate(tqdm(data_loader)):
            self.optim.zero_grad()
            image, label = batch['img'].cuda(), batch['label'].cuda()
            bs_size = image.shape[0]
            loss, visual_embedding, logit = self.model(image, label, cal_gradient=True)
            embed_dim = visual_embedding.shape[-1]
            with torch.no_grad():
                bias_parameters_grads = torch.autograd.grad(loss, logit)[0]
                weight_parameters_grads = visual_embedding.view(bs_size, 1,
                                                                -1).repeat(1, self.num_classes, 1) * \
                                          bias_parameters_grads.view(bs_size, self.num_classes,
                                                                     1).repeat(1, 1, embed_dim)
                gradients.append(torch.cat([bias_parameters_grads, weight_parameters_grads.flatten(1)],
                                           dim=1).cpu().numpy())

        gradients = np.concatenate(gradients, axis=0, dtype=np.float32)
        print('Finish Gradient Calculation')
        self.model.train()
        return gradients

    def finish_run(self):
        # self.model.embedding_recorder.record_embedding = True  # recording embedding vector

        gradients = self.calc_gradient()
        self.norm_matrix[:,0] = np.linalg.norm(gradients,axis=1)



        # embedding_dim = self.model.get_last_layer().in_features
        # data_loader = self.select_dm(self.dst_train, None, is_train=False)
        # sample_num = self.n_train
        #
        # for i, batch in enumerate(data_loader):
        #     self.optim.zero_grad()
        #     image, target,batch_inds = batch['img'].cuda(), batch['label'].cuda(), batch['index'].cuda()
        #
        #     outputs = self.model(image)
        #     loss = self.criterion(outputs.requires_grad_(True),
        #                           targets.to(self.args.device)).sum()
        #     batch_num = targets.shape[0]
        #     with torch.no_grad():
        #         bias_parameters_grads = torch.autograd.grad(loss, outputs)[0]
        #         self.norm_matrix[i * self.args.selection_batch:min((i + 1) * self.args.selection_batch, sample_num),
        #         self.cur_repeat] = torch.norm(torch.cat([bias_parameters_grads, (
        #                 self.model.embedding_recorder.embedding.view(batch_num, 1, embedding_dim).repeat(1,
        #                                      self.args.num_classes, 1) * bias_parameters_grads.view(
        #                                      batch_num, self.args.num_classes, 1).repeat(1, 1, embedding_dim)).
        #                                      view(batch_num, -1)], dim=1), dim=1, p=2)
        #
        # self.model.train()


    def select(self, **kwargs):
        # Initialize a matrix to save norms of each sample on idependent runs
        self.norm_matrix = np.zeros([self.n_train, self.repeat])

        # for self.cur_repeat in range(self.repeat):
        self.run()
            # self.random_seed = self.random_seed + 5

        self.norm_mean = np.mean(self.norm_matrix, axis=1)
        if not self.balance:
            top_examples = self.train_indx[np.argsort(self.norm_mean)][::-1][:self.coreset_size]
        else:
            top_examples = np.array([], dtype=np.int64)
            for c in tqdm(range(self.num_classes)):
                c_indx = self.train_indx[self.dst_train_label == c]
                budget = round(self.fraction * len(c_indx))
                top_examples = np.append(top_examples, c_indx[np.argsort(self.norm_mean[c_indx])[::-1][:budget]])

        return {"indices": top_examples, "scores": self.norm_mean}

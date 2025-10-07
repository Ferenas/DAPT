from .earlytrain import EarlyTrain
import numpy as np
import torch
from .methods_utils import cossim_np, submodular_function, submodular_optimizer
from ..nets.nets_utils import MyDataParallel


class Submodular(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200, specific_model=None, balance=True,
                 function="GraphCut", greedy="LazyGreedy", metric="cossim", **kwargs):
        super(Submodular, self).__init__(dst_train, args, fraction, random_seed, epochs, specific_model, **kwargs)

        if greedy not in submodular_optimizer.optimizer_choices:
            raise ModuleNotFoundError("Greedy optimizer not found.")
        print(f"The Submodular Method is {function}")
        self._greedy = greedy
        self._metric = metric
        self._function = function

        self.balance = balance

    def before_train(self):
        pass

    def after_loss(self, outputs, loss, targets, batch_inds, epoch):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def before_run(self):
        pass

    def num_classes_mismatch(self):
        raise ValueError("num_classes of pretrain dataset does not match that of the training dataset.")


    def calc_gradient(self, index=None):
        '''
        Calculate gradients matrix on current network for specified training dataset.
        '''
        self.model.eval()
        data_loader = self.select_dm(self.dst_train, index, is_train=False)
        # Initialize a matrix to save gradients.
        # (on cpu)
        gradients = []

        for i, batch in enumerate(data_loader):

            self.optim.zero_grad()
            image, label = batch['img'].cuda(), batch['label'].cuda()
            bs_size = image.shape[0]
            loss,visual_embedding,logit = self.model(image,label,cal_gradient=True)
            embed_dim = visual_embedding.shape[-1]
            with torch.no_grad():
                bias_parameters_grads = torch.autograd.grad(loss, logit)[0]
                weight_parameters_grads = visual_embedding.view(bs_size, 1,
                                        -1).repeat(1, self.num_classes, 1) *\
                                        bias_parameters_grads.view(bs_size, self.num_classes,
                                        1).repeat(1, 1, embed_dim)
                gradients.append(torch.cat([bias_parameters_grads, weight_parameters_grads.flatten(1)],
                                            dim=1).cpu().numpy())

        gradients = np.concatenate(gradients, axis=0,dtype=np.float32)
        print('Finish Gradient Calculation')
        return gradients

    def finish_run(self):
        if isinstance(self.model, MyDataParallel):
            self.model = self.model.module

        # Turn on the embedding recorder and the no_grad flag

        self.model.no_grad = True
        self.train_indx = np.arange(self.n_train)

        gradients = self.calc_gradient(index=None)

        if self.balance:
            selection_result = np.array([], dtype=np.int64)
            for c in range(self.num_classes):
                print(f'class {c}')
                c_indx = self.train_indx[self.dst_train_label == c]
                # Calculate gradients into a matrix
                c_gradients = gradients[c_indx]
                # Instantiate a submodular function
                submod_function = submodular_function.__dict__[self._function](index=c_indx,
                                    similarity_kernel=lambda a, b:cossim_np(c_gradients[a], c_gradients[b]))
                submod_optimizer = submodular_optimizer.__dict__[self._greedy](args=self.args,
                                    index=c_indx, budget=round(self.fraction * len(c_indx)), already_selected=[])

                c_selection_result = submod_optimizer.select(gain_function=submod_function.calc_gain,
                                                             update_state=submod_function.update_state)
                selection_result = np.append(selection_result, c_selection_result)
        else:
            # Calculate gradients into a matrix
            gradients = self.calc_gradient()
            # Instantiate a submodular function
            submod_function = submodular_function.__dict__[self._function](index=self.train_indx,
                                        similarity_kernel=lambda a, b: cossim_np(gradients[a], gradients[b]))
            submod_optimizer = submodular_optimizer.__dict__[self._greedy](args=self.args, index=self.train_indx,
                                                                              budget=self.coreset_size)
            selection_result = submod_optimizer.select(gain_function=submod_function.calc_gain,
                                                       update_state=submod_function.update_state)

        self.model.no_grad = False
        return {"indices": selection_result}

    def select(self, **kwargs):
        selection_result = self.run()
        return selection_result



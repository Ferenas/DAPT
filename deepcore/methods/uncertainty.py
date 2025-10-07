from .earlytrain import EarlyTrain
import torch
import numpy as np
from datasets.data_manager import select_dm_loader
import time

class Uncertainty(EarlyTrain):
    def __init__(self, dst_train, args,fraction=0.5, random_seed=None, epochs=200, selection_method="Margin",
                 specific_model=None, balance=False, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs, specific_model, **kwargs)

        selection_choices = ["LeastConfidence",
                             "Entropy",
                             "Margin"]
        if selection_method not in selection_choices:
            raise NotImplementedError("Selection algorithm unavailable.")
        self.selection_method = selection_method

        self.epochs = epochs
        self.balance = balance

    def before_train(self):
        pass

    def after_loss(self, outputs, loss, targets, batch_inds, epoch):
        pass


    def after_epoch(self):
        pass

    def before_run(self):
        pass

    def num_classes_mismatch(self):
        raise ValueError("num_classes of pretrain dataset does not match that of the training dataset.")

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        pass

    def finish_run(self):
        if self.balance:
            selection_result = np.array([], dtype=np.int64)
            scores = []
            for c in range(self.num_classes):
                print(f"Balance Processing on the train set class {c}")
                class_index = np.arange(self.n_train)[self.dst_train_label == c]
                scores.append(self.rank_uncertainty_clip(class_index))
                selection_result = np.append(selection_result, class_index[np.argsort(scores[-1])[
                                                               :round(len(class_index) * self.fraction)]])
        else:
            print(f"Imbalance Processing on the train set class")
            scores = self.rank_uncertainty_clip()
            selection_result = np.argsort(scores)[::-1][:self.coreset_size]
        return {"indices": selection_result, "scores": scores}

    def rank_uncertainty(self,index=None):
        self.specific_model.eval()
        with torch.no_grad():
            train_loader = torch.utils.data.DataLoader(
                self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
                batch_size=self.args.selection_batch,
                num_workers=self.args.workers)

            scores = np.array([])
            batch_num = len(train_loader)

            for i, (input, _) in enumerate(train_loader):
                if i % self.args.print_freq == 0:
                    print("| Selecting for batch [%3d/%3d]" % (i + 1, batch_num))
                if self.selection_method == "LeastConfidence":
                    scores = np.append(scores, self.model(input.to(self.args.device)).max(axis=1).values.cpu().numpy())
                elif self.selection_method == "Entropy":
                    preds = torch.nn.functional.softmax(self.model(input.to(self.args.device)), dim=1).cpu().numpy()
                    scores = np.append(scores, (np.log(preds + 1e-6) * preds).sum(axis=1))
                elif self.selection_method == 'Margin':
                    preds = torch.nn.functional.softmax(self.model(input.to(self.args.device)), dim=1)
                    preds_argmax = torch.argmax(preds, dim=1)
                    max_preds = preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax].clone()
                    preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax] = -1.0
                    preds_sub_argmax = torch.argmax(preds, dim=1)
                    scores = np.append(scores, (max_preds - preds[
                        torch.ones(preds.shape[0], dtype=bool), preds_sub_argmax]).cpu().numpy())
        return scores


    def rank_uncertainty_clip(self,index=None):
        self.model.eval()
        with torch.no_grad():
            train_loader = select_dm_loader(self.args,self.dst_train,index)
            scores = np.array([])

            for i, batch in enumerate(train_loader):
                # if i % self.args.print_freq == 0:
                #     print("| Selecting for batch [%3d/%3d]" % (i + 1, batch_num))
                image, label = batch['img'].cuda(), batch['label'].cuda()
                logits = self.model(image,label)  ##Eval mode
                if self.selection_method == "LeastConfidence":
                    scores = np.append(scores, logits.max(axis=1).values.cpu().numpy())
                elif self.selection_method == "Entropy":
                    preds = torch.softmax(logits, dim=1).cpu().numpy()
                    scores = np.append(scores, (np.log(preds + 1e-6) * preds).sum(axis=1))
                elif self.selection_method == 'Margin':
                    preds = torch.softmax(logits, dim=1)
                    preds_argmax = torch.argmax(preds, dim=1)
                    max_preds = preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax].clone()
                    preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax] = -1.0
                    preds_sub_argmax = torch.argmax(preds, dim=1)
                    scores = np.append(scores, (max_preds - preds[torch.ones(preds.shape[0], dtype=bool), preds_sub_argmax]).cpu().numpy())
        self.model.train()
        return scores


    def select(self, **kwargs):
        selection_result = self.run()
        return selection_result

    def select_without_train(self):
        selection_result = self.finish_run()
        return selection_result

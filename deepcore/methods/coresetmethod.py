import numpy as np
import os

class CoresetMethod(object):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None,**kwargs):
        if fraction <= 0.0 or fraction > 1.0:
            raise ValueError("Illegal Coreset Size.")

        self.dm = dst_train
        self.dst_train = dst_train.dataset.train_x
        self.num_classes = dst_train.dataset.num_classes
        self.fraction = fraction
        self.random_seed = random_seed
        self.index = []
        self.args = args
        self.dst_train_label = self.get_train_label(self.dst_train)
        self.n_train = len(self.dst_train)
        self.coreset_size = round(self.n_train * fraction)
        self.max_epoch = self.args.OPTIM_SELECTION.MAX_EPOCH

    def select(self, **kwargs):
        return

    def get_train_label(self,dst_train):
        ####Readable
        ind = []
        for i,item in enumerate(dst_train):
            ind.append(item.label)
        return np.asarray(ind)
    def pre_run(self):
        self.train_indx = np.arange(self.n_train)
        print(f'Start pre-funing CLIP with all datasets by {self.max_epoch} epoch')
        file_save_name = self.args.DATASET.NAME + '_' + str(self.args.SEED) + '.pth'
        output_checkpoint_dir = os.path.join('checkpoints', file_save_name)
        if self.max_epoch > 0:

            if os.path.exists(output_checkpoint_dir):
                print(f'The checkpiont exists! Load that shit')
                ckpt = torch.load(output_checkpoint_dir)
                self.model.load_state_dict(ckpt)
            else:
                for epoch in range(self.epoch, self.max_epoch):
                    # list_of_train_idx = np.random.choice(np.arange(self.n_pretrain if self.if_dst_pretrain else self.n_train),
                    #                                      self.n_pretrain_size, replace=False)
                    self.before_epoch()  # PASS
                    self.train(epoch)
                    self.test(epoch)
                    self.after_epoch()
        torch.save(self.model.state_dict(), output_checkpoint_dir)

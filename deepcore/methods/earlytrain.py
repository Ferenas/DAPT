from .coresetmethod import CoresetMethod
import torch, time
from torch import nn
import numpy as np
from copy import deepcopy
from .. import nets
from torchvision import transforms
from datasets.data_manager import select_dm_loader
from dassl.utils import MetricMeter, AverageMeter
from torch.cuda.amp import GradScaler, autocast
import datetime
from tqdm import tqdm
import os

class EarlyTrain(CoresetMethod):
    '''
    Core code for training related to coreset selection methods when pre-training is required.
    '''

    def __init__(self, dst_train, args,fraction=0.5, random_seed=None, epochs=200, specific_model=None,
                 torchvision_pretrain: bool = False, dst_pretrain_dict: dict = {}, fraction_pretrain=1., dst_test=None,
                 **kwargs):
        super().__init__(dst_train, args, fraction, random_seed)
        self.epochs = epochs
        self.n_train = len(self.dst_train)
        self.coreset_size = round(self.n_train * fraction)
        self.model = specific_model
        self.train_loader = self.dm.train_loader_x
        self.test_loader = self.dm.test_loader


        if kwargs:
            # self.text_feature = kwargs['text_feature']
            self.optim = kwargs['optim']
            self.sche = kwargs['schedule']
            self.scar = kwargs['scar']



        self.start_epoch = self.epoch = 0
        self.max_epoch = self.args.OPTIM_SELECTION.MAX_EPOCH

        if fraction_pretrain <= 0. or fraction_pretrain > 1.:
            raise ValueError("Illegal pretrain fraction value.")
        self.fraction_pretrain = fraction_pretrain

        if dst_pretrain_dict.__len__() != 0:
            dict_keys = dst_pretrain_dict.keys()
            if 'im_size' not in dict_keys or 'channel' not in dict_keys or 'dst_train' not in dict_keys or \
                    'num_classes' not in dict_keys:
                raise AttributeError(
                    'Argument dst_pretrain_dict must contain imszie, channel, dst_train and num_classes.')
            if dst_pretrain_dict['im_size'][0] != args.im_size[0] or dst_pretrain_dict['im_size'][0] != args.im_size[0]:
                raise ValueError("im_size of pretrain dataset does not match that of the training dataset.")
            if dst_pretrain_dict['channel'] != args.channel:
                raise ValueError("channel of pretrain dataset does not match that of the training dataset.")
            if dst_pretrain_dict['num_classes'] != args.num_classes:
                self.num_classes_mismatch()

        self.dst_pretrain_dict = dst_pretrain_dict
        self.torchvision_pretrain = torchvision_pretrain
        self.if_dst_pretrain = (len(self.dst_pretrain_dict) != 0)

        if torchvision_pretrain:
            # Pretrained models in torchvision only accept 224*224 inputs, therefore we resize current
            # datasets to 224*224.
            if args.im_size[0] != 224 or args.im_size[1] != 224:
                self.dst_train = deepcopy(dst_train)
                self.dst_train.transform = transforms.Compose([self.dst_train.transform, transforms.Resize(224)])
                if self.if_dst_pretrain:
                    self.dst_pretrain_dict['dst_train'] = deepcopy(dst_pretrain_dict['dst_train'])
                    self.dst_pretrain_dict['dst_train'].transform = transforms.Compose(
                        [self.dst_pretrain_dict['dst_train'].transform, transforms.Resize(224)])
        if self.if_dst_pretrain:
            self.n_pretrain = len(self.dst_pretrain_dict['dst_train'])
        self.n_pretrain_size = round(
            self.fraction_pretrain * (self.n_pretrain if self.if_dst_pretrain else self.n_train))
        self.dst_test = dst_test


    def train(self, epoch, list_of_train_idx=None, **kwargs):
        """ Train model for one epoch """

        self.before_train()
        self.model.train()

        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()


        end = time.time()

        print('\n=> Training Pre-tuning Epoch #%d' % epoch)
        train_loader = select_dm_loader(self.args,self.dst_train,is_train=True)
        self.num_batches = len(train_loader)

        # trainset_permutation_inds = np.random.permutation(list_of_train_idx)
        # batch_sampler = torch.utils.data.BatchSampler(trainset_permutation_inds, batch_size=self.args.selection_batch,
        #                                               drop_last=False)
        # trainset_permutation_inds = list(batch_sampler)
        #
        # train_loader = torch.utils.data.DataLoader(self.dst_pretrain_dict['dst_train'] if self.if_dst_pretrain
        #                                            else self.dst_train, shuffle=False, batch_sampler=batch_sampler,
        #
        #
        #                                            num_workers=self.args.workers, pin_memory=True)

        for i, batch in enumerate(train_loader):
            data_time.update(time.time() - end)
            image, label,real_ind = batch['img'].cuda(),batch['label'].cuda(),batch['index'].cuda()

            model = self.model
            optim = self.optim
            scaler = self.scar

            prec = self.args.TRAINER.MAPLE.PREC
            if prec == "amp":
                with autocast():
                    loss,outputs = model(image, label)
                optim.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss,outputs = model(image, label)
                optim.zero_grad()
                loss.backward()
                optim.step()

            self.after_loss(outputs, loss, label, real_ind, epoch)
            self.while_update(outputs, loss, label, epoch, i, self.args.DATALOADER.TRAIN_X.BATCH_SIZE)

            loss_summary = {"loss": loss.item()}

            if (i + 1) == self.num_batches:
                self.sche.step()
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (i + 1) % self.args.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.args.TRAIN.PRINT_FREQ

            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - i - 1
                nb_remain += (self.max_epoch - self.epoch - 1) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{i + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {optim.param_groups[0]['lr']:.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            # n_iter = self.epoch * self.num_batches + i
            # for name, meter in losses.meters.items():
            #     self.write_scalar("train/" + name, meter.avg, n_iter)
            # self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

        return self.finish_train()

    def run(self):
        self.train_indx = np.arange(self.n_train)
        self.before_run()
        print(f'Start pre-funing CLIP with all datasets by {self.max_epoch} epoch')
        file_save_name = self.args.DATASET.NAME + '_' + str(self.args.SEED) + '.pth'
        output_checkpoint_dir = os.path.join('checkpoints', file_save_name)
        if self.max_epoch > 0:

            if os.path.exists(output_checkpoint_dir):
                print(f'The checkpiont exists! Load that shit')
                ckpt = torch.load(output_checkpoint_dir)
                self.model.load_state_dict(ckpt)
            else:
                for epoch in range(self.epoch,self.max_epoch):
                    # list_of_train_idx = np.random.choice(np.arange(self.n_pretrain if self.if_dst_pretrain else self.n_train),
                    #                                      self.n_pretrain_size, replace=False)
                    self.before_epoch()  #PASS
                    self.train(epoch)
                    self.test(epoch)
                    self.after_epoch()
        torch.save(self.model.state_dict(),output_checkpoint_dir)

        return self.finish_run()

    def test(self, epoch):
        self.model.no_grad = True
        self.model.eval()


        correct = 0.
        total = 0.

        print('\n=> Testing Tuning Epoch #%d' % epoch)

        for batch_idx, batch in enumerate(self.test_loader):
            image, target = batch['img'].cuda(), batch['label']
            output = self.model(image, target.cuda())


            predicted = torch.max(output.data, 1).indices.cpu()
            correct += predicted.eq(target).sum().item()
            total += target.size(0)

            # if batch_idx % self.args.print_freq == 0:
            #     print('| Test Epoch [%3d/%3d] Iter[%3d/%3d]\t\t Test Acc: %.3f%%' % (
            #         epoch, self.epochs, batch_idx + 1, (round(len(self.dst_test) * self.args.selection_test_fraction) //
            #                                             self.args.selection_batch) + 1, loss.item(),
            #         100. * correct / total))
        print(f'| Test Epoch {epoch} Test Acc: {100. * correct / total:.3f}%')
        self.model.no_grad = False

    def num_classes_mismatch(self):
        pass

    def before_train(self):
        pass

    def after_loss(self, outputs, loss, targets, batch_inds, epoch):
        pass

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        pass

    def finish_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def before_run(self):
        pass

    def finish_run(self):
        pass

    def select(self, **kwargs):
        selection_result = self.run()
        return selection_result

    def select_without_train(self, **kwargs):
        return self.finish_run()

    @torch.no_grad()
    def calcluate_clip_probability(self,batch):
        input = batch["img"].cuda()

        self.specific_model = self.specific_model.cuda()
        image_features = self.specific_model.encode_image(input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.specific_model.logit_scale.exp()
        return logit_scale * image_features @ self.text_feature.t()

    # using the defined select_dm
    def select_dm(self,data,ind=None,is_train=None):
        return select_dm_loader(self.args,data,ind,is_train)


    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]

        input = input.cuda()
        label = label.cuda()

        return input, label

    def parse_batch_train(self, batch):
        input = batch["img"].cuda()
        label = batch["label"].cuda()
        domain = batch["index"].cuda()

        return input, label, domain



    def calc_gradient(self, index=None):
        '''
        Calculate gradients matrix on current network for specified training dataset.
        '''
        self.model.eval()
        data_loader = self.select_dm(self.dst_train, index, is_train=False)
        # Initialize a matrix to save gradients.
        # (on cpu)
        gradients = []
        lam = 0.5
        for i, batch in enumerate(tqdm(data_loader)):
            self.optim.zero_grad()
            image, label = batch['img'].cuda(), batch['label'].cuda()
            bs_size = image.shape[0]
            loss, visual_embedding, logit= self.model(image, label, cal_gradient=True)
            embed_dim = visual_embedding.shape[-1]
            with torch.no_grad():
                bias_parameters_grads = torch.autograd.grad(loss, logit)[0]
                weight_parameters_grads = visual_embedding.view(bs_size, 1,
                                                                -1).repeat(1, self.num_classes, 1) * \
                                          bias_parameters_grads.view(bs_size, self.num_classes,
                                                                     1).repeat(1, 1, embed_dim)
                # weight_parameters_grads_t = text_embedding.view(bs_size, 1,
                #                                                 -1).repeat(1, self.num_classes, 1) * \
                #                           bias_parameters_grads.view(bs_size, self.num_classes,
                #                                                      1).repeat(1, 1, embed_dim)
                # final_weight = torch.abs(weight_parameters_grads-weight_parameters_grads_t)
                gradients.append(torch.cat([bias_parameters_grads, weight_parameters_grads.flatten(1)],
                                           dim=1).cpu().numpy())

        gradients = np.concatenate(gradients, axis=0, dtype=np.float32)
        print('Finish Gradient Calculation')
        self.model.train()
        return gradients


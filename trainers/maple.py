import os.path as osp
import random
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
import time
import os
import pickle
import deepcore.methods as s_method
import numpy as np

from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint, mkdir_if_missing
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.evaluation import Classification,EvaluatorBase
from pygrad.pcgrad import PCGrad
from datasets.data_manager import DataManager
from dassl.data.datasets import build_dataset


from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from trainers.zsclip import CUSTOM_TEMPLATES
from .coop import load_clip_to_cpu as lcp
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix
from collections import OrderedDict, defaultdict
from .util import GradCAM,denorm
import cv2
_tokenizer = _Tokenizer()

BACKGROUND_CATEGORY = ['ground','land','grass','tree','building','wall','sky','lake','water','river','sea','railway','railroad','keyboard','helmet',
                        'cloud','house','mountain','ocean','road','rock','street','valley','bridge','sign',]


#['ground','land','grass','tree','building','wall','sky','lake','water','river','sea','railway','railroad','keyboard','helmet',
                        #'cloud','house','mountain','ocean','road','rock','street','valley','bridge','sign',
                        #]

BACKGROUND_CATEGORY_FOOD = ['table','forks','tablecloth','hands','spoon','glasses','dishes']

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg.TRAINER.MAPLE.N_CTX}

    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model





class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.MAPLE.N_CTX  # n_ctx
        ctx_init = cfg.TRAINER.MAPLE.CTX_INIT  # a photo of
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]  #512
        clip_imsize = clip_model.visual.input_resolution  #224
        cfg_imsize = cfg.INPUT.SIZE[0]  #224
        # Default is 1, which is compound shallow prompting
        assert cfg.TRAINER.MAPLE.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.TRAINER.MAPLE.PROMPT_DEPTH  #9 # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 768)
        self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)  #[2 512]
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)







        classnames = [name.replace("_", " ") for name in classnames]

        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)


        ###Introduce Background
        bg_template = 'a clean origami {}.'

        bg_classesnames = [bg_template.format(name) for name in  BACKGROUND_CATEGORY +BACKGROUND_CATEGORY_FOOD ]
        tokenized_bg_prompts = torch.cat([clip.tokenize(bg) for bg in bg_classesnames])
        bg_num =  len(BACKGROUND_CATEGORY) + len(BACKGROUND_CATEGORY_FOOD)
        tokenized_prompts = torch.cat((tokenized_prompts,tokenized_bg_prompts),dim=0)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            self.bg_embeding = embedding[-bg_num:]

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:-bg_num, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:-bg_num, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor [class_num 77]  [:-bg_num]
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)



        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        final_prompts = torch.cat((prompts,self.bg_embeding.cuda()),dim=0)
        return final_prompts

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts  # pass here original, as for visual 768 is required


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.image_encoder_ori = clip_model.visual_ori
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.txt_f = []
        self.img_f = []
        self.one_hot_label = []
        self.vtx = []
        self.loaded_mask = None
        # self.loss_weights = torch.nn.Parameter(torch.tensor([0.8,0.03],dtype=self.dtype))



    def get_uniform_ball_noise(self,input_shape,radius=1.0):
        uniform_noise_ball = torch.randn(input_shape).cuda()
        uniform_noise_sphere = F.normalize(uniform_noise_ball,dim=1)
        u = torch.rand(input_shape[0]).cuda()
        u = u **(1. / input_shape[1])
        uniform_noise_ball = (uniform_noise_sphere.T *u *radius).T
        return uniform_noise_ball.type(self.dtype)


    def get_learnable_noise(self,input_shape):
        para = 0.05
        noise = torch.nn.Parameter(torch.randn(input_shape)*para).cuda()

        return noise.type(self.dtype)

    def cos_sim(self,a,b):
        return F.cosine_similarity(a,b)

    def forward(self, image, label=None,record=False,cal_gradient=False,weight=None,epoch=None,index=None,cfg=None,mask=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        text_features_fg = text_features[:-len(BACKGROUND_CATEGORY)]
        ori_image_input = image.type(self.dtype)
        # text_features = text_features + self.get_learnable_noise(text_features.shape)

        text_features_fg = text_features_fg / text_features_fg.norm(dim=-1, keepdim=True)

        image_features, visual_ctx, mask_similarity = self.image_encoder(ori_image_input, shared_ctx,
                                                                         deep_compound_prompts_vision)



        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # if label is not None:
        #     image_features = image_features + self.get_uniform_ball_noise(image_features.shape)

        logits = logit_scale * image_features @ text_features_fg.t()





        if mask != None:



            text_features_bg = text_features[-len(BACKGROUND_CATEGORY):]
            text_features_bg = text_features_bg / text_features_bg.norm(dim=-1, keepdim=True)
            image_features_fg,_,_ = self.image_encoder(ori_image_input*mask, shared_ctx, deep_compound_prompts_vision)  #, shared_ctx, deep_compound_prompts_vision


            image_features_fg = image_features_fg / image_features_fg.norm(dim=-1, keepdim=True)
            image_features_bg,_,_ = self.image_encoder(ori_image_input*(1-mask), shared_ctx, deep_compound_prompts_vision)
            image_features_bg = image_features_bg / image_features_bg.norm(dim=-1, keepdim=True)


            loss_re1 = F.triplet_margin_loss(image_features,image_features_fg.detach(),image_features_bg.detach(),margin=1.5)

            # image_features_fg_ori = self.image_encoder_ori(ori_image_input*mask_random)
            # image_features_bg_ori = self.image_encoder_ori(ori_image_input*(1-mask_random))
            # image_features_fg_ori = image_features_fg_ori / image_features_fg_ori.norm(dim=-1, keepdim=True)
            # image_features_bg_ori = image_features_bg_ori / image_features_bg_ori.norm(dim=-1,keepdim=True)
            # image_features_all_ori = image_features_fg_ori + image_features_bg_ori
            # image_features_all_ori = image_features_all_ori / image_features_all_ori.norm(dim=-1,keepdim=True)
            # loss_reo = torch.abs(image_features_all_ori.detach() - image_features).mean()

            foreground_score = logit_scale*image_features_fg.detach()@text_features_fg.t()
            pseudo_label = torch.argmax(image_features_bg @ text_features_bg.t(), dim=-1)
            logits_bg = logit_scale*(image_features_bg) @ text_features_bg.t()

            para_bg = 0.5
            para_fg = 0.1
            para_vd = 0.8


            loss_bg = F.cross_entropy(logits_bg,pseudo_label)
            loss_fg = F.cross_entropy(foreground_score,label)

            if epoch > 6:  #Tunable parameters
                loss_re =  para_fg*loss_fg + para_bg*loss_bg
            else:
                loss_re =  para_vd*loss_re1 #loss_reo would be effective in base2novel setting


        if self.prompt_learner.training:
            if weight is None:
                return F.cross_entropy(logits,label)+loss_re,logits,{'loss_vd':loss_re1.item(),'loss_bg':loss_bg.item(),'loss_fg':loss_fg.item()}
            else:
                return F.cross_entropy(weight.unsqueeze(-1)*logits,label), logits

        if record: #store the embeeding
            one_hot_label = F.one_hot(label,num_classes=text_features.shape[0]).to(torch.float16)
            return image_features.detach(),(one_hot_label @ text_features).detach(), logits

        if cal_gradient:
            #Treating this as initial gradient
            # one_hot_label = F.one_hot(label,num_classes=text_features.shape[0]).to(torch.float16)
            return F.cross_entropy(logits.requires_grad_(True), label), image_features.detach(), logits #,(one_hot_label @ text_features).detach()
        return logits

    def grad_norm(self,loss_group,original_loss_group):
        alpha = 0.10
        self.loss_weights.grad.data = self.loss_weights.grad.data * 0.0
        W = self.prompt_learner.compound_prompt_projections[0]
        norms = []
        for i in range(len(loss_group)):
            gygw = torch.autograd.grad(loss_group[i],W.parameters(),retain_graph=True)
            norms.append(torch.norm(torch.mul(self.loss_weights[i],gygw[0])))
        norms = torch.stack(norms)
        loss_ratio = loss_group.data.cpu().numpy() / original_loss_group
        inverse_train_rate = loss_ratio / np.mean(loss_ratio)
        mean_norm = np.mean(norms.data.cpu().numpy())
        constant_norm = torch.tensor(mean_norm*(inverse_train_rate**alpha),requires_grad=False).cuda()
        grad_norm_loss = torch.sum(torch.abs(norms - constant_norm))



        self.loss_weights.grad = torch.autograd.grad(grad_norm_loss,self.loss_weights)[0]




    def forward_test(self, image, label=None,record=False,cal_gradient=False,weight=None,cfg=None,attn_mask=False):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        image_features,visual_ctx,mask = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)


        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()


        if self.prompt_learner.training:
            if weight is None:
                return F.cross_entropy(logits, label),logits
            else:
                return F.cross_entropy(weight.unsqueeze(-1)*logits,label), logits

        if record: #store the embeeding
            one_hot_label = F.one_hot(label,num_classes=text_features.shape[0]).to(torch.float16)
            return image_features.detach(),(one_hot_label @ text_features).detach(), logits
        if attn_mask:
            return logits,mask
        if cal_gradient:
            #Treating this as initial gradient
            # one_hot_label = F.one_hot(label,num_classes=text_features.shape[0]).to(torch.float16)
            return F.cross_entropy(logits.requires_grad_(True), label), image_features.detach(), logits #,(one_hot_label @ text_features).detach()
        return logits

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@TRAINER_REGISTRY.register()
class MaPLe(TrainerX):



    def check_cfg(self, cfg):
        assert cfg.TRAINER.MAPLE.PREC in ["fp16", "fp32", "amp"]



    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.MAPLE.PREC == "fp32" or cfg.TRAINER.MAPLE.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():


            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)


        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # self.model.loss_weights.requires_grad_(True)  #open gradient for loss_weights
        # NOTE: only give prompt_learner to the optimizer


        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)



        self.selected_optim = build_optimizer(self.model, cfg.OPTIM_SELECTION)
        self.selected_sched = build_lr_scheduler(self.optim, cfg.OPTIM_SELECTION)

        self.register_model("MultiModalPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.MAPLE.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)


    # def generate_text_feature(self):
    #     cfg = self.cfg
    #     classnames = self.dm.dataset.classnames
    #     #
    #     # print(f"Loading Custom CLIP (backbone: {cfg.MODEL.BACKBONE.NAME}) for selection")
    #     # clip_model = lcp(cfg)
    #     # clip_model.to(self.device)
    #
    #     temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
    #     prompts = [temp.format(c.replace("_", " ")) for c in classnames]
    #     print(f"Prompts: {prompts}")
    #     prompts = torch.cat([clip.tokenize(p) for p in prompts])
    #     prompts = prompts.to(self.device)
    #
    #     p, _, deep_compound_prompts_text, _ = self.model.prompt_learner()
    #     with torch.no_grad():
    #         text = self.model.text_encoder(prompts)
    #         text_features = self.model.encode_text(prompts, tokenized_prompts, deep_compound_prompts_text)
    #         text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    #
    #     self.ori_text_features = text_features






    def forward_backward(self, batch):
        if self.sample_weights is not None:
            image, label,index,mask = self.parse_batch_train_pair(batch)
        else:
            image, label,index,mask = self.parse_batch_train_pair(batch)
            weight = None

        model = self.model
        optim = self.optim
        scaler = self.scaler


        prec = self.cfg.TRAINER.MAPLE.PREC
        if prec == "amp":
            with autocast():
                loss,_ = model(image, label, weight=weight,mask=mask)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss,_,loss_dict = model(image, label, weight=weight,epoch=self.epoch,index=index,cfg=self.cfg,mask=mask)
            optim.zero_grad()

            # optim.pc_backward(loss_task)
            loss.backward()
            # if self.epoch == 0:
            #     self.loss_o1 = loss_task.data.cpu().numpy()
            # model.grad_norm(loss_task,self.loss_o1)

            optim.step()

        # normalized_coeff = 2 / torch.sum(model.loss_weights.data,dim=0)
        # model.loss_weights.data *= normalized_coeff




        loss_summary = loss_dict

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()


        return loss_summary

    def parse_batch_train_pair(self, batch):
        input = batch["img"]
        label = batch["label"]
        index = batch["index"]
        mask = batch['mask']
        input = input.to(self.device)
        label = label.to(self.device)
        mask = mask.to(self.device)

        if self.sample_weights is not None:
            # weight = batch['weight'].cuda()
            return input, label,index,mask
        else:
            return input, label,index,mask



    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        index = batch["index"]
        input = input.to(self.device)
        label = label.to(self.device)


        if self.sample_weights is not None:
            weight = batch['weight'].cuda()
            return input, label,weight,index
        else:
            return input, label,index

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    def before_train(self):
        directory = self.cfg.OUTPUT_DIR
        if self.cfg.RESUME:
            directory = self.cfg.RESUME
        # self.start_epoch = self.resume_model_if_exist(directory)  #in case of loading pre-trained weight


        # Redefine the dataloader
        selected_res = self.selector()
        if 'weights' in selected_res:
            c_weight = np.zeros(len(self.dm.dataset.train_x))
            c_weight[selected_res['indices']] = selected_res['weights']
            self.sample_weights = c_weight[selected_res['indices']]
        else:
            self.sample_weights = None



        self.build_final_data_loader(selected_res['indices'],self.sample_weights)
        print(f'Finish the selecting process, now continue tune CLIP')
        # Initialize summary writer
        writer_dir = osp.join(self.output_dir, "tensorboard")
        mkdir_if_missing(writer_dir)
        self.init_writer(writer_dir)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()

        print(f"Now generate the attentive masking in {self.cfg.TRAINER.DAPT_MODE} \n")


        if self.cfg.TRAINER.DAPT_MODE == 'dapt-s':
            self.generate_mask_train()
        else:
            self.generate_gradcam_train(split='train')



    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False)

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar"
                )

        # if meet_checkpoint_freq or last_epoch:
        #     self.save_model(self.epoch, self.output_dir)

        print(f"Now generate the attentive masking in {self.cfg.TRAINER.DAPT_MODE} \n")


        if self.cfg.TRAINER.DAPT_MODE == 'dapt-s':
            self.generate_mask_train()
        else:
            self.generate_gradcam_train(split='train')




    def build_final_data_loader(self,selected_ind=None,weight=None):
        new_dm = DataManager(self.cfg,self.dm.dataset,selected_ind,weight=weight)
        self.train_loader_x = new_dm.train_loader_x
        self.train_loader_xmore = new_dm.train_loader_xmore  #for generate the attentive masking
        self.mask_list = torch.zeros((selected_ind.shape[0], 1, *self.cfg.INPUT.SIZE),dtype=torch.float16)

    def selector(self):
        selection_ratio = self.cfg.DATASET.SELECTION_RATIO
        seed = self.cfg.SEED
        method = self.cfg.DATASET.SELECTION_METHOD
        print(f"Selecting {selection_ratio*100}% data by {method}")

        if self.cfg.DATASET.SELECTION_METHOD == 'Uniform':


            selector = s_method.Uniform(self.dm, self.cfg,selection_ratio, seed)
        else:

            selector = s_method.__dict__[method](dst_train=self.dm,
                                                 args=self.cfg,
                                                 fraction=selection_ratio,
                                                 random_seed=seed,
                                                 specific_model=self.model,
                                                 optim = self.selected_optim,
                                                 schedule = self.selected_sched,
                                                 scar = self.scaler,
                                                 balance = True
                                                 )


        return selector.select()

    @torch.no_grad()
    def test_withlabel(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        new_estimate = NewClassification(self.cfg,self.evaluator._lab2cname)
        new_estimate.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)

            output = self.model.forward_test(input,label,cfg = self.cfg)
            new_estimate.process(output, label)

        results = new_estimate.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]


    def generate_gradcam(self, split=None,attn_mask=False):
        """A generic pipeline for generating GradCAM"""
        self.set_model_mode("eval")
        model_dict = {'arch':self.model,'layer_name':'target.layer'}
        cam = GradCAM(model_dict)
        # new_estimate = NewClassification(self.cfg,self.evaluator._lab2cname)
        # new_estimate.reset()

        img_split = 'wrong'  #true/wrong
        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Generate GradCAM on the *{split}* set")

        save_path = self.cfg.OUTPUT_DIR + '/'+f'{split}_{img_split}_promptcamother'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            img_name = batch['impath'][0].split('/')[-1]
            img_save_path = os.path.join(save_path, img_name)
            img0 = denorm(batch['img0'].numpy(),self.cfg.INPUT.PIXEL_MEAN,self.cfg.INPUT.PIXEL_STD)
            saliency_map = cam.forward(input,label,cfg = self.cfg,split=img_split,attn_mask=attn_mask)
            if saliency_map != None:
                final_map = cam.show_cam(img0,saliency_map.detach().cpu(),img_save_path)




    def generate_mask_train(self):
        for batch_idx, batch in enumerate(tqdm(self.train_loader_xmore)):
            input, _, index = self.parse_batch_train(batch)
            b,c,h,w = input.shape
            mask = torch.ones((1,h,w),dtype=torch.float16)
            grid_sizes = [32,16]
            hide_prob = 0.5
            grid_size = grid_sizes[torch.randint(0,len(grid_sizes),size=(1,))]

            if (grid_size != 0):
                for x in range(0,h,grid_size):
                    for y in range(0,w,grid_size):
                        x_end,y_end = min(h, x+grid_size),min(w,y+grid_size)
                        if (random.random() <= hide_prob):
                            mask[:,x:x_end,y:y_end] = 0
            self.mask_list[index, :] = mask
        self.model.loaded_mask = self.mask_list


    def generate_mask_bg(self):
        for batch_idx, batch in enumerate(tqdm(self.train_loader_xmore)):
            input, _, index = self.parse_batch_train(batch)
            b,c,h,w = input.shape
            mask = torch.ones((1,h,w),dtype=torch.float16)
            grid_sizes = [64,128]
            hide_prob = 0.5
            grid_size = grid_sizes[torch.randint(0,len(grid_sizes),size=(1,))]

            if (grid_size != 0):
                for x in range(0,h,grid_size):
                    for y in range(0,w,grid_size):
                        x_end,y_end = min(h, x+grid_size),min(w,y+grid_size)
                        if (random.random() <= hide_prob):
                            mask[:,x:x_end,y:y_end] = 0
            self.mask_list[index, :] = mask
        self.model.loaded_mask = self.mask_list


    def generate_gradcam_train(self, split=None,attn_mask=False):
        """A generic pipeline for generating GradCAM"""
        self.set_model_mode("eval")
        model_dict = {'arch':self.model,'layer_name':'target.layer'}
        cam = GradCAM(model_dict)
        # new_estimate = NewClassification(self.cfg,self.evaluator._lab2cname)
        # new_estimate.reset()

        print(f"Generate GradCAM on the *{split}* set")

        # save_path = self.cfg.OUTPUT_DIR + '/'+f'{split}_{img_split}_promptcamother'
        # if not os.path.exists(save_path):
        #     os.mkdir(save_path)
        for batch_idx, batch in enumerate(tqdm(self.train_loader_xmore)):
            input, label, index = self.parse_batch_train(batch)
            # img0 = denorm(batch['img0'].numpy(),self.cfg.INPUT.PIXEL_MEAN,self.cfg.INPUT.PIXEL_STD)
            saliency_map = cam.forward_train(input,label,cfg = self.cfg,attn_mask=attn_mask)
            self.mask_list[index,:] = saliency_map.detach().cpu()
            # if saliency_map != None:
            #     final_map = cam.show_cam(img0,saliency_map.detach().cpu(),img_save_path)
        self.model.loaded_mask = self.mask_list


class NewClassification(Classification):
    def __init__(self, cfg, lab2cname=None, **kwargs):
        super(NewClassification, self).__init__(cfg,lab2cname)
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)

    def evaluate(self):
        results = OrderedDict()
        acc = 100.0 * self._correct / self._total
        err = 100.0 - acc
        macro_f1 = 100.0 * f1_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true)
        )

        # The first value will be returned by trainer.test()
        results["accuracy"] = acc
        results["error_rate"] = err
        results["macro_f1"] = macro_f1

        wrong_ind = np.array(self._y_true) != np.array(self._y_pred)
        np.save(self.cfg.OUTPUT_DIR + '/'+'wrongind.npy',wrong_ind)
        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* correct: {self._correct:,}\n"
            f"* accuracy: {acc:.1f}%\n"
            f"* error: {err:.1f}%\n"
            f"* macro_f1: {macro_f1:.1f}%"
        )

        if self._per_class_res is not None:
            labels = list(self._per_class_res.keys())
            labels.sort()

            print("=> per-class result")
            accs = []

            for label in labels:
                classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                acc = 100.0 * correct / total
                accs.append(acc)

                print(
                    f"* class: {label} ({classname})\t"
                    f"total: {total:,}\t"
                    f"correct: {correct:,}\t"
                    f"acc: {acc:.1f}%"
                )

            mean_acc = np.mean(accs)
            np.save(self.cfg.OUTPUT_DIR + '/'+'per-class.npy',{'per_cls':accs, 'mean_acc':mean_acc})
            print(f"* average: {mean_acc:.1f}%")

            results["perclass_accuracy"] = mean_acc

        if self.cfg.TEST.COMPUTE_CMAT:
            cmat = confusion_matrix(
                self._y_true, self._y_pred, normalize="true"
            )
            save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
            torch.save(cmat, save_path)
            print(f"Confusion matrix is saved to {save_path}")

        return results

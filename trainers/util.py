import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import os

BACKGROUND_CATEGORY = ['ground','land','grass','tree','building','wall','sky','lake','water','river','sea','railway','railroad','keyboard','helmet',
                        'cloud','house','mountain','ocean','road','rock','street','valley','bridge','sign',
                        ]

class GradCAM(object):
    def __init__(self,model_dict):
        layer_name = model_dict['layer_name']
        self.model_arch = model_dict['arch']

        self.gradient = dict()
        self.activation = dict()

        self.gradient_t = dict()
        self.activation_t = dict()

        def backward_hook(module,grad_input,grad_output):
            self.gradient['value'] = grad_output[0]
            return None

        def forward_hook(module,input,output):
            self.activation['value'] = output
            return None

        def backward_hook_t(module,grad_input,grad_output):
            self.gradient_t['value'] = grad_output[0]
            return None

        def forward_hook_t(module,input,output):
            self.activation_t['value'] = output
            return None

        target_layer = self.model_arch.image_encoder.transformer.resblocks[-1].ln_1
        # target_layer_t = self.model_arch.image_encoder.transformer.resblocks[-2].mlp.c_proj
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        # target_layer_t.register_forward_hook(forward_hook_t)
        # target_layer_t.register_backward_hook(backward_hook_t)

    def forward(self,input,labels,cfg=None,retain_graph=False,split=None,attn_mask=False):




        b,c,h,w = input.shape
        patch_num,ori_size = self.model_arch.image_encoder.patch_num, self.model_arch.image_encoder.input_resolution

        if attn_mask:
            logit,mask = self.model_arch.forward_test(input,labels,cfg=cfg,attn_mask=attn_mask)
            cls_mask = mask[:,1:-self.model_arch.prompt_learner.n_ctx,:1].reshape(b,-1,patch_num,patch_num) #+ mask[:,1:-self.model_arch.prompt_learner.n_ctx,:1].permute(0,2,1)
            aff = mask[:,1:-self.model_arch.prompt_learner.n_ctx, 1:-self.model_arch.prompt_learner.n_ctx]
            # aff = (aff + aff.permute(0,2,1)) / 2
            aff = aff / (aff.sum(dim=1,keepdim=True) + 1e-6)
            # aff = aff / (aff.sum(dim=1,keepdim=True) + 1e-6)


            # aff = (aff + aff.permute(0,2,1)) / 2

            # aff = torch.bmm(aff,aff)

            # aff = F.softmax(aff,dim=1)
            # cls_mask = torch.bmm(cls_mask, aff).reshape(b,-1,patch_num,patch_num)


            # cls_mask = mask[:,1:-self.model_arch.prompt_learner.n_ctx,:1].permute(0,2,1).reshape(b,-1,patch_num,patch_num)
            # # cls_mask = mask[:,-self.model_arch.prompt_learner.n_ctx:,1:-self.model_arch.prompt_learner.n_ctx].reshape(b,-1,patch_num,patch_num).mean(dim=1,keepdim=True)
            # final_cls_mask = F.upsample(cls_mask, size=(ori_size, ori_size), mode='bilinear',
            #                                   align_corners=True)
            # final_cls_feature_min, final_cls_feature_max = final_cls_mask.min(), final_cls_mask.max()
            # final_cls_mask = (final_cls_mask - final_cls_feature_min) / (
            #             final_cls_feature_max - final_cls_feature_min + 1e-6)
            # final_cls_mask = final_cls_mask / (final_cls_mask.max() + 1e-6)

        else:
            logit = self.model_arch.forward_test(input,labels,cfg=cfg)
        pred_label = torch.argmax(logit[:,:-len(BACKGROUND_CATEGORY)])
        sign = pred_label == labels
        # if (split == 'true' and sign == False) or (split == 'wrong' and sign == True):
        #     print(f'Ignore the not {split} sample')
        #     return None

        # if attn_mask:
        #     return final_cls_mask
        pred = logit[:,:-len(BACKGROUND_CATEGORY)].argmax(dim=-1)
        background_logit = logit[:,-len(BACKGROUND_CATEGORY):]
        one_hot_labels = F.one_hot(labels, num_classes=logit.shape[1]-len(BACKGROUND_CATEGORY)).to(torch.float16)

        loss = (F.softmax(logit[:,:-len(BACKGROUND_CATEGORY)])*one_hot_labels).mean() #+ background_logit.mean() #(logit[:,:-len(BACKGROUND_CATEGORY)]*one_hot_labels).mean() #F.cross_entropy(logit.requires_grad_(True), labels)

        # score = logit[:,labels]
        self.model_arch.zero_grad()
        loss.backward(retain_graph=retain_graph)
        gradients = self.gradient['value']
        activations = self.activation['value']

        # gradients_t = self.gradient_t['value']
        # activations_t = self.activation_t['value']

        visual_feature = activations[1:-self.model_arch.prompt_learner.n_ctx]
        # visual_feature = activations[1:-self.model_arch.prompt_learner.n_ctx]
        # cls = gradients[1:-self.model_arch.prompt_learner.n_ctx,:,:]
        # cls_token_gradient = gradients[-self.model_arch.prompt_learner.n_ctx:,:,:].mean(dim=0,keepdim=True)#gradients[:1,:,:]
        cls_token_gradient,prompt_gradient = gradients[:1,:,:], gradients[-self.model_arch.prompt_learner.n_ctx:,:,:].mean(keepdim=True,dim=0)
        visual_gradient = torch.mean(gradients[1:-self.model_arch.prompt_learner.n_ctx],keepdim=True,dim=0)

        lam = 0.5
        # cls_token_gradient = cls_token_gradient / (cls_token_gradient.max(dim=-1,keepdim=True)[0] + 1e-6)
        # prompt_gradient = prompt_gradient / (prompt_gradient.max(dim=-1,keepdim=True)[0] + 1e-6)

        # sim = F.cosine_similarity(prompt_gradient.mean(dim=0,keepdim=True),cls_token_gradient,dim=-1)
        # print(sim)
        # cls_token_gradient = gradients[-self.model_arch.prompt_learner.n_ctx:,:,:].max(dim=0,keepdim=True)[0]#gradients[:1,:,:]

        # token_gradient = cls_token_gradient
        # token_gradient = cls_token_gradient#*(prompt_gradient.mean(dim=0,keepdim=True))
        # propmt_mean = prompt_gradient.mean(dim=0,keepdim=True)
        token_gradient = visual_gradient

        final_visual_feature = torch.bmm(visual_feature.permute(1,0,2),token_gradient.permute(1,2,0))
        final_visual_feature = F.relu(final_visual_feature).permute(0,2,1)
        # if attn_mask:
        #     final_visual_feature = torch.bmm(final_visual_feature, aff)

        final_visual_feature = final_visual_feature.reshape(final_visual_feature.shape[0],1, patch_num, patch_num)
        final_visual_feature = F.upsample(final_visual_feature,size=(ori_size,ori_size),mode='bilinear',align_corners=True)

        # saliency_map = final_visual_feature / final_visual_feature.max()
        final_visual_feature_min, final_visual_feature_max = final_visual_feature.min(), final_visual_feature.max()
        saliency_map = final_visual_feature / (final_visual_feature_max + 1e-6)#(final_visual_feature-final_visual_feature_min) / (final_visual_feature_max - final_visual_feature_min + 1e-6)

        threshold = 0.5
        # saliency_map[saliency_map >= threshold] = 1
        saliency_map[saliency_map < threshold] = 0

        return saliency_map


    def forward_train(self,input,labels,cfg=None,retain_graph=False,split=None,attn_mask=False):




        b,c,h,w = input.shape
        patch_num,ori_size = self.model_arch.image_encoder.patch_num, self.model_arch.image_encoder.input_resolution

        if attn_mask:
            logit,mask = self.model_arch.forward_test(input,labels,cfg=cfg,attn_mask=attn_mask)
            cls_mask = mask[:,1:-self.model_arch.prompt_learner.n_ctx,:1].reshape(b,-1,patch_num,patch_num) #+ mask[:,1:-self.model_arch.prompt_learner.n_ctx,:1].permute(0,2,1)
            aff = mask[:,1:-self.model_arch.prompt_learner.n_ctx, 1:-self.model_arch.prompt_learner.n_ctx]
            # aff = (aff + aff.permute(0,2,1)) / 2
            aff = aff / (aff.sum(dim=1,keepdim=True) + 1e-6)
            # aff = aff / (aff.sum(dim=1,keepdim=True) + 1e-6)


            # aff = (aff + aff.permute(0,2,1)) / 2

            # aff = torch.bmm(aff,aff)

            # aff = F.softmax(aff,dim=1)
            # cls_mask = torch.bmm(cls_mask, aff).reshape(b,-1,patch_num,patch_num)


            # cls_mask = mask[:,1:-self.model_arch.prompt_learner.n_ctx,:1].permute(0,2,1).reshape(b,-1,patch_num,patch_num)
            # # cls_mask = mask[:,-self.model_arch.prompt_learner.n_ctx:,1:-self.model_arch.prompt_learner.n_ctx].reshape(b,-1,patch_num,patch_num).mean(dim=1,keepdim=True)
            # final_cls_mask = F.upsample(cls_mask, size=(ori_size, ori_size), mode='bilinear',
            #                                   align_corners=True)
            # final_cls_feature_min, final_cls_feature_max = final_cls_mask.min(), final_cls_mask.max()
            # final_cls_mask = (final_cls_mask - final_cls_feature_min) / (
            #             final_cls_feature_max - final_cls_feature_min + 1e-6)
            # final_cls_mask = final_cls_mask / (final_cls_mask.max() + 1e-6)

        else:
            logit = self.model_arch.forward_test(input,labels,cfg=cfg)
        pred_label = torch.argmax(logit)
        sign = pred_label == labels
        # if (split == 'true' and sign == False) or (split == 'wrong' and sign == True):
        #     print(f'Ignore the not {split} sample')
        #     return None

        # if attn_mask:
        #     return final_cls_mask
        # pred = logit[:,-len(BACKGROUND_CATEGORY):].argmax(dim=-1)
        # background_logit = logit[:,-len(BACKGROUND_CATEGORY):]
        one_hot_labels = F.one_hot(labels, num_classes=logit.shape[1]).to(torch.float16)
        loss = (logit*one_hot_labels).mean() #+ background_logit.mean() #(logit[:,:-len(BACKGROUND_CATEGORY)]*one_hot_labels).mean() #F.cross_entropy(logit.requires_grad_(True), labels)
        # score = logit[:,labels]
        self.model_arch.zero_grad()
        loss.backward(retain_graph=retain_graph)
        gradients = self.gradient['value']
        activations = self.activation['value']

        # gradients_t = self.gradient_t['value']
        # activations_t = self.activation_t['value']

        visual_feature = activations[1:-self.model_arch.prompt_learner.n_ctx]
        # visual_feature = activations[1:-self.model_arch.prompt_learner.n_ctx]
        # cls = gradients[1:-self.model_arch.prompt_learner.n_ctx,:,:]
        # cls_token_gradient = gradients[-self.model_arch.prompt_learner.n_ctx:,:,:].mean(dim=0,keepdim=True)#gradients[:1,:,:]
        cls_token_gradient,prompt_gradient = gradients[:1,:,:], gradients[-self.model_arch.prompt_learner.n_ctx:,:,:].mean(keepdim=True,dim=0)
        visual_gradient = torch.mean(gradients[1:-self.model_arch.prompt_learner.n_ctx],keepdim=True,dim=0)

        lam = 0.5
        # cls_token_gradient = cls_token_gradient / (cls_token_gradient.max(dim=-1,keepdim=True)[0] + 1e-6)
        # prompt_gradient = prompt_gradient / (prompt_gradient.max(dim=-1,keepdim=True)[0] + 1e-6)

        # sim = F.cosine_similarity(prompt_gradient.mean(dim=0,keepdim=True),cls_token_gradient,dim=-1)
        # print(sim)
        # cls_token_gradient = gradients[-self.model_arch.prompt_learner.n_ctx:,:,:].max(dim=0,keepdim=True)[0]#gradients[:1,:,:]

        # token_gradient = cls_token_gradient
        # token_gradient = cls_token_gradient#*(prompt_gradient.mean(dim=0,keepdim=True))
        # propmt_mean = prompt_gradient.mean(dim=0,keepdim=True)
        token_gradient = visual_gradient

        final_visual_feature = torch.bmm(visual_feature.permute(1,0,2),token_gradient.permute(1,2,0))
        final_visual_feature = F.relu(final_visual_feature).permute(0,2,1)
        # if attn_mask:
        #     final_visual_feature = torch.bmm(final_visual_feature, aff)

        final_visual_feature = final_visual_feature.reshape(final_visual_feature.shape[0],1, patch_num, patch_num)
        final_visual_feature = F.upsample(final_visual_feature,size=(ori_size,ori_size),mode='bilinear',align_corners=True)

        # saliency_map = final_visual_feature / final_visual_feature.max()
        final_visual_feature_min, final_visual_feature_max = final_visual_feature.min(), final_visual_feature.max()
        saliency_map = final_visual_feature / (final_visual_feature_max + 1e-6)#(final_visual_feature-final_visual_feature_min) / (final_visual_feature_max - final_visual_feature_min + 1e-6)

        threshold = 0.5
        saliency_map[saliency_map >= threshold] = 1
        saliency_map[saliency_map < threshold] = 0

        return saliency_map


    def show_cam(self,img,mask,save_path=None):

        heat_map = cv2.applyColorMap(np.uint8(255*mask.squeeze()), cv2.COLORMAP_JET)
        heatmap = torch.from_numpy(heat_map).permute(2,0,1).float().div(255)
        b,g,r = heatmap.split(1)
        heatmap = torch.cat([r,g,b])
        rate = 0.5
        res = rate*heatmap + (1-rate)*img
        res = res.div(res.max()).squeeze()
        res = np.transpose(np.uint8(255*res),(1,2,0))

        pil_image = Image.fromarray(res)
        # pil_image.save('test1.jpg')
        pil_image.save(save_path)
        return pil_image



def denorm(img,mean,std):
    mean,std = np.array(mean),np.array(std)
    img = img*std[:, None, None] + mean[:, None, None]
    # img = np.clip(img*255, 0, 255)  #.clamp(0,255)
    # img = img / 255
    return img


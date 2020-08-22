"""
Adapted from: https://github.com/NVlabs/SPADE/blob/c0d50e6c4c106c0e88974f445f435b8ba5f4ccf6/models/pix2pix_model.py
"""

import torch
import models.networks as networks
import util.util as util
import math

class Pix2PixModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netGbg, self.netG, self.netD = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            if opt.two_step_model:
                self.criterionMask = torch.nn.BCEWithLogitsLoss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
                
        if opt.embed_captions and not opt.use_precomputed_captions:
            print('Loading pretrained BERT model...')
            from transformers import BertModel
            self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True).eval()
            if self.use_gpu():
                self.bert = self.bert.cuda()

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode, mask_decay=None):
        input_semantics_bg, input_semantics_fg, input_semantics, input_attr_bg, input_attr_fg, input_attr, real_mask, \
            real_image, sentence, sentence_mask = self.preprocess_input(data)

        if mode == 'generator':
            g_loss, generated_bg, generated = self.compute_generator_loss(
                input_semantics_bg, input_semantics_fg, input_semantics, input_attr_bg, input_attr_fg, input_attr, real_mask,
                real_image, mask_decay, sentence, sentence_mask)
            return g_loss, generated_bg, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_semantics_bg, input_semantics_fg, input_semantics, input_attr_bg, input_attr_fg, input_attr,
                real_image, real_mask, sentence, sentence_mask)
            return d_loss
        elif mode == 'encode_only':
            z, mu, logvar = self.encode_z(real_image)
            return mu, logvar
        elif mode == 'inference':
            with torch.no_grad():
                fake_image_bg, fake_image, mask = self.generate_fake(input_semantics_bg, input_semantics_fg, input_semantics,
                                                                        input_attr_bg, input_attr_fg, input_attr,
                                                                        real_image, real_mask, sentence, sentence_mask)
            return fake_image_bg, fake_image, mask
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        if self.opt.two_step_model:
            G_params = list(self.netG.parameters()) + list(self.netGbg.parameters())
        else:
            G_params = list(self.netG.parameters())

        if opt.isTrain:
            D_params = list(self.netD.parameters())

        if opt.no_TTUR:
            beta1, beta2 = opt.beta1, opt.beta2
            G_lr, D_lr = opt.lr, opt.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        if self.opt.two_step_model:
            util.save_network(self.netGbg, 'Gbg', epoch, self.opt)
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        if self.opt.two_step_model:
            netGbg = networks.define_Gbg(opt)
        else:
            netGbg = None
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None

        if not opt.isTrain or opt.continue_train:
            if self.opt.two_step_model:
                netGbg = util.load_network(netGbg, 'Gbg', opt.which_epoch, opt)
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)

        return netGbg, netG, netD

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        # move to GPU and change data types
        data['label'] = data['label'].long()
        if self.opt.two_step_model:
            data['label_bg'] = data['label_bg'].long()
            data['label_fg'] = data['label_fg'].long()
            
        if self.opt.embed_attributes:
            data['attributes'] = data['attributes'].long()
            if self.opt.two_step_model:
                data['attributes_bg'] = data['attributes_bg'].long()
                data['attributes_fg'] = data['attributes_fg'].long()
        if self.use_gpu():
            data['image'] = data['image'].cuda()
            data['label'] = data['label'].cuda()
            if self.opt.two_step_model:
                data['label_bg'] = data['label_bg'].cuda()
                data['label_fg'] = data['label_fg'].cuda()
            
            if self.opt.embed_attributes:
                data['attributes'] = data['attributes'].cuda()
                if self.opt.two_step_model:
                    data['attributes_bg'] = data['attributes_bg'].cuda()
                    data['attributes_fg'] = data['attributes_fg'].cuda()
            if self.opt.embed_captions:
                if self.opt.use_precomputed_captions:
                    data['sentence_emb'] = data['sentence_emb'].cuda()
                else:
                    data['sentence_tokens'] = data['sentence_tokens'].cuda()
                data['sentence_len'] = data['sentence_len'].cuda()
            
        with torch.no_grad():
            label_map = data['label']
            bs, _, h, w = label_map.size()
            nc = self.opt.label_nc
            input_label = self.FloatTensor(bs, nc, h, w).zero_()
            input_semantics = input_label.scatter_(1, label_map, 1.0)

            if self.opt.two_step_model:
                label_map = data['label_bg']
                bs, _, h, w = label_map.size()
                nc = self.opt.label_nc
                input_label_bg = self.FloatTensor(bs, nc, h, w).zero_()
                input_semantics_bg = input_label_bg.scatter_(1, label_map, 1.0)

                label_map = data['label_fg']
                bs, _, h, w = label_map.size()
                nc = self.opt.label_nc
                input_label_fg = self.FloatTensor(bs, nc, h, w).zero_()
                input_semantics_fg = input_label_fg.scatter_(1, label_map, 1.0)

                real_mask = (data['label_fg'] != 0).float()
            else:
                input_semantics_bg = None
                input_semantics_fg = None
                real_mask = None

            if self.opt.embed_attributes:
                attr_map = data['attributes']
                bs, nl, _, h, w = attr_map.size()
                nc = self.opt.attr_nc + 1
                input_attr = self.FloatTensor(bs, nl, nc, h, w).zero_()
                input_attr = input_attr.scatter_(2, attr_map, 1.0)
                input_attr = input_attr.sum(dim=1)
                input_attr = input_attr[:, 1:] # Discard no-label
                
                if self.opt.two_step_model:
                    attr_map = data['attributes_bg']
                    bs, nl, _, h, w = attr_map.size()
                    nc = self.opt.attr_nc + 1
                    input_attr_bg = self.FloatTensor(bs, nl, nc, h, w).zero_()
                    input_attr_bg = input_attr_bg.scatter_(2, attr_map, 1.0)
                    input_attr_bg = input_attr_bg.sum(dim=1)
                    input_attr_bg = input_attr_bg[:, 1:] # Discard no-label

                    attr_map = data['attributes_fg']
                    bs, nl, _, h, w = attr_map.size()
                    nc = self.opt.attr_nc + 1
                    input_attr_fg = self.FloatTensor(bs, nl, nc, h, w).zero_()
                    input_attr_fg = input_attr_fg.scatter_(2, attr_map, 1.0)
                    input_attr_fg = input_attr_fg.sum(dim=1)
                    input_attr_fg = input_attr_fg[:, 1:] # Discard no-label
                else:
                    input_attr_fg = None
                    input_attr_bg = None
            else:
                input_attr_fg = None
                input_attr_bg = None
                input_attr = None
            
        if self.opt.embed_captions:
            sentence_lengths = data['sentence_len']
            if self.opt.use_precomputed_captions:
                sentence = data['sentence_emb']
                sentence_mask = torch.arange(sentence.shape[1], device=sentence_lengths.device)[None, :] < sentence_lengths
            else:
                # Infer sentence representation using BERT
                with torch.no_grad():
                    tokens = data['sentence_tokens']
                    tokens_batch_size = tokens.shape[0]
                    captions_per_sample = tokens.shape[1]
                    tokens = tokens.view(-1, tokens.shape[2])
                    sentence_lengths = sentence_lengths.view(-1, sentence_lengths.shape[2])
                    attention_mask = torch.arange(tokens.shape[1], device=sentence_lengths.device)[None, :] < sentence_lengths
                    _, _, hidden_states = self.bert(input_ids=tokens, attention_mask=attention_mask)
                    sentences = hidden_states[self.opt.bert_embedding_level]
                    sentence = sentences.view(tokens_batch_size, captions_per_sample*tokens.shape[1], -1)
                    sentence_mask = attention_mask.view(tokens_batch_size, -1)
        else:
            sentence = None
            sentence_mask = None

        return input_semantics_bg, input_semantics_fg, input_semantics, input_attr_bg, input_attr_fg, \
                input_attr, real_mask, data['image'], sentence, sentence_mask

    def compute_generator_loss(self, input_semantics_bg, input_semantics_fg, input_semantics,
                               input_attr_bg, input_attr_fg, input_attr, real_mask, real_image, mask_decay,
                               sentence, sentence_mask):
        G_losses = {}

        fake_image_bg, fake_image, mask = self.generate_fake(
            input_semantics_bg, input_semantics_fg, input_semantics, input_attr_bg, input_attr_fg, input_attr,
            real_image, real_mask, sentence, sentence_mask)

        pred_fake, pred_real = self.discriminate(
            input_semantics, input_attr, fake_image, real_image, sentence, sentence_mask)

        G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                            for_discriminator=False)

        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image) \
                * self.opt.lambda_vgg
            
        if self.opt.two_step_model:
            G_losses['Mask'] = mask_decay * self.criterionMask(mask, real_mask)

        return G_losses, fake_image_bg, fake_image

    def compute_discriminator_loss(self, input_semantics_bg, input_semantics_fg, input_semantics,
                                   input_attr_bg, input_attr_fg, input_attr, real_image, real_mask, sentence, sentence_mask):
        D_losses = {}
        with torch.no_grad():
            _, fake_image, _ = self.generate_fake(input_semantics_bg, input_semantics_fg, input_semantics, input_attr_bg, input_attr_fg,
                                                     input_attr, real_image, real_mask, sentence, sentence_mask)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        pred_fake, pred_real = self.discriminate(
            input_semantics, input_attr, fake_image, real_image, sentence, sentence_mask)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True)

        return D_losses


    def generate_fake(self, input_semantics_bg, input_semantics_fg, input_semantics, input_attr_bg, input_attr_fg, input_attr, real_image,
                      real_mask, sentence, sentence_mask):

        if self.opt.two_step_model:
            fake_image_bg = self.netGbg(input_semantics_bg, input_attr_bg,
                                        input_semantics, input_attr, # These will be global-average-pooled
                                        sentence, sentence_mask)
            fake_image, mask = self.netG(fake_image_bg,
                                         input_semantics_fg, input_attr_fg,
                                         sentence, sentence_mask)
        else:
            fake_image = self.netG(input_semantics, input_attr,
                                   None, None,
                                   sentence, sentence_mask)
            fake_image_bg = None
            mask = None

        return fake_image_bg, fake_image, mask

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, attr_masks, fake_image, real_image, sentence, sentence_mask):
        fake_and_real_image = torch.cat([fake_image, real_image], dim=0)
        fake_and_real_mask = torch.cat([input_semantics, input_semantics], dim=0)
        if self.opt.embed_attributes:
            fake_and_real_attr = torch.cat([attr_masks, attr_masks], dim=0)
        else:
            fake_and_real_attr = None
            
        if self.opt.embed_captions:
            fake_and_real_sentence = torch.cat([sentence, sentence], dim=0)
            fake_and_real_sentence_mask = torch.cat([sentence_mask, sentence_mask], dim=0)
        else:
            fake_and_real_sentence = None
            fake_and_real_sentence_mask = None

        discriminator_out = self.netD(fake_and_real_image, fake_and_real_mask, fake_and_real_attr,
                                      fake_and_real_sentence, fake_and_real_sentence_mask)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

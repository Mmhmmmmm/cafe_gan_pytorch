# Copyright (C) 2018 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
#
# This work is licensed under the MIT License. To view a copy of this license,
# visit https://opensource.org/licenses/MIT.

"""AttGAN, generator, and discriminator."""

import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
import torch
from torch.functional import norm
import torch.nn as nn
from nn import LinearBlock, Conv2dBlock, ConvTranspose2dBlock
#from torchsummary import summary
#from torch import linalg as LA

# This architecture is for images of 128x128
# In the original AttGAN, slim.conv2d uses padding 'same'
MAX_DIM = 64 * 16  # 1024


class ConvGRUCell(nn.Module):
    def __init__(self, n_attrs, in_dim, out_dim, kernel_size=3):
        super(ConvGRUCell, self).__init__()
        self.n_attrs = n_attrs
        self.upsample = nn.ConvTranspose2d(
            in_dim * 2 + n_attrs, out_dim, 4, 2, 1, bias=False)
        self.reset_gate = nn.Sequential(
            nn.Conv2d(in_dim + out_dim, out_dim, kernel_size,
                      1, (kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Conv2d(in_dim + out_dim, out_dim, kernel_size,
                      1, (kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Sigmoid()
        )
        self.hidden = nn.Sequential(
            nn.Conv2d(in_dim + out_dim, out_dim, kernel_size,
                      1, (kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Tanh()
        )

    def forward(self, input, old_state, attr):
        n, _, h, w = old_state.size()
        attr = attr.view((n, self.n_attrs, 1, 1)).expand(
            (n, self.n_attrs, h, w))
        state_hat = self.upsample(torch.cat([old_state, attr], 1))
        r = self.reset_gate(torch.cat([input, state_hat], dim=1))
        z = self.update_gate(torch.cat([input, state_hat], dim=1))
        new_state = r * state_hat
        hidden_info = self.hidden(torch.cat([input, new_state], dim=1))
        output = (1-z) * state_hat + z * hidden_info
        return output, new_state


class Generator(nn.Module):
    def __init__(self, attr_dim, conv_dim=64, n_layers=5, shortcut_layers=2, stu_kernel_size=3, use_stu=True, one_more_conv=True):
        super(Generator, self).__init__()
        self.n_attrs = 13
        self.n_layers = n_layers
        self.shortcut_layers = min(shortcut_layers, n_layers - 1)
        self.use_stu = use_stu

        self.encoder = nn.ModuleList()
        in_channels = 3
        for i in range(self.n_layers):
            self.encoder.append(nn.Sequential(
                nn.Conv2d(in_channels, conv_dim * 2 ** i, 4, 2, 1, bias=False),
                nn.BatchNorm2d(conv_dim * 2 ** i),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ))
            in_channels = conv_dim * 2 ** i

        if use_stu:
            self.stu = nn.ModuleList()
            for i in reversed(range(self.n_layers - 1 - self.shortcut_layers, self.n_layers - 1)):
                self.stu.append(ConvGRUCell(
                    self.n_attrs, conv_dim * 2 ** i, conv_dim * 2 ** i, stu_kernel_size))

        self.decoder = nn.ModuleList()
        for i in range(self.n_layers):
            if i < self.n_layers - 1:
                if i == 0:
                    self.decoder.append(nn.Sequential(
                        nn.ConvTranspose2d(conv_dim * 2 ** (self.n_layers - 1) + attr_dim,
                                           conv_dim * 2 ** (self.n_layers - 1), 4, 2, 1, bias=False),
                        nn.BatchNorm2d(in_channels),
                        nn.ReLU(inplace=True)
                    ))
                elif i <= self.shortcut_layers:     # not <
                    self.decoder.append(nn.Sequential(
                        nn.ConvTranspose2d(conv_dim * 3 * 2 ** (self.n_layers - 1 - i),
                                           conv_dim * 2 ** (self.n_layers - 1 - i), 4, 2, 1, bias=False),
                        nn.BatchNorm2d(
                            conv_dim * 2 ** (self.n_layers - 1 - i)),
                        nn.ReLU(inplace=True)
                    ))
                else:
                    self.decoder.append(nn.Sequential(
                        nn.ConvTranspose2d(conv_dim * 2 ** (self.n_layers - i),
                                           conv_dim * 2 ** (self.n_layers - 1 - i), 4, 2, 1, bias=False),
                        nn.BatchNorm2d(
                            conv_dim * 2 ** (self.n_layers - 1 - i)),
                        nn.ReLU(inplace=True)
                    ))
            else:
                in_dim = conv_dim * 3 if self.shortcut_layers == self.n_layers - 1 else conv_dim * 2
                if one_more_conv:
                    self.decoder.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            in_dim, conv_dim // 4, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(conv_dim // 4),
                        nn.ReLU(inplace=True),

                        nn.ConvTranspose2d(
                            conv_dim // 4, 3, 3, 1, 1, bias=False),
                        nn.Tanh()
                    ))
                else:
                    self.decoder.append(nn.Sequential(
                        nn.ConvTranspose2d(in_dim, 3, 4, 2, 1, bias=False),
                        nn.Tanh()
                    ))

    def forward(self, x, a):
        # propagate encoder layers
        y = []
        x_ = x
        for layer in self.encoder:
            x_ = layer(x_)
            y.append(x_)

        out = y[-1]
        n, _, h, w = out.size()
        attr = a.view((n, self.n_attrs, 1, 1)).expand((n, self.n_attrs, h, w))
        out = self.decoder[0](torch.cat([out, attr], dim=1))
        stu_state = y[-1]

        # propagate shortcut layers
        for i in range(1, self.shortcut_layers + 1):
            if self.use_stu:
                stu_out, stu_state = self.stu[i-1](y[-(i+1)], stu_state, a)
                out = torch.cat([out, stu_out], dim=1)
                out = self.decoder[i](out)
            else:
                out = torch.cat([out, y[-(i+1)]], dim=1)
                out = self.decoder[i](out)

        # propagate non-shortcut layers
        for i in range(self.shortcut_layers + 1, self.n_layers):
            out = self.decoder[i](out)

        return out


class Generator(nn.Module):
    def __init__(self, enc_dim=64, enc_layers=5, enc_norm_fn='batchnorm', enc_acti_fn='lrelu',
                 dec_dim=64, dec_layers=5, dec_norm_fn='batchnorm', dec_acti_fn='relu',
                 n_attrs=13, shortcut_layers=2, inject_layers=0, img_size=128, stu_kernel_size=3, use_stu=True):
        super(Generator, self).__init__()
        self.shortcut_layers = min(shortcut_layers, dec_layers - 1)
        self.inject_layers = min(inject_layers, dec_layers - 1)
        self.f_size = img_size // 2**enc_layers  # f_size = 4 for 128x128
        self.use_stu = use_stu
        layers = []
        n_in = 3
        for i in range(enc_layers):
            n_out = min(enc_dim * 2**i, MAX_DIM)
            layers += [Conv2dBlock(
                n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=enc_norm_fn, acti_fn=enc_acti_fn
            )]
            n_in = n_out
        self.enc_layers = nn.ModuleList(layers)

        if use_stu:
            self.stu = nn.ModuleList()
            for i in reversed(range(dec_layers - 1 - self.shortcut_layers, dec_layers - 1)):
                self.stu.append(ConvGRUCell(n_attrs, enc_dim *
                                            2 ** i, enc_dim * 2 ** i, stu_kernel_size))

        layers = []
        n_in = n_in + n_attrs  # 1024 + 13
        for i in range(dec_layers):
            if i < dec_layers - 1:
                n_out = min(dec_dim * 2**(dec_layers-i-1), MAX_DIM)
                layers += [ConvTranspose2dBlock(
                    n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=dec_norm_fn, acti_fn=dec_acti_fn
                )]
                n_in = n_out
                n_in = n_in + n_in//2 if self.shortcut_layers > i else n_in
                n_in = n_in + n_attrs if self.inject_layers > i else n_in
            else:
                layers += [ConvTranspose2dBlock(
                    n_in, 3, (4, 4), stride=2, padding=1, norm_fn='none', acti_fn='tanh'
                )]
        self.dec_layers = nn.ModuleList(layers)

    def encode(self, x):
        z = x
        zs = []
        for layer in self.enc_layers:
            z = layer(z)
            zs.append(z)
        return zs

    def decode(self, zs, a):
        a_tile = a.view(a.size(0), -1, 1, 1).repeat(1,
                                                    1, self.f_size, self.f_size)
        z = self.dec_layers[0](torch.cat([zs[-1], a_tile], dim=1))
        stu_state = zs[-1]

        # for i, layer in enumerate(self.dec_layers):
        for i in range(1, self.shortcut_layers+1):
            # z = layer(z)
            # if self.shortcut_layers > i:  # Concat 1024 with 512
            if self.use_stu:
                stu_out, stu_state = self.stu[i -
                                              1](zs[-(i+1)], stu_state, a)
                z = torch.cat([z, stu_out], dim=1)
                z = self.dec_layers[i](z)
            else:

                z = torch.cat([z, zs[-(i+1)]], dim=1)
                z = self.dec_layers[i](z)

            # if self.inject_layers > i:
            #     a_tile = a.view(a.size(0), -1, 1, 1) \
            #               .repeat(1, 1, self.f_size * 2**(i+1), self.f_size * 2**(i+1))
            #     z = torch.cat([z, a_tile], dim=1)
        for i in range(self.shortcut_layers+1, len(self.dec_layers)):
            z = self.dec_layers[i](z)

        return z

    def forward(self, x, a=None, mode='enc-dec'):
        if mode == 'enc-dec':
            assert a is not None, 'No given attribute.'
            return self.decode(self.encode(x), a)
        if mode == 'enc':
            return self.encode(x)
        if mode == 'dec':
            assert a is not None, 'No given attribute.'
            return self.decode(x, a)
        raise Exception('Unrecognized mode: ' + mode)


class Discriminators(nn.Module):
    # No instancenorm in fcs in source code, which is different from paper.
    def __init__(self, dim=64, norm_fn='instancenorm', acti_fn='lrelu',
                 fc_dim=1024, fc_norm_fn='none', fc_acti_fn='lrelu', n_layers=5, img_size=128):
        super(Discriminators, self).__init__()
        self.f_size = img_size // 2**(n_layers+1)

        layers = []
        n_in = 3
        for i in range(3):
            n_out = min(dim * 2**i, MAX_DIM)
            layers += [Conv2dBlock(
                n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
            )]
            n_in = n_out
        self.conv1 = nn.Sequential(*layers)
        layers2 = []
        for i in range(3, n_layers):
            n_out = min(dim*2**i, MAX_DIM)
            layers2 += [Conv2dBlock(
                n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
            )]
            n_in = n_out
        self.conv2 = nn.Sequential(*layers2)

        # cls1 net
        # n_in = 256
        # layers3 = []
        # for i in range(3, n_layers):
        #     n_out = min(dim*2**i, MAX_DIM)
        #     layers3 += [Conv2dBlock(
        #         n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
        #     )]
        #     n_in = n_out
        self.convcls1 = nn.Sequential(Conv2dBlock(
                256, 512, (3, 3), stride=1, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
            ),Conv2dBlock(
                512, 1024, (4, 4), stride=2, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
            ),nn.AvgPool2d(7,stride=1)
            )
        # cls2 net
        # n_in = 13*256
        # layers4 = []
        # for i in range(3, n_layers):
        #     n_out = min(dim*2**i, MAX_DIM)
        #     layers4 += [Conv2dBlock(
        #         n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
        #     )]
        #     n_in = n_out
        # self.convcls2 = nn.Sequential(*layers4)
        self.convcls2 = nn.Sequential(Conv2dBlock(
                256, 512, (3, 3), stride=1, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
            ),Conv2dBlock(
                512, 1024, (4, 4), stride=2, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
            ),nn.AvgPool2d(7,stride=1)
            )
        # abv's fc
        self.fc_adv = nn.Sequential(
            LinearBlock(1024 * 4 * 4,
                        fc_dim, fc_norm_fn, fc_acti_fn),
            LinearBlock(fc_dim, 1, 'none', 'none')
        )
        # cls1 and cls2's fc
        fc_cls1 = [nn.Sequential(
            LinearBlock(1024 * self.f_size * self.f_size, 1, 'none', 'sigmoid')) for _ in range(13)]
        fc_cls2 = [nn.Sequential(
            LinearBlock(1024 * self.f_size * self.f_size, 1, 'none', 'sigmoid')) for _ in range(13)]
        self.fc_cls1 = nn.ModuleList(fc_cls1)
        self.fc_cls2 = nn.ModuleList(fc_cls2)
        # att
        self.att_conv = nn.Sequential(Conv2dBlock(256, 512, (3, 3), stride=1, padding=1, norm_fn=norm_fn, acti_fn='none'), Conv2dBlock(
            512, 512, (3, 3), stride=1, padding=1, norm_fn=norm_fn, acti_fn='none'))
        self.att_convab1 = Conv2dBlock(
            512, 13, (1, 1), stride=1, padding=0, norm_fn='none', acti_fn='none')
        self.att_convab2 = nn.Sequential(Conv2dBlock(13, 13, (1, 1), stride=1, padding=0, norm_fn='none', acti_fn='none'), Conv2dBlock(
            13, 13, (1, 1), stride=1, padding=0, norm_fn='none', acti_fn='sigmoid'))
        self.att_convab3 = nn.Sequential(Conv2dBlock(
            13, 13, (1, 1), stride=1, padding=0, norm_fn='none', acti_fn='none'), nn.AdaptiveAvgPool2d(1))
        self.att_convcab1 = Conv2dBlock(
            512, 13, (1, 1), stride=1, padding=0, norm_fn='none', acti_fn='none')
        self.att_convcab2 = nn.Sequential(Conv2dBlock(13, 13, (1, 1), stride=1, padding=0, norm_fn='none', acti_fn='none'), Conv2dBlock(
            13, 13, (1, 1), stride=1, padding=0, norm_fn='none', acti_fn='sigmoid'),)
        self.att_convcab3 = nn.Sequential(Conv2dBlock(
            13, 13, (1, 1), stride=1, padding=0, norm_fn='none', acti_fn='none'), nn.AdaptiveAvgPool2d(1))

        # cls

    def forward(self, x):
        h = self.conv1(x)
        fe = h
        ax = self.att_conv(h)
        # attention branch net
        af = self.att_convab1(ax)
        att = self.att_convab2(af)
        as1 = self.att_convab3(af)
        as1 = as1.view(as1.size(0), -1)
        as1 = torch.sigmoid(as1)
        # co-attention branch net
        caf = self.att_convcab1(ax)
        catt = self.att_convcab2(caf)
        as2 = self.att_convcab3(caf)
        as2 = as2.view(as2.size(0), -1)
        as2 = torch.sigmoid(as2)
        # class
        b,c,hs,ws = h.shape
        cls1 = []
        cls2 = []
        for i in range(13):
            itematt = att[:,i].view(b,1,hs,ws)
            itemcls1 = h * itematt
            itemcls1 = self.convcls1(itemcls1)
            itemcls1 = itemcls1.view(b,-1)
            cls1.append(self.fc_cls1[i](itemcls1))

            itemcatt = catt[:,i].viiew(b,1,hs,ws)
            itemcls2 = h* itemcatt
            itemcls2 = self.convcls2(itemcls2)
            itemcls2 = itemcls2.view(b,-1)
            cls2.append(self.fc_cls2[i](itemcls2))
        
        cls1 = torch.cat(cls1,dim=1)
        cls2 = torch.cat(cls2,dim=1)




        # cls1 = torch.einsum('binm,bjnm->bijnm', h, self.att)
        # cls1 = cls1.view(cls1.size(0), -1, 16, 16)
        # # print(h.size(), self.att.size(), cls1.size())

        # per1 = cls1
        # cls1 = self.convcls1(cls1)
        # cls1 = cls1.view(cls1.size(0), -1)
        # cls1 = self.fc_cls1(cls1)
        # cls2 = torch.einsum('binm,bjnm->bijnm', h, self.catt)
        # cls2 = cls2.view(cls2.size(0), -1, 16, 16)
        # per2 = cls2
        # cls2 = self.convcls2(cls2)
        # cls2 = cls2.view(cls2.size(0), -1)
        # cls2 = self.fc_cls2(cls2)

        # abv
        h = self.conv2(h)

        h = h.view(h.size(0), -1)
        return self.fc_adv(h), as1, as2, cls1, cls2, [att, catt, fe]


# multilabel_soft_margin_loss = sigmoid + binary_cross_entropy

class AttGAN():
    def __init__(self, args):
        self.mode = args.mode
        self.gpu = args.gpu
        # self.gpunum = args.gpunum
        self.multi_gpu = args.multi_gpu if 'multi_gpu' in args else False
        self.lambda_1 = args.lambda_1
        self.lambda_2 = args.lambda_2
        self.lambda_3 = args.lambda_3
        self.lambda_gp = args.lambda_gp
        # self.gpunum=args.gpunum
        self.G = Generator(
            args.enc_dim, args.enc_layers, args.enc_norm, args.enc_acti,
            args.dec_dim, args.dec_layers, args.dec_norm, args.dec_acti,
            args.n_attrs, args.shortcut_layers, args.inject_layers, args.img_size
        )
        self.G.train()
        if self.gpu:
            self.G.cuda()
        # summary(self.G, [(3, args.img_size, args.img_size), (args.n_attrs,
                                                             #1, 1)], batch_size=4, device='cuda(args.gpunum)' if args.gpu else 'cpu')

        self.D = Discriminators(
            args.dis_dim, args.dis_norm, args.dis_acti,
            args.dis_fc_dim, args.dis_fc_norm, args.dis_fc_acti, args.dis_layers, args.img_size
        )
        self.D.train()
        if self.gpu:
            self.D.cuda()
        # summary(self.D, [(3, args.img_size, args.img_size)],
                # batch_size=4, device='cuda(args.gpunum)' if args.gpu else 'cpu')

        if self.multi_gpu:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)

        self.optim_G = optim.Adam(
            self.G.parameters(), lr=args.lr, betas=args.betas)
        self.optim_D = optim.Adam(
            self.D.parameters(), lr=args.lr, betas=args.betas)

    def set_lr(self, lr):
        for g in self.optim_G.param_groups:
            g['lr'] = lr
        for g in self.optim_D.param_groups:
            g['lr'] = lr

    def trainG(self, img_a, att_a, att_a_, att_b, att_b_):
        for p in self.D.parameters():
            p.requires_grad = False

        zs_a = self.G(img_a, mode='enc')
        img_fake = self.G(zs_a, att_b-att_a, mode='dec')
        img_recon = self.G(zs_a, att_a-att_a, mode='dec')
        d_real, da_real, dca_real, dc1_real, dc2_real, [
            att_, catt_, fe_] = self.D(img_a)
        d_fake, da_fake, dca_fake, dc1_fake, dc2_fake, [
            att, catt, fe] = self.D(img_fake)

        if self.mode == 'wgan':
            gf_loss = -d_fake.mean()
        if self.mode == 'lsgan':  # mean_squared_error
            gf_loss = F.mse_loss(d_fake, torch.ones_like(d_fake))
        if self.mode == 'dcgan':  # sigmoid_cross_entropy
            gf_loss = F.binary_cross_entropy_with_logits(
                d_fake, torch.ones_like(d_fake))
        gc1_loss = F.binary_cross_entropy(torch.cat((dc1_fake,dc2_fake),1), torch.cat((att_b,att_b),1))
        # gc2_loss = F.binary_cross_entropy_with_logits(dc2_fake, 1-att_b)
        att_tmp = torch.zeros_like(att_)
        catt_tmp = torch.zeros_like(catt_)
        # cm_loss = torch.zeros(1)

        att_c = torch.abs(att_b-att_a).view(-1, 13, 1, 1)

        att_tmp = att_*(1-att_c) + catt_*(att_c)
        catt_tmp = att_*(att_c) + catt_*(1-att_c)
        # for i in range(len(att_a)):
        #     if att_a[i] == att_b[i]:
        #         cm_loss =cm_loss + LA.norm(att_[i]-att[i],1)+LA.norm(catt_[i]-catt[i],1)
        #     else:
        #         cm_loss =cm_loss + LA.norm(catt_[i]-att[i],1)+LA.norm(att_[i]-catt[i],1)

        # cm_loss = (att_ + catt_ - att-catt)/256
        cm_loss = torch.norm(att_tmp-att, p=1, dim=(2, 3)) + \
            torch.norm(catt_tmp-catt, p=1, dim=(2, 3))

        cm_loss = torch.mean(torch.sum(cm_loss,dim=(1))/256)

        gr_loss = F.l1_loss(img_recon, img_a)
        g_loss = gf_loss + 10 * (gc1_loss) + 100 * gr_loss + cm_loss

        self.optim_G.zero_grad()
        g_loss.backward()
        self.optim_G.step()

        errG = {
            'g_loss': g_loss.item(), 'gf_loss': gf_loss.item(),
            'gc_loss': gc1_loss.item(), 'gr_loss': gr_loss.item(),
            'cm_loss': cm_loss.item()
        }
        return errG

    def trainD(self, img_a, att_a, att_a_, att_b, att_b_):
        for p in self.D.parameters():
            p.requires_grad = True

        img_fake = self.G(img_a, att_b-att_a).detach()
        d_real, da_real, dca_real, dc1_real, dc2_real, _ = self.D(img_a)
        d_fake, da_fake, dca_fake, dc1_fake, dc2_fake, _ = self.D(img_fake)

        def gradient_penalty(f, real, fake=None):
            def interpolate(a, b=None):
                if b is None:  # interpolation in DRAGAN
                    beta = torch.rand_like(a)
                    b = a + 0.5 * a.var().sqrt() * beta
                alpha = torch.rand(a.size(0), 1, 1, 1)
                alpha = alpha.cuda() if self.gpu else alpha
                inter = a + alpha * (b - a)
                return inter
            x = interpolate(real, fake).requires_grad_(True)
            pred = f(x)
            if isinstance(pred, tuple):
                pred = pred[0]
            grad = autograd.grad(
                outputs=pred, inputs=x,
                grad_outputs=torch.ones_like(pred),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            grad = grad.view(grad.size(0), -1)
            norm = grad.norm(2, dim=1)
            gp = ((norm - 1.0) ** 2).mean()
            return gp

        if self.mode == 'wgan':
            wd = d_real.mean() - d_fake.mean()
            df_loss = -wd
            df_gp = gradient_penalty(self.D, img_a, img_fake)
        if self.mode == 'lsgan':  # mean_squared_error
            df_loss = F.mse_loss(d_real, torch.ones_like(d_fake)) + \
                F.mse_loss(d_fake, torch.zeros_like(d_fake))
            df_gp = gradient_penalty(self.D, img_a)
        if self.mode == 'dcgan':  # sigmoid_cross_entropy
            df_loss = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) + \
                F.binary_cross_entropy_with_logits(
                    d_fake, torch.zeros_like(d_fake))
            df_gp = gradient_penalty(self.D, img_a)
        da_loss = F.binary_cross_entropy(da_real, att_a)
        dca_loss = F.binary_cross_entropy(dca_real, 1-att_a)
        dc1_loss = F.binary_cross_entropy(torch.cat((dc1_real,dc2_real),1), torch.cat((att_a,att_a),1))
        # dc2_loss = F.binary_cross_entropy_with_logits(dc2_real, 1-att_a)
        d_loss = df_loss + self.lambda_gp * df_gp + \
            da_loss + dca_loss + (dc1_loss )#+ dc2_loss)/2

        self.optim_D.zero_grad()
        d_loss.backward()
        self.optim_D.step()

        errD = {
            'd_loss': d_loss.item(), 'df_loss': df_loss.item(),
            'df_gp': df_gp.item(), 'datt_loss': da_loss.item()+dca_loss.item(),
            'dcls_loss': dc1_loss.item()
        }
        return errD

    def train(self):
        self.G.train()
        self.D.train()

    def eval(self):
        self.G.eval()
        self.D.eval()

    def save(self, path):
        states = {
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'optim_G': self.optim_G.state_dict(),
            'optim_D': self.optim_D.state_dict()
        }
        torch.save(states, path)

    def load(self, path):
        states = torch.load(path, map_location=lambda storage, loc: storage)
        if 'G' in states:
            self.G.load_state_dict(states['G'])
        if 'D' in states:
            self.D.load_state_dict(states['D'])
        if 'optim_G' in states:
            self.optim_G.load_state_dict(states['optim_G'])
        if 'optim_D' in states:
            self.optim_D.load_state_dict(states['optim_D'])

    def saveG(self, path):
        states = {
            'G': self.G.state_dict()
        }
        torch.save(states, path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', dest='img_size', type=int, default=128)
    parser.add_argument('--shortcut_layers',
                        dest='shortcut_layers', type=int, default=1)
    parser.add_argument('--inject_layers',
                        dest='inject_layers', type=int, default=0)
    parser.add_argument('--enc_dim', dest='enc_dim', type=int, default=64)
    parser.add_argument('--dec_dim', dest='dec_dim', type=int, default=64)
    parser.add_argument('--dis_dim', dest='dis_dim', type=int, default=64)
    parser.add_argument('--dis_fc_dim', dest='dis_fc_dim',
                        type=int, default=1024)
    parser.add_argument('--enc_layers', dest='enc_layers', type=int, default=5)
    parser.add_argument('--dec_layers', dest='dec_layers', type=int, default=5)
    parser.add_argument('--dis_layers', dest='dis_layers', type=int, default=5)
    parser.add_argument('--enc_norm', dest='enc_norm',
                        type=str, default='batchnorm')
    parser.add_argument('--dec_norm', dest='dec_norm',
                        type=str, default='batchnorm')
    parser.add_argument('--dis_norm', dest='dis_norm',
                        type=str, default='instancenorm')
    parser.add_argument('--dis_fc_norm', dest='dis_fc_norm',
                        type=str, default='none')
    parser.add_argument('--enc_acti', dest='enc_acti',
                        type=str, default='lrelu')
    parser.add_argument('--dec_acti', dest='dec_acti',
                        type=str, default='relu')
    parser.add_argument('--dis_acti', dest='dis_acti',
                        type=str, default='lrelu')
    parser.add_argument('--dis_fc_acti', dest='dis_fc_acti',
                        type=str, default='relu')
    parser.add_argument('--lambda_1', dest='lambda_1',
                        type=float, default=100.0)
    parser.add_argument('--lambda_2', dest='lambda_2',
                        type=float, default=10.0)
    parser.add_argument('--lambda_3', dest='lambda_3', type=float, default=1.0)
    parser.add_argument('--lambda_gp', dest='lambda_gp',
                        type=float, default=10.0)
    parser.add_argument('--mode', dest='mode', default='wgan',
                        choices=['wgan', 'lsgan', 'dcgan'])
    parser.add_argument('--lr', dest='lr', type=float,
                        default=0.0002, help='learning rate')
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.5)
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.999)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    args.n_attrs = 13
    args.betas = (args.beta1, args.beta2)
    attgan = AttGAN(args)

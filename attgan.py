# Copyright (C) 2018 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the MIT License. To view a copy of this license,
# visit https://opensource.org/licenses/MIT.

"""AttGAN, generator, and discriminator."""

import torch
import torch.nn as nn
from nn import LinearBlock, Conv2dBlock, ConvTranspose2dBlock, ResNet, BasicBlock
from torchsummary import summary

# This architecture is for images of 128x128
# In the original AttGAN, slim.conv2d uses padding 'same'
MAX_DIM = 64 * 16  # 1024

class ConvGRUCell(nn.Module):
    def __init__(self, n_attrs, in_dim, out_dim, kernel_size=3):
        super(ConvGRUCell, self).__init__()
        self.n_attrs = n_attrs
        self.upsample = nn.ConvTranspose2d(in_dim * 2 + n_attrs, out_dim, 4, 2, 1, bias=False)
        self.reset_gate = nn.Sequential(
            nn.Conv2d(in_dim + out_dim, out_dim, kernel_size, 1, (kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Conv2d(in_dim + out_dim, out_dim, kernel_size, 1, (kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Sigmoid()
        )
        self.hidden = nn.Sequential(
            nn.Conv2d(in_dim + out_dim, out_dim, kernel_size, 1, (kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Tanh()
        )

    def forward(self, input, old_state, attr):
        n, _, h, w = old_state.size()
        attr = attr.view((n, self.n_attrs, 1, 1)).expand((n, self.n_attrs, h, w))
        state_hat = self.upsample(torch.cat([old_state, attr], 1))
        r = self.reset_gate(torch.cat([input, state_hat], dim=1))
        z = self.update_gate(torch.cat([input, state_hat], dim=1))
        new_state = r * state_hat
        hidden_info = self.hidden(torch.cat([input, new_state], dim=1))
        output = (1-z) * state_hat + z * hidden_info
        return output, new_state

class Generator(nn.Module):
    def __init__(self, enc_dim=64, enc_layers=5, enc_norm_fn='batchnorm', enc_acti_fn='lrelu',
                 dec_dim=64, dec_layers=5, dec_norm_fn='batchnorm', dec_acti_fn='relu',
                 n_attrs=13, shortcut_layers=1, inject_layers=0, img_size=128, use_stu=True):
        super(Generator, self).__init__()
        self.shortcut_layers = min(shortcut_layers, dec_layers - 1)
        self.inject_layers = min(inject_layers, dec_layers - 1)
        self.f_size = img_size // 2**enc_layers  # f_size = 4 for 128x128
        self.n_attrs = n_attrs
        self.use_stu = use_stu
        self.dec_layers = dec_layers
        
        layers = []
        n_in = 3
        for i in range(enc_layers):
            n_out = min(enc_dim * 2**i, MAX_DIM)
            layers += [Conv2dBlock(
                n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=enc_norm_fn, acti_fn=enc_acti_fn
            )]
            n_in = n_out
        self.Encode = nn.ModuleList(layers)

        if use_stu:
            layers = []
            for i in reversed(range(enc_layers-1-self.shortcut_layers, enc_layers-1)):
                conv_dim = min(enc_dim * 2**i, MAX_DIM)
                layers += [ConvGRUCell(n_attrs, conv_dim, conv_dim, 3)]
            self.stu = nn.ModuleList(layers)

        layers = []
        n_in = n_in + n_attrs  # 1024 + 13
        for i in range(dec_layers):
            if i < dec_layers - 1:
                n_out = min(dec_dim * 2 ** (dec_layers - i - 1), MAX_DIM)
                if i == 0:
                    layers += [ConvTranspose2dBlock(
                        n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=dec_norm_fn, acti_fn=dec_acti_fn
                    )]
                elif i <= self.shortcut_layers:
                    n_in = n_in + n_in//2 if self.shortcut_layers >= i else n_in
                    layers += [ConvTranspose2dBlock(
                        n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=dec_norm_fn, acti_fn=dec_acti_fn
                    )]
                else:
                    layers += [ConvTranspose2dBlock(
                        n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=dec_norm_fn, acti_fn=dec_acti_fn
                    )]
                n_in = n_out
            else:
                layers += [ConvTranspose2dBlock(
                    n_in, n_in // 4, (4, 4), stride=2, padding=1, norm_fn=dec_norm_fn, acti_fn=dec_acti_fn
                )]
                layers += [ConvTranspose2dBlock(
                    n_in // 4, 3, (3, 3), stride=1, padding=1, norm_fn='none', acti_fn='tanh'
                )]
        self.Decode = nn.ModuleList(layers)
    
    def encode(self, x):
        z = x
        zs = []
        for layer in self.Encode:
            z = layer(z)
            zs.append(z)
        return zs
    
    def decode(self, zs, a):

        out = zs[-1]
        n, _, h, w = out.size()
        attr = a.view((n, self.n_attrs, 1, 1)).expand((n, self.n_attrs, h, w))
        out = self.Decode[0](torch.cat([out, attr], dim=1))
        stu_state = zs[-1]

        #propagate shortcut layers
        for i in range(1, self.shortcut_layers + 1):
            if self.use_stu:
                stu_out, stu_state =self.stu[i-1](zs[-(i+1)], stu_state, a)
                out = torch.cat([out, stu_out], dim=1)
                out = self.Decode[i](out)
            else:
                out = torch.cat([out, zs[-(i+1)]], dim=1)
                out = self.Decode[i](out)

        for i in range(self.shortcut_layers + 1, self.dec_layers + 1):
            out = self.Decode[i](out)

        return out
    
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
                 fc_dim=1024, fc_norm_fn='none', fc_acti_fn='lrelu', n_layers=2):
        super(Discriminators, self).__init__()

        layers = []
        n_in = 256 * BasicBlock.expansion
        for i in range(3, 3 + n_layers):
            n_out = min(dim * 2**i, MAX_DIM)
            layers += [Conv2dBlock(
                n_in, n_out, 4, stride=2, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
            )]
            n_in = n_out
        self.conv = nn.Sequential(*layers)
        self.fc_adv = nn.Sequential(
            LinearBlock(4 * fc_dim, fc_dim, fc_norm_fn, fc_acti_fn),
            LinearBlock(fc_dim, 1, 'none', 'none')
        )

        self.att_cls = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=13)
    
    def forward(self, x ,type=None):
        if type is None:
            x, abn_ax, cabn_ax, cls1_rx, cls2_rx, abn_att, cabn_att = self.att_cls(x)
            h = self.conv(x)
            h = h.view(h.size(0), -1)
            return self.fc_adv(h), abn_ax, cabn_ax, cls1_rx, cls2_rx, abn_att, cabn_att
        elif type is 'adv':
            x = self.att_cls(x, 'adv')
            h = self.conv(x)
            h = h.view(h.size(0), -1)
            return self.fc_adv(h)
        elif type is 'abn':
            abn_att, cabn_att = self.att_cls(x, 'abn')
            return abn_att, cabn_att

import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim

# multilabel_soft_margin_loss = sigmoid + binary_cross_entropy

class AttGAN():
    def __init__(self, args):
        self.mode = args.mode
        self.gpu = args.gpu
        self.multi_gpu = args.multi_gpu if 'multi_gpu' in args else False
        self.lambda_Datt = args.lambda_Datt
        self.lambda_Dcls = args.lambda_Dcls
        self.lambda_Gcls = args.lambda_Gcls
        self.lambda_rec = args.lambda_rec
        self.lambda_CM = args.lambda_CM
        self.lambda_gp = args.lambda_gp
        
        self.G = Generator(
            args.enc_dim, args.enc_layers, args.enc_norm, args.enc_acti,
            args.dec_dim, args.dec_layers, args.dec_norm, args.dec_acti,
            args.n_attrs, args.shortcut_layers, args.inject_layers, args.img_size
        )
        self.G.train()
        if self.gpu: self.G.cuda()
        # summary(self.G, [(3, args.img_size, args.img_size), (args.n_attrs, 1, 1)], batch_size=4, device='cuda' if args.gpu else 'cpu')
        
        self.D = Discriminators(
            args.dis_dim, args.dis_norm, args.dis_acti,
            args.dis_fc_dim, args.dis_fc_norm, args.dis_fc_acti, args.dis_layers
        )
        self.D.train()
        if self.gpu: self.D.cuda()
        # summary(self.D, [(3, args.img_size, args.img_size)], batch_size=4, device='cuda' if args.gpu else 'cpu')
        
        if self.multi_gpu:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)
        
        self.optim_G = optim.Adam(self.G.parameters(), lr=args.lr, betas=args.betas)
        self.optim_D = optim.Adam(self.D.parameters(), lr=args.lr, betas=args.betas)
    
    def set_lr(self, lr):
        for g in self.optim_G.param_groups:
            g['lr'] = lr
        for g in self.optim_D.param_groups:
            g['lr'] = lr
    
    def trainG(self, img_real, label_trg, attr_diff):
        for p in self.D.parameters():
            p.requires_grad = False

        zs = self.G(img_real, mode='enc')
        img_fake = self.G(zs, attr_diff, mode='dec')
        img_recon = self.G(zs, torch.zeros_like(attr_diff), mode='dec')
        abn_att_real, cabn_att_real = self.D(img_real, 'abn')
        adv_fake, _, _, cls1_rx_fake, cls2_rx_fake, abn_att_fake, cabn_att_fake = self.D(img_fake)

        num, _, fh, fw = abn_att_real[1].size()
        N_ele = fh * fw

        if self.mode == 'wgan':
            gf_loss = -adv_fake.mean()
        if self.mode == 'lsgan':  # mean_squared_error
            gf_loss = F.mse_loss(adv_fake, torch.ones_like(adv_fake))
        if self.mode == 'dcgan':  # sigmoid_cross_entropy
            gf_loss = F.binary_cross_entropy_with_logits(adv_fake, torch.ones_like(adv_fake))

        gc_loss = 0
        for i in range(len(cls1_rx_fake)):
            gc_loss += F.binary_cross_entropy_with_logits(cls1_rx_fake[i], label_trg[:,i].view(label_trg.size(0), 1)) + \
                       F.binary_cross_entropy_with_logits(cls2_rx_fake[i], 1 - label_trg[:,i].view(label_trg.size(0), 1))

        gr_loss = F.l1_loss(img_recon, img_real)
        gcm_loss = 0
        for i in range(len(abn_att_real)):
            for j in range(num):
                if attr_diff[j, i] == 0:
                    gcm_loss += F.l1_loss(abn_att_real[i][j], abn_att_fake[i][j]) / N_ele + \
                                F.l1_loss(cabn_att_real[i][j], cabn_att_fake[i][j]) / N_ele
                else:
                    gcm_loss += F.l1_loss(abn_att_real[i][j], cabn_att_fake[i][j]) / N_ele + \
                                F.l1_loss(cabn_att_real[i][j], abn_att_fake[i][j]) / N_ele

        gcm_loss /= num
        g_loss = gf_loss + self.lambda_Gcls * gc_loss +\
                 self.lambda_rec * gr_loss + self.lambda_CM * gcm_loss
        
        self.optim_G.zero_grad()
        g_loss.backward()
        self.optim_G.step()
        
        errG = {
            'g_loss': g_loss.item(), 'gf_loss':gf_loss.item(),'gc_loss': gc_loss.item(),
            'gr_loss': gr_loss.item(), 'gcm_loss': gcm_loss.item()
        }
        return errG
    
    def trainD(self, img_real, label_org, attr_diff):
        for p in self.D.parameters():
            p.requires_grad = True


        img_fake = self.G(img_real, attr_diff).detach()
        adv_real, abn_ax_real, cabn_ax_real, cls1_rx_real, cls2_rx_real, _, _ = self.D(img_real)
        adv_fake = self.D(img_fake, 'adv')
        
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
            wd = adv_real.mean() - adv_fake.mean()
            df_loss = -wd
            df_gp = gradient_penalty(self.D, img_real, img_fake)
        if self.mode == 'lsgan':  # mean_squared_error
            df_loss = F.mse_loss(adv_real, torch.ones_like(adv_fake)) + \
                      F.mse_loss(adv_fake, torch.zeros_like(adv_fake))
            df_gp = gradient_penalty(self.D, img_real)
        if self.mode == 'dcgan':  # sigmoid_cross_entropy
            df_loss = F.binary_cross_entropy_with_logits(adv_real, torch.ones_like(adv_real)) + \
                      F.binary_cross_entropy_with_logits(adv_fake, torch.zeros_like(adv_fake))
            df_gp = gradient_penalty(self.D, img_real)

        datt_loss = F.binary_cross_entropy_with_logits(abn_ax_real, label_org) + \
                   F.binary_cross_entropy_with_logits(cabn_ax_real, 1 - label_org)

        dc_loss = 0
        for i in range(len(cls1_rx_real)):
            dc_loss += F.binary_cross_entropy_with_logits(cls1_rx_real[i], label_org[:,i].view(label_org.size(0), 1)) + \
                       F.binary_cross_entropy_with_logits(cls2_rx_real[i], 1 - label_org[:,i].view(label_org.size(0), 1))
        d_loss = df_loss + self.lambda_gp * df_gp + self.lambda_Datt * datt_loss\
                 + self.lambda_Dcls * dc_loss

        self.optim_D.zero_grad()
        d_loss.backward()
        self.optim_D.step()
        
        errD = {
            'd_loss': d_loss.item(), 'df_loss': df_loss.item(),
            'df_gp': df_gp.item(), 'datt_loss': datt_loss.item(),
            'dc_loss': dc_loss.item()
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


    def saveG_D(self, path, flag=None):
        states = {
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
        }
        if flag is None:
            torch.save(states, path)
        elif flag == 'unzip':
            torch.save(states, f=path, _use_new_zipfile_serialization=False)
    
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

    def saveD(self, path):
        states = {
            'D': self.G.state_dict()
        }
        torch.save(states, path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', dest='img_size', type=int, default=128)
    parser.add_argument('--shortcut_layers', dest='shortcut_layers', type=int, default=1)
    parser.add_argument('--inject_layers', dest='inject_layers', type=int, default=0)
    parser.add_argument('--enc_dim', dest='enc_dim', type=int, default=64)
    parser.add_argument('--dec_dim', dest='dec_dim', type=int, default=64)
    parser.add_argument('--dis_dim', dest='dis_dim', type=int, default=64)
    parser.add_argument('--dis_fc_dim', dest='dis_fc_dim', type=int, default=1024)
    parser.add_argument('--enc_layers', dest='enc_layers', type=int, default=5)
    parser.add_argument('--dec_layers', dest='dec_layers', type=int, default=5)
    parser.add_argument('--dis_layers', dest='dis_layers', type=int, default=5)
    parser.add_argument('--enc_norm', dest='enc_norm', type=str, default='batchnorm')
    parser.add_argument('--dec_norm', dest='dec_norm', type=str, default='batchnorm')
    parser.add_argument('--dis_norm', dest='dis_norm', type=str, default='instancenorm')
    parser.add_argument('--dis_fc_norm', dest='dis_fc_norm', type=str, default='none')
    parser.add_argument('--enc_acti', dest='enc_acti', type=str, default='lrelu')
    parser.add_argument('--dec_acti', dest='dec_acti', type=str, default='relu')
    parser.add_argument('--dis_acti', dest='dis_acti', type=str, default='lrelu')
    parser.add_argument('--dis_fc_acti', dest='dis_fc_acti', type=str, default='relu')
    parser.add_argument('--lambda_1', dest='lambda_1', type=float, default=100.0)
    parser.add_argument('--lambda_2', dest='lambda_2', type=float, default=10.0)
    parser.add_argument('--lambda_3', dest='lambda_3', type=float, default=1.0)
    parser.add_argument('--lambda_gp', dest='lambda_gp', type=float, default=10.0)
    parser.add_argument('--mode', dest='mode', default='wgan', choices=['wgan', 'lsgan', 'dcgan'])
    parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.5)
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.999)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    args.n_attrs = 13
    args.betas = (args.beta1, args.beta2)
    attgan = AttGAN(args)

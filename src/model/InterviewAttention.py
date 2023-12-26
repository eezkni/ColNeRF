import torch
import torch.nn.init as init
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

def norm(norm_type, out_ch):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = torch.nn.BatchNorm2d(out_ch, affine=True)
    elif norm_type == 'instance':
        layer = torch.nn.InstanceNorm2d(out_ch, affine=False)
    else:
        raise NotImplementedError('Normalization layer [{:s}] is not found'.format(norm_type))
    return layer

def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for lrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = torch.nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = torch.nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = torch.nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('Activation layer [{:s}] is not found'.format(act_type))
    return layer

def sequential(*args):
    modules = []
    for module in args:
        if isinstance(module, torch.nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, torch.nn.Module):
            modules.append(module)
    return torch.nn.Sequential(*modules)
    
# conv norm activation
def conv_block(in_ch, out_ch, kernel_size, stride=1, dilation=1, padding=0, padding_mode='zeros', norm_type=None,
               act_type='relu', groups=1, inplace=True):
    c = torch.nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding,
                  padding_mode=padding_mode, groups=groups)
    n = norm(norm_type, out_ch) if norm_type else None
    a = act(act_type, inplace) if act_type else None
    return sequential(c, n, a)

class AFGSA(torch.nn.Module):
    def __init__(self, ch, block_size=5, halo_size=3, num_heads=4, sr = 1, bias=False):
        super(AFGSA, self).__init__()
        self.block_size = block_size
        self.halo_size = halo_size
        self.num_heads = num_heads
        self.head_ch = ch // num_heads
        assert ch % num_heads == 0, "ch should be divided by # heads"

        # relative positional embedding: row and column embedding each with dimension 1/2 head_ch
        self.rel_h = torch.nn.Parameter(torch.randn(1, block_size+2*halo_size, 1, self.head_ch//2), requires_grad=True)
        self.rel_w = torch.nn.Parameter(torch.randn(1, 1, block_size+2*halo_size, self.head_ch//2), requires_grad=True)

        self.conv_map = conv_block(ch*2, ch, kernel_size=1, act_type='relu',)
        self.q_conv = torch.nn.Conv2d(ch, ch, kernel_size=1, bias=bias)
        self.k_conv = torch.nn.Conv2d(ch, ch, kernel_size=1, bias=bias)
        self.v_conv = torch.nn.Conv2d(ch, ch, kernel_size=1, bias=bias)
        self.sr = sr
        if self.sr > 1:
            self.sampler = torch.nn.MaxPool2d(2, sr)
            self.LocalProp= torch.nn.ConvTranspose2d(ch, ch, kernel_size=sr, stride=sr, groups=ch)
            torch.nn.init.kaiming_normal_(self.LocalProp.weight, mode='fan_in', nonlinearity='relu')

        self.reset_parameters()
        
    def reset_parameters(self):
        init.kaiming_normal_(self.q_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.k_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.v_conv.weight, mode='fan_out', nonlinearity='relu')
        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)

    def forward(self, noisy, aux):
        n_aux = self.conv_map(torch.cat([noisy, aux], dim=1))
        
        if self.sr > 1.:
            n_aux = self.sampler(n_aux)
            noisy = self.sampler(noisy)
            
        b, c, h, w, block, halo, heads = *noisy.shape, self.block_size, self.halo_size, self.num_heads
        assert h % block == 0 and w % block == 0, 'feature map dimensions must be divisible by the block size'

        q = self.q_conv(n_aux)
        q = rearrange(q, 'b c (h k1) (w k2) -> (b h w) (k1 k2) c', k1=block, k2=block)
        q *= self.head_ch ** -0.5  # b*#blocks, flattened_query, c

        k = self.k_conv(n_aux)
        k = F.unfold(k, kernel_size=block+halo*2, stride=block, padding=halo)
        k = rearrange(k, 'b (c a) l -> (b l) a c', c=c)

        v = self.v_conv(noisy)
        v = F.unfold(v, kernel_size=block+halo*2, stride=block, padding=halo)
        v = rearrange(v, 'b (c a) l -> (b l) a c', c=c)

        # b*#blocks*#heads, flattened_vector, head_ch
        q, v = map(lambda i: rearrange(i, 'b a (h d) -> (b h) a d', h=heads), (q, v))
        # positional embedding
        k = rearrange(k, 'b (k1 k2) (h d) -> (b h) k1 k2 d', k1=block+2*halo, h=heads)
        k_h, k_w = k.split(self.head_ch//2, dim=-1)
        k = torch.cat([k_h+self.rel_h, k_w+self.rel_w], dim=-1)
        k = rearrange(k, 'b k1 k2 d -> b (k1 k2) d')

        # b*#blocks*#heads, flattened_query, flattened_neighborhood
        sim = torch.einsum('b i d, b j d -> b i j', q, k)
        attn = F.softmax(sim, dim=-1)
        # b*#blocks*#heads, flattened_query, head_ch
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h w n) (k1 k2) d -> b (n d) (h k1) (w k2)', b=b, h=(h//block), w=(w//block), k1=block, k2=block)
        
        if self.sr > 1.:
            # out = F.interpolate(out, size=(h * self.sr, w * self.sr), mode='bilinear', align_corners=True)
            out = self.LocalProp(out)
        return out

class TransformerBlock(torch.nn.Module):
    def __init__(self, ch, block_size=5, halo_size=3, num_heads=4, sr = 1, checkpoint=True):
        super(TransformerBlock, self).__init__()
        self.checkpoint = checkpoint
        self.attention = AFGSA(ch, block_size=block_size, halo_size=halo_size, num_heads=num_heads, sr = sr)
        self.feed_forward = torch.nn.Sequential(
            conv_block(ch, ch, kernel_size=3, padding=1, padding_mode='reflect', act_type='relu'),
            conv_block(ch, ch, kernel_size=3, padding=1, padding_mode='reflect', act_type='relu')
        )

    def forward(self, x):
        if self.checkpoint:
            noisy = x[0] + checkpoint(self.attention, x[0], x[1])
            noisy = noisy + checkpoint(self.feed_forward,noisy)
        else:
            noisy = x[0] + self.attention(x[0], x[1])
            noisy = noisy + self.feed_forward(noisy)
        return noisy

class InterviewFusion(torch.nn.Module):
    def __init__(self, base_ch=512, nviews=1, num_heads=4, block_size=8, halo_size=3, sr = 1, checkpoint=False, **kwargs,):
        super(InterviewFusion, self).__init__()
        self.trans = TransformerBlock(base_ch, block_size=block_size, halo_size=halo_size,
                                                           num_heads=num_heads, sr = sr, checkpoint=checkpoint)
        self.nviews = nviews
        
    def forward(self, latents):
        initial_latent = latents
        H = latents.shape[-2]
        W = latents.shape[-1]
        C = latents.shape[-3]
        x = latents.reshape(-1, self.nviews, C, H, W)# B , Nviews, ...
        
        if self.nviews > 1:
            x = torch.unbind(x, dim=1)
            inter_value = []
            for i in range(self.nviews):
                src = x[i]
                ref = x[:i] + x[i+1:]
                aux = torch.zeros_like(src)
                for j in range(self.nviews-1):
                    aux = aux + ref[j]
                inter_value.append(self.trans([src, aux]))
                
        return inter_value
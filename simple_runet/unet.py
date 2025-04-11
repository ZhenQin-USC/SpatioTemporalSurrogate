import torch
import torch.nn as nn
from os.path import join
from typing import Callable, List, Optional, Sequence, Tuple, Union


class ConvLSTM3DCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, 
                 kernel_size=3, padding=1, padding_mode='zeros', bias=True, 
                 ):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTM3DCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding_mode = padding_mode
        self.bias = bias

        self.input_conv = nn.Conv3d(in_channels=self.input_dim,
                                    out_channels=4*self.hidden_dim,
                                    kernel_size=self.kernel_size,
                                    padding=padding,
                                    padding_mode=self.padding_mode,
                                    stride=(1,1,1),
                                    bias=self.bias)
        
        self.recurrent_conv = nn.Conv3d(in_channels=self.hidden_dim, 
                                        out_channels=4*self.hidden_dim,
                                        kernel_size=self.kernel_size, 
                                        padding=padding,
                                        padding_mode=self.padding_mode,
                                        stride=(1,1,1),
                                        bias=False)
        
        self.recurrent_activation = nn.Sigmoid()
        self.activation = nn.Tanh()

    def forward(self, inputs, states):
        h_tm1, c_tm1 = states

        x = self.input_conv(inputs)
        x_i, x_f, x_c, x_o = torch.split(x, self.hidden_dim, dim=1)

        h = self.recurrent_conv(h_tm1)
        h_i, h_f, h_c, h_o = torch.split(h, self.hidden_dim, dim=1)
        
        f = self.recurrent_activation(x_f + h_f)
        i = self.recurrent_activation(x_i + h_i)
        c = f * c_tm1 + i * self.activation(x_c + h_c)
        o = self.recurrent_activation(x_o + h_o)
        h = o * self.activation(c)

        return h, c

    def init_hidden(self, batch_size, input_size):
        depth, height, width = input_size
        return (torch.zeros(batch_size, self.hidden_dim, depth, height, width, 
                            device=self.input_conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, depth, height, width, 
                            device=self.input_conv.weight.device))


class Conv_block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, 
                 act_layer: Callable = nn.GELU, stride=1, padding_mode='zeros', 
                 norm_type='group', num_groups=None):

        super(Conv_block, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              padding_mode=padding_mode)
        if norm_type == 'batch': 
            self.norm = nn.BatchNorm3d(out_channels)
        elif norm_type == 'group':
            if num_groups == None:
                num_groups = out_channels
            self.norm = nn.GroupNorm(num_groups=num_groups, 
                                     num_channels=out_channels)
        self.nonlinear = act_layer()

    def forward(self, x):
        x = self.norm(self.conv(x))
        return self.nonlinear(x)


class ResConv_block(nn.Module):

    def __init__(self, in_channels, kernel_size=3, padding=1, 
                 stride=1, padding_mode='zeros', norm_type='group', num_groups=None):
        
        super(ResConv_block, self).__init__()

        self.conv0 = nn.Conv3d(in_channels=in_channels,
                               out_channels=in_channels,
                               kernel_size=kernel_size,
                               padding=padding,
                               stride=stride,
                               padding_mode=padding_mode)
        
        self.conv1 = nn.Conv3d(in_channels=in_channels,
                               out_channels=in_channels,
                               kernel_size=kernel_size,
                               padding=padding,
                               stride=stride,
                               padding_mode=padding_mode)

        if norm_type == 'batch': 
            self.norm0 = nn.BatchNorm3d(in_channels)
            self.norm1 = nn.BatchNorm3d(in_channels)
        elif norm_type == 'group':
            if num_groups == None:
                num_groups = in_channels
            self.norm0 = nn.GroupNorm(num_groups=num_groups, 
                                      num_channels=in_channels)
            self.norm1 = nn.GroupNorm(num_groups=num_groups, 
                                      num_channels=in_channels)
        
        self.nonlinear = nn.ReLU()

    def forward(self, x):
        a = self.conv0(x)
        a = self.nonlinear(self.norm0(a))
        a = self.conv1(a)
        y = self.norm1(a)
        return x + y


class Deconv_block(nn.Module):
    def __init__(self, scale_factor, in_channels, out_channels, kernel_size=3, padding=0, 
                 stride=1, padding_mode='reflect', norm_type='group', num_groups=None):
        super(Deconv_block, self).__init__()
        if isinstance(scale_factor, list):
            scale_factor = tuple(scale_factor)
        elif not isinstance(scale_factor, (int, tuple)):
            raise ValueError("scale_factor must be an int, tuple, or list.")

        self.deconv = nn.Upsample(scale_factor=scale_factor)
        self.conv = nn.Conv3d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=stride,
                              padding_mode=padding_mode)
        if norm_type == 'batch': 
            self.norm = nn.BatchNorm3d(out_channels)
        elif norm_type == 'group':
            if num_groups == None:
                num_groups = out_channels
            self.norm = nn.GroupNorm(num_groups=num_groups, 
                                     num_channels=out_channels)
        self.nonlinear = nn.ReLU()

    def forward(self, x):
        x = self.deconv(x)
        x = self.norm(self.conv(x))
        return self.nonlinear(x)


class Encoder(nn.Module):
    def __init__(self, 
                 ninputs, 
                 filters=16, 
                 norm_type='group', 
                 num_groups=4, 
                 kernel_size=3, 
                 padding=1, 
                 strides=None):
        
        super(Encoder, self).__init__()

        if strides == None:
            stride0, stride1 = 2, 2
        else:
            stride0, stride1 = strides

        self.conv = nn.Conv3d(ninputs, 1*filters, kernel_size=kernel_size, padding=padding, stride=1)
        
        self.conv0 = Conv_block(
            in_channels=1*filters, out_channels=2*filters, kernel_size=kernel_size, 
            padding=padding, stride=stride0, norm_type=norm_type, num_groups=num_groups)
        
        self.conv1 = Conv_block(
            in_channels=2*filters, out_channels=2*filters, kernel_size=kernel_size, 
            padding=padding, stride=1, norm_type=norm_type, num_groups=num_groups)
        
        self.conv2 = Conv_block(
            in_channels=2*filters, out_channels=4*filters, kernel_size=kernel_size, 
            padding=padding, stride=stride1, norm_type=norm_type, num_groups=num_groups)
        
        self.conv3 = Conv_block(
            in_channels=4*filters, out_channels=4*filters, kernel_size=kernel_size, 
            padding=padding, stride=1, norm_type=norm_type, num_groups=num_groups)


    def forward(self, inputs):
        x0 = self.conv(inputs)
        x1 = self.conv0(x0)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        return [x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, noutputs, 
                 filters=16, 
                 norm_type='group', 
                 num_groups=4, 
                 kernel_size=3, 
                 padding=1, 
                 with_control=False, 
                 strides=None):
        
        super(Decoder, self).__init__()
        if strides == None:
            stride1, stride0 = 2, 2
        else:
            stride1, stride0 = strides
            
        self.with_control = with_control 
        if self.with_control:
            coeff=3
        else:
            coeff=2

        self.deconv4 = Deconv_block(
            1, in_channels=coeff*4*filters, out_channels=4*filters, 
            kernel_size=kernel_size, padding=padding, norm_type=norm_type, num_groups=num_groups)
        
        self.deconv3 = Deconv_block(
            stride0, in_channels=coeff*4*filters, out_channels=2*filters, 
            kernel_size=kernel_size, padding=padding, norm_type=norm_type, num_groups=num_groups)
        
        self.deconv2 = Deconv_block(
            1, in_channels=coeff*2*filters, out_channels=2*filters, 
            kernel_size=kernel_size, padding=padding, norm_type=norm_type, num_groups=num_groups)
        
        self.deconv1 = Deconv_block(
            stride1, in_channels=coeff*2*filters, out_channels=1*filters, 
            kernel_size=kernel_size, padding=padding, norm_type=norm_type, num_groups=num_groups)
        
        self.conv = nn.Conv3d(1*filters, noutputs, kernel_size=kernel_size, padding=padding, stride=1)
        
    def forward(self, x, x_enc, u_enc):
        x1, x2, x3, x4 = x_enc
        if self.with_control:
            u1, u2, u3, u4 = u_enc
            x4_ = self.deconv4(torch.cat(( x,  x4, u4), dim=1))
            x3_ = self.deconv3(torch.cat((x4_, x3, u3), dim=1))
            x2_ = self.deconv2(torch.cat((x3_, x2, u2), dim=1))
            x1_ = self.deconv1(torch.cat((x2_, x1, u1), dim=1))
        else:          
            x4_ = self.deconv4(torch.cat(( x,  x4), dim=1))
            x3_ = self.deconv3(torch.cat((x4_, x3), dim=1))
            x2_ = self.deconv2(torch.cat((x3_, x2), dim=1))
            x1_ = self.deconv1(torch.cat((x2_, x1), dim=1))
        outputs = self.conv(x1_)
        return outputs


class SimpleRUNet(nn.Module):
    def __init__(self, 
                 steps: int = 1, 
                 filters: int = 8, 
                 units: List[int] = [2, 1], 
                 with_control: bool = False, 
                 norm_type: str = 'group', 
                 num_groups: int = 4, 
                 strides: List[int] = [2, 2], 
                 **kwargs
                 ):
        
        super().__init__()
        self.steps = steps
        nx, nm = units
        self.with_control = with_control
        self.x_encoder = Encoder(nx, filters=filters, norm_type=norm_type, 
                                 num_groups=num_groups, strides=strides)
        self.m_encoder = Encoder(nm, filters=filters, norm_type=norm_type, 
                                 num_groups=num_groups, strides=strides)
        self.resnet0 = ResConv_block(4*filters, norm_type=norm_type, num_groups=num_groups)
        self.resnet1 = ResConv_block(4*filters, norm_type=norm_type, num_groups=num_groups)
        self.resnet2 = ResConv_block(4*filters, norm_type=norm_type, num_groups=num_groups)
        self.resnet3 = ResConv_block(4*filters, norm_type=norm_type, num_groups=num_groups)
        self.m_conv  = Conv_block(in_channels=4*filters, out_channels=4*filters, 
                                  kernel_size=3, padding=1, stride=1,
                                  norm_type=norm_type, num_groups=num_groups)

        self.convlstm = ConvLSTM3DCell(4*filters, 4*filters)
        self.x_decoder  = Decoder(nx, filters=filters, with_control=with_control, 
                                  norm_type=norm_type, num_groups=num_groups, strides=strides)

    def forward(self, _states, _static):

        output_seq = []

        # Encoder for Dynamic States:
        x_enc1, x_enc2, x_enc3, x_enc4 = self.x_encoder(_states)
        zx = self.resnet0(x_enc4)
        zx = self.resnet1(zx)
        
        # Encoder for Static States:
        m_enc1, m_enc2, m_enc3, m_enc4 = self.m_encoder(_static)
        zm = self.m_conv(m_enc4)

        # Initial States for RNN:
        hidden_states = [zx, zx]

        # ConvLSTM:
        for step in range(self.steps):            
            h, c = self.convlstm(zm, states=hidden_states)
            hidden_states = [h, c]

            # ResNet Before Decoder:
            h = self.resnet2(h)
            h = self.resnet3(h)
            out = self.x_decoder(h, 
                               [x_enc1, x_enc2, x_enc3, x_enc4],
                               [m_enc1, m_enc2, m_enc3, m_enc4]
                               )
            output_seq.append(out)

        preds = torch.stack(output_seq, dim=1)
        return preds


class RUNetSequential(nn.Module):
    def __init__(self, 
                 filters: int, 
                 units: List[int] = [10, 2, 2], 
                 with_control: bool = False, 
                 with_states: bool = True,
                 norm_type: str = 'group', 
                 num_groups: int = 4, 
                 kernel_size: int = 3,
                 padding: int = 1,
                 strides: List[int] = [2, 2], 
                 ):
        
        super().__init__()
        nstatic, ncontrols, noutputs = units
        self.with_control = with_control
        self.with_states = with_states

        self.x_encoder = Encoder(nstatic+noutputs, filters=filters, norm_type=norm_type, 
                                 kernel_size=kernel_size, padding=padding, 
                                 num_groups=num_groups, strides=strides)
        
        self.m_encoder = Encoder(nstatic, filters=filters, norm_type=norm_type, 
                                 kernel_size=kernel_size, padding=padding, 
                                 num_groups=num_groups, strides=strides)

        self.u_encoder = Encoder(ncontrols, filters=filters, norm_type=norm_type, 
                                 kernel_size=kernel_size, padding=padding, 
                                 num_groups=num_groups, strides=strides)
        
        self.resnet0 = ResConv_block(4*filters, kernel_size=kernel_size, padding=padding, norm_type=norm_type, num_groups=num_groups)
        self.resnet1 = ResConv_block(4*filters, kernel_size=kernel_size, padding=padding, norm_type=norm_type, num_groups=num_groups)
        self.resnet2 = ResConv_block(4*filters, kernel_size=kernel_size, padding=padding, norm_type=norm_type, num_groups=num_groups)
        self.resnet3 = ResConv_block(4*filters, kernel_size=kernel_size, padding=padding, norm_type=norm_type, num_groups=num_groups)
        
        factor = 2 if with_states else 1
        self.x_conv  = Conv_block(in_channels=factor*4*filters, out_channels=4*filters, 
                                  kernel_size=kernel_size, padding=padding, stride=1,
                                  norm_type=norm_type, num_groups=num_groups)

        self.convlstm = ConvLSTM3DCell(4*filters, 4*filters, kernel_size=kernel_size, padding=padding)

        self.decoder  = Decoder(noutputs, filters=filters, with_control=with_control, 
                                kernel_size=kernel_size, padding=padding, 
                                norm_type=norm_type, num_groups=num_groups, strides=strides)

    def forward(self, contrl, _states, _static):
        B, T, _, Nx, Ny, Nz = contrl.size()

        states = torch.cat((_states, _static), dim=1)
        
        # Encoder:
        x_enc1, x_enc2, x_enc3, x_enc4 = self.x_encoder(states)
        m_enc1, m_enc2, m_enc3, m_enc4 = self.m_encoder(_static)

        # ResNet After Encoder:
        x4 = self.resnet0(x_enc4)
        x4 = self.resnet1(x4)
        hidden_states = [x4, x4]
        
        # ConvLSTM:
        output_seq = []
        for step in range(T):
            _contrl = contrl[:,step,...]
            u_enc1, u_enc2, u_enc3, u_enc4 = self.u_encoder(_contrl)
            if self.with_states:
                x = self.x_conv(torch.cat((u_enc4, x4), dim=1))
            else:
                x = self.x_conv(u_enc4)
            x, c = self.convlstm(x, states=hidden_states)
            hidden_states = [x, c]

            # ResNet Before Decoder:
            x = self.resnet2(x)
            x = self.resnet3(x)
            out = self.decoder(x, 
                               [m_enc1, m_enc2, m_enc3, m_enc4],
                               [u_enc1, u_enc2, u_enc3, u_enc4]
                               )
            output_seq.append(out)

        preds = torch.stack(output_seq, dim=1)
        return preds


class RUNet(RUNetSequential):
    """
    A parallel variant of RUNet where control encoder and decoder operate independently,
    useful when control signal needs to be disentangled or multi-streamed.
    """
    def __init__(self, 
                 filters: int, 
                 units: List[int] = [10, 2, 2], 
                 with_control: bool = False, 
                 with_states: bool = True,
                 norm_type: str = 'group', 
                 num_groups: int = 4, 
                 kernel_size: int = 3,
                 padding: int = 1,
                 strides: List[int] = [2, 2]):
        super().__init__(filters=filters, units=units, with_control=with_control,
                         with_states=with_states, norm_type=norm_type, num_groups=num_groups,
                         kernel_size=kernel_size, padding=padding, strides=strides)
        
    def forward(self, contrl, _states, _static):
        B, T, C, X, Y, Z = contrl.shape

        states = torch.cat((_states, _static), dim=1)

        # Encoder:
        x_enc1, x_enc2, x_enc3, x_enc4 = self.x_encoder(states)
        m_enc1, m_enc2, m_enc3, m_enc4 = self.m_encoder(_static)

        # ResNet After Encoder:
        x4 = self.resnet0(x_enc4)
        x4 = self.resnet1(x4)
        hidden_states = [x4, x4]

        # Contrl Encoder: (B, T, C, X, Y, Z) -> (B*T, C, X, Y, Z)
        u_enc1, u_enc2, u_enc3, u_enc4 = self.u_encoder(contrl.reshape(B * T, C, X, Y, Z)) 
        u4s = u_enc4.reshape(B, T, *u_enc4.shape[1:]) # (B, T, C, X, Y, Z)

        # Conv-LSTM for latent dynamics:
        latent_seq = []
        for step in range(T):
            u4 = u4s[:, step, ...]
            if self.with_states:
                x = self.x_conv(torch.cat((u4, x4), dim=1))
            else:
                x = self.x_conv(u4)

            h, c = self.convlstm(x, states=hidden_states)
            hidden_states = [h, c]
            latent_seq.append(h)

        latent_seq = torch.stack(latent_seq, dim=1)  # (B, T, C, X, Y, Z)

        preds = self.parallel_decoder(latent_seq,  # (B, T, C, X, Y, Z)
                                      [m_enc1, m_enc2, m_enc3, m_enc4],
                                      [u_enc1, u_enc2, u_enc3, u_enc4])
        
        return preds
    
    def parallel_decoder(self, latent_seq, m_encs, u_encs):
        B, T, C, X, Y, Z = latent_seq.shape

        x = latent_seq.reshape(B * T, C, X, Y, Z)

        x_encs_repeated = [torch.repeat_interleave(enc, repeats=T, dim=0) 
                           for i, enc in enumerate(m_encs)]

        out = self.decoder(x, x_encs_repeated, u_encs) # (B*T, C, X, Y, Z)

        output_seq = out.reshape(B, T, *out.shape[1:])
        return output_seq


''' from original FNO repo '''
import torch
import torch.nn as nn
from .basics import SpectralConv2dV2, _get_act


def _coerce_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ['1', 'true', 'yes', 'y', 'on']
    return bool(val)


def _resolve_zero_mode_enforcement(params):
    enforcement = getattr(params, 'constraint_zero_mode_enforcement', None)
    if enforcement is not None:
        if isinstance(enforcement, bool):
            return 'hard' if enforcement else 'off'
        enforcement = str(enforcement).strip().lower()
        if enforcement in ['true', 'false']:
            return 'hard' if enforcement == 'true' else 'off'
        if enforcement not in ['off', 'hard', 'soft']:
            raise ValueError("constraint_zero_mode_enforcement must be one of ['off', 'hard', 'soft']")
        return enforcement

    return 'hard' if _coerce_bool(getattr(params, 'constraint_zero_mode_enable', False)) else 'off'


class FNN2d(nn.Module):
    def __init__(self, modes1, modes2,
                 width=64, fc_dim=128,
                 layers=None,
                 in_dim=3, out_dim=1,
                 dropout=0,
                 activation='tanh',
                 mean_constraint=False,
                 mean_constraint_mode='all',
                 mean_constraint_omega_channel=None,
                 mean_constraint_omega_tol=1.0e-8):
        super(FNN2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        # input channel is 3: (a(x, y), x, y)
        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.fc0 = nn.Linear(in_dim, self.layers[0])

        self.sp_convs = nn.ModuleList([SpectralConv2dV2(
            in_size, out_size, mode1_num, mode2_num)
            for in_size, out_size, mode1_num, mode2_num
            in zip(self.layers, self.layers[1:], self.modes1, self.modes2)])

        self.dropout = nn.Dropout(p=dropout)

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.layers, self.layers[1:])])

        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)
        self.activation = _get_act(activation)
        self.mean_constraint = mean_constraint
        self.mean_constraint_mode = mean_constraint_mode
        self.mean_constraint_omega_channel = mean_constraint_omega_channel
        self.mean_constraint_omega_tol = mean_constraint_omega_tol

    def _mean_projection_mask(self, inputs):
        if self.mean_constraint_mode == 'all':
            return None

        if self.mean_constraint_mode == 'gauge_aware':
            if self.mean_constraint_omega_channel is None:
                return None
            omega = inputs[:, self.mean_constraint_omega_channel:self.mean_constraint_omega_channel+1]
            return (torch.abs(omega) <= self.mean_constraint_omega_tol)

        return None

    def forward(self, x):
        '''
        (b,c,h,w) -> (b,1,h,w)
        '''
        input_tensor = x
        length = len(self.ws)
        batchsize = x.shape[0]
        size_x, size_y = x.shape[2], x.shape[3]

        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x) # project
        x = x.permute(0, 3, 1, 2)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x.view(batchsize, self.layers[i], -1)).view(batchsize, self.layers[i+1], size_x, size_y)
            x = x1 + x2
            if i != length - 1:
                x = self.activation(x)
            x = self.dropout(x)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x.permute(0, 3, 1, 2)

        if self.mean_constraint:
            dc = torch.mean(x, dim=(-2, -1), keepdim=True)
            mean_mask = self._mean_projection_mask(input_tensor)
            if mean_mask is None:
                x = x - dc
            else:
                x = x - dc * mean_mask.to(x.dtype)

        return x

def fno(params):
    if params.mode_cut > 0:
        params.modes1 = [params.mode_cut]*len(params.modes1)
        params.modes2 = [params.mode_cut]*len(params.modes2)

    if params.embed_cut > 0:
        params.layers = [params.embed_cut]*len(params.layers)

    if params.fc_cut > 0 and params.embed_cut > 0:
        params.fc_dim = params.embed_cut * params.fc_cut

    input_dim = params.in_dim

    zero_mode_enforcement = _resolve_zero_mode_enforcement(params)
    zero_mode_enable = (zero_mode_enforcement == 'hard')
    zero_mode_mode = str(getattr(params, 'constraint_zero_mode_mode', 'all')).lower()
    zero_mode_tol = float(getattr(params, 'constraint_zero_mode_omega_tol', 1.0e-8))

    omega_channel = getattr(params, 'constraint_zero_mode_omega_channel', None)
    if omega_channel is None:
        system = str(getattr(params, 'system', '')).lower()
        if input_dim >= 7:
            omega_channel = 6
        elif system == 'helmholtz' and input_dim >= 3:
            omega_channel = 2
    else:
        omega_channel = int(omega_channel)

    return FNN2d(params.modes1, params.modes2, layers=params.layers, fc_dim=params.fc_dim,
                in_dim=input_dim, out_dim=params.out_dim, dropout=params.dropout,
                activation='gelu', mean_constraint=zero_mode_enable,
                mean_constraint_mode=zero_mode_mode,
                mean_constraint_omega_channel=omega_channel,
                mean_constraint_omega_tol=zero_mode_tol)

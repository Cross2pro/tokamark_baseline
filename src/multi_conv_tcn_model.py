"""
Multi-branch CNN front-end + Temporal Convolutional Network (TCN) backbone.

Same encoder/decoder branch layout as the CNN and LSTM baselines; replaces the
flatten-MLP / LSTM middle with dilated causal convolutions so the time axis is
kept (addresses the CNN backbone flattening shortcoming).
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.utils.checkpoint import checkpoint
from torchinfo import summary

from src.model_transform import _make_dummy_outputs
from src.conv_encoders_decoders import (
    Conv1DEncoder,
    Conv2DEncoder,
    Conv3DEncoder,
    Conv1DDecoder,
    Conv2DDecoder,
    Conv3DDecoder,
)
from tokamark.tools.utils import get_device

device = get_device()

padding = 1
kernel_size = 3
stride = 3
layers_encoder = 3
layers_decoder = 3
bb_factor = 2


class Chomp1d(nn.Module):
    """Trim causal padding on the right so length matches the input."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """One residual dilated causal conv block (Bai et al. TCN)."""

    def __init__(self, n_inputs, n_outputs, k, stride_, dilation, dropout=0.2):
        super().__init__()
        pad = (k - 1) * dilation
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, k, stride=stride_, padding=pad, dilation=dilation)
        )
        self.chomp1 = Chomp1d(pad)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, k, stride=stride_, padding=pad, dilation=dilation)
        )
        self.chomp2 = Chomp1d(pad)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Stack of TemporalBlocks; input/output shaped (B, C, T)."""

    def __init__(self, num_inputs, num_channels, kernel_size_=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    in_ch,
                    out_ch,
                    kernel_size_,
                    stride_=1,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def create_tcn_architecture(dataloader_, dict_metadata, D=16, verbose=True):
    if verbose:
        print("\n\n----------TCN MODEL INITIALIZATION----------\n")

    input_shapes = []
    exogenous_shapes = []
    output_shapes = []

    for l, first_window in enumerate(dataloader_.dataset):
        try:
            input_shapes = [arr.shape for arr in first_window["input"]]
            exogenous_shapes = [arr.shape for arr in first_window["exogenous"]]
            output_shapes = [arr.shape for arr in first_window["y"]]

            if verbose:
                print(f"Shot {first_window['shot_id']} used as reference")
                print(f"Input shapes are: {input_shapes}")
                print(f"Actuator future shapes are: {exogenous_shapes}")
                print(f"Output shapes are: {output_shapes}")
            break
        except Exception as e:
            print(f"Skipping sample {l} because not trainable: {e}")
            continue

    model = MultiConv_TCN(
        input_shapes=input_shapes,
        exogenous_shapes=exogenous_shapes,
        output_shapes=output_shapes,
        dict_metadata=dict_metadata,
        D=D,
    ).to(device)

    if verbose:
        input_sizes = []
        for shape in input_shapes + exogenous_shapes:
            input_sizes.append((2,) + shape)
        summary(model, input_size=input_sizes)

    return model


class MultiConv_TCN(nn.Module):
    """
    Multi-branch CNN encoders -> TCN encoder (keeps time) -> fuse future actuators
    -> TCN decoder over forecast horizon -> multi-branch CNN decoders.
    """

    def __init__(
        self,
        input_shapes,
        exogenous_shapes,
        output_shapes,
        dict_metadata,
        D=16,
        tcn_levels=3,
        tcn_kernel=3,
        tcn_dropout=0.2,
    ):
        super().__init__()

        self.D = D
        self.W_in = input_shapes[0][0]
        self.hidden = D * bb_factor

        self.output_shapes = output_shapes
        y = _make_dummy_outputs(output_shapes, dict_metadata)
        output_latent_shapes = [arr.shape for arr in y]
        self.W_out = output_latent_shapes[0][0]

        self.input_branches = nn.ModuleList()
        for var_shape in input_shapes:
            if len(var_shape) == 5:
                branch = Conv3DEncoder(var_shape[1:], D, layers_encoder, kernel_size, stride, padding, bb_factor)
            elif len(var_shape) == 4:
                branch = Conv2DEncoder(var_shape[1:], D, layers_encoder, kernel_size, stride, padding, bb_factor)
            elif len(var_shape) == 3:
                branch = Conv1DEncoder(var_shape[1:], D, layers_encoder, kernel_size, stride, padding, bb_factor)
            else:
                raise ValueError(f"Unsupported input shape: {var_shape[1:]}")
            self.input_branches.append(branch)

        enc_in_ch = self.hidden * len(self.input_branches)
        tcn_channels = [self.hidden] * tcn_levels
        self.encoder_tcn = TemporalConvNet(
            num_inputs=enc_in_ch,
            num_channels=tcn_channels,
            kernel_size_=tcn_kernel,
            dropout=tcn_dropout,
        )

        self.exogenous_branches = nn.ModuleList()
        for var_shape in exogenous_shapes:
            if len(var_shape) == 5:
                branch = Conv3DEncoder(var_shape[1:], D, layers_encoder, kernel_size, stride, padding, bb_factor)
            elif len(var_shape) == 4:
                branch = Conv2DEncoder(var_shape[1:], D, layers_encoder, kernel_size, stride, padding, bb_factor)
            elif len(var_shape) == 3:
                branch = Conv1DEncoder(var_shape[1:], D, layers_encoder, kernel_size, stride, padding, bb_factor)
            else:
                raise ValueError(f"Unsupported input shape: {var_shape[1:]}")
            self.exogenous_branches.append(branch)

        n_exo = len(exogenous_shapes)
        if n_exo > 0:
            dec_in_ch = self.hidden + self.hidden * n_exo
        else:
            dec_in_ch = self.hidden

        self.decoder_tcn = TemporalConvNet(
            num_inputs=dec_in_ch,
            num_channels=tcn_channels,
            kernel_size_=tcn_kernel,
            dropout=tcn_dropout,
        )

        self.start_token = nn.Parameter(torch.randn(1, 1, self.hidden))

        self.output_branches = nn.ModuleList()
        for var_shape in output_latent_shapes:
            if len(var_shape) == 5:
                branch = Conv3DDecoder(var_shape[1:], D, layers_decoder, kernel_size, stride, padding, bb_factor)
            elif len(var_shape) == 4:
                branch = Conv2DDecoder(var_shape[1:], D, layers_decoder, kernel_size, stride, padding, bb_factor)
            elif len(var_shape) == 3:
                branch = Conv1DDecoder(var_shape[1:], D, layers_decoder, kernel_size, stride, padding, bb_factor)
            else:
                raise ValueError(f"Unsupported input shape: {var_shape[1:]}")
            self.output_branches.append(branch)

    @staticmethod
    def _run_cnn_encoder(branch, x):
        B, W = x.shape[:2]
        x = x.view(B * W, *x.shape[2:])
        out = branch(x)
        out = out.view(B, W, -1)
        return out

    @staticmethod
    def _run_cnn_decoder(branch, goal_shape, x):
        B, W = x.shape[:2]
        x = x.reshape(B * W, *x.shape[2:])
        out = branch(x)
        out = out.reshape(B, W * out.shape[2], *out.shape[3:])
        target_len = goal_shape[0]
        out = out[:, -target_len:]
        return out

    def forward(self, *args):
        n_in = len(self.input_branches)
        n_exo = len(self.exogenous_branches)

        inputs = args[:n_in]
        exogenous = args[n_in : n_in + n_exo]

        input_branch_outputs = []
        for branch, x in zip(self.input_branches, inputs):
            out = checkpoint(self._run_cnn_encoder, branch, x, use_reentrant=False)
            input_branch_outputs.append(out)
        input_seq = torch.cat(input_branch_outputs, dim=2)

        exo_branch_outputs = []
        for branch, x in zip(self.exogenous_branches, exogenous):
            out = checkpoint(self._run_cnn_encoder, branch, x, use_reentrant=False)
            exo_branch_outputs.append(out)
        exo_seq = torch.cat(exo_branch_outputs, dim=2) if exo_branch_outputs else None

        enc = self.encoder_tcn(input_seq.transpose(1, 2))
        context = enc[:, :, -1]

        B = context.size(0)
        ctx_steps = context.unsqueeze(1).expand(B, self.W_out, self.hidden)
        if exo_seq is not None:
            if exo_seq.size(1) != self.W_out:
                exo_t = exo_seq.transpose(1, 2)
                exo_t = torch.nn.functional.interpolate(
                    exo_t, size=self.W_out, mode="linear", align_corners=False
                )
                exo_seq = exo_t.transpose(1, 2)
            dec_in = torch.cat([ctx_steps, exo_seq], dim=2)
        else:
            start = self.start_token.to(context.device).expand(B, self.W_out, self.hidden)
            dec_in = ctx_steps + start

        dec = self.decoder_tcn(dec_in.transpose(1, 2))
        dec_out = dec.transpose(1, 2)

        outputs = []
        for branch, goal_shape in zip(self.output_branches, self.output_shapes):
            out = checkpoint(self._run_cnn_decoder, branch, goal_shape, dec_out, use_reentrant=False)
            outputs.append(out)

        return outputs

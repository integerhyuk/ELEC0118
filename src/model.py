import math

import torch
from torch import nn
from transformers import BertConfig

from src.const import audio_processor
from src.mmi_module import MMI_Model


class Conv2dSubsampling(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]


class Conv2dSubsampling2(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/2 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling2 object."""
        super(Conv2dSubsampling2, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 1),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 2)), odim),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:1]

    class Conv2dSubsampling(torch.nn.Module):
        """Convolutional 2D subsampling (to 1/4 length).

        Args:
            idim (int): Input dimension.
            odim (int): Output dimension.
            dropout_rate (float): Dropout rate.
            pos_enc (torch.nn.Module): Custom position encoding layer.

        """

        def __init__(self, idim, odim, dropout_rate, pos_enc=None):
            """Construct an Conv2dSubsampling object."""
            super(Conv2dSubsampling, self).__init__()
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(1, odim, 3, 2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(odim, odim, 3, 2),
                torch.nn.ReLU(),
            )
            self.out = torch.nn.Sequential(
                torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
            )

        def forward(self, x, x_mask):
            """Subsample x.

            Args:
                x (torch.Tensor): Input tensor (#batch, time, idim).
                x_mask (torch.Tensor): Input mask (#batch, 1, time).

            Returns:
                torch.Tensor: Subsampled tensor (#batch, time', odim),
                    where time' = time // 4.
                torch.Tensor: Subsampled mask (#batch, 1, time'),
                    where time' = time // 4.

            """
            x = x.unsqueeze(1)  # (b, c, t, f)
            x = self.conv(x)
            b, c, t, f = x.size()
            x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
            if x_mask is None:
                return x, None
            return x, x_mask[:, :, :-2:2][:, :, :-2:2]


class StatisticsPooling(nn.Module):
    """This class implements a statistic pooling layer.
    It returns the mean and/or std of input tensor.
    Arguments
    ---------
    return_mean : True
         If True, the average pooling will be returned.
    return_std : True
         If True, the standard deviation will be returned.
    Example
    -------
    >>> inp_tensor = torch.rand([5, 100, 50])
    >>> sp_layer = StatisticsPooling()
    >>> out_tensor = sp_layer(inp_tensor)
    >>> out_tensor.shape
    torch.Size([5, 1, 100])
    """

    def __init__(self, return_mean=True, return_std=True):
        super().__init__()

        # Small value for GaussNoise
        self.eps = 1e-5
        self.return_mean = return_mean
        self.return_std = return_std
        if not (self.return_mean or self.return_std):
            raise ValueError(
                "both of statistics are equal to False \n"
                "consider enabling mean and/or std statistic pooling"
            )

    def forward(self, x, lengths=None):
        """Calculates mean and std for a batch (input tensor).
        Arguments
        ---------
        x : torch.Tensor
            It represents a tensor for a mini-batch.
        """
        if lengths is None:
            if self.return_mean:
                mean = x.mean(dim=1)
            if self.return_std:
                std = x.std(dim=1)
        else:
            mean = []
            std = []
            for snt_id in range(x.shape[0]):
                # Avoiding padded time steps
                actual_size = int(torch.round(lengths[snt_id] * x.shape[1]))

                # computing statistics
                if self.return_mean:
                    mean.append(
                        torch.mean(x[snt_id, 0:actual_size, ...], dim=0)
                    )
                if self.return_std:
                    std.append(torch.std(x[snt_id, 0:actual_size, ...], dim=0))
            if self.return_mean:
                mean = torch.stack(mean)
            if self.return_std:
                std = torch.stack(std)

        if self.return_mean:
            gnoise = self._get_gauss_noise(mean.size(), device=mean.device)
            gnoise = gnoise
            mean += gnoise
        if self.return_std:
            std = std + self.eps

        # Append mean and std of the batch
        if self.return_mean and self.return_std:
            pooled_stats = torch.cat((mean, std), dim=1)
            pooled_stats = pooled_stats.unsqueeze(1)
        elif self.return_mean:
            pooled_stats = mean.unsqueeze(1)
        elif self.return_std:
            pooled_stats = std.unsqueeze(1)

        return pooled_stats


class ActivateFun(nn.Module):
    def __init__(self, activate_fun):
        super(ActivateFun, self).__init__()
        self.activate_fun = activate_fun

    def _gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x):
        if self.activate_fun == 'relu':
            return torch.relu(x)
        elif self.activate_fun == 'gelu':
            return self._gelu(x)


class FuseModel(nn.Module):

    def __init__(self, text_config):

        super().__init__()

        tran_dim = 768

        self.config_mmi = BertConfig('config.json')
        self.model_mmi = MMI_Model(self.config_mmi,len(audio_processor.tokenizer),4)

        self.temperature = 0.07

        self.orgin_linear_change = nn.Sequential(
            nn.Linear(tran_dim*2, tran_dim),
            ActivateFun('gelu'),
            nn.Linear(tran_dim, tran_dim)
        )

        self.augment_linear_change = nn.Sequential(
            nn.Linear(tran_dim*2, tran_dim),
            ActivateFun('gelu'),
            nn.Linear(tran_dim, tran_dim)
        )

    # def forward_encoder(self, text_output, attention_mask, audio_inputs, audio_length, ctc_labels, emotion_labels, augmentation = False):
    def forward_encoder(self, text_output, attention_mask, audio_inputs, audio_length):

        #bert_attention_mask, audio_input, audio_length, ctc_labels, emotion_labels, text_output, augmentation = False
        # emotion_logits, logits, loss_cls, loss_ctc = self.model_mmi(text_output, attention_mask, audio_inputs, audio_length, ctc_labels, emotion_labels, augmentation = augmentation)
        emotion_logits, logits = self.model_mmi(text_output, attention_mask, audio_inputs, audio_length)
        # return emotion_logits, logits, loss_cls, loss_ctc
        return emotion_logits, logits

    # def forward(self, text_output, attention_mask, audio_inputs, audio_length, ctc_labels, emotion_labels, ctc_labels, emotion_labels):
    def forward(self, text_output, attention_mask, audio_inputs, audio_length):

        # emotion_logits, logits, loss_cls, loss_ctc = self.forward_encoder(text_output, attention_mask, audio_inputs, audio_length, ctc_labels, emotion_labels)
        emotion_logits, logits = self.forward_encoder(text_output, attention_mask, audio_inputs, audio_length)

        return emotion_logits


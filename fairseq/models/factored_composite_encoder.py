# Based on CompositeEncoder. Factored. Jordi Armengol Estapé.

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from . import FairseqEncoder
import torch


class FactoredCompositeEncoder(FairseqEncoder):
    """
    A wrapper around a dictionary of :class:`FairseqEncoder` objects.

    We run forward on each encoder and return their concatenation.

    Args:
        encoders (dict): a dictionary of :class:`FairseqEncoder` objects.
    """

    def __init__(self, encoders):
        super().__init__(None)
        self.encoders = encoders
        for key in self.encoders:
            self.add_module(key, self.encoders[key])

    def forward(self, src_tokens, src_lengths):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(#factors, batch, src_len)`
            src_lengths (LongTensor): lengths of each source sentence of shape
                `(batch)`

        Returns:
            dict:
                the outputs from each Encoder
        """
        concat_encoder = None
        for index, key in enumerate(self.encoders):
            encoder_out = self.encoders[key](src_tokens[index], src_lengths)
            if concat_encoder is None:
                concat_encoder = encoder_out#['encoder_out']
            else:
                concat = torch.cat((concat_encoder['encoder_out'], encoder_out['encoder_out']), 2)
                concat_encoder['encoder_out'] = concat
        return concat_encoder

    def reorder_encoder_out(self, encoder_out, new_order):
        """Reorder encoder output according to new_order."""
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out
        # changed for inference, not training (factored)

    def max_positions(self):
        return min([self.encoders[key].max_positions() for key in self.encoders])

    def upgrade_state_dict(self, state_dict):
        for key in self.encoders:
            self.encoders[key].upgrade_state_dict(state_dict)
        return state_dict

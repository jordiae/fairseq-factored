# Based on Fairseq's Transformer. By Jordi Armengol Estapé.
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import OrderedDict

from fairseq import utils
from fairseq.tasks.factored_translation import FactoredTranslationTask

from . import FairseqFactoredMultiModel, register_model, register_model_architecture

from .transformer import (
    base_architecture,
    Embedding,
    TransformerModel,
    TransformerEncoder,
    TransformerDecoder,
)

from copy import deepcopy

@register_model('factored_transformer')
class FactoredTransformerModel(FairseqFactoredMultiModel):
    """Train a factored Transformer model.

    Requires `--task factored_translation`.

    We inherit all arguments from TransformerModel and assume that all language
    pairs use a single Transformer architecture. In addition, we provide several
    options that are specific to the multilingual setting.

    Args: for factored, encoder never shared, decoder always shared
        --share-encoder-embeddings: share encoder embeddings across all source languages
        --share-decoder-embeddings: share decoder embeddings across all target languages
        --share-encoders: share all encoder params (incl. embeddings) across all source languages
        --share-decoders: share all decoder params (incl. embeddings) across all target languages
    """

    def __init__(self, encoders, decoders):
        super().__init__(encoders, decoders)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        pass
        TransformerModel.add_args(parser)
        parser.add_argument('--share-encoder-embeddings', action='store_true',
                            help='share encoder embeddings across languages')
        parser.add_argument('--share-decoder-embeddings', action='store_true',
                            help='share decoder embeddings across languages')
        parser.add_argument('--share-encoders', action='store_true',
                            help='share encoders across languages')
        parser.add_argument('--share-decoders', action='store_true',
                            help='share decoders across languages')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        assert isinstance(task, FactoredTranslationTask)

        # make sure all arguments are present in older models
        base_factored_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_langs = [lang_pair.split('-')[0] for lang_pair in args.lang_pairs]
        tgt_langs = [lang_pair.split('-')[1] for lang_pair in args.lang_pairs]

        if args.share_encoders:
            args.share_encoder_embeddings = True
        if args.share_decoders:
            args.share_decoder_embeddings = True

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        # build shared embeddings (if applicable)
        shared_encoder_embed_tokens, shared_decoder_embed_tokens = None, None
        if args.share_all_embeddings:
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            shared_encoder_embed_tokens = FairseqFactoredMultiModel.build_shared_embeddings(
                dicts=task.dicts,
                langs=task.langs,
                embed_dim=args.encoder_embed_dim,
                build_embedding=build_embedding,
                pretrained_embed_path=args.encoder_embed_path,
            )
            shared_decoder_embed_tokens = shared_encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            if args.share_encoder_embeddings:
                shared_encoder_embed_tokens = (
                    FairseqFactoredMultiModel.build_shared_embeddings(
                        dicts=task.dicts,
                        langs=src_langs,
                        embed_dim=args.encoder_embed_dim,
                        build_embedding=build_embedding,
                        pretrained_embed_path=args.encoder_embed_path,
                    )
                )
            if args.share_decoder_embeddings:
                shared_decoder_embed_tokens = (
                    FairseqFactoredMultiModel.build_shared_embeddings(
                        dicts=task.dicts,
                        langs=tgt_langs,
                        embed_dim=args.decoder_embed_dim,
                        build_embedding=build_embedding,
                        pretrained_embed_path=args.decoder_embed_path,
                    )
                )

        # encoders/decoders for each language
        lang_encoders, lang_decoders = {}, {}

        def get_encoder(lang):
            if lang not in lang_encoders:
                if shared_encoder_embed_tokens is not None:
                    encoder_embed_tokens = shared_encoder_embed_tokens
                else:
                    if lang == args.factor:
                        args2 = deepcopy(args)
                        args2.encoder_embed_dim = args.factor_encoder_embed_dim
                        args2.encoder_ffn_embed_dim = args.factor_encoder_embed_dim * 2
                        encoder_embed_tokens = build_embedding(
                            task.dicts[lang], args2.encoder_embed_dim, args.encoder_embed_path
                        )
                        lang_encoders[lang] = TransformerEncoder(args2, task.dicts[lang], encoder_embed_tokens)
                    else:
                        encoder_embed_tokens = build_embedding(
                            task.dicts[lang], args.encoder_embed_dim, args.encoder_embed_path
                        )
                        lang_encoders[lang] = TransformerEncoder(args, task.dicts[lang], encoder_embed_tokens)
            return lang_encoders[lang]

        def get_decoder(lang):
            if lang not in lang_decoders:
                if shared_decoder_embed_tokens is not None:
                    decoder_embed_tokens = shared_decoder_embed_tokens
                else:
                    decoder_embed_tokens = build_embedding(
                        task.dicts[lang], args.decoder_embed_dim, args.decoder_embed_path
                    )
                lang_decoders[lang] = TransformerDecoder(args, task.dicts[lang], decoder_embed_tokens)
            return lang_decoders[lang]

        # shared encoders/decoders (if applicable)
        shared_encoder, shared_decoder = None, None
        if args.share_encoders:
            shared_encoder = get_encoder(src_langs[0])
        if args.share_decoders:
            shared_decoder = get_decoder(tgt_langs[0])

        encoders, decoders = OrderedDict(), OrderedDict()
        for lang_pair, src, tgt in zip(args.lang_pairs, src_langs, tgt_langs):
            encoders[lang_pair] = shared_encoder if shared_encoder is not None else get_encoder(src)
            decoders[lang_pair] = shared_decoder if shared_decoder is not None else get_decoder(tgt)
        #return FactoredTransformerModel(encoders, decoders)
        return FactoredTransformerModel(encoders, shared_decoder)


@register_model_architecture('factored_transformer', 'factored_transformer')
def base_factored_architecture(args):
    base_architecture(args)
    '''
    args.share_encoder_embeddings = False
    args.share_decoder_embeddings = True
    args.share_encoders = False
    args.share_decoders = True
    '''
    args.share_encoder_embeddings = getattr(args, 'share_encoder_embeddings', False)
    args.share_decoder_embeddings = getattr(args, 'share_decoder_embeddings', True)
    args.share_encoders = getattr(args, 'share_encoders', False)
    args.share_decoders = getattr(args, 'share_decoders', True)


@register_model_architecture('factored_transformer', 'factored_transformer_iwslt_de_en')
def factored_transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.factor_encoder_embed_dim = 32
    args.factor = 'de_postags_at'
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 544)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1088)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    base_factored_architecture(args)

@register_model_architecture('factored_transformer', 'test_factored_transformer_iwslt_de_en')
def test_factored_transformer_iwslt_de_en(args):
    '''
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 64)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 128)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 1)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 64)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 128)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 1)
    args.decoder_layers = getattr(args, 'decoder_layers', 1)
    base_factored_architecture(args)
    '''
    '''
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 16)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 32)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 1)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 16)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 32)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 1)
    args.decoder_layers = getattr(args, 'decoder_layers', 1)
    base_factored_architecture(args)
    '''
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 32)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 64)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 1)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)

    args.factor_encoder_embed_dim = 32
    args.factor = 'de_postags_at'

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 64)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 12)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 1)
    args.decoder_layers = getattr(args, 'decoder_layers', 1)
    base_factored_architecture(args)

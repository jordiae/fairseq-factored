# Based on Fairseq's Multilingual translation task. By Jordi Armengol Estapé.

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import OrderedDict
import os

import torch
import numpy as np

from fairseq import options
from fairseq.data import (
    Dictionary,
    IndexedCachedDataset,
    IndexedDataset,
    IndexedRawTextDataset,
    #LanguagePairDataset,
    FactoredLanguagePairDataset,
    RoundRobinZipDatasets,
)
from fairseq.models import FairseqFactoredMultiModel, FairseqFactoredOneEncoderModel, FairseqFactoredMultiSumModel

from . import FairseqTask, register_task


@register_task('factored_translation')
class FactoredTranslationTask(FairseqTask):
    """A task for factored/multi-source translation.
    It can be used both with one or multiple encoders.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', metavar='DIR', help='path to data directory')
        parser.add_argument('--lang-pairs', default=None, metavar='PAIRS',
                            help='comma-separated list of language pairs (in training order): en-de,en-fr,de-fr')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language (only needed for inference)')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language (only needed for inference)')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left (default: True)')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left (default: False)')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')

        parser.add_argument('--factors-to-freeze', default=None, metavar='PAIRS',
                            help='Comma-separated list of the factors to be frozen (default: None): de_postags-en,...')
        #parser.add_argument('--reduced-factor-embed', default=None, type=str,
        #                    help='name of the factor the embeddings of which will be reduced. Default: undefined (no reduction). Notice that it only works with factored_transformer_iwslt_de_en architecture.')
        parser.add_argument('--freeze-factors-epoch', default=10, type=int, metavar='N',
                            help='Freeze training of factors starting at the required epoch (only for --factors-to-freeze). Default: 10.')
        parser.add_argument('--multiple-encoders', default='True', type=str, metavar='BOOL',
                            help='whether each factor has its own encoder (default: True).')
        '''
        parser.add_argument('--sum-instead-of-cat', default='False', type=str, metavar='BOOL',
                            help='whether factors should be added instead of concatenated (default: False).')
        '''
        # fmt: on

    def __init__(self, args, dicts, training):
        super().__init__(args)
        self.dicts = dicts
        self.langs = list(dicts.keys())
        self.training = training
        self.seed_for_factored = args.seed  # np.random.randint(2**32 - 1)#np.random.RandomState(seed=None)

    @classmethod
    def setup_task(cls, args, **kwargs):
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        '''
        if args.source_lang is not None or args.target_lang is not None:
            if args.lang_pairs is not None:
                raise ValueError(
                    '--source-lang/--target-lang implies generation, which is '
                    'incompatible with --lang-pairs'
                )
            training = False
            args.lang_pairs = ['{}-{}'.format(args.source_lang, args.target_lang)]
        '''
        if args.target_lang is not None:
            training = False
            args.lang_pairs = args.lang_pairs.split(',')
            args.source_lang, args.target_lang = args.lang_pairs[0].split('-')
        else:
            training = True
            args.lang_pairs = args.lang_pairs.split(',')
            args.source_lang, args.target_lang = args.lang_pairs[0].split('-')

        langs = list({x for lang_pair in args.lang_pairs for x in lang_pair.split('-')})

        # load dictionaries
        dicts = OrderedDict()
        for lang in langs:
            dicts[lang] = Dictionary.load(os.path.join(args.data, 'dict.{}.txt'.format(lang)))
            if len(dicts) > 0:
                assert dicts[lang].pad() == dicts[langs[0]].pad()
                assert dicts[lang].eos() == dicts[langs[0]].eos()
                assert dicts[lang].unk() == dicts[langs[0]].unk()
            print('| [{}] dictionary: {} types'.format(lang, len(dicts[lang])))

        return cls(args, dicts, training)

    def load_dataset(self, split, **kwargs):
        """Load a dataset split."""

        def split_exists(split, src, tgt, lang):
            filename = os.path.join(self.args.data, '{}.{}-{}.{}'.format(split, src, tgt, lang))
            if self.args.raw_text and IndexedRawTextDataset.exists(filename):
                return True
            elif not self.args.raw_text and IndexedDataset.exists(filename):
                return True
            return False

        def indexed_dataset(path, dictionary):
            if self.args.raw_text:
                return IndexedRawTextDataset(path, dictionary)
            elif IndexedDataset.exists(path):
                if self.args.lazy_load:
                    return IndexedDataset(path, fix_lua_indexing=True)
                else:
                    return IndexedCachedDataset(path, fix_lua_indexing=True)
            return None

        def sort_lang_pair(lang_pair):
            return '-'.join(sorted(lang_pair.split('-')))

        src_datasets, tgt_datasets = {}, {}
        for lang_pair in set(map(sort_lang_pair, self.args.lang_pairs)):
            src, tgt = lang_pair.split('-')
            if split_exists(split, src, tgt, src):
                prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, src, tgt))
            elif split_exists(split, tgt, src, src):
                prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, tgt, src))
            else:
                continue
            src_datasets[lang_pair] = indexed_dataset(prefix + src, self.dicts[src])
            tgt_datasets[lang_pair] = indexed_dataset(prefix + tgt, self.dicts[tgt])
            print('| {} {} {} examples'.format(self.args.data, split, len(src_datasets[lang_pair])))

        if len(src_datasets) == 0:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, self.args.data))

        def language_pair_dataset(lang_pair):
            src, tgt = lang_pair.split('-')
            if lang_pair in src_datasets:
                src_dataset, tgt_dataset = src_datasets[lang_pair], tgt_datasets[lang_pair]
            else:
                lang_pair = sort_lang_pair(lang_pair)
                tgt_dataset, src_dataset = src_datasets[lang_pair], tgt_datasets[lang_pair]
            return FactoredLanguagePairDataset(
                src_dataset, src_dataset.sizes, self.dicts[src],
                tgt_dataset, tgt_dataset.sizes, self.dicts[tgt],
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
                seed_for_factored = self.seed_for_factored
            )

        self.datasets[split] = RoundRobinZipDatasets(
            OrderedDict([
                (lang_pair, language_pair_dataset(lang_pair))
                for lang_pair in self.args.lang_pairs
            ]),
            eval_key=None if self.training else ['factored']#self.args.lang_pairs[0],
        )

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)
        if args.multiple_encoders == 'False' and not isinstance(model, FairseqFactoredOneEncoderModel):
            raise ValueError('FactoredTranslationTask with one encoder requires a '
                             'FairseqFactoredOneEncoderModel architecture')
        if not args.multiple_encoders == 'False' and (not isinstance(model, FairseqFactoredMultiModel) and not isinstance(model, FairseqFactoredMultiSumModel)):
            raise ValueError('FactoredTranslationTask with multiple encoders requires a '
                             'FairseqFactoredMultiModel or FairseqFactoredMultiSumModel architecture')
        return model

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        model.train()
        agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}
        mixed_sample = {}
        for lang_pair in self.args.lang_pairs:
            if sample[lang_pair] is None or len(sample[lang_pair]) == 0:
                continue
            if len(mixed_sample) == 0:
                mixed_sample = sample[lang_pair]
                src_tokens = mixed_sample['net_input']['src_tokens']
                ##### MASS
                print(src_tokens.shape)
                d1, d2 = src_tokens.shape
                for i in range(0, d1):
                    #print(src_tokens[i][int(np.round(np.random.uniform(0, len(src_tokens[i])-1)))],'->',self.dicts[lang_pair.split('-')[0]].unk())
                    #index_to_mask = int(np.round(np.random.uniform(0, len(src_tokens[i])-1)))
                    index_to_mask = torch.LongTensor(1,device='cuda').random_(0, len(src_tokens[i]))[0]
                    print(src_tokens[i][index_to_mask])
                    src_tokens[i][index_to_mask] = 3#torch.tensor(self.dicts[lang_pair.split('-')[0]].unk())
                    print(src_tokens[i][index_to_mask])
                #if torch.cuda.is_available(): src_tokens.cuda()

                #print(self.dicts[lang_pair.split('-')[0]].unk())
                print(src_tokens.shape)
                print()
                #exit()
                #####
                mixed_sample['net_input']['src_tokens'] = torch.unsqueeze(src_tokens, 0) #torch.tensor(src_tokens)#.clone().detach()
            else:
                mixed_sample['net_input']['src_tokens'] = torch.cat((mixed_sample['net_input']['src_tokens'], torch.unsqueeze(sample[lang_pair]['net_input']['src_tokens'], 0)))
        #mixed_sample['net_input']['src_tokens'][0]
        #int(np.round(np.random.uniform(0, l - 1)))
        loss, sample_size, logging_output = criterion(model, mixed_sample)
        #print(sample_size)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        agg_loss += loss.detach().item()
        # TODO make summing of the sample sizes configurable
        agg_sample_size += sample_size
        agg_logging_output = logging_output
        return agg_loss, agg_sample_size, agg_logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}
            mixed_sample = {}
            for lang_pair in self.args.lang_pairs:
                if sample[lang_pair] is None or len(sample[lang_pair]) == 0:
                    continue
                if len(mixed_sample) == 0:
                    mixed_sample = sample[lang_pair]
                    src_tokens = mixed_sample['net_input']['src_tokens']
                    mixed_sample['net_input']['src_tokens'] = torch.unsqueeze(src_tokens, 0)  # torch.tensor(src_tokens)#.clone().detach()
                else:
                    mixed_sample['net_input']['src_tokens'] = torch.cat((mixed_sample['net_input']['src_tokens'], torch.unsqueeze(sample[lang_pair]['net_input']['src_tokens'],0)))
        loss, sample_size, logging_output = criterion(model, mixed_sample)
        agg_loss += loss.data.item()
        # TODO make summing of the sample sizes configurable
        agg_sample_size += sample_size
        agg_logging_output = logging_output
        return agg_loss, agg_sample_size, agg_logging_output

    def init_logging_output(self, sample):
        return {
            'ntokens': sum(
                sample_lang.get('ntokens', 0)
                for sample_lang in sample.values()
            ) if sample is not None else 0,
            'nsentences': sum(
                sample_lang['target'].size(0) if 'target' in sample_lang else 0
                for sample_lang in sample.values()
            ) if sample is not None else 0,
        }

    def grad_denom(self, sample_sizes, criterion):
        return criterion.__class__.grad_denom(sample_sizes)

    '''
    def aggregate_logging_outputs(self, logging_outputs, criterion):
        # aggregate logging outputs for each language pair
        agg_logging_outputs = {
            lang_pair: criterion.__class__.aggregate_logging_outputs([
                logging_output.get(lang_pair, {}) for logging_output in logging_outputs
            ])
            for lang_pair in self.args.lang_pairs
        }

        def sum_over_languages(key):
            return sum(logging_output[key] for logging_output in agg_logging_outputs.values())

        # flatten logging outputs
        flat_logging_output = {
            '{}:{}'.format(lang_pair, k): v
            for lang_pair, agg_logging_output in agg_logging_outputs.items()
            for k, v in agg_logging_output.items()
        }
        flat_logging_output['loss'] = sum_over_languages('loss')
        flat_logging_output['nll_loss'] = sum_over_languages('nll_loss')
        flat_logging_output['sample_size'] = sum_over_languages('sample_size')
        flat_logging_output['nsentences'] = sum_over_languages('nsentences')
        flat_logging_output['ntokens'] = sum_over_languages('ntokens')
        return flat_logging_output
    '''

    @property
    def source_dictionary(self):
        return self.dicts[self.args.source_lang]

    @property
    def target_dictionary(self):
        return self.dicts[self.args.target_lang]

    def sort_lang_pair(self, lang_pair):
        return '-'.join(sorted(lang_pair.split('-')))
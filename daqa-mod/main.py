#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os

# import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets  # NOQA F401

from data import DAQA
from models import LSTMN, FCNLSTMN, CONVLSTMN, FCNLSTMNSA, CONVLSTMNSA
from film import FiLM
from malimo import MALiMo

# Training settings
parser = argparse.ArgumentParser()

# Input
parser.add_argument('--audio-training-set', type=str,
                    default='daqa_audio_train.h5',
                    help='Path to training data.')
parser.add_argument('--qa-training-set', type=str,
                    default='daqa_train_questions_answers.json',
                    help='Path to training data.')
parser.add_argument('--audio-test-set', type=str,
                    default='daqa_audio_val.h5',
                    help='Path to test data.')
parser.add_argument('--qa-test-set', type=str,
                    default='daqa_val_questions_answers.json',
                    help='Path to test data.')

# Settings
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='Random seed.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA.')
parser.add_argument('--multi-gpus', action='store_true', default=False,
                    help='Use all available GPUs.')
parser.add_argument('--distributed-parallel', action='store_true', default=False,
                    help='Distributed data parallel mode.')
parser.add_argument('--resume', action='store_true', default=False,
                    help='Resume training.')
parser.add_argument('--model', type=str, default='malimo',
                    help='Model to train.')

parser.add_argument('--embedding-dim', type=int, default=256,
                    help='Size of embedding layer.')
parser.add_argument('--lstm-hidden-dim-q', type=int, default=128,
                    help='Size of layer(s) in LSTM.')
parser.add_argument('--num-lstm-layers-q', type=int, default=1,
                    help='Number of layers in LSTM.')
parser.add_argument('--bidirectional', action='store_true', default=False,
                    help='Bidirectional LSTM.')
parser.add_argument('--num-conv-filts', type=int, default=16,
                    help='Number of filters in first layer in ConvNet.')
parser.add_argument('--num-conv-layers', type=int, default=5,
                    help='Number of layers in ConvNet.')
parser.add_argument('--stride', type=int, default=1,
                    help='Convolution stride.')
parser.add_argument('--dilation', type=int, default=1,
                    help='Convolution dilation.')
parser.add_argument('--fcn-output-dim', type=int, default=256,
                    help='Number of filters in final FCN layer.')
parser.add_argument('--fcn-coeff-dim', type=int, default=1,
                    help='Dimension along coefficients in adaptive pooling.')
parser.add_argument('--fcn-temp-dim', type=int, default=1,
                    help='Dimension along time in adaptive pooling.')
parser.add_argument('--aggregate', type=str, default='mean',
                    help='Function to aggregate over variable size input.')
parser.add_argument('--lstm-hidden-dim-a', type=int, default=128,
                    help='Size of layer(s) in LSTM.')
parser.add_argument('--num-lstm-layers-a', type=int, default=1,
                    help='Number of layers in LSTM.')
parser.add_argument('--stacked-att-dim', type=int, default=512,
                    help='Stacked attention layer dimension.')
parser.add_argument('--num-stacked-att', type=int, default=2,
                    help='Number of stacked attention layers.')
parser.add_argument('--use-coordinates', action='store_true', default=False,
                    help='Append coordinates to feature maps.')
parser.add_argument('--num-conv-filts-film', type=int, default=64,
                    help='Number of filters in first layer in film ConvNet.')
parser.add_argument('--num-conv-layers-film', type=int, default=2,
                    help='Number of layers in film ConvNet.')
parser.add_argument('--output-hidden-dim', type=int, default=1024,
                    help='Dimension of hidden layer before output layer.')

parser.add_argument('--optimizer', type=str, default='adam',
                    help='Optimzer.')
parser.add_argument('--lr', type=float, default=0.0001, metavar='L',
                    help='Learning rate.')
parser.add_argument('--l2', type=float, default=0.0001, metavar='M',
                    help='Weight decay.')
parser.add_argument('--dropout', type=float, default=0.0, metavar='R',
                    help='Dropout rate.')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='Batch size for training.')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='Batch size for testing.')
parser.add_argument('--epochs', type=int, default=10, metavar='T',
                    help='Number of epochs to train.')
parser.add_argument('--early-stopping', action='store_true', default=False,
                    help='Early stopping.')
parser.add_argument('--anneal-learning-rate', action='store_true', default=False,
                    help='Anneal Learning Rate.')
parser.add_argument('--patience', type=int, default=10, metavar='P',
                    help='Number of epochs before early stopping.')
# Output
parser.add_argument('--show-log', action='store_true', default=False,
                    help='Log training status.')
parser.add_argument('--log-interval', type=int, default=1000, metavar='I',
                    help='Number of batches to logging status.')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='Save current model.')
parser.add_argument('--model-dir', type=str, default='models',
                    help='Path to model.')
parser.add_argument('--model-name', type=str, default='model.pt',
                    help='Model name.')
parser.add_argument('--infer-only', action='store_true', default=False,
                    help='Run in test mode only.')


def build_model(args, vocab_dim, padding_idx, input_dim, output_dim):
    if args.model == 'lstmn':
        model = LSTMN(vocab_dim=vocab_dim,
                      embedding_dim=args.embedding_dim,
                      padding_idx=padding_idx,
                      lstm_hidden_dim_q=args.lstm_hidden_dim_q,
                      num_lstm_layers_q=args.num_lstm_layers_q,
                      bidirectional=args.bidirectional,
                      input_dim=input_dim,
                      lstm_hidden_dim_a=args.lstm_hidden_dim_a,
                      num_lstm_layers_a=args.num_lstm_layers_a,
                      output_hidden_dim=args.output_hidden_dim,
                      output_dim=output_dim)
    elif args.model == 'fcnlstmn':
        model = FCNLSTMN(vocab_dim=vocab_dim,
                         embedding_dim=args.embedding_dim,
                         padding_idx=padding_idx,
                         lstm_hidden_dim_q=args.lstm_hidden_dim_q,
                         num_lstm_layers_q=args.num_lstm_layers_q,
                         bidirectional=args.bidirectional,
                         num_conv_filts=args.num_conv_filts,
                         num_conv_layers=args.num_conv_layers,
                         stride=args.stride,
                         dilation=args.dilation,
                         fcn_output_dim=args.fcn_output_dim,
                         fcn_coeff_dim=args.fcn_coeff_dim,
                         fcn_temp_dim=args.fcn_temp_dim,
                         aggregate=args.aggregate,
                         output_hidden_dim=args.output_hidden_dim,
                         output_dim=output_dim)
    elif args.model == 'convlstmn':
        model = CONVLSTMN(vocab_dim=vocab_dim,
                          embedding_dim=args.embedding_dim,
                          padding_idx=padding_idx,
                          lstm_hidden_dim_q=args.lstm_hidden_dim_q,
                          num_lstm_layers_q=args.num_lstm_layers_q,
                          bidirectional=args.bidirectional,
                          input_dim=input_dim,
                          num_conv_filts=args.num_conv_filts,
                          num_conv_layers=args.num_conv_layers,
                          stride=args.stride,
                          dilation=args.dilation,
                          lstm_hidden_dim_a=args.lstm_hidden_dim_a,
                          num_lstm_layers_a=args.num_lstm_layers_a,
                          output_hidden_dim=args.output_hidden_dim,
                          output_dim=output_dim)
    elif args.model == 'fcnlstmnsa':
        model = FCNLSTMNSA(vocab_dim=vocab_dim,
                           embedding_dim=args.embedding_dim,
                           padding_idx=padding_idx,
                           lstm_hidden_dim_q=args.lstm_hidden_dim_q,
                           num_lstm_layers_q=args.num_lstm_layers_q,
                           bidirectional=args.bidirectional,
                           num_conv_filts=args.num_conv_filts,
                           num_conv_layers=args.num_conv_layers,
                           stride=args.stride,
                           dilation=args.dilation,
                           fcn_output_dim=args.fcn_output_dim,
                           fcn_coeff_dim=args.fcn_coeff_dim,
                           fcn_temp_dim=args.fcn_temp_dim,
                           aggregate=args.aggregate,
                           use_coordinates=args.use_coordinates,
                           stacked_att_dim=args.stacked_att_dim,
                           num_stacked_att=args.num_stacked_att,
                           output_hidden_dim=args.output_hidden_dim,
                           output_dim=output_dim)
    elif args.model == 'convlstmnsa':
        model = CONVLSTMNSA(vocab_dim=vocab_dim,
                            embedding_dim=args.embedding_dim,
                            padding_idx=padding_idx,
                            lstm_hidden_dim_q=args.lstm_hidden_dim_q,
                            num_lstm_layers_q=args.num_lstm_layers_q,
                            bidirectional=args.bidirectional,
                            input_dim=input_dim,
                            num_conv_filts=args.num_conv_filts,
                            num_conv_layers=args.num_conv_layers,
                            stride=args.stride,
                            dilation=args.dilation,
                            lstm_hidden_dim_a=args.lstm_hidden_dim_a,
                            num_lstm_layers_a=args.num_lstm_layers_a,
                            stacked_att_dim=args.stacked_att_dim,
                            num_stacked_att=args.num_stacked_att,
                            output_hidden_dim=args.output_hidden_dim,
                            output_dim=output_dim)
    elif args.model == 'film':
        model = FiLM(vocab_dim=vocab_dim,
                     embedding_dim=args.embedding_dim,
                     padding_idx=padding_idx,
                     lstm_hidden_dim_q=args.lstm_hidden_dim_q,
                     num_lstm_layers_q=args.num_lstm_layers_q,
                     bidirectional=args.bidirectional,
                     num_conv_filts_base=args.num_conv_filts,
                     num_conv_layers_base=args.num_conv_layers,
                     stride_base=args.stride,
                     dilation_base=args.dilation,
                     use_coordinates=args.use_coordinates,
                     num_conv_filts_film=args.num_conv_filts_film,
                     num_conv_layers_film=args.num_conv_layers_film,
                     stride_film=args.stride,
                     dilation_film=args.dilation,
                     fcn_output_dim=args.fcn_output_dim,
                     fcn_coeff_dim=args.fcn_coeff_dim,
                     fcn_temp_dim=args.fcn_temp_dim,
                     aggregate=args.aggregate,
                     output_hidden_dim=args.output_hidden_dim,
                     output_dim=output_dim)
    elif args.model == 'malimo':
            model = MALiMo(vocab_dim=vocab_dim,
                           embedding_dim=args.embedding_dim,
                           padding_idx=padding_idx,
                           lstm_hidden_dim_q=args.lstm_hidden_dim_q,
                           num_lstm_layers_q=args.num_lstm_layers_q,
                           bidirectional=args.bidirectional,
                           num_conv_filts_base=args.num_conv_filts,
                           num_conv_layers_base=args.num_conv_layers,
                           stride_base=args.stride,
                           dilation_base=args.dilation,
                           input_dim=input_dim,
                           a_aggregate=args.aggregate,
                           lstm_hidden_dim_a=args.lstm_hidden_dim_a,
                           num_lstm_layers_a=args.num_lstm_layers_a,
                           use_coordinates=args.use_coordinates,
                           num_conv_filts_film=args.num_conv_filts_film,
                           num_conv_layers_film=args.num_conv_layers_film,
                           stride_film=args.stride,
                           dilation_film=args.dilation,
                           fcn_output_dim=args.fcn_output_dim,
                           fcn_coeff_dim=args.fcn_coeff_dim,
                           fcn_temp_dim=args.fcn_temp_dim,
                           aggregate=args.aggregate,
                           output_hidden_dim=args.output_hidden_dim,
                           output_dim=output_dim)
    else:
        assert False, 'Unknown model.'
    return model


def save_state(args, epoch, model, optimizer, scheduler, train_loss, train_perf,
               test_loss, test_perf, best_perf, patience, early_stopping, best=False):
    checkpoint = os.path.join(args.model_dir, args.model_name)
    kwargs = {
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'train_perf': train_perf,
        'test_loss': test_loss,
        'test_perf': test_perf,
        'best_perf': best_perf,
        'patience': patience,
        'early_stopping': early_stopping,
    }
    if best:
        checkpoint += '.best'
        kwargs['model_state_dict'] = model.module.state_dict()  # unwrap model
        torch.save(kwargs, checkpoint)
    else:
        kwargs['model_state_dict'] = model.state_dict()
        torch.save(kwargs, checkpoint)


def load_state(args, model, optimizer, scheduler):
    checkpoint = torch.load(os.path.join(args.model_dir, args.model_name))
    model.load_state_dict(checkpoint['model_state_dict'])
    sepoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    train_loss = checkpoint['train_loss'] if 'train_loss' in checkpoint else 0
    train_perf = checkpoint['train_perf'] if 'train_perf' in checkpoint else 0
    test_loss = checkpoint['test_loss'] if 'test_loss' in checkpoint else 0
    test_perf = checkpoint['test_perf'] if 'test_perf' in checkpoint else 0
    best_perf = checkpoint['best_perf'] if 'best_perf' in checkpoint else 0
    patience = checkpoint['patience'] if 'patience' in checkpoint else 0
    if 'early_stopping' in checkpoint:
        early_stopping = checkpoint['early_stopping']
    else:
        early_stopping = False
    return sepoch, model, optimizer, scheduler, train_loss, train_perf, \
        test_loss, test_perf, best_perf, patience, early_stopping


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (a, len_a, q, len_q, target) in enumerate(train_loader):
        a = a.to(device)
        len_a = len_a.to(device)
        q = q.to(device)
        len_q = len_q.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(a, len_a, q, len_q)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if args.show_log and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss, correct, examples = 0., 0, 0
    with torch.no_grad():
        for a, len_a, q, len_q, target in test_loader:
            a = a.to(device)
            len_a = len_a.to(device)
            q = q.to(device)
            len_q = len_q.to(device)
            target = target.to(device)
            output = model(a, len_a, q, len_q)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            label = output.argmax(dim=1, keepdim=True)
            correct += label.eq(target.view_as(label)).sum().item()
            examples += len(a)
    test_loss /= examples
    perf = correct / examples
    # print('Average loss: {:.4f}, perf: {:.4f}%'.format(test_loss, 100. * perf))
    return test_loss, perf


def main(id, args):  # noqa C901

    # Infra
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    dist_parallel_mode = (use_cuda
                          and args.multi_gpus
                          and args.distributed_parallel
                          and torch.cuda.device_count() > 1)
    if dist_parallel_mode:
        dist.init_process_group(backend='nccl',
                                init_method='tcp://127.0.0.1:23456',
                                world_size=torch.cuda.device_count(),
                                rank=id)
        torch.cuda.set_device(id)
        device = torch.device('cuda:%d' % id)
    else:
        device = torch.device('cuda' if use_cuda else 'cpu')
    if id == 0 and args.save_model:
        if not os.path.isdir(args.model_dir):
            os.makedirs(args.model_dir)

    # Dataset
    train_set = DAQA(args.audio_training_set, args.qa_training_set)
    test_set = DAQA(args.audio_test_set,
                    args.qa_test_set,
                    train_set.stats,
                    train_set.word_to_ix,
                    train_set.answer_to_ix)
    if dist_parallel_mode:
        sampler_kwargs = {'num_replicas': torch.cuda.device_count(), 'rank': id}
        train_sampler = torch.utils.data.DistributedSampler(train_set, **sampler_kwargs)
        # test_sampler = torch.utils.data.DistributedSampler(test_set, **sampler_kwargs)
        # The above is commented out because we only evaluate on the main process
        # Note also that this means that evaluation using train_sampler will lead to
        # evaluation on a subset of the training set which is advantageous.
        test_sampler = torch.utils.data.RandomSampler(test_set)
        batch_size = int(args.batch_size / torch.cuda.device_count())
    else:
        train_sampler = torch.utils.data.RandomSampler(train_set)
        test_sampler = torch.utils.data.RandomSampler(test_set)
        batch_size = args.batch_size
    assert batch_size == 1, 'Batch size / number of GPUs != 1.'
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               collate_fn=DAQA.pad_collate_fn,
                                               **loader_kwargs)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              sampler=test_sampler,
                                              collate_fn=DAQA.pad_collate_fn,
                                              **loader_kwargs)

    # Model
    model = build_model(args,
                        vocab_dim=len(train_set.word_to_ix),
                        padding_idx=train_set.word_to_ix['<pad>'],
                        input_dim=train_set.stats['mean'].shape[0],
                        output_dim=len(train_set.answer_to_ix))
    model = model.to(device)

    # GPU / multi-GPU / distributed multi-GPU
    if dist_parallel_mode:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[id],
                                                          output_device=id,
                                                          check_reduction=True,
                                                          broadcast_buffers=False)
        if id == 0:
            print('DistributedDataParallel! Using', device)
    elif (use_cuda
          and args.multi_gpus
          and torch.cuda.device_count() > 1):
        model = nn.DataParallel(model)
        print('DataParallel! Using', torch.cuda.device_count(), 'GPUs!')
    else:
        print('Single CPU/GPU! Using', device)

    # Optimizer and scheduler
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr,
                               weight_decay=args.l2)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(),
                               lr=args.lr,
                               weight_decay=args.l2)
    else:
        assert False, 'Unknown optimizer.'
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 0.1,
                                                     int(args.patience / 2),
                                                     verbose=True)

    checkpoint_pt = os.path.join(args.model_dir, args.model_name)

    # Inference
    if args.infer_only:
        if id == 0:
            if os.path.isfile(checkpoint_pt):
                print('Testing: ' + checkpoint_pt)
                _, model, optimizer, scheduler, _, _, _, _, _, _, _ = \
                    load_state(args, model, optimizer, scheduler)
                print(' ')
                print('Hyperparamters')
                print(args)
                print(' ')
                print('Model')
                print(model)
                print(' ')
                print('Start testing.')
                test_loss, test_perf = test(args, model, device, test_loader)
                print(('Test loss: {:.3f}, Test Perf: {:.3f}%.').format(
                      test_loss,
                      100. * test_perf))
            else:
                print('Could not find model to test.')
        return  # inference done, nothing else to do here.

    # Initialize or load from exisiting checkpoint
    if (args.resume and os.path.isfile(checkpoint_pt)):
        if id == 0:
            print('Continue training from: ' + checkpoint_pt)
        sepoch, model, optimizer, scheduler, train_loss, train_perf, \
            test_loss, test_perf, best_perf, patience, early_stopping = \
            load_state(args, model, optimizer, scheduler)
    else:
        sepoch = 0
        best_perf, patience = 0., 0
        early_stopping = False
        if id == 0:  # evaluate only on main process
            print(' ')
            print('Hyperparamters')
            print(args)
            print(' ')
            print('Model')
            print(model)
            print(' ')
            print('Start training.')
            train_loss, train_perf = test(args, model, device, train_loader)
            test_loss, test_perf = test(args, model, device, test_loader)
            print(('Epoch {:03d}. Train loss: {:.3f}, Train Perf: {:.3f}%'
                   + '. Test loss: {:.3f}, Test Perf: {:.3f}%.').format(sepoch,
                    train_loss,
                    100. * train_perf,
                    test_loss,
                    100. * test_perf))
        else:  # Other processes don't need this
            train_loss, train_perf, test_loss, test_perf = 0, 0, 0, 0

    # Force other processes to wait
    if dist_parallel_mode or dist.is_initialized():
        dist.barrier()

    # Training loop
    for epoch in range(sepoch + 1, args.epochs + 1):

        # Load latest checkpoint to synchronize optimizer, early stopping, etc.
        if dist_parallel_mode and epoch > sepoch + 1:
            if args.anneal_learning_rate or args.early_stopping:
                checkpoint = torch.load(checkpoint_pt)
            if args.anneal_learning_rate:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if args.early_stopping:
                early_stopping = checkpoint['early_stopping']
                if early_stopping:
                    print('Early Stopping! Id: ' + id)
                    break

        # DistributedSampler requires manually seting the epoch for randomization
        if dist_parallel_mode:
            train_loader.sampler.set_epoch(epoch)
            # test_loader is RandomSampler doesnt require this

        # Train
        train(args, model, device, train_loader, optimizer, epoch)

        # Force other processes to wait
        if dist_parallel_mode or dist.is_initialized():
            dist.barrier()

        # Eval
        if id == 0:  # evaluate only on main process
            train_loss, train_perf = test(args, model, device, train_loader)
            test_loss, test_perf = test(args, model, device, test_loader)
            print(('Epoch {:03d}. Train loss: {:.3f}, Train Perf: {:.3f}%'
                   + '. Test loss: {:.3f}, Test Perf: {:.3f}%.').format(epoch,
                    train_loss,
                    100. * train_perf,
                    test_loss,
                    100. * test_perf))

            if args.anneal_learning_rate:
                scheduler.step(test_perf)

            # Monitor best performance so far assuming higher better
            if test_perf > best_perf:
                best_perf, patience = test_perf, 0
                print('Best Model at Epoch ' + str(epoch))
                if args.save_model:
                    save_state(args, epoch, model, optimizer, scheduler,
                               train_loss, train_perf, test_loss, test_perf,
                               best_perf, patience, early_stopping, best=True)
            else:
                patience += 1

            if args.early_stopping and (patience >= args.patience):
                early_stopping = True

            if (args.save_model):
                save_state(args, epoch, model, optimizer, scheduler,
                           train_loss, train_perf, test_loss, test_perf,
                           best_perf, patience, early_stopping)

            # If there is only a single process then break now
            # If > a single process then all processes break start of next epoch
            if not dist_parallel_mode and early_stopping:
                print('Early Stopping!')
                break

        # Force other processes to wait
        if dist_parallel_mode or dist.is_initialized():
            dist.barrier()


def union(args):
    # Set seed
    # np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if (not args.no_cuda
          and torch.cuda.is_available()
          and args.multi_gpus
          and args.distributed_parallel
          and torch.cuda.device_count() > 1):

        assert args.batch_size == torch.cuda.device_count(), \
            'Batch size must equal to number of GPUs.'
        if not args.save_model:
            assert not args.anneal_learning_rate, \
                'Checkpoints are used to synchronize learning rate.'
            assert not args.early_stopping, \
                'Checkpoints are used to synchronize early stopping flag.'

        print('Distributed!')
        mp.spawn(main, nprocs=torch.cuda.device_count(), args=(args,), daemon=False)
    else:
        assert args.batch_size == 1, 'Illegal batch size > 1 for undistributed mode.'
        main(0, args)


if __name__ == '__main__':
    args = parser.parse_args()
    union(args)
    print('Success!')

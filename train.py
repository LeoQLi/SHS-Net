import os, sys
import argparse
import torch
import torch.utils.data
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

# torch.autograd.set_detect_anomaly(True)   # slower! to show more details about errors

from misc import *
from net.network import Network
from dataset import PointCloudDataset, PatchDataset, RandomPointcloudPatchSampler


def parse_arguments():
    parser = argparse.ArgumentParser()
    ## Training
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--lr_gamma', type=float, default=0.2)
    parser.add_argument('--lr_min', type=float, default=1e-6)
    parser.add_argument('--scheduler_epoch', type=int, nargs='+', default=[400,600,800])
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
    parser.add_argument('--log_root', type=str, default='./log')
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--nepoch', type=int, default=800)
    parser.add_argument('--interval', type=int, default=100)
    parser.add_argument('--max_grad_norm', type=float, default=float("inf"))
    ## Dataset and loader
    parser.add_argument('--dataset_root', type=str, default='')
    parser.add_argument('--data_set', type=str, default='PCPNet')
    parser.add_argument('--trainset_list', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--patch_size', type=int, default=0)
    parser.add_argument('--sample_size', type=int, default=0)
    parser.add_argument('--encode_knn', type=int, default=16)
    parser.add_argument('--patches_per_shape', type=int, default=1000,
                        help='The number of patches sampled from each shape in an epoch')
    args = parser.parse_args()
    return args


def get_data_loaders(args):
    def worker_init_fn(worker_id):
        random.seed(args.seed)
        np.random.seed(args.seed)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_dset = PointCloudDataset(
            root=args.dataset_root,
            mode='train',
            data_set=args.data_set,
            data_list=args.trainset_list,
        )
    train_set = PatchDataset(
            datasets=train_dset,
            patch_size=args.patch_size,
            sample_size=args.sample_size,
            seed=args.seed,
        )
    train_datasampler = RandomPointcloudPatchSampler(train_set, patches_per_shape=args.patches_per_shape, seed=args.seed)
    train_dataloader = torch.utils.data.DataLoader(
            train_set,
            sampler=train_datasampler,
            batch_size=args.batch_size,
            num_workers=int(args.num_workers),
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            generator=g,
        )

    return train_dataloader, train_datasampler


### Arguments
args = parse_arguments()
seed_all(args.seed)

assert args.gpu >= 0, "ERROR GPU ID!"
_device = torch.device('cuda:%d' % args.gpu)
PID = os.getpid()

### Model
print('Building model ...')
model = Network(num_pat=args.patch_size,
                num_sam=args.sample_size,
                encode_knn=args.encode_knn,
            ).to(_device)

### Datasets and loaders
print('Loading datasets ...')
train_dataloader, train_datasampler = get_data_loaders(args)
train_num_batch = len(train_dataloader)

### Optimizer and Scheduler
optimizer = optim.Adam(model.parameters(), lr=args.lr)

#### Logging
if args.logging:
    log_path, log_dir_name = get_new_log_dir(args.log_root, prefix='',
                                            postfix='_' + args.tag if args.tag is not None else '')
    sub_log_dir = os.path.join(log_path, 'log')
    os.makedirs(sub_log_dir)
    logger = get_logger(name='train(%d)(%s)' % (PID, log_dir_name), log_dir=sub_log_dir)
    ckpt_dir = os.path.join(log_path, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)
    git_commit(logger=logger, log_dir=sub_log_dir, git_name=log_dir_name)
else:
    logger = get_logger('train', None)

refine_epoch = -1
if args.resume != '':
    assert os.path.exists(args.resume), 'ERROR path: %s' % args.resume
    logger.info('Resume from: %s' % args.resume)
    load_model = ''

    if load_model == 'pretrained':
        ### only load common model parameters
        pretrained_dict = torch.load(args.resume, map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        ### filter out unnecessary keys
        load_dict = {k: v for k, v in pretrained_dict.items()  \
                        if (k in model_dict) and (v.size() == model_dict[k].size()) and (k.startswith('pointEncoder'))
                    }

        ### overwrite entries in the existing state dict
        model_dict.update(load_dict)
        model.load_state_dict(model_dict)

        # for param in model.pointEncoder.parameters():
        #     param.requires_grad = False
        # for k, v in model.named_parameters():
        #     if k in load_dict:# and not k.startswith('conv_n') and not k.startswith('mlp_n'):
        #         v.requires_grad = False
        #         logger.info(k)
        # logger.info('\n')
        # for k, v in model.named_parameters():
        #     logger.info('%s: %s' % (k, v.requires_grad))
        # logger.info('\n')

        for k, v in load_dict.items():
            logger.info(k)
        logger.info('Number of loaded model dict from pretrained model: %d\n' % len(load_dict))
    else:
        model.load_state_dict(torch.load(args.resume))
        # refine_epoch = ckpt['others']['epoch']
        _, it = os.path.split(args.resume)[1].split('_')
        refine_epoch = int(it.split('.')[0])
    logger.info('Load pretrained mode: %s' % args.resume)


if args.logging:
    code_dir = os.path.join(log_path, 'code')
    os.makedirs(code_dir, exist_ok=True)
    os.system('cp %s %s' % ('*.py', code_dir))
    os.system('cp -r %s %s' % ('net', code_dir))


### Arguments
logger.info('Command: {}'.format(' '.join(sys.argv)))
arg_str = '\n'.join(['    {}: {}'.format(op, getattr(args, op)) for op in vars(args)])
logger.info('Arguments:\n' + arg_str)
logger.info(repr(model))
logger.info('training set: %d patches (in %d batches)' % (len(train_datasampler), len(train_dataloader)))


def train(epoch):

    for train_batchind, batch in enumerate(train_dataloader, 0):
        pcl_pat = batch['pcl_pat'].to(_device)
        normal_pat = batch['normal_pat'].to(_device)
        normal_center = batch['normal_center'].to(_device).squeeze()  # (B, 3)
        pcl_sample = batch['pcl_sample'].to(_device) if 'pcl_sample' in batch else None

        ### Reset grad and model state
        model.train()
        optimizer.zero_grad()

        ### Forward
        pred_point, weights, pred_neighbor = model(pcl_pat, pcl_sample=pcl_sample)
        loss, loss_tuple = model.get_loss(q_target=normal_center, q_pred=pred_point,
                                            ne_target=normal_pat, ne_pred=pred_neighbor,
                                            pred_weights=weights, pcl_in=pcl_pat,
                                        )

        ### Backward and optimize
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        ### Logging
        s = ''
        for l in loss_tuple:
            s += '%.5f+' % l.item()
        logger.info('[Train] [%03d: %03d/%03d] | Loss: %.6f(%s) | Grad: %.6f' % (
                    epoch, train_batchind, train_num_batch-1, loss.item(), s[:-1], orig_grad_norm)
                )

    return


def scheduler_fun():
    pre_lr = optimizer.param_groups[0]['lr']
    current_lr = pre_lr * args.lr_gamma
    if current_lr < args.lr_min:
        current_lr = args.lr_min
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    logger.info('Update learning rate: %f => %f \n' % (pre_lr, current_lr))


if __name__ == '__main__':
    logger.info('Start training ...')
    try:
        for epoch in range(1, args.nepoch+1):
            logger.info('### Epoch %d ###' % epoch)
            if epoch <= refine_epoch:
                if epoch in args.scheduler_epoch:
                    scheduler_fun()
                continue

            start_time = time.time()
            train(epoch)
            end_time = time.time()
            logger.info('Time cost: %.1f s \n' % (end_time-start_time))

            if epoch in args.scheduler_epoch:
                scheduler_fun()

            if epoch % args.interval == 0 or epoch == args.nepoch-1:
                if args.logging:
                    model_filename = os.path.join(ckpt_dir, 'ckpt_%d.pt' % epoch)
                    torch.save(model.state_dict(), model_filename)

    except KeyboardInterrupt:
        logger.info('Terminating ...')

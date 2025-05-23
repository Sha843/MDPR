import torch
import argparse
import os
import sys
import warnings
import json
import time
import numpy as np
from torch.amp import GradScaler, autocast
from torchvision import datasets, transforms
from mdpr_clip import MDPRPluginMaPLe, MDPRLoss, MDPRTrainer, load_semantic_prompts_from_json, load_matrix_pretrain
from mdpr_clip.utils import AverageMeter, accuracy, write_log
from clip.clip_maple import load as maple_clip_load
from imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
import ImageNet_LT
import Places_LT


torch.autograd.set_detect_anomaly(True)


base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added project root to sys.path: {project_root}")

parser = argparse.ArgumentParser(description='MDPR Plugin with MaPLe')
parser.add_argument('--dataset', default='cifar100', type=str, help='dataset (cifar10, cifar100, imagenet_lt, places_lt)')
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_factor', type=float, default=0.01)
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--test-batch-size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (reduced for stability)')
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--semantic_prompts_path', type=str, default=os.path.join(project_root, '', '.json'))
parser.add_argument('--matrix_prior_path', type=str, default=os.path.join(project_root, '', '.pt'))
parser.add_argument('--encoded_knowledge_path', type=str, default=os.path.join(project_root, '', '.pt'))
parser.add_argument('--num_classes', type=int, default=100)
parser.add_argument('--coop_n_ctx', type=int, default=16)
parser.add_argument('--maple_prompt_depth', type=int, default=9)
parser.add_argument('--loss_base_cls_weight', type=float, default=1.0)
parser.add_argument('--loss_sem_cls_weight', type=float, default=0.1)
parser.add_argument('--loss_pa_weight', type=float, default=0.0)
parser.add_argument('--loss_ka_weight', type=float, default=0.0)
parser.add_argument('--ka_temperature', type=float, default=0.0)
parser.add_argument('--attn_num_heads', type=int, default=8)
parser.add_argument('--attn_dropout', type=float, default=0.2)
parser.add_argument('--ka_projection_dim', type=int, default=128)
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
parser.add_argument('--save_name', default='best', type=str, help='name for saved checkpoint')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
args = parser.parse_args()
def save_path(args):
    save_dir = os.path.join(project_root, 'checkpoints', args.dataset, 'test0.0')
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device
print(f"Using device: {device}")

def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    checkpoint_path = os.path.join(save_path(args), filename)
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(save_path(args), f'{args.save_name}_best.pth.tar')
        torch.save(state, best_path)

data_root = os.path.join(project_root, 'data')
transform_train = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_val = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

print(f"Loading dataset: {args.dataset} from {data_root}")
if args.dataset == 'cifar10':
    args.num_classes = 10
    train_dataset = IMBALANCECIFAR10(root=data_root, imb_type=args.imb_type, imb_factor=args.imb_factor,
                                     rand_number=args.rand_number, train=True, download=True, transform=transform_train)
    val_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_val)
elif args.dataset == 'cifar100':
    args.num_classes = 100
    train_dataset = IMBALANCECIFAR100(root=data_root, imb_type=args.imb_type, imb_factor=args.imb_factor,
                                      rand_number=args.rand_number, train=True, download=True, transform=transform_train)
    val_dataset = datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform_val)
elif args.dataset == 'imagenet_lt':
    args.num_classes = 1000
    import ImageNet_LT

    train_dataset, _, val_dataset = ImageNet_LT.get_imagenet_lt(args, root_image='/root/autodl-tmp/data/imagenet/',
                                                            path_imbalance='../ImageNet_LT/subsets/')
elif args.dataset == 'places_lt':
    args.num_classes = 365
    if Places_LT is None: raise ImportError("Places_LT module not available.")
    places_root_image_path = os.path.join(project_root, 'places365_standard')
    places_path_imbalance = os.path.join(project_root, 'Places_LT_files', 'places365_challenge_lt')
    train_dataset, _, val_dataset = Places_LT.get_places(args, root_image=places_root_image_path,
                                                         path_imbalance=places_path_imbalance)
else:
    warnings.warn(f'Dataset {args.dataset} is not supported.')
    sys.exit(1)

if hasattr(train_dataset, 'get_cls_num_list'):
    args.img_num_list = train_dataset.get_cls_num_list()


kwargs_dataloader = {'num_workers': 4, 'pin_memory': True if device.type == 'cuda' else False}
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs_dataloader)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs_dataloader)
print("DataLoaders created.")

classnames_path = os.path.join(project_root, 'clip', 'data', f'prompt_{args.dataset}.txt')
if not os.path.exists(classnames_path):
    print(f"Error: Classnames file not found: {classnames_path}")
    sys.exit(1)
with open(classnames_path, 'r') as f:
    classnames = eval(f.read())
    if not isinstance(classnames, (list, tuple)) or len(classnames) != args.num_classes:
        print(f"Error: Invalid classnames format or count in {classnames_path}")
        sys.exit(1)
print(f"Loaded {len(classnames)} class names.")

print(f"Loading semantic prompts from {args.semantic_prompts_path}")
semantic_prompts_data = load_semantic_prompts_from_json(args.semantic_prompts_path, classnames)
if semantic_prompts_data is None:
    print(f"Error: Failed to load semantic prompts from {args.semantic_prompts_path}")
    sys.exit(1)
print("Semantic prompts data loaded.")

print(f"Loading matrix prior data from {args.matrix_prior_path}")
matrix_prior_data = load_matrix_pretrain(args.matrix_prior_path)
if matrix_prior_data is None:
    print(f"Error: Failed to load matrix prior from {args.matrix_prior_path}")
    sys.exit(1)
print("Matrix prior data loaded.")


design_details = {
    "trainer": 'MaPLe',
    "vision_depth": 0,
    "language_depth": args.maple_prompt_depth,
    "vision_ctx": args.coop_n_ctx,
    "language_ctx": args.coop_n_ctx,
    "maple_length": args.coop_n_ctx
}
print(f"Loading CLIP model ('ViT-B/16') with MaPLe design details: {design_details}")
clip_model, preprocess = maple_clip_load("ViT-B/16", device=device, design_details=design_details)
print("CLIP model loaded and moved to device.")

print("Initializing MDPRPluginMaPLe...")
mdpr_clip_model = MDPRPluginMaPLe(args, classnames, clip_model, matrix_prior_data)
mdpr_clip_model.to(device)
print("MDPRPluginMaPLe initialized.")


print("Initializing MDPRLoss...")
mdpr_loss = MDPRLoss(args, args.img_num_list)
print("MDPRLoss initialized.")


print("Setting up optimizer and scheduler...")
trainable_params = filter(lambda p: p.requires_grad, mdpr_clip_model.parameters())
optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)
print("Optimizer and scheduler ready.")


print("Initializing MDPRTrainer...")
trainer = MDPRTrainer(mdpr_clip_model, mdpr_loss, optimizer, scheduler, args, device)
print("MDPRTrainer ready.")

def main():
    global args
    best_prec1 = 0
    best_text = ''
    log_file_path = os.path.join(save_path(args), 'mdpr_clip_maple_log.txt')
    log_best_path = os.path.join(save_path(args), 'mdpr_clip_maple_best.txt')
    log_file = open(log_file_path, 'a')
    log_best = open(log_best_path, 'a')

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_acc1']
            trainer.model.load_state_dict(checkpoint['state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    print(f"Starting training for {args.epochs} epochs. Logs will be written to {log_file_path}")
    for epoch in range(args.start_epoch, args.epochs):
        start_train = time.time()
        trainer.train_epoch(train_loader, epoch, log_file)
        end_train = time.time()

        start_test = time.time()
        test_acc, _, _, test_text = trainer.evaluate(val_loader, epoch, log_file,mode_name="VALIDATION")
        end_test = time.time()

        is_best = test_acc > best_prec1
        best_prec1 = max(test_acc, best_prec1)

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'state_dict': trainer.model.state_dict(),
            'best_acc1': best_prec1,
            'optimizer': trainer.optimizer.state_dict(),
        }, is_best)

        time_train = round(end_train - start_train, 2)
        time_test = round(end_test - start_test, 2)
        time_text = f'time of training: {time_train}, time of testing: {time_test}'
        write_log(log_file, time_text + "\n")

        if is_best:
            best_text = f'epoch:{epoch}\n{test_text}'
            write_log(log_best, best_text + "\n")

        scheduler.step()

    log_file.close()
    log_best.close()

if __name__ == '__main__':
    main()
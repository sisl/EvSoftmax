# Code adapted from: https://github.com/ritheshkumar95/pytorch-vqvae

import torch
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid

from modules import VectorQuantizedVAE, GatedPixelCNN
from datasets import MiniImagenet

from tensorboardX import SummaryWriter

from tqdm import tqdm

import csv

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def test(data_loader, model, prior, args, writer):
    with torch.no_grad():
        i = 0
        for images, labels in data_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)

            latents = model.encode(images)
            gen_image = model.decode(latents)

            save_image(images[0].cpu(), './VQVAE/models/{0}/images_test_gt/'.format(args.output_folder) + str(i) + ".png")
            save_image(gen_image[0].cpu(), './VQVAE/models/{0}/images_test_vqvae/'.format(args.output_folder) + str(i) + ".png")
            i += 1

    return loss.item()

def main(args):

    if args.dataset == 'tinyimagenet':
        readable_labels = {38: 'organ', 42: 'prayer_rug', 31: 'file', 61: 'cliff', 58: 'consomme', 59: 'hotdog',
                        21: 'aircraft_carrier', 14: 'French_bulldog', 28: 'cocktail_shaker', 63: 'ear', 
                        3: 'green_mamba', 4: 'harvestman', 17: 'Arctic_fox', 32: 'fire_screen', 11: 'komondor',
                        43: 'reel', 18: 'ladybug', 45: 'snorkel', 24: 'beer_bottle', 36: 'lipstick', 5: 'toucan',
                        0: 'house_finch', 16: 'miniature_poodle', 50: 'tile_roof', 15: 'Newfoundland', 
                        46: 'solar_dish', 10: 'Gordon_setter', 7: 'dugong', 52: 'unicycle', 20: 'rock_beauty',
                        48: 'stage', 22: 'ashcan', 34: 'hair_slide', 30: 'dome', 13: 'Tibetan_mastiff',
                        53: 'upright', 62: 'bolete', 2: 'triceratops', 40: 'pencil_box', 26: 'chime',
                        47: 'spider_web', 51: 'tobacco_shop', 60: 'orange', 49: 'tank', 8: 'Walker_hound',
                        23: 'barrel', 6: 'jellyfish', 33: 'frying_pan', 9: 'Saluki', 37: 'oboe', 1: 'robin',
                        19: 'three-toed_sloth', 39: 'parallel_bars', 55: 'worm_fence', 27: 'clog', 41: 'photocopier',
                        25: 'carousel', 29: 'dishrag', 57: 'street_sign', 35: 'holster', 12: 'boxer', 56: 'yawl',
                        54: 'wok', 44: 'slot'}

    elif args.dataset == 'cifar10':
        readable_labels = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

    writer = SummaryWriter('./VQVAE/logs/{0}'.format(args.output_folder))
    save_filename = './VQVAE/models/{0}/prior.pt'.format(args.output_folder)

    if args.dataset in ['mnist', 'fashion-mnist', 'cifar10', 'tinyimagenet']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if args.dataset == 'mnist':
            # Define the train & test datasets
            train_dataset = datasets.MNIST(args.data_folder, train=True,
                download=True, transform=transform)
            test_dataset = datasets.MNIST(args.data_folder, train=False,
                transform=transform)
            num_channels = 1
        elif args.dataset == 'fashion-mnist':
            # Define the train & test datasets
            train_dataset = datasets.FashionMNIST(args.data_folder,
                train=True, download=True, transform=transform)
            test_dataset = datasets.FashionMNIST(args.data_folder,
                train=False, transform=transform)
            num_channels = 1
        elif args.dataset == 'cifar10':
            # Define the train & test datasets
            train_dataset = datasets.CIFAR10(args.data_folder,
                train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10(args.data_folder,
                train=False, transform=transform)
            num_channels = 3
        elif args.dataset == 'tinyimagenet':
            # Define the train & test datasets
            train_dataset = datasets.ImageFolder(
                os.path.join(args.data_folder, "train"),
                transform=transform)
            test_dataset = datasets.ImageFolder(
                os.path.join(args.data_folder, "test"),
                transform=transform)
            num_channels = 3
        valid_dataset = test_dataset
    elif args.dataset == 'miniimagenet':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # Define the train, valid & test datasets
        train_dataset = MiniImagenet(args.data_folder, train=True,
            download=True, transform=transform)
        valid_dataset = MiniImagenet(args.data_folder, valid=True,
            download=True, transform=transform)
        test_dataset = MiniImagenet(args.data_folder, test=True,
            download=True, transform=transform)
        num_channels = 3

    # Define the data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
        batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=1, shuffle=True)

    # Fixed images for Tensorboard
    fixed_images, _ = next(iter(test_loader))
    fixed_grid = make_grid(fixed_images, nrow=8, range=(-1, 1), normalize=True)
    writer.add_image('original', fixed_grid, 0)

    model = VectorQuantizedVAE(num_channels, args.hidden_size_vae, args.k).to(args.device)
    with open(args.model, 'rb') as f:
        state_dict = torch.load(f)
        model.load_state_dict(state_dict)
    model.eval()

    if args.dataset == 'tinyimagenet':
        print("number of training classes:", len(train_dataset.class_to_idx))
        n_classes = len(train_dataset.class_to_idx)
        shape = (16, 16)
        yrange = range(n_classes)
        csv_filename = 'tinyimagenet_generated.csv'
        sample_size = 25
    elif args.dataset == 'cifar10':
        print("number of training classes:", 10)
        n_classes = 10
        shape = (8, 8)
        yrange = range(n_classes)
        csv_filename = 'cifar10_generated.csv'
        sample_size = 1000

    prior = GatedPixelCNN(args.k, args.hidden_size_prior,
        args.num_layers, n_classes=n_classes).to(args.device)

    with open(args.prior, 'rb') as f:
        state_dict = torch.load(f)
        prior.load_state_dict(state_dict)
    prior.eval()

    # maximum number of kept dimensions
    max_num_dst = 0
    max_num_sparsemax = 0

    with torch.no_grad():

        f = open('./VQVAE/models/{0}/{1}'.format(args.output_folder, csv_filename), 'w')

        with f:

            writer = csv.writer(f)
            writer.writerow(['filename', 'label'])

            for y in tqdm(yrange):
                label = torch.tensor([y])
                label = label.to(args.device)
                sample_size = 1
                z = prior.generate(label=label, shape=shape, batch_size=sample_size, normalization=args.normalization)
                x = model.decode(z)
                import pdb
                pdb.set_trace()
                
                for im in range(x.shape[0]):
                    save_image(x.cpu()[im], './data/{0}/dataset/{1}_{2}.jpg'.format(args.output_folder, str(y).zfill(2), str(im).zfill(3)), range=(-1, 1), normalize=True)
                    if args.dataset == 'tinyimagenet':
                        y_str = list(train_dataset.class_to_idx.keys())[list(train_dataset.class_to_idx.values()).index(y)]
                        writer.writerow(['{0}_{1}.jpg'.format(str(y).zfill(2), str(im).zfill(3)), str(y_str)])
                    elif args.dataset == 'cifar10':
                        writer.writerow(['{0}_{1}.jpg'.format(str(y).zfill(2), str(im).zfill(3)), str(y)])

if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='PixelCNN Prior for VQ-VAE')

    # General
    parser.add_argument('--data-folder', type=str,
        help='name of the data folder')
    parser.add_argument('--dataset', type=str,
        help='name of the dataset (mnist, fashion-mnist, cifar10, tinyimagenet)')
    parser.add_argument('--model', type=str,
        help='filename containing the model')
    parser.add_argument('--prior', type=str,
        help='filename containing the model')

    # Latent space
    parser.add_argument('--hidden-size-vae', type=int, default=256,
        help='size of the latent vectors (default: 256)')
    parser.add_argument('--hidden-size-prior', type=int, default=64,
        help='hidden size for the PixelCNN prior (default: 64)')
    parser.add_argument('--k', type=int, default=512,
        help='number of latent vectors (default: 512)')
    parser.add_argument('--num-layers', type=int, default=15,
        help='number of layers for the PixelCNN prior (default: 15)')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=128,
        help='batch size (default: 128)')
    parser.add_argument('--num-epochs', type=int, default=100,
        help='number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=3e-4,
        help='learning rate for Adam optimizer (default: 3e-4)')

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='prior',
        help='name of the output folder (default: prior)')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
    parser.add_argument('--device', type=str, default='cuda',
        help='set the device (cpu or cuda, default: cpu)')
    parser.add_argument('--normalization', type=str, default='softmax', choices=['softmax', 'esoftmax', 'sparsemax', 'tsallis15'],
                        help='Normalzation function')
    parser.add_argument('--regularization', type=float, default=0, help='L2 regularization factor')

    args = parser.parse_args()

    # Create logs and models folder if they don't exist
    if not os.path.exists('./VQVAE/logs'):
        os.makedirs('./VQVAE/logs')
    if not os.path.exists('./VQVAE/models'):
        os.makedirs('./VQVAE/models')

    if not os.path.exists('./data'):
        os.makedirs('./data')
    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')
    # Slurm
    args.output_folder += f"_{args.normalization}_reg_{args.regularization}_lr_{args.lr}"
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])
    if not os.path.exists('./VQVAE/models/{0}'.format(args.output_folder)):
        os.makedirs('./VQVAE/models/{0}'.format(args.output_folder))
    if not os.path.exists('./data/{0}/dataset'.format(args.output_folder)):
        os.makedirs('./data/{0}/dataset'.format(args.output_folder))
    if not os.path.exists('./data/{0}/images_test_vqvae'.format(args.output_folder)):
        os.makedirs('./data/{0}/images_test_vqvae'.format(args.output_folder))
    if not os.path.exists('./data/{0}/images_test_gt'.format(args.output_folder)):
        os.makedirs('./data/{0}/images_test_gt'.format(args.output_folder))
    args.steps = 0

    main(args)

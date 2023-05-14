import os
import torch
import time
import torch.optim as optim
from absl import flags, app
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tqdm import trange
import source.models.legan as models
import source.losses as losses
from torchvision.datasets import ImageFolder
from tensorboardX import SummaryWriter
from source.utils import generate_imgs, infiniteloop, set_seed
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = torch.load('logs/AE/pathmnist.pth', map_location=device)
cf = torch.load('logs/KC/pathmnist/8.pkl', map_location=device)
net_G_models = {
    'cnn28': models.Generator28,
}

net_D_models = {
    'cnn28': models.Discriminator28,
}

loss_fns = {
    'bce': losses.BCEWithLogits,
    'hinge': losses.Hinge,
    'was': losses.Wasserstein,
    'softplus': losses.Softplus,
    'entorpy' : losses.Entorpy
}

FLAGS = flags.FLAGS
# model and training
flags.DEFINE_enum('arch', 'cnn28', net_G_models.keys(), "architecture")
flags.DEFINE_integer('total_steps', 10000, "total number of training steps")
flags.DEFINE_integer('batch_size', 64, "batch size")
flags.DEFINE_float('lr_G', 2e-4, "Generator learning rate")
flags.DEFINE_float('lr_D', 2e-4, "Discriminator learning rate")
flags.DEFINE_multi_float('betas', [0.0, 0.9], "for Adam")
flags.DEFINE_integer('n_dis', 5, "update Generator every this steps")
flags.DEFINE_integer('z_dim', 128, "latent space dimension")
flags.DEFINE_float('alpha', 10, "gradient penalty")
flags.DEFINE_enum('loss', 'was', loss_fns.keys(), "loss function")
flags.DEFINE_enum('loss_e', 'entorpy', loss_fns.keys(), "loss function")
flags.DEFINE_integer('seed', 0, "random seed")
# logging
flags.DEFINE_integer('eval_step', 5000, "evaluate FID and Inception Score")
flags.DEFINE_integer('sample_step', 200, "sample image every this steps")
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_string('logdir', './logs/pathmnist/LEGAN', 'log folder')
# flags.DEFINE_enum('kc', 'cifar10', kc_model.keys(), 'kc folder')
flags.DEFINE_bool('record', False, "record inception score and FID")
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', 'FID cache')
# generate
flags.DEFINE_bool('generate', True, 'generate images')
flags.DEFINE_string('pretrain', './logs/pathmnist/LEGAN/model.pt', 'path to test model')
flags.DEFINE_string('output', './outputs/pathmnist/LEGAN', 'path to output dir')
flags.DEFINE_integer('num_images', 1000, 'the number of generated images')

def generate():
    assert FLAGS.pretrain is not None, "set model weight by --pretrain [model]"

    net_G = net_G_models[FLAGS.arch](FLAGS.z_dim).to(device)
    net_G.load_state_dict(torch.load(FLAGS.pretrain, map_location=device)['net_G'])
    net_G.eval()

    counter = 0
    if os.path.exists(FLAGS.output):
        True
    else:
        os.makedirs(FLAGS.output)
    with torch.no_grad():
        for start in trange(
                0, FLAGS.num_images, FLAGS.batch_size, dynamic_ncols=True):
            batch_size = min(FLAGS.batch_size, FLAGS.num_images - start)
            z = torch.randn(batch_size, FLAGS.z_dim).to(device)
            x = net_G(z).cpu()
            x = (x + 1) / 2
            for image in x:
                save_image(
                    image, os.path.join(FLAGS.output, '%d.png' % counter))
                counter += 1

def cacl_gradient_penalty(net_D, real, fake):
    t = torch.rand(real.size(0), 1, 1, 1).to(real.device)
    t = t.expand(real.size())

    interpolates = t * real + (1 - t) * fake
    interpolates.requires_grad_(True)
    disc_interpolates = net_D(interpolates)
    grad = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True, retain_graph=True)[0]

    grad_norm = torch.norm(torch.flatten(grad, start_dim=1), dim=1)
    loss_gp = torch.mean((grad_norm - 1) ** 2)
    return loss_gp


def train():
    data_transform = transforms.Compose([
        # transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    dataset = ImageFolder("./data/pathmnist/0", transform=data_transform)
    # sampler = sampler_(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=True, drop_last=True)

    # net_G = net_G_models[FLAGS.arch](FLAGS.z_dim).to(device)
    # net_D = net_D_models[FLAGS.arch]().to(device)
    net_G = net_G_models[FLAGS.arch](FLAGS.z_dim).to(device)
    net_G.load_state_dict(torch.load(FLAGS.pretrain, map_location=device)['net_G'])
    net_D = net_D_models[FLAGS.arch]().to(device)
    net_D.load_state_dict(torch.load(FLAGS.pretrain, map_location=device)['net_D'])
    loss_fn = loss_fns[FLAGS.loss]()
    loss_en = loss_fns[FLAGS.loss_e]()

    optim_G = optim.Adam(net_G.parameters(), lr=FLAGS.lr_G, betas=FLAGS.betas)
    optim_D = optim.Adam(net_D.parameters(), lr=FLAGS.lr_D, betas=FLAGS.betas)
    sched_G = optim.lr_scheduler.LambdaLR(
        optim_G, lambda step: 1 - step / FLAGS.total_steps)
    sched_D = optim.lr_scheduler.LambdaLR(
        optim_D, lambda step: 1 - step / FLAGS.total_steps)

    if os.path.exists(os.path.join(FLAGS.logdir, 'sample')):
        True
    else:
        os.makedirs(os.path.join(FLAGS.logdir, 'sample'))
    writer = SummaryWriter(os.path.join(FLAGS.logdir))
    # sample_z = torch.randn(FLAGS.sample_size, FLAGS.z_dim).to(device)
    with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
        f.write(FLAGS.flags_into_string())
    writer.add_text(
        "flagfile", FLAGS.flags_into_string().replace('\n', '  \n'))

    real, _ = next(iter(dataloader))
    grid = (make_grid(real[:FLAGS.sample_size]) + 1) / 2
    writer.add_image('real_sample', grid)

    looper = infiniteloop(dataloader)
    with trange(1, FLAGS.total_steps + 1, desc='Training', ncols=0) as pbar:
        for step in pbar:
            # Discriminator
            for _ in range(FLAGS.n_dis):
                with torch.no_grad():
                    z = torch.randn(FLAGS.batch_size, FLAGS.z_dim).to(device)
                    fake = net_G(z).detach()
                real = next(looper).to(device)
                net_D_real = net_D(real)
                net_D_fake = net_D(fake)
                loss = loss_fn(net_D_real, net_D_fake)
                loss_gp = cacl_gradient_penalty(net_D, real, fake)
                loss_all = loss + FLAGS.alpha * loss_gp

                optim_D.zero_grad()
                loss_all.backward()
                optim_D.step()

                if FLAGS.loss == 'was':
                    loss = -loss
                pbar.set_postfix(loss='%.4f' % loss)
            writer.add_scalar('loss', loss, step)
            writer.add_scalar('loss_gp', loss_gp, step)

            # Generator
            for p in net_D.parameters():
                # reduce memory usage
                p.requires_grad_(False)
            z = torch.randn(FLAGS.batch_size, FLAGS.z_dim).to(device)
            gen_loss = loss_fn(net_D(net_G(z)))
            H_loss = loss_en(encoder, net_G, cf, z, FLAGS.batch_size)
            R_loss = torch.abs(net_G(z) - real).mean()
            loss = gen_loss - (H_loss - R_loss) * 0.1
            H.append(H_loss)

            optim_G.zero_grad()
            loss.backward()
            optim_G.step()
            for p in net_D.parameters():
                p.requires_grad_(True)

            sched_G.step()
            sched_D.step()

            if step == 1 or step % FLAGS.sample_step == 0:
                sample_z = torch.randn(FLAGS.sample_size, FLAGS.z_dim).to(device)
                fake = net_G(sample_z).cpu()
                fake = net_G(sample_z).cpu()
                grid = (make_grid(fake) + 1) / 2
                writer.add_image('sample', grid, step)
                save_image(grid, os.path.join(
                    FLAGS.logdir, 'sample', '%d.png' % step))

            if step == 1 or step % FLAGS.eval_step == 0 or step == FLAGS.total_steps:
                torch.save({
                    'net_G': net_G.state_dict(),
                    'net_D': net_D.state_dict(),
                    'optim_G': optim_G.state_dict(),
                    'optim_D': optim_D.state_dict(),
                    'sched_G': sched_G.state_dict(),
                    'sched_D': sched_D.state_dict(),
                }, os.path.join(FLAGS.logdir, 'model.pt'))
    writer.close()


def main(argv):
    set_seed(FLAGS.seed)
    if FLAGS.generate:
        generate()
    else:
        train()


if __name__ == '__main__':
    app.run(main)
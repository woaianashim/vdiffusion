import os 
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from tqdm import tqdm
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from generate import generate


@hydra.main(config_path="config", config_name="config_vae", version_base=None)
def train(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    writer = SummaryWriter()

    to_pil_image = transforms.ToPILImage()
    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                                transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
    ds = hydra.utils.instantiate(cfg.dataset, path=hydra.utils.get_original_cwd())
    model = hydra.utils.instantiate(cfg.model).cuda()

    # dummy = torch.tensor(ds[0][0])
    print(model)

    if cfg.resume:
        model.load_state_dict(torch.load(cfg.resume))

    pbar = tqdm(range(cfg.epochs))
    for i in pbar:
        model.train()
        dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

        running_loss = 0
        for ind, data in enumerate(dl):
            image = data[0].cuda()
            rec, feats, q_loss = model(image)
            torch.save(feats.squeeze(0), os.path.join(hydra.utils.get_original_cwd(),f"converted/{ind}.pt"))


        break
        pbar.set_postfix(loss=running_loss / len(dl))
        writer.add_scalar("Loss/train", running_loss, i)


train()

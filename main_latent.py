import os 
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from tqdm import tqdm
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from generate import generate


@hydra.main(config_path="config", config_name="config_latent", version_base=None)
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
    vae_model = hydra.utils.instantiate(cfg.vae_model).cuda()

    # dummy = torch.tensor(ds[0][0])
    print(model)

    if cfg.resume:
        model.load_state_dict(torch.load(cfg.resume))
    if cfg.vae_resume:
        vae_model.load_state_dict(torch.load(cfg.vae_resume))
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    pbar = tqdm(range(cfg.epochs))
    for i in pbar:
        model.train()
        dl = torch.utils.data.DataLoader(ds, batch_size=12)

        running_loss = 0
        for data in dl:
            optim.zero_grad()
            image, noise, time = data[0].cuda(), data[1].cuda(), data[3].cuda() 
            rec, feats = model(image, time)
            loss = torch.nn.functional.mse_loss(noise, rec)
            loss.backward()
            running_loss += loss.item()
            optim.step()

        pbar.set_postfix(loss=running_loss / len(dl))
        writer.add_scalar("Loss/train", running_loss, i)

        if(i%10==0):
            if "outputs" not in os.listdir("."):
                os.mkdir("outputs")
            with torch.no_grad():
                for j in range(1, 11):
                    if j<10:
                        data = ds.__getitem__((i//10)%100, time=j*10)
                        x_time, noise, image, time = data[0].unsqueeze(0).cuda(), data[1].unsqueeze(0).cuda(), data[2].unsqueeze(0).cuda(), data[3].cuda()
                    else:
                        x_time = torch.randn_like(ds[0][0].unsqueeze(0)).cuda()
                        time = torch.tensor(99).cuda()
                    lat = generate(model, x_time, time)
                    print(lat.shape, x_time.shape, image.shape)
                    rec = vae_model.decoder(lat)
                    image_r = vae_model.decoder(image)
                    x_time = vae_model.decoder(x_time)
                    example = invTrans(torch.cat([image_r.squeeze(0), x_time.squeeze(0), rec.squeeze(0)], dim=1).cpu())
                    writer.add_image(f"Noisiness {time.cpu().item()}", example, global_step=i)
                to_pil_image(example).save(f"outputs/sample_{i}.jpg")
            torch.save(model.state_dict(), f"outputs/weights_{i}.pt")

train()

import os 
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from tqdm import tqdm
from torchvision import transforms
from generate import generate


@hydra.main(config_path="config", config_name="config", version_base=None)
def train(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    to_pil_image = transforms.ToPILImage()
    ds = hydra.utils.instantiate(cfg.dataset, path=hydra.utils.get_original_cwd())
    model = hydra.utils.instantiate(cfg.model).cuda()

    # dummy = torch.tensor(ds[0][0])
    print(model)

    if cfg.resume:
        model.load_state_dict(torch.load(cfg.resume))
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    pbar = tqdm(range(cfg.epochs))
    for i in pbar:
        model.train()
        dl = torch.utils.data.DataLoader(ds, batch_size=32)

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

        if(i%10==0):
            if "outputs" not in os.listdir("."):
                os.mkdir("outputs")
            with torch.no_grad():
                # x_time, noise, image, time = ds[0][0].unsqueeze(0).cuda(), ds[1][0].unsqueeze(0).cuda(), ds[2][0].unsqueeze(0).cuda(), ds[3][0].cuda()
                # rec = model(x_time, time)
                # example = torch.cat([image.squeeze(0), x_time.squeeze(0), x_time - rec])
                rec = generate(model)
                sample = rec.transpose(0,1).view(3,-1, 64).cpu()
                to_pil_image(sample).save(f"outputs/sample_{i}.jpg")
            torch.save(model.state_dict(), f"outputs/weights_{i}.pt")

train()

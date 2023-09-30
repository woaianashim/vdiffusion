import os 
import torch
import hydra
from tqdm import tqdm
from torchvision import transforms


@hydra.main(config_path="config", config_name="config", version_base=None)
def train(cfg):
    print(cfg)

    
    to_pil_image = transforms.ToPILImage()
    ds = hydra.utils.instantiate(cfg.dataset, path=hydra.utils.get_original_cwd())
    model = hydra.utils.instantiate(cfg.model).cuda()
    print(torch.tensor(ds[0][0]).shape)
    if cfg.resume:
        model.load_state_dict(torch.load(cfg.resume))
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for i in tqdm(range(cfg.epochs)):
        dl = torch.utils.data.DataLoader(ds, batch_size=32)
        for data in dl:
            optim.zero_grad()
            image = data[0].cuda()
            rec, feats = model(image, torch.zeros(image.shape[0]).cuda())
            loss = torch.nn.functional.mse_loss(image, rec)
            loss.backward()
            optim.step()
        if(i%10==0):
            if "outputs" not in os.listdir("."):
                os.mkdir("outputs")
            with torch.no_grad():
                image = ds[0][0].unsqueeze(0).cuda()
                rec=model(image,torch.zeros(image.shape[0]).cuda())[0].squeeze(0)
                sample = torch.cat([image.squeeze(0), rec], dim=2).cpu()
                to_pil_image(sample).save(f"outputs/sample_{i}.jpg")
            torch.save(model.state_dict(), f"outputs/weights_{i}.pt")




train()

import torch

def generate(model, n_samples=1, size=(3,64,64), noise_steps=100, beta_init=1e-4, beta_final=2e-2):
    sample = torch.randn(n_samples, *size).cuda()
    betas = torch.linspace(beta_init, beta_final, noise_steps).cuda()
    alphas = 1. - betas
    alphas_hat = alphas.cumprod(0)
    with torch.no_grad():
        model.eval()
        for i in reversed(range(noise_steps)):
            time = i*torch.ones(n_samples).cuda()
            epsilon, _ = model(sample, time)
            alpha = alphas[i]
            alpha_hat = alphas_hat[i]
            beta = betas[i]
            noise = beta.sqrt()*torch.randn_like(sample) if i>0 else 0.
            mu_theta = (sample - epsilon*(1.-alpha)/(1.-alpha_hat).sqrt())/alpha.sqrt()
            sample = mu_theta + noise
    return sample




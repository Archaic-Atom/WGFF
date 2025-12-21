import os.path

import numpy as np
import random
import torch
from PIL import Image
import einops
from pytorch_lightning import seed_everything
from utils import create_model, load_state_dict
from WGFF.sampler import WGFF_Sampler


class WGFFPipeline():
    def __init__(
            self,
            model_path,
            model_cfg,
            device='cuda',
            inversion_steps=999,
            sampling_steps=100,
    ):

        self.device = device
        self.inversion_steps = inversion_steps
        self.sampling_steps = sampling_steps

        # load model weight
        self.model = self.create_WGFF_model(model_cfg, model_path)
        self.sampler = self.create_sampler(self.model)

        self.un_cond = {"c_crossattn": [self.model.get_learned_conditioning([''])]}

    def create_WGFF_model(self, model_cfg, model_path):
        print(f"loading SD model from path: {model_path}")
        model = create_model(model_cfg).to(self.device)
        model.load_state_dict(load_state_dict(model_path, location=self.device), strict=False)
        return model

    def create_sampler(self, model):
        print(f"creating WGFF Sampler")
        return WGFF_Sampler(model, self.device)

    def get_latent(self, img):
        def get_image_tensor(img):
            img = (np.array(img).astype(np.float32) / 127.5) - 1.0
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)[None, ...].repeat(1, 1, 1, 1).to(self.device)
            return img_tensor

        print(f"creating latent")
        encoder_posterior = self.model.encode_first_stage(get_image_tensor(img))
        z = self.model.get_first_stage_encoding(encoder_posterior).detach()
        self.sampler.make_schedule(ddim_num_steps=self.inversion_steps, verbose=False)
        latent, _ = self.sampler.encode(x0=z, cond=self.un_cond, t_enc=self.inversion_steps)
        return latent

    def inference(self, prompt, img, weights, tau_f=0.5, unconditional_guidance_scale=7.5, seed=2025):
        with torch.no_grad():
            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            ref_latent = self.get_latent(img)

            self.sampler.make_schedule(ddim_num_steps=self.sampling_steps, verbose=False)

            print(f"Encoding prompt '{prompt}'")
            cond = {"c_crossattn": [self.model.get_learned_conditioning([prompt])]}
            # sampling
            output = self.sampler.sampling( ref_latent=ref_latent,
                                            weights=weights,
                                            cond=cond,
                                            t_dec=self.sampling_steps,
                                            unconditional_conditioning=self.un_cond,
                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                            end_step=tau_f * self.inversion_steps)


            x_samples = torch.clip(self.model.decode_first_stage(output), min=-1, max=1)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().astype(
                np.uint8)

            output_image=None
            for sample in x_samples:
                Image.fromarray(sample).save('output.png')
                output_image = Image.fromarray(sample)

        return output_image





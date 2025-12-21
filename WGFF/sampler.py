import torch,os
import numpy as np
from tqdm import tqdm
from WGFF.ddim_sampler import DDIM_Sampler
from utils import create_wavelet_filter, wavelet_transform, inverse_wavelet_transform

class WGFF_Sampler(DDIM_Sampler):

    def __init__(self, model, device='cuda', schedule="linear", **kwargs):
        super(WGFF_Sampler, self).__init__(model, schedule, **kwargs)
        self.wt_filter, self.iwt_filter = create_wavelet_filter('haar', 4, 4, device, type=torch.float32)

    @torch.no_grad()
    def sampling(self, ref_latent, weights, cond, t_dec, unconditional_conditioning, unconditional_guidance_scale=7.5,
                        use_original_steps=False, end_step=500):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_dec]
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps, colour="green")

        t2i_latent = torch.randn_like(ref_latent, device=ref_latent.device)

        ll, lh, hl, hh = weights['ll'], weights['lh'], weights['hl'], weights['hh']

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((ref_latent.shape[0], ), step, device=ref_latent.device, dtype=torch.long)
            if step > end_step:

                t2i_ll, t2i_lh, t2i_hl, t2i_hh = wavelet_transform(t2i_latent, self.wt_filter)
                ref_ll, ref_lh, ref_hl, ref_hh = wavelet_transform(ref_latent, self.wt_filter)

                t_ll = ref_ll * ll + t2i_ll * (1 - ll)
                t_lh = ref_lh * lh + t2i_lh * (1 - lh)
                t_hl = ref_hl * hl + t2i_hl * (1 - hl)
                t_hh = ref_hh * hh + t2i_hh * (1 - hh)

                t2i_latent = inverse_wavelet_transform(t_ll, t_lh, t_hl, t_hh, self.iwt_filter)

                ref_latent, _, _ = self.p_sample_ddim( ref_latent, unconditional_conditioning, ts, index=index,
                                                       use_original_steps=use_original_steps,
                                                       unconditional_guidance_scale=1.0,
                                                       unconditional_conditioning=None)

                t2i_latent, _, _ = self.p_sample_ddim( t2i_latent, cond, ts, index=index, use_original_steps=use_original_steps,
                                                       unconditional_guidance_scale=unconditional_guidance_scale,
                                                       unconditional_conditioning=unconditional_conditioning)

            else:
                t2i_latent, _, _ = self.p_sample_ddim(t2i_latent, cond, ts, index=index, use_original_steps=use_original_steps,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=unconditional_conditioning)

        return t2i_latent
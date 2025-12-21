import torch
from PIL import Image
from accelerate.utils import set_seed
from WGFF import WGFFPipeline

class Runner:
    def __init__(self):
        self.sd15 = None
        self.model_path = "models/v1-5-pruned-emaonly.ckpt"
        self.model_cfg = "models/model_ldm_v15.yaml"
        self.pipeline = self.load_pipeline()
    def load_pipeline(self):
        return WGFFPipeline(self.model_path, self.model_cfg)

    def preprocecss(self, image: Image.Image, height=None, width=None):
        image = image.resize((height, width), Image.Resampling.LANCZOS)
        return image

    def run_WGFF(self, content_image, prompt, ll_weight, lh_weight, hl_weight, hh_weight, tau_f, unconditional_guidancd_scale, seed):

        set_seed(seed)

        weights = {'ll': ll_weight, 'lh': lh_weight, 'hl': hl_weight, 'hh': hh_weight}

        output_image = self.pipeline.inference(
            prompt, content_image, weights, tau_f=tau_f, unconditional_guidance_scale=unconditional_guidancd_scale, seed=seed
        )

        # torch.cuda.empty_cache()

        return [output_image]
# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path


from functools import partial
# from pathlib import Path
import os
import sys
sys.path.append('/src/taming-transformers')
sys.path.append('/src/latent-diffusion')
sys.path.append('/src/v-diffusion-pytorch')
sys.path.append('/src/cloob-training')
sys.path.append('/src/CLIP')
from omegaconf import OmegaConf
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from typing import Optional
import train_latent_diffusion as train
from cloob_training import model_pt, pretrained
import ldm.models.autoencoder
from diffusion import sampling, utils


def resize_and_center_crop(image, size):
    fac = max(size[0] / image.size[0], size[1] / image.size[1])
    image = image.resize((int(fac * image.size[0]), int(fac * image.size[1])), Image.LANCZOS)
    return TF.center_crop(image, size[::-1])


class LatentDiffusionGenerator(BasePredictor):
    device = 'cuda:0'
    # This is just where I've put the model checkpoints and 
    diff_list = {
        'yfcc': {'channel_multipliers': [4, 4, 8, 8], 'base_channels': 192, 'model_fp': 'checkpoints/yfcc-latent-diffusion-f8-e2-s250k.ckpt'},
        'danbooru': {'channel_multipliers': [4, 4, 8, 8], 'base_channels': 128, 'model_fp': 'checkpoints/danbooru-latent-diffusion-e88.ckpt'}
    }
    ae_list = {
        # 2.55 is the hard-coded scale in `cfg_sample.py`. I think that it's supposed to correspond to this autoencoder, but I'm not sure
        # TODO: Check this
        'yfcc': {'scale': 2.55, 'config_fp': 'configs/autoencoder_kl_32x32x4.yaml', 'model_fp': 'checkpoints/kl-f8.ckpt'},
        'laion5b': {'scale': 8.0779, 'config_fp': 'configs/laion-5B-kl-f8.yaml', 'model_fp': 'checkpoints/laion-5B-kl-f8.ckpt'},
        'danbooru': {'scale': 9.3154, 'config_fp': 'configs/danbooru-kl-f8.yaml', 'model_fp': 'checkpoints/danbooru-kl-f8.ckpt'}
    }
    def setup(self):
        cloob_config_fp = 'cloob-training/cloob_training/pretrained_configs/cloob_laion_400m_vit_b_16_16_epochs.json'
        cloob_checkpoint_fp = 'checkpoints/cloob_laion_400m_vit_b_16_16.pkl'
        cloob_config = pretrained.load_config(cloob_config_fp)
        self.cloob = model_pt.get_pt_model(cloob_config)
        self.cloob.load_state_dict(model_pt.get_pt_params(cloob_config, cloob_checkpoint_fp))
        self.cloob.eval().requires_grad_(False).to(self.device)
        self.channel_multipliers = [4, 4, 8, 8]
        self.base_channels = 192

    def predict(
        self,
        prompt: str = Input(description='Text prompt'),
        image: Path = Input(description='Input image prompt', default=None),
        # cond_scale: float = Input(
        #    description='The degree to which the image will be "conditioned" on the prompt', ge=1, le=100, default=3
        # ),
        diffusion_model: str = Input(choices=['yfcc', 'danbooru'], default='yfcc'),
        autoencoder_model: str = Input(choices=['yfcc', 'danbooru', 'laion5b'], default='yfcc'),
        steps: int = Input(description='The number of steps to sample for', default=25),
        noise: float = Input(description='The amount of noise to add during sampling, on a scale of 0 (none) to 1 (lots).', default=1.0),
        sampling_method: str = Input(choices=['ddpm', 'ddim', 'prk', 'plms', 'pie', 'plms2'], default='plms'),
        seed: int = Input(description='Random seed', default=2)
    ) -> Path:
        """Run a single prediction on the model"""
        torch.manual_seed(seed)
        diffusion, autoencoder = self.load_models(diffusion_model, autoencoder_model)
        n_ch, side_y, side_x = 4, 32, 32
        target_embeds = self.get_target_embeddings(prompt, image)

        def cfg_model_fn(x, t):
            n = x.shape[0]
            n_conds = len(target_embeds)
            x_in = x.repeat([n_conds, 1, 1, 1])
            t_in = t.repeat([n_conds])
            clip_embed_in = torch.cat([*target_embeds]).repeat_interleave(n, 0)
            vs = diffusion(x_in, t_in, clip_embed_in).view([n_conds, n, *x.shape[1:]])
            v = vs.sum(0)
            return v
        
        sampling_fn = self.make_sample_fn(cfg_model_fn, sampling_method, noise)
        n = 1
        x = torch.randn([n, n_ch, side_y, side_x], device=self.device)
        t = torch.linspace(1, 0, steps + 1, device=self.device)[:-1]
        steps = utils.get_spliced_ddpm_cosine_schedule(t)
        out_latents = sampling_fn(x, steps)
        outs = autoencoder.decode(out_latents * torch.tensor(2.55).to("cuda:0"))
        output_path = f'out.png'
        utils.to_pil_image(outs).save(output_path)
        return Path(output_path)

    def get_target_embeddings(self, prompt: str, image: Optional[Path]):
        zero_embed = torch.zeros([1, self.cloob.config['d_embed']], device=self.device)
        target_embeds = [zero_embed]
        target_embeds.append(self.cloob.text_encoder(self.cloob.tokenize(prompt).to(self.device)).float())
        if image is not None:
            img = Image.open(utils.fetch(image)).convert('RGB')
            cloob_size = self.cloob.config['image_encoder']['image_size']
            img = resize_and_center_crop(img, (cloob_size, cloob_size))
            batch = TF.to_tensor(img)[None].to(self.device)
            img_embed = F.normalize(self.cloob.image_encoder(self.cloob.normalize(batch)).float(), dim=-1)
            target_embeds.append(img_embed)
        return target_embeds

    def load_models(self, diffusion_type: str, autoencoder_type: str):
        diff_args = self.diff_list[diffusion_type]
        ae_args = self.ae_list[autoencoder_type]
        ae_config = OmegaConf.load(ae_args['config_fp'])
        ae_model = ldm.models.autoencoder.AutoencoderKL(**ae_config.model.params)
        ae_model.eval().requires_grad_(False).to(self.device)
        ae_model.init_from_ckpt(ae_args['model_fp'])
        model = train.DiffusionModel(
            diff_args['base_channels'],
            diff_args['channel_multipliers'],
            autoencoder_scale=torch.tensor(ae_args['scale'])
        )
        model.load_state_dict(torch.load(diff_args['model_fp'], map_location='cpu'))
        model = model.to(self.device).eval().requires_grad_(False)
        return model, ae_model
    
    def make_sample_fn(self, cfg_model_fn, method: str, noise: float):
        if method == 'ddpm':
            return partial(sampling.sample, cfg_model_fn, eta=1., extra_args={})
        if method == 'ddim':
            return partial(sampling.sample, cfg_model_fn, eta=noise, extra_args={})
        if method == 'prk':
            return partial(sampling.prk_sample, cfg_model_fn, extra_args={})
        if method == 'plms':
            return partial(sampling.plms_sample, cfg_model_fn, extra_args={})
        if method == 'pie':
            return partial(sampling.pie_sample, cfg_model_fn, extra_args={})
        if method == 'plms2':
            return partial(sampling.plms2_sample, cfg_model_fn, extra_args={})
        raise ValueError(f'Sampling method {method} is not implemented.')


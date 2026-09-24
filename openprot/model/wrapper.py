import pytorch_lightning as pl
from ..utils.logger import Logger
import torch
from .model import OpenProtModel
from abc import abstractmethod
from ..utils.misc_utils import autoimport
import os
import tqdm
import torch
import torch.distributed as dist
import pytorch_lightning as pl
from ..tracks.manager import OpenProtTrackManager
from ..utils import residue_constants as rc
from ..data.data import OpenProtData
from ..utils.tensor_utils import tensor_tree_map
from omegaconf import OmegaConf
from .ema import ExponentialMovingAverage


def is_oom_error(e):
    return isinstance(e, RuntimeError) and "out of memory" in str(e)


class Wrapper(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters("cfg")
        self.cfg = cfg
        self._logger = Logger(cfg.logger)
        self.ema = None
        self.cached_weights = None
    
    def training_step(self, batch, batch_idx):
        self._logger.prefix = "train"

        out = self.general_step(batch)
        self._logger.step(self.trainer, "train")
        return out

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if self.cfg.model.ema:
            if self.cached_weights is None:
                self.load_ema_weights()
        self._logger.prefix = "val"
        # self.general_step(batch)
        self.validation_step_extra(batch, batch_idx)
        self._logger.step(self.trainer, "val")

    @abstractmethod
    def general_step(self, batch):
        NotImplemented

    def validation_step_extra(self, batch, batch_idx):
        pass

    def on_train_epoch_end(self):
        self._logger.epoch_end(self.trainer, prefix="train")

    def on_validation_epoch_end(self):
        self._logger.epoch_end(self.trainer, prefix="val")
        if self.cfg.model.ema:
            self.restore_cached_weights()

    def load_ema_weights(self):
        clone_param = lambda t: t.detach().clone()
        self.cached_weights = tensor_tree_map(clone_param, self.model.state_dict())
        self.model.load_state_dict(self.ema.state_dict()["params"])

    def restore_cached_weights(self):
        self.model.load_state_dict(self.cached_weights)
        self.cached_weights = None
        
    def on_before_zero_grad(self, *args, **kwargs):
        if self.cfg.model.ema:
            self.ema.update(self.model)
            
    # uncomment this to debug
    def on_before_optimizer_step(self, optimizer):
        quit = False
        for name, p in self.model.named_parameters():
            if p.requires_grad and p.grad is None:
                print(name, "has no grad")
                quit = True
            
        if quit:
            exit()

    def configure_optimizers(self):
        cls = getattr(torch.optim, self.cfg.optimizer.type)
        all_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = cls(all_params, lr=self.cfg.optimizer.lr)

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-12,
            end_factor=1.0,
            total_iters=self.cfg.optimizer.scheduler.warmup_steps,
        )
        decay = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=self.cfg.optimizer.scheduler.end_factor,
            total_iters=int(0.9 * self.cfg.optimizer.scheduler.total_steps),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, decay],
            milestones=[self.cfg.optimizer.scheduler.start_decay],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

        return optimizer


class OpenProtWrapper(Wrapper):
    def __init__(self, cfg, tracks: OpenProtTrackManager, evals: dict):
        super().__init__(cfg)
        
        self.model = OpenProtModel(cfg.model)
        tracks.add_modules(self.model)
        
        self.tracks = tracks
        self.evals = evals
        if self.cfg.model.ema:
            self.ema = ExponentialMovingAverage(self.model, self.cfg.model.ema)
        self.generator = torch.Generator().manual_seed(cfg.data.seed)
        
    def on_save_checkpoint(self, checkpoint):
        esm_keys = {k for k in checkpoint['state_dict'].items() if "model.esm." in k}
        checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items() if k not in esm_keys}

        if self.cached_weights is not None:
            self.restore_cached_weights()
        if self.cfg.model.ema:
            checkpoint["ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        state_dict = self.state_dict()
        esm_keys = {k for k in state_dict.items() if "model.esm." in k}
        checkpoint['state_dict'] |= {k: state_dict[k] for k in esm_keys}
        if self.cfg.model.ema:
            ema = checkpoint["ema"]
            self.ema.load_state_dict(ema)
            
        
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if self.cfg.model.ema:
            if self.ema.device != device:
                self.ema.to(device)
        return batch.to(device)

    def get_lr(self):
        for param_group in self.optimizers().param_groups:
            return param_group["lr"]

    def forward(self, noisy_batch):
        
        ## embed the tracks into an input dict
        inp = self.tracks.embed(self.model, noisy_batch)

        ## account for self cond
        if self.cfg.model.self_cond:
            inp['sc'] = noisy_batch.get('sc', torch.zeros_like(inp['x']))
            
        ## run it thorugh the model
        out = self.model(inp)

        ## place the readouts in a dict
        readout = self.tracks.readout(self.model, inp, out)

        return out, readout

    def general_step(self, batch):
        
        self._logger.register_masks(batch)
        self._logger.masked_log("toks", batch["pad_mask"], sum=True)
        
        ## corrupt all the tracks
        noisy_batch, target = self.tracks.corrupt(batch, logger=self._logger)

        
        
        if torch.rand(1, generator=self.generator).item() < self.cfg.model.self_cond_prob:
            with torch.cuda.amp.autocast(enabled=False): 
                with torch.no_grad():
                    out, readout = self.forward(OpenProtData(**noisy_batch))
            noisy_batch['sc'] = out['x']

        out, readout = self.forward(noisy_batch)

        ## compute the loss
        loss = self.tracks.compute_loss(
            readout, target, logger=self._logger, step=self.trainer.global_step
        )

        ## log some metrics
        self._logger.masked_log("loss", loss, batch["pad_mask"])
        self._logger.log("lr", self.get_lr())

        self._logger.clear_masks()

        return (loss * batch["pad_mask"]).sum() / batch["pad_mask"].sum()

    def on_validation_epoch_end(self):
        for name, eval_ in self.evals.items():
            if not eval_.cfg.get('run_metrics', True):
                continue
            savedir = (
                f'{os.environ["MODEL_DIR"]}/eval_step{self.trainer.global_step}/{name}'
            )
            savedir = eval_.cfg.get('savedir', savedir)
        
            eval_.compute_metrics(
                rank=self.trainer.global_rank,
                world_size=self.trainer.world_size,
                device=self.device,
                savedir=savedir,
                logger=self._logger,
            )
        super().on_validation_epoch_end()

    def validation_step_extra(self, batch, batch_idx):
        
        name = batch["dataset"][0]
        if not self.evals[name].cfg.get('run_inf', True):
            return
            
        savedir = (
            f'{os.environ["MODEL_DIR"]}/eval_step{self.trainer.global_step}/{name}'
        )
        savedir = self.evals[name].cfg.get('savedir', savedir)
        os.makedirs(savedir, exist_ok=True)
        if self.trainer.global_rank == 0:
            OmegaConf.save(self.evals[name].cfg, f"{savedir}/config.yaml")
    
        noisy_batch = batch.copy()
        for track in self.tracks.values():
            track.corrupt(batch, noisy_batch, {})
        self.evals[name].run_batch(
            self,
            batch,
            noisy_batch,
            savedir=savedir,
            logger=self._logger
        )

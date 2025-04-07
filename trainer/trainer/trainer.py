import torch
from pathlib import Path
from contextlib import nullcontext
from ..utils.checkpoint import load_checkpoint, save_checkpoint
from ..utils.progressbar import ProgressBar
import torch.distributed as dist

def example_to_device(example, device, non_blocking=True):
    example_torch = {}
    for k, v in example.items():
        if k in ['token']:
            example_torch[k] = v
        elif isinstance(v, list):
            example_torch[k] = [res.to(device, non_blocking=non_blocking)
                                for res in v if isinstance(res, torch.Tensor)]
        elif isinstance(v, torch.Tensor):
            example_torch[k] = v.to(device, non_blocking=non_blocking)
        else:
            example_torch[k] = v
    return example_torch


class Trainer(object):
    def __init__(self, model, train_dataloader=None, val_dataloader=None,
                 optimizer=None, lr_scheduler=None, clip_grad_val=0.0, max_epochs=0,
                 eval_every_nepochs=1, eval_epochs=None, logger=None, log_every_niters=2000,
                 accumulation_steps=8):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.log_every_niters = log_every_niters if log_every_niters is not None else max(1,len(train_dataloader) // 10)

        self.accumulation_steps = accumulation_steps  # 初始化梯度累积步数
        self.accumulation_count = 0  # 当前累积计数

        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        self.epoch = 0
        self.global_step = 0
        self.inner_iter = 0
        self.max_epochs = max_epochs

        self.clip_grad_val = clip_grad_val
        self.eval_every_nepochs = eval_every_nepochs
        self.eval_epochs = eval_epochs
        self.logger = logger


    @property
    def current_lr(self):
        if self.optimizer is None:
            raise RuntimeError("lr is not applicable because optimizer does not exist.")
        return [group["lr"] for group in self.optimizer.param_groups]

    @property
    def device(self):
        return next(self.model.parameters()).device

    def load_checkpoint(self, filename, map_location="cpu", strict=False):
        self.logger.info("load checkpoint from %s", filename)
        try:
            return load_checkpoint(self.model, filename, map_location, strict)
        except FileNotFoundError:
            self.logger.error(f"Checkpoint file {filename} not found.")
            raise
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            raise

    def save_checkpoint(self, filename_tmpl="epoch_{}.pth", save_optimizer=True):
        meta = dict(epoch=self.epoch, iter=self.global_step)
        filepath = filename_tmpl.format(self.epoch)
        optimizer = self.optimizer if save_optimizer else None
        scheduler = self.lr_scheduler if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, scheduler=scheduler, meta=meta)

    def resume(self, checkpoint, resume_optimizer=True, map_location=torch.device("cpu")):
        try:
            checkpoint = self.load_checkpoint(checkpoint, map_location=map_location, strict=True)
            self.epoch = checkpoint['meta']['epoch']
            self.global_step = checkpoint['meta']['iter']
            if 'optimizer' in checkpoint and resume_optimizer:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint and resume_optimizer:
                self.lr_scheduler.load_state_dict(checkpoint['scheduler'])
            self.logger.info("resumed epoch %d, iter %d", self.epoch, self.global_step)
        except KeyError as e:
            self.logger.error(f"Checkpoint missing required key: {e}")
            raise

    def optimize_step(self, loss):
        # 判断是否处于梯度累积阶段
        is_accumulating = (self.accumulation_count + 1) % self.accumulation_steps != 0

        # 在DDP模式下且需要累积时使用no_sync上下文
        if self.world_size > 1 and is_accumulating:
            context = self.model.no_sync()
        else:
            context = nullcontext()

        # 按累积步数缩放损失
        scaled_loss = loss / self.accumulation_steps

        with context:
            scaled_loss.backward()  # 累积梯度

        self.accumulation_count += 1

        # 达到累积步数时执行参数更新
        if not is_accumulating:
            self._update_parameters()

    def _update_parameters(self):
        # 梯度裁剪
        if self.clip_grad_val > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_val)
        # 参数更新
        self.optimizer.step()
        # 学习率调度
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        # 清空梯度
        self.optimizer.zero_grad()
        self.accumulation_count = 0  # 重置累积计数器

    def train_epoch(self):
        self.model.train()
        if self.world_size > 1:
            self.train_dataloader.sampler.set_epoch(self.epoch)

        for i, data_batch in enumerate(self.train_dataloader):
            self.inner_iter = i
            data_batch = example_to_device(data_batch, self.device, non_blocking=True)
            loss, logs = self.model(data_batch)
            self.optimize_step(loss)
            if (self.inner_iter + 1) % 50 == 0:
                torch.cuda.empty_cache()
            if (self.inner_iter + 1) % self.log_every_niters == 0:
                log_str = f"Epoch [{self.epoch + 1}/{self.max_epochs}][{self.inner_iter + 1}/{len(self.train_dataloader)}]\tlr: {self.current_lr[0]:.8f}"
                self.logger.info(log_str)
                self.logger.info(self._convert_to_str(logs))
                torch.cuda.empty_cache()

            self.global_step += 1

        # 在 epoch 结束时，检查是否有未处理的累积梯度
        if self.accumulation_count > 0:
            self._update_parameters()

        torch.cuda.empty_cache()

        self.epoch += 1
        if self.world_size > 1:
            dist.barrier()
        if self.rank == 0:
            self.save_checkpoint()
            
    # def optimize_step(self, loss):
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     if self.clip_grad_val > 0:
    #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_val)
    #     self.optimizer.step()
    #     if self.lr_scheduler is not None:
    #         self.lr_scheduler.step()
    #
    # def train_iter(self, data_batch):
    #     data_batch = example_to_device(data_batch, self.device, non_blocking=True)
    #     loss, logs = self.model(data_batch)
    #     self.optimize_step(loss)
    #     if (self.inner_iter + 1) % self.log_every_niters == 0:
    #         log_str = f"Epoch [{self.epoch + 1}/{self.max_epochs}][{self.inner_iter + 1}/{len(self.train_dataloader)}]\tlr: {self.current_lr[0]:.8f}"
    #         self.logger.info(log_str)
    #         self.logger.info(self._convert_to_str(logs))
    #         torch.cuda.empty_cache()
    #
    #     self.global_step += 1
    #
    # def train_epoch(self):
    #     self.model.train()
    #     if self.world_size > 1:
    #         self.train_dataloader.sampler.set_epoch(self.epoch)
    #
    #     for i, data_batch in enumerate(self.train_dataloader):
    #         self.inner_iter = i
    #         self.train_iter(data_batch)
    #
    #     # 在 epoch 结束时，检查是否有未处理的累积梯度
    #     if self.accumulation_count > 0:
    #         self._update_parameters()
    #     torch.cuda.empty_cache()
    #
    #     self.epoch += 1
    #     if self.world_size > 1:
    #         dist.barrier()
    #     if self.rank == 0:
    #         self.save_checkpoint()

    @torch.no_grad()
    def val_epoch(self):
        self.model.eval()
        if self.rank == 0:
            prog_bar = ProgressBar(len(self.val_dataloader))

        results = {}
        for i, data_batch in enumerate(self.val_dataloader):
            self._inner_iter = i
            data_batch = example_to_device(data_batch, self.device, non_blocking=True)
            res = self.model(data_batch)
            results.update(res)
            if self.rank == 0:
                prog_bar.update()

        torch.cuda.empty_cache()

        if self.world_size > 1:
            dist.barrier()
            all_predictions = [None for _ in range(self.world_size)]
            dist.all_gather_object(all_predictions, results)
            predictions = {}
            for p in all_predictions:
                predictions.update(p)
        else:
            predictions = results

        if self.rank == 0:
            output_dir = Path("results") / f"epoch_{self.epoch}"
            output_dir.mkdir(parents=True, exist_ok=True)
            result_dict = self.val_dataloader.dataset.evaluation(predictions, output_dir)
            self.logger.info("\n")
            for k, v in result_dict.items():
                self.logger.info(f"Evaluation {k}: {v}")

    def fit(self):
        self.logger.info("max: %d epochs", self.max_epochs)
        while self.epoch < self.max_epochs:
            self.train_epoch()
            if (self.epoch > 2):
                self.val_epoch()
            # if (self.eval_every_nepochs > 0 and self.epoch % self.eval_every_nepochs == 0) or \
            #    (self.eval_epochs is not None and self.epoch in self.eval_epochs):
            #     self.val_epoch()

    def _convert_to_str(self, log_dict):
        def _convert_to_precision4(val):
            if isinstance(val, torch.Tensor):
                val = val.detach().cpu().tolist()
            if isinstance(val, float):
                val = "{:.4f}".format(val)
            elif isinstance(val, list):
                val = [_convert_to_precision4(v) for v in val]
            return val

        def _convert_dict_to_str(log_vars):
            log_items = []
            for name, val in log_vars.items():
                log_items.append(f"{name}: {_convert_to_precision4(val)}")
            return ", ".join(log_items)

        if isinstance(log_dict, list):
            logs = [_convert_dict_to_str(log_vars) for log_vars in log_dict]
            log_str = '\n'.join(logs)
        else:
            log_str = _convert_dict_to_str(log_dict)
        log_str += "\n"
        return log_str

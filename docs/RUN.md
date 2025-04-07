# 🚀 Training and Evaluation

## 📚 nuScenes Dataset

Once your nuScenes data is prepared, you can start training the model using the script below. By default, the model is trained on **2 GPUs** with **gradient accumulation**:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    ./tools/train.py --config-name nusc_det_pillaramf_base \
    data.train_dataset.root_path=/root/to/nuscenes/ \
    dataloader.train.batch_size=2 \
    scheduler.max_lr=0.001 \
    trainer.max_epochs=20 \
    hydra.run.dir=outputs/nusc_pillaramf_base
```

---

## 🎯 Training Strategy: *Faded Strategy*

For the final reported results, we apply a **Faded Strategy**, where the **copy-and-paste augmentation is disabled during the last 5 epochs**.

To disable copy-and-paste manually, add the following override:

```bash
+data.train_dataset.use_gt_sampling=False
```

---

## ⏸️ Resuming Training from Epoch 15

In our experiments, training was manually stopped at **epoch 15**. To resume training with copy-and-paste disabled, use the script below:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    ./tools/train.py --config-name nusc_det_pillaramf_base \
    data.train_dataset.root_path=/root/to/nuscenes/ \
    dataloader.train.batch_size=2 \
    scheduler.max_lr=0.001 \
    trainer.max_epochs=20 \
    hydra.run.dir=outputs/nusc_pillaramf_base \
    +data.train_dataset.use_gt_sampling=False \
    +resume_from=epoch_15.pth
```

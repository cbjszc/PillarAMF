# 📁 Data Preparation

## 📦 nuScenes

### 1. Download and Organize the Dataset

After downloading the nuScenes dataset, structure the directory as follows:

```
NUSCENES_DATASET_ROOT/
├── samples        # Key frames with annotations
├── sweeps         # Additional frames without annotations
├── maps           # (Optional) Not used
├── v1.0-trainval  # Metadata and labels
```

> ⚠️ Make sure the folder names exactly match the above structure.

---

### 2. Generate Processed Data

Run the following command to preprocess the data:

```bash
python tools/create_data nuscenes_data_prep \
    --root_path /path/to/nuscenes
```

Replace `/path/to/nuscenes` with the actual path to your `NUSCENES_DATASET_ROOT`.

The system currently is trained on 2 class (pothole, crack) and 4 class (pothole, longitudinal crack, latitude crack, and alligator crack).

## Overview

1. Detection: YOLO11s - Detects and localizes road damage in images/video.
2. VLM Caption: Qwen2-VL-2B - Generates Indonesian-language damage description for each detected region.
3. Finetuning: LoRA - adapts Qwen2-VL to domain-specific road damage captioning via LLaMA-Factory

---

## Project Structure
allycia/
|---dataset-2class/
|   |---cropped_images/         # Cropped road damage images
|   |---adas_caption_train.json
|---dataset-4class/
|   |---cropped_images/
|   |---adas_caption_train.json
|---vlm/
|   |---qwen2-2b-2class/
|   |   |---checkpoint-XXXXX/
|   |   |---adapter_config.json
|   |---qwen2-2b-4class/
|   |   |---checkpoint-XXXXX/
|   |   |---adapter_config.json
|---vid/
|---inf/
|   |---*.mp4               # Annotated output videos
|   |---*.logs.txt          # Detection + caption logs
|---model/
|   |---yolo-2class.pt
|   |---yolo-4class.pt
|---code/
|   |---yolo.ipynb          # Code for detection
|   |---finetuning.ipynb    # Code for finetuning using LoRA
|   |---inference.ipynb     # Code for inferencing video using YOLO and Qwen2

---

## Pipeline Flow

Input Video/Image
|
V
YOLO11s Detection
- Runs on every fram
- Detects either 2 class or 4 class
- Outputs: bbox, class, confidence
|
V
Bounding Box Crops
|
V
Crop + Pad: Crops detected region + 10px padding
Region: Skip crops < 32*32px
|
V
Qwen2-VL-2B: Fine-tuned with LoRA on dataset
Captioning: Generates Indonesian damage descriptiong
Also appends position label (kiri/tengah/kanan lajur) according to the bbox coordinates
|
V
IoU Cache: Reuses captions for same object across frames
Matching: Avoids redundant Qwen calls (IoU treshold)
|
V
Output:
- Annotated Video (.mp4): BBox + Label + Caption Overlay
- Detection Log (.txt): Per-detection record with timing

---

## Dataset

### Format
The dataset follows ShareGPT format compatibility with LLaMA-Factory:

```json
{
    "id": "pothole_0_v1",
    "image": "cropped_images/crop_0_AA_10800_jpeg.rf.0ffd63a44a4e8d68df26630ad4caa019.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nLakukan analisis pada gambar ini dan sebutkan jenis kerusakan beserta tingkat keparahannya."
      },
      {
        "from": "gpt",
        "value": "Terdapat retak melintang yang tergolong ringan."
      }
    ]
}
```

### Splits
| Split      | File               | Samples |
|------------|--------------------|---------|
| Train      | qwen_train.json    | ~13,368 |
| Validation | qwen_val.json      | ~1671   |
| Test       | qwen_test.json     | ~1671   |

---

## Training

### Environment
- Platform:     Google Colab (T4 GPU, 15GB VRAM)
- Framework:    [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- Base Model:   Qwen/Qwen2-VL-2B-Instruct

### Fine-tuning Config
| Parameter                | Value              |
|--------------------------|--------------------|
| Method                   | LoRA (SFT)         |
| LoRA Rank                | 8                  |
| LoRA Target              | All linear layers  |
| Learning Rate            | 5e-5               |
| LR Scheduler             | Cosine             |
| Epochs                   | 10                 |
| Batch Size               | 1                  |
| Gradient Accumulation    | 4                  |
| Effective Batch Size     | 4                  |
| Quantization             | 4-bit (QLoRA)      |
| Precision                | BF16               |
| Cutoff Length            | 1024 tokens        |
| Eval Strategy            | Every 100 steps    |
| Save Strategy            | Every 200 steps    |

### Training Command
```bash
"llamafactory-cli", "train",
    "--stage", "sft",
    "--do_train", "True",
    "--model_name_or_path", "Qwen/Qwen2-VL-2B-Instruct",
    "--dataset", "adas_qwen_train",  
    "--eval_dataset", "adas_qwen_val",
    "--template", "qwen2_vl",
    "--finetuning_type", "lora",
    "--lora_target", "all",
    "--lora_rank", "8",
    "--dataset_dir", "/content/LLaMA-Factory/data",
    "--output_dir", output_dir,
    "--overwrite_cache", "True",
    "--overwrite_output_dir", "False",
    "--cutoff_len", "1024",
    "--per_device_train_batch_size", "1",
    "--gradient_accumulation_steps", "4",
    "--lr_scheduler_type", "cosine",
    "--logging_steps", "10",
    "--eval_strategy", "steps",
    "--eval_steps", "100",
    "--save_strategy", "steps",
    "--save_steps", "200",
    "--learning_rate", "5e-5",
    "--num_train_epochs", "10",
    "--quantization_bit", "4",
    "--bf16", "True",
    "--load_best_model_at_end", "True",
    "--metric_for_best_model", "eval_loss",
```

---

## Inference

### Video Inference
```python
YOLO_MODEL_PATH   = "yolo-2class.pt"
QWEN_BASE_MODEL   = "Qwen/Qwen2-VL-2B-Instruct"
QWEN_LORA_PATH    = "vlm/qwen2-2b-2class"
CONFIDENCE_THRESHOLD   = 0.25
CAPTION_EVERY_N_FRAMES = 15   # Qwen runs every 15 frames
MAX_NEW_TOKENS         = 40   # Short captions for speed
```

### Position Detection
Position is inferred from bounding box center relative to frame dimensions:

Frame Width divided into 3 horizontal zones:
┌──────────┬──────────┬──────────┐
│   Kiri   │  Tengah  │  Kanan   │
│  Lajur   │  Lajur   │  Lajur   │
├──────────┴──────────┴──────────┤
│         Sisi Dekat             │  ← Lower half
└────────────────────────────────┘
│         Sisi Jauh              │  ← Upper half
└────────────────────────────────┘

Sample caption output:
Terdapat retakan yang tergolong ringan di tengah lajur (sisi dekat).
Terdapat lubang yang tergolong kecil di tengah lajur (sisi dekat).

### Output Log Format
- 2 Class:
[#1] Frame 15 | Time 00:00.50 | Video FPS: 13.1
  Class      : retakan (conf: 0.540)
  BBox       : [708, 596, 869, 908]
  Caption    : Terdapat retakan yang tergolong sedang di tengah lajur (sisi dekat).
  Qwen time  : 1.228s

- 4 Class:
[#1] Frame 15 | Time 00:00.50 | Video FPS: 14.1
  Class      : longitudinal_cracking (conf: 0.253)
  BBox       : [294, 788, 499, 946]
  Caption    : Terdapat retak memanjang yang tergolong ringan di kiri lajur (sisi dekat).
  Qwen time  : 1.936s

---

## Performance
- 2 Class:
Summary
  Total frames processed : 365
  Total captions logged  : 21
  Total Qwen time        : 28.47s
  Avg Qwen time/caption  : 1.356s
  Total pipeline time    : 56.58s

- 4 Class:
Summary
  Total frames processed : 365
  Total captions logged  : 20
  Total Qwen time        : 29.96s
  Avg Qwen time/caption  : 1.498s
  Total pipeline time    : 58.54s
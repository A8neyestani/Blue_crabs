# Blue crab segmentation and dimension estimation

I wanted to see whether a camera could help with two parts of studying blue crabs: identifying what is in the image and estimating how big it is. This project uses YOLO11m-seg to find crabs and egg clusters, then combines the masks with depth and camera calibration to estimate dimensions in millimeters.

I completed the original experiment in July 2025. The repository includes the dataset, inference code, training settings and examples from that work, along with a later check of the saved model on the test split.

**Author:** [Arman Neyestani](https://github.com/A8neyestani) · **Stack:** Python, Ultralytics, PyTorch, OpenCV, NumPy

## Motivation

The starting point was a practical task in marine monitoring and aquaculture: recording the type and size of an animal. Doing that by hand means reviewing images, marking boundaries or taking physical measurements. I wanted to explore a camera-based workflow that could eventually be used in an underwater observation or robotic system.

A segmentation mask gives a useful outline, but it still measures the crab in pixels. The same animal looks smaller as it moves away from the camera. That is why I added depth-based measurement: use the mask to select the animal, recover metric coordinates from depth, and use PCA to estimate its extent along its main axes. Combining these steps is the main contribution of the project.

## Demo

[![Underwater segmentation demo: a male blue crab with an instance mask](assets/demo-preview.gif)](assets/demo.mp4)

**[Watch the full 24-second demo](assets/demo.mp4)** · [Download the MP4](assets/demo.mp4?raw=true)

Click the preview to watch the original demonstration. The full clip also shows frames where the crab becomes obscured and detections are lost. It illustrates the model's behavior; it was not recorded as a speed benchmark.

| Example input | Current pipeline output |
| --- | --- |
| ![Original example image](assets/example.jpg) | ![Female blue crab segmentation with confidence and instance ID](assets/segmentation.jpg) |

This is one of the original project images. The mask follows the visible body and appendages. The score above it is the detection confidence.

## What I worked on

I fine-tuned a COCO-pretrained YOLO11m-seg model on a four-class dataset and used the predicted masks as the input to the measurement code. Starting with pretrained weights let me focus on the crab images, the quality of the masks and the geometry needed to turn them into useful measurements.

The current code saves a mask for each detected animal, an overlay image and a JSON file with the predictions. It runs without opening a display window. When depth is supplied, each instance gets its own dimension estimate.

## Experimental dimension estimation

The measurement step needs three things: an RGB image, a depth map aligned to it, and the camera's intrinsic matrix. The figure below shows how these inputs fit together in the original prototype.

![Original workflow figure: RGB image and depth illustration above the segmented crab and dimension output](assets/measurement-workflow.png)

*Figure 3 from the original project report. YOLO processes the RGB image; the depth map is used afterward to calculate metric coordinates.*

### From a mask to millimeters

1. **Segment the RGB image.** YOLO11m-seg returns a class, confidence score and mask for each detected instance. The mask must remain aligned with the original RGB pixel coordinates.
2. **Select valid depth samples.** For every mask pixel `(u, v)`, read the aligned depth `Z` in meters. Exclude missing, non-positive and non-finite depths.
3. **Back-project into camera coordinates.** Use the calibrated focal lengths `(fx, fy)` and principal point `(cx, cy)` to obtain the metric coordinates below. Each pixel defines a point `(X, Y, Z)`; the current estimator uses its `(X, Y)` projection.

```text
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
```

4. **Find the main axes.** Center the `(X, Y)` points by subtracting their mean, then use singular value decomposition to find the two PCA directions. This follows the orientation of the projected shape instead of using the horizontal and vertical image axes.
5. **Measure the extents.** Project the centered points onto those directions. For each axis, subtract the minimum coordinate from the maximum and multiply by 1,000 to convert meters to millimeters.

```text
centered_points = points_xy - mean(points_xy)
axis_coordinates = centered_points @ pca_axes.T
extent_mm = (max(axis_coordinates) - min(axis_coordinates)) * 1000
```

The last line is evaluated separately for each PCA axis. The current outputs are named `pca_axis_1_mm` and `pca_axis_2_mm`: they are projected whole-mask extents, not anatomical length/width or full 3D surface measurements. Limb position, camera angle, occlusion and depth outliers all affect them. The original prototype merged masks by class; this repository applies the geometry to each instance so two crabs of the same class are measured separately.

### Original measurement demonstration

![Original prototype screenshot showing a female crab mask and displayed dimensions of approximately 207.8 mm and 73.5 mm](assets/measurement-example.jpg)

*The original prototype displayed about 207.8 mm and 73.5 mm. These are demonstration values, not verified physical measurements: the saved 8-bit depth preview and calibration information are insufficient to validate them. The current code uses PCA-axis names in place of the old Width/Length labels.*

### Run with calibrated depth

Supply an RGB-aligned metric depth map and the camera matrix for that exact RGB resolution. `intrinsics.json` must contain a numeric 3×3 array with rows `[fx, 0, cx]`, `[0, fy, cy]`, `[0, 0, 1]`. Calibration values must come from your camera; no sample calibration is assumed.

```bash
python crab.py predict --image your_rgb.jpg --depth your_depth.png --depth-scale 0.001 --intrinsics intrinsics.json --output outputs/measurement
```

`--depth-scale` is meters per stored unit: use `0.001` for millimeter-valued depth or `1` for a `.npy` depth array already in meters. The loader accepts single-channel depth images or numeric `.npy` arrays and rejects 8-bit depth previews. Zero, negative and non-finite depths are excluded; fewer than ten valid mask pixels yields `null` dimensions rather than a misleading zero measurement.

I have not established the physical accuracy of this measurement method. That needs a comparison with reference measurements using calibrated depth. Underwater testing would also need calibration for the actual imaging setup and optical conditions.

## Results

The two columns below refer to different splits and evaluation runs. Values are percentages.

| Metric | Original validation, 53 images | Test re-evaluation, 26 images |
| --- | ---: | ---: |
| Box mAP@50 | 80.71 | 90.8 |
| Box mAP@50–95 | 69.59 | 78.2 |
| Mask mAP@50 | 79.13 | 85.7 |
| Mask mAP@50–95 | 59.23 | 67.6 |
| Mask precision | 80.30 | 84.5 |
| Mask recall | 72.50 | 85.5 |

The validation values come from metadata stored in `best.pt`. The test values were measured again on 11 September 2026 using that same checkpoint, Ultralytics 8.3.167, Python 3.12.14, PyTorch 2.14.0 on CPU, 640-pixel inputs and batch size 1. Evaluation uses Ultralytics' validation thresholds; the demo uses a confidence threshold of 0.30.

| Test class | Annotated instances | Mask mAP@50 | Mask mAP@50–95 |
| --- | ---: | ---: | ---: |
| Female blue crab | 12 | 90.1 | 72.5 |
| Male blue crab | 8 | 98.2 | 57.6 |
| Eggs | 4 | 99.5 | 96.2 |
| Other crabs | 7 | 55.1 | 44.1 |

The small test set matters when reading these numbers. There are just four egg instances, so I would not draw a broad conclusion from that class's high score. Other crabs are the weakest class. For male crabs, the gap between mAP@50 and mAP@50–95 also shows that locating an animal is easier than getting its boundary consistently right. A larger test set from a new source would give a better picture of generalization.

![Original training history: segmentation loss and mask average precision across 300 epochs](assets/training.png)

These curves come from the original 300-epoch training history stored in the checkpoint. The later evaluation used the saved weights without retraining.

## Run it locally

Use **Python 3.12**. The dependency versions below were exercised on Windows with CPU inference.

```bash
git clone https://github.com/A8neyestani/Blue_crabs.git
cd Blue_crabs
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or `source .venv/bin/activate` on macOS/Linux. Then:

```bash
python -m pip install -r requirements.txt
```

Download **[best.pt from the v1.0.0 release](https://github.com/A8neyestani/Blue_crabs/releases/download/v1.0.0/best.pt)** and place it at `weights/best.pt`. The checkpoint is approximately 45 MB and is kept outside the Git history.

SHA-256:

```text
2a6c38cfd73837393e3eefbd0455a3f25f14625fc2cbe2f8cf64a90c8e7cbb73
```

```bash
python crab.py predict --image assets/example.jpg --weights weights/best.pt --output outputs/example
```

The output directory contains `overlay.jpg`, one binary PNG mask per instance, and `predictions.json` with class IDs, names, confidence scores and mask filenames. An image with no detections produces an unchanged overlay and an empty JSON list. Use a new output directory for each image.

Use `--confidence 0.50` to change the detection threshold, `--iou` to change suppression, or `--device 0` when using a compatible CUDA-enabled PyTorch installation. GPU execution was not tested for this release.

## Dataset and training

**[Download Dataset.zip](Dataset.zip?raw=true)** (about 45 MB), or use the copy included when you clone this repository. It is the original Roboflow export used for the results above, with images, polygon labels and `data.yaml`.

The dataset contains **631 images** across four classes:

| Split | Images | Polygon instances |
| --- | ---: | ---: |
| Training | 552 | 687 |
| Validation | 53 | 68 |
| Test | 26 | 31 |

Class IDs are `0: Blue_crab_Female`, `1: Blue_crab_Male`, `2: Eggs`, and `3: Other_crabs`. These are object/region labels: an egg cluster can be annotated separately from the animal carrying it.

The export came from [Roboflow blue-crab, version 1](https://universe.roboflow.com/neyestanisetelco/blue-crab/dataset/1). You do not need a Roboflow account to use the ZIP included here. The collection includes web-sourced images with different viewpoints, backgrounds and image quality, as well as augmented training examples. The 631 images are therefore not 631 independent captures or a uniform set of underwater observations.

Checks found no byte-identical images or shared Roboflow source-name stems across splits. That does not rule out visually similar images or copies saved in another format. The polygon coordinates are within the expected normalized range, and every split has matching image and label counts.

From the repository root, extract the dataset with:

```bash
python -m zipfile -e Dataset.zip data
```

In the extracted `data/data.yaml`, add `path` with the absolute path to your `data` directory. Replace the three split paths with the values below and keep the class definitions:

```yaml
train: train/images
val: valid/images
test: test/images
nc: 4
names: [Blue_crab_Female, Blue_crab_Male, Eggs, Other_crabs]
```

Keep these splits unchanged when comparing your run with the reported results.

```bash
python crab.py evaluate --data data/data.yaml --weights weights/best.pt --split test
python crab.py train --data data/data.yaml --device 0
```

Evaluation saves plots and sample predictions under `runs/evaluate/`. Training uses [train.yaml](train.yaml), recovered from the checkpoint: COCO-pretrained `yolo11m-seg.pt`, 300 epochs, batch 16, image size 640, AdamW, initial learning rate 0.005, five warm-up epochs and a linear learning-rate schedule. Mosaic, horizontal flips and HSV augmentation were enabled; training-time MixUp was zero. Roboflow preprocessing/augmentation and Ultralytics training augmentation are separate stages.

The training command downloads the upstream pretrained weights if they are missing. The configuration matches the saved experiment settings; hardware and library differences can still affect the result.

## Checks and next experiments

```bash
python -m unittest -v
```

The regression tests cover known metric geometry, depth units, invalid depth and calibration, missing images, empty detections, and separate animals of the same class. The release was also checked with real checkpoint inference and evaluation on the full archived test split.

I would extend the work by collecting a larger test set from separate sources, reviewing the other-crab failures, and comparing the dimension estimates with physical measurements. Before using it on a robot, I would also benchmark it on the target hardware.

## Attribution and reuse

The model builds on [Ultralytics YOLO11 segmentation](https://docs.ultralytics.com/tasks/segment/) and COCO-pretrained weights. Its checkpoint carries Ultralytics' AGPL-3.0 metadata. The included Roboflow export declares MIT; individual source images and video may have their own attribution and reuse terms, which that declaration does not resolve. The internal project report is kept outside this repository.

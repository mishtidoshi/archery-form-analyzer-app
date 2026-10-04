# Models

This directory holds the pose model weights and (optionally) a trained fallback
shot-classifier. Nothing here is tracked in git — everything is either downloaded
automatically or generated locally.

## Pose model (required)

`yolo26m-pose.pt` auto-downloads from Ultralytics the first time any tool that needs
it runs (e.g. `archery_analyzer_v2.py`, `tools/analyze_face_yolo.py`). No manual step
needed with a working internet connection.

## Fine-tuning the pose model on your own footage (optional)

The stock model is a general-purpose human pose model, never adapted to this app's
own recording conditions (camera angles, distances, archery stance, bow/arrow
occlusion). You can fine-tune it on your own verified full-draw frames once you have
a few sessions' worth of data:

1. `python3 tools/export_pose_dataset.py` — builds a training set from every
   `session_history/*.json`'s `metrics.verified_frames`, using the *current* pose
   model's own predictions as pseudo-labels (there's no independent ground truth).
   **Look at the preview sheet it writes to `output/pose_finetune_preview.png` before
   training** — every skeleton drawn on it is what the model will be told is correct,
   so this can only adapt the model to your recording setup, not fix cases where it's
   already wrong.
2. `python3 tools/finetune_pose_model.py` — continues training from the pretrained
   checkpoint (training from random initialization would need a vastly larger dataset
   than any single archer's session history will produce). Runs on CPU by default —
   as of `ultralytics==8.4.31`, YOLO26's pose loss crashes on Apple Silicon's MPS
   backend (a hard float64/MPS incompatibility, not a missing-op fallback case), so
   CPU is the safe default on a Mac; CUDA is auto-detected and used if available.

Once `models/archery_pose_finetuned.pt` exists, every tool in the pipeline picks it
up automatically (`tools/yolo_pose_adapter.resolve_pose_model_path()`) — no flags to
pass. Delete it to go back to the stock model. More verified sessions (ideally from
several different clips, not just one) make for a meaningfully better fine-tune —
a couple of frames from a single clip is only good for a mechanical smoke test, not
a real improvement, and can actually make things worse by overfitting.

## RandomForest shot-classifier fallback (optional)

`ml_shot_detector/` implements a fallback shot-phase classifier
(IDLE / DRAWING / FULL_DRAW / RELEASE / FOLLOW_THROUGH) used when the primary
audio-visual detector (`tools/audio_visual_shot_detector.py`) needs a second opinion.
It falls back to a velocity heuristic automatically if no trained model is present
here, so this is entirely optional to start.

To train your own:
1. Label some clips with `ml_shot_detector/labeler.py`
2. Extract features with `ml_shot_detector/extract_training_data.py`
3. Train with `ml_shot_detector/train_model.py`

This writes the trained model (and its metadata) into this directory.

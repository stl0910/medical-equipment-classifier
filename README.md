# Medical Equipment Classifier

An image classifier that identifies the type of medical equipment shown in a photo. Built as a hands-on project while working through fast.ai's *Practical Deep Learning for Coders* (Lesson 1).

## What it does

Given a photo, the model predicts which of six equipment categories it belongs to:

- Stethoscope
- Wheelchair
- Syringe
- Hospital bed
- Defibrillator
- X-ray machine

## Results

Final validation error rate: **2.86%** (~97% accuracy) after fine-tuning a pretrained ResNet18 for 3 epochs.

## How it works

1. **Data collection** — images for each category were pulled from web search and saved into per-category folders.
2. **Cleaning** — corrupted or unreadable downloads were filtered out automatically.
3. **Training** — a ResNet18 model, pretrained on ImageNet, was fine-tuned on the equipment images using fastai's `vision_learner`.
4. **Evaluation** — accuracy was checked using a validation split, a confusion matrix, and fastai's `most_confused()` to spot any categories the model mixes up.

## A lesson learned along the way

Early versions of this project used the search query `"<item> medical equipment photo"`, which pulled in irrelevant results for some categories — including a sports mascot photo mistakenly downloaded into the "syringe" folder, which a model can't learn anything useful from. Switching to a simpler, more literal query (`"<item> photo"`) fixed this and noticeably improved the data quality and resulting accuracy. Worth remembering for future projects: it's worth spot-checking a handful of images in each class folder before training, since search-based datasets can include unexpected noise.

## Built with

- [fastai](https://www.fast.ai/) / PyTorch
- Kaggle Notebooks
- DuckDuckGo image search for data collection

## Possible next steps

- Add more images per category to improve robustness
- Try a larger backbone (e.g. ResNet34) for comparison
- Expand to more equipment categories

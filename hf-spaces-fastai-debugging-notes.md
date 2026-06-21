# Deploying a fast.ai Model to Hugging Face Spaces: Debugging Notes

Reference notes from deploying the medical equipment classifier (fast.ai + Gradio + HF Spaces).
Four separate, unrelated bugs stacked on top of each other. Keep this for the next deployment.

---

## Issue 1: PowerShell install verification error (cosmetic, not a real failure)

**Symptom:**
```
[ERROR] Installation verification failed: Error: Invalid value. 'charmap' codec can't
encode character '\u2713' in position 5: character maps to <undefined>
```

**Cause:** The HF CLI installer tries to print a checkmark (✓) character. Windows
PowerShell's default console codepage (cp1252) can't display it, so the *verification
step* crashes — even though the actual install usually succeeds.

**Fix:** Ignore it, or force UTF-8 first:
```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001
```

**How to confirm it actually installed:**
```powershell
hf --version
```
If that returns a version number, the install worked regardless of the error.

---

## Issue 2: Gradio `examples=[...]` crashes app startup entirely

**Symptom (the big one — showed up 3 separate times in different forms):**
```
TypeError: unsupported operand type(s) for +: 'PILImage' and 'dict'
...
Exception: Couldn't start the app because 'http://localhost:7860/gradio_api/startup-events'
failed (code 500).
```

This happens during `Caching examples at: '/app/.gradio/cached_examples/14'` — Gradio
pre-runs your `predict()` function on every example image at startup, *before* the app
finishes booting. If anything in that path is broken, the **entire app fails to start**,
not just the examples feature.

**Root cause (in this case):** the model's exported pickle (`export.pkl`) carried its
training-time **y-pipeline** (`parent_label -> Categorize`) inside it. `learn.predict()`
re-runs that pipeline even at inference time. `parent_label` does `Path(x).parent.name` —
on a bare filename with no real labeled folder structure, this returns garbage that
breaks deep inside fastai's `fasttransform` internals.

**How we diagnosed it:**
```python
# Inspect what's inside the exported learner
from fastai.vision.all import *
learn = load_learner('export.pkl')
print(learn.dls.tfms)          # shows the x-pipeline AND y-pipeline
print(learn.dls.tls[1].tfms.fs) # the actual y-transforms: [parent_label, Categorize]

# Confirm parent_label returns garbage for a bare filename
print(repr(parent_label(Path('test.jpg'))))  # → '' (empty string)
```

**Fix that actually worked: bypass `learn.predict()` entirely, do raw PyTorch inference.**
Don't try to patch the fastai pipeline — route around it completely.

```python
from fastai.vision.all import *
import torch
import torchvision.transforms as T

learn = load_learner('export.pkl')
labels = learn.dls.vocab   # grab BEFORE touching anything else
learn.model.eval()

# Replicate training-time preprocessing manually.
# Check your own values via: learn.dls.tfms / learn.dls.after_batch
preprocess = T.Compose([
    T.Resize((192, 192)),   # match your training Resize size/method
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
])

def predict(img_path):
    img = PILImage.create(img_path).convert('RGB')
    x = preprocess(img).unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        out = learn.model(x)
        probs = torch.softmax(out, dim=1)[0]
    return {labels[i]: float(probs[i]) for i in range(len(labels))}
```

**Things that did NOT fix it (tried and ruled out, in order):**
- Switching `gr.Image(type="pil")` → `type="filepath"` — no effect, the crash wasn't
  about input type.
- Pinning `fastai==2.8.7` to match the training environment — no effect.
- Pinning `fasttransform==0.0.2` to match exactly too — no effect. This ruled out
  version drift as the cause entirely.
- Manually emptying the y-transform list (`learn.dls.tls[1].tfms.fs = []`) — worked
  locally in the training notebook, but **did not work identically on the Space**
  despite identical package versions. Never fully explained; abandoned in favor of
  bypassing fastai's predict path altogether.

**Lesson:** when a fix that "should" work doesn't reproduce in a different environment
even with matched dependency versions, stop trying to patch the framework's internals —
route around them.

---

## Issue 3: Examples directory blocked by Gradio's file security check

**Symptom (after fixing Issue 2 and re-adding `examples=[...]`):**
```
gradio.exceptions.InvalidPathError: Cannot move /app/examples/hospital_bed.jpg to the
gradio cache dir because it was not uploaded by a user.
```

**Cause:** Gradio blocks serving/caching local server files as examples by default,
for security (prevents an app from exposing arbitrary server files to users).

**Attempted fix:**
```python
demo.launch(allowed_paths=["examples"])
```
This is the documented fix for this exact error — but it **did not work** in our case;
`allowed_paths` is checked at request-serving time, not during the startup example-cache
step, so the same crash happened anyway.

**What we actually did:** dropped the `examples=[...]` block entirely. The feature
wasn't worth a fourth debugging round-trip given the core app already worked. If you
want clickable examples later, this needs a fresh, isolated investigation — possibly
pre-caching differently or using `cache_examples=False`.

---

## General lessons for next time

1. **Read tracebacks bottom-up.** The real exception is usually the last "real" line
   near the bottom — everything above it is framework/server plumbing (uvicorn →
   starlette → fastapi → gradio → your code).

2. **Test locally/in the original notebook first.** A full HF Space rebuild takes
   minutes per attempt. Reproducing the failure in Kaggle/Colab (where you trained
   the model) gives you a feedback loop measured in seconds instead.

3. **Don't trust a fix until you've seen it actually run.** Several "should work"
   theories here didn't pan out in practice — only directly printing/inspecting
   actual objects (`print(repr(...))`, `print(type(...))`) cut through the guessing.

4. **Know when to cut a feature.** Three failed attempts at `examples=[...]` was the
   signal to drop it rather than chase a fourth fix. The core deliverable (working
   classifier UI) mattered more than the nice-to-have.

5. **fast.ai's pickle export (`export.pkl`) is brittle across environments** because
   it serializes live Python objects (transforms, classes), not just model weights.
   This is a known community pain point, not a sign of doing something wrong. For
   future projects, consider exporting just the model weights + rebuilding the
   architecture in plain PyTorch for deployment, to sidestep this category of bug
   entirely.

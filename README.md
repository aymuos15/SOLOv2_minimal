# PyTorch only implementation of SOLOv2

Essentially, Cleaned the code and minor changes from: `https://github.com/feiyuhuahuo/SOLOv2_minimal`.

The repo has the dummy dataset. So:

1. To train as is: `python train.py`
2. To test as is: `python val.py`
3. To visualise: run `detect.py` --> Visualisations saved at `results`
4. For all, run: `python train.py && python val.py && python detect.py`

My goal was here to get the minimal viable product at potentially the cost of some performance. 

## Notes

1. The `detect.py` from the original code was giving some errors. So I have chosen to remove that. I think everything is logically correct.
2. Sometimes, even on the dummy dataset, 2 epochs are not enough and the model does not learn. I am not sure why there is so much randomness.
3. For curve metrics, refer to original repo. Have removed it from here for now.
4. `/weigths/weights/backbone_resnet34.pth` is from the orginal repo. Check all the weights there.
    - Apparently the version outperforms original. I have not dug deep into my changes to gauge a performance deficit. You can see the commit difs as I started with the original clone. It is the first commit with the message "Minimalisation Push!".

### Dummy Datasets

- `data/dummy/makedata.py` to build the dummy dataset. Look into the file to set up the params. This is inherently coco style.
- `data/dummy/lookdata.py` to verify your data.

## Immediate todos

- [ ] Improve visualisation code.
- [ ] Improve logging and experiment tracking. --> This is actually done in the orig repo but I do not like how its done at the moment.
- [ ] Yet to check what this `results/val/Solov2_light_res34.json` val results actually store.
- [ ] Test on COCO.
- [ ] Add a requirements.txt file.
- [ ] Make a light config of UNet
- [ ] Unit Tests

### Misc

To suppress coco dataset outputs:

```python
import sys
import os
from pycocotools.coco import COCO
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# Use this when loading COCO
with suppress_stdout():
    coco = COCO('path/to/annotations.json')
```
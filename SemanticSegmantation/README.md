


## Course Project

- Semantic Segmentation using CNN
- Data description
    - All images are sampled from VOC 2007 & 2012
        - target classes: ['person', 'cat', 'plane', 'car', 'bird']
        - class ids in segmentation annotation: [1, 2, 3, 4, 5]
        - image size: 256 x 256
        - mask format: H x W
        - total image number: 6624
    - Data structure
        - JPEGImages: images folder
        - ImageSets: image sets of train and validation
        - Annotations: annotation folder
    - Suggestion: Use PIL.Image library to read the annotation file
- Evaluation metric: mean IOU, please refer to the theory: https://ilmonteux.github.io/2019/05/10/segmentation-metrics.html and the code in CCNet: https://github.com/speedinghzl/CCNet/blob/e3976eefed81aadc1754a2075ccf87a521fedc83/evaluate.py#L244

- Step #1: implement a small segmentation network using naive pixel-wise classification
- Step #2: try various network structures to improve performance
- Step #3: evaluate model on real world scenes


## Code Instruction

- `train_seg.py` implements the training process with PSPNet. It will automatically invoke cuda: `python3 train_seg.py`
- `test_seg.py` tests specified images: `python3 test_seg.py IMG_PATH MODEL_PATH`







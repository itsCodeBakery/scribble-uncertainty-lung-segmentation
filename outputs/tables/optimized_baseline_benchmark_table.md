| Method                 | Postprocessing        | TTA   |   Dice | IoU    | Sensitivity   | Specificity   | Precision   | MAE    |
|:-----------------------|:----------------------|:------|-------:|:-------|:--------------|:--------------|:------------|:-------|
| UNet++ EfficientNet-B0 | Threshold + Lung Mask | Yes   | 0.7856 | 0.6964 | 0.8125        | 0.9968        | 0.8269      | 0.0063 |
| UNet++ EfficientNet-B0 | Threshold + Lung Mask | No    | 0.767  | 0.6750 | 0.7595        | 0.9977        | 0.8634      | 0.0067 |
| UNet++ EfficientNet-B0 | No                    | No    | 0.6956 | -      | -             | -             | -           | -      |
| U-Net                  | No                    | No    | 0.5897 | -      | -             | -             | -           | -      |
| Attention U-Net        | No                    | No    | 0.5772 | -      | -             | -             | -           | -      |
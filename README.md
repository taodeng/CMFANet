# Coordinative Multi-Scale Feature Augmentation Network for Multi-Task Driving Perception
## Tao Deng, Haoran Liu, Wenbo Liu, and Fei Yan
### 2025 under review

Abstract: This work proposes a multi-task network model based on a coordinative multi-scale feature augmentation network (CMFANet), capable of simultaneously addressing traffic object detection, drivable area segmentation, and lane line detection, thereby enabling comprehensive perception of traffic scenarios. The proposed CMFANet consists of a shared backbone, three independent neck layers, and task heads. To achieve the lightweight and high-precision perception of the driving environment, we design a powerful backbone network for feature extraction and develop an efficient feature fusion strategy tailored to the specific requirements of each task. We construct two versions of the proposed model: nano and small. The nano version features a parameter count of only 3.8 M, significantly lower than existing models such as YOLOP. The model was evaluated on the challenging BDD100K dataset, achieving outstanding results: a mAP50 of 82.5\% for traffic object detection, a mIoU of 91.5\% for drivable area segmentation, and an IoU of 29.8\% for lane line detection. Experiments confirm our model's balance of efficiency, lightweight design, and accuracy for real-world driving scenarios.

## Proposed CMFANet
<img width="2024" height="990" alt="image" src="https://github.com/user-attachments/assets/0350c0db-f9ab-4e78-b6e5-2c80429de546" />

<img width="2064" height="878" alt="image" src="https://github.com/user-attachments/assets/18101448-7316-4105-9c3e-38f95cce6b82" />
<img width="2042" height="918" alt="image" src="https://github.com/user-attachments/assets/41bfb127-cdd9-4a1c-8cf0-6074701b5044" />


### Training
It will be updated soon.

### Testing demo
```
python test.py
```
Note: The weights [best.pt] of CMFANet(n) can be downloaded in ```/weights/``` folder.

### The complete code will be released soon.

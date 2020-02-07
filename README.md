# Paper_Reading👨‍🎓

#### If it has a problem, I 'll delete!

---

### S^3 DNN
Supervised Streaming and Scheduling for GPU - Accelerated Real-Time DNN Workloads

**Abstract:**

Deep Neural Networks (DNNs) are being widely applied in many advanced embedded systems that require autonomous decision making, e.g., autonomous driving and robotics. To handle resource-demanding DNN workloads, graphic processing units (GPUs) have been used as the main acceleration engine. Although much research has been conducted to algorithmically optimize the efficiency of applying DNN to applications such as object recognition, limited attention has been given to optimizing the execution of GPU-accelerated DNN workloads at the system level. In this paper, we propose S^3DNN, a system solution that optimizes the execution of DNN workloads on GPU in a real-time multi-tasking environment, which simultaneously optimizes the two (sometimes) conflicting goals of real-time correctness and throughput. S^3DNN contains a governor that selectively gathers system-wide DNN requests to perform smart data fusion, and a novel supervised streaming and scheduling framework that combines a deadline-aware scheduler with the concurrency-enabled CUDA stream technique. To simultaneously maximize concurrency-induced benefits and real-time performance, S^3DNN explores a rather interesting and unique characteristic of DNN workloads, where multiple layers of a DNN instance often exhibit a gradually decreased GPU resource utilization pattern. We have fully implemented S^3DNN in a GPU-accelerated system and have conducted extensive sets of experiments evaluating the efficacy of S^3DNN under a wide range of system and workload scenarios. The results show that S^3DNN significantly improves upon state-of-the-art GPU-accelerated DNN processing frameworks, e.g., up to 37% and over 40% improvements in real-time performance and throughput, respectively.

---

### YOLOv3
An Incremental Improvement 

**Abstract:**

We present some updates to YOLO! We made a bunch of little design changes to make it better. We also trained this new network that’s pretty swell. It’s a little bigger than last time but more accurate. It’s still fast though, don’t worry. At 320 x 320 YOLOv3 runs in 22 ms at 28.2 mAP, as accurate as SSD but three times faster. When we look at the old .5 IOU mAP detection metric YOLOv3 is quite good. It achieves 57.9 AP50 in 51 ms on a Titan X, com- pared to 57.5 AP50 in 198 ms by RetinaNet, similar perfor- mance but 3.8⇥ faster. As always, all the code is online at https://pjreddie.com/yolo/.

* 1 Stage Detector
* 단 하나의 네트워크가 한 번에 객체의 위치와 분류가 동시에 이루어진다. 중대한 localization error (작은 BB와 큰 BB를 동일하게 처리하기 때문), 앵커박스의 개념 도입으로부터 사전에 좋은 앵커 박스를 미리 지정해줘야한다. 
* 바운딩 박스의 형태가 트레이닝 데이터를 통해서만 학습되므로, 새로운 / 독특한 형태의 바운딩 박스인 경우 정확하게 예측하지 못한다. 겹쳐진 사물의 구분은 어렵다.
* 7x7 그리드로 영상을 나눈뒤, 각 그리드에서 중심을 그리드 안쪽으로 하면서 크기가 일정하지 않은 경계박스를 2개씩 생성.
* 각 박스는 하나의 box confidence score를 가지고 있다. 
* 한 그리드 셀이 하나의 클래스만 예측할 수 있다. 작은 객체 여러개가 다닥다닥 붙으면 정확하게 예측할 수 없다. 
* YOLO는 객체탐지를 회귀 문제로 접근하며 별도의 지역 제안을 위한 구조 없이 한번에 전체 이미지로부터 어떤 객체들이 어디에 위치하고 있는지 예측할 수 있다. 
* YOLO는 모든 클래스에 대한 모든 바운딩 박스를 동시에 예측한다.
* NMS - confidence score순으로 예측을 정렬. 예측 중에서 신뢰도가 가장 큰 것 하나만 남기고 나머지는 모두 지운다. 임계값 미만도 무시
* YOLO-9000 : COCO 데이터셋과 ImageNet 데이터셋을 Joint 하여 기존의 Object Detection을 다루는 모델들 보다 많은 데이터를 활용하겠다!
* YOLO-v2는 모델의 마지막 부분 Fully Connected Layer를 Convolution Layer로 바꿈
* YOLOv3 : 기존의 YOLO의 recall 문제를 해결하겠다. recall 이란 기존의 YOLO에서 recall 이란 detection rate와 일치. 사전에 정의된 Anchor Box를 시므도이드 함수를 통하여 BB 를 다시 찾아내고 가장 GT와 가까운 값을 Confidence 를 1로 준다. 

![img](https://blogfiles.pstatic.net/MjAxNzA0MjhfMTgy/MDAxNDkzMzYyNDk1MTE5.sJhub9RA2DgRz3-ziXYL-UfX1VcnPcpdxqzYoWnyTk4g.RtfPbkK1GajeoYLlXBpxotv6KZE_an22ns7C1OHlq00g.PNG.sogangori/last0.PNG?type=w1)

> 그림) 네트워크 예측

왼쪽 빨간점으로 표시한 부분은 7x7 그리드셀중에 하나로 이미지에서 개의 중앙 부분에 해당한다.

그리고 빨간색 박스보다 큰 **노란색 박스**가 바로 빨간색 그리드셀에서 예측한 경계 박스이다.

7x7 은 영상을 7x7 의 격자로 나눈것이다.

30개의 채널은 (경계 박스의 정보 4개 , 경계 박스안에 오브젝트가 있을 확률(confidence)) x 2, 어떤 클래스일 확률 20개 로 구성된다.

경계 박스 정보 x, y : 노란색 경계 박스의 중심이 빨간 격자 셀의 중심에서 어디에 있는가.

경계 박스 정보 w,h : 노란색 경계 박스의 가로 세로 길이가 전체 이미지 크기에 어느 정도 크기를 갖는가

만약 경계박스가 위의 그림처럼 되었다면 x,y는 모두 0.5 정도이고 w,h는 각각  2/7, 4/7 정도가 될 것이다.

노란색 경계 박스는 반드시 그 중심이 빨간 그리드 셀 안에 있어야 하며, 가로와 세로길이는 빨간 그리드 셀보다 작을 수도 있고 그림처럼 클 수도 있다.

또한 정사각형일 필요도 없다.

빨간 그리드 셀 내부 어딘가를 중심으로 하여 근처에 있는 어떤 오브젝트를 둘러싸는 직사각형의 노란색 경계 박스를 그리는 것이 목표이다.

노란색 경계 박스가 바로 ROI , 오브젝트 후보이다.

이것을 2개 만든다.

Reference : [PR-207: YOLOv3: An Incremental Improvement](https://www.youtube.com/watch?v=HMgcvgRrDcA&list=PLqAFpvtCnrySi60YxMXf45YAyY9X24hLO&index=2)



---

### Mask R-CNN

**Abstract:**

We present a conceptually simple, flexible, and general framework for object instance segmentation. Our approach
efficiently detects objects in an image while simultaneously generating a high-quality segmentation mask for each in- stance. The method, called Mask R-CNN, extends Faster R-CNN by adding a branch for predicting an object mask in* parallel *with the existing branch for bounding box recogni- tion. Mask R-CNN is simple to train and adds only a small overhead to Faster R-CNN, running at 5 fps. Moreover, Mask R-CNN is easy to generalize to other tasks,* e.g*., al- lowing us to estimate human poses in the same framework. We show top results in all three tracks of the COCO suite of challenges, including instance segmentation, bounding-box object detection, and person keypoint detection. Without tricks, Mask R-CNN outperforms all existing, single-model entries on every task, including the COCO 2016 challenge winners. We hope our simple and effective approach will serve as a solid baseline and help ease future research in instance-level recognition. Code will be made available.

Reference : [PR-057: Mask R-CNN](https://www.youtube.com/watch?v=RtSZALC9DlU&t=881s)

---

### CornerNet

**Abstract:**

We propose CornerNet, a new approach to object detection where we detect an object bounding box as a pair of keypoints, the top-left corner and the bottom-right corner, using a single convolution neural network. By detecting objects as paired keypoints, we eliminate the need for designing a set of anchor boxes commonly used in prior single-stage detectors. In addi- tion to our novel formulation, we introduce corner pool- ing, a new type of pooling layer that helps the network better localize corners. Experiments show that Corner- Net achieves a 42.2% AP on MS COCO, outperforming all existing one-stage detectors.

Reference : [PR-146: CornerNet](https://www.youtube.com/watch?v=6OYmOtivQY8&t=1433s)

---

### Fully Convolutional Networks for Semantic Segmentation 

**Abstract:**

Convolutional networks are powerful visual models that yield hierarchies of features. We show that convolu- tional networks by themselves, trained end-to-end, pixels- to-pixels, exceed the state-of-the-art in semantic segmen- tation. Our key insight is to build “fully convolutional” networks that take input of arbitrary size and produce correspondingly-sized output with efficient inference and learning. We define and detail the space of fully convolu- tional networks, explain their application to spatially dense prediction tasks, and draw connections to prior models. We adapt contemporary classification networks (AlexNet [22], the VGG net [34], and GoogLeNet [35]) into fully convolu- tional networks and transfer their learned representations by fine-tuning [5] to the segmentation task. We then define a skip architecture that combines semantic information from a deep, coarse layer with appearance information from a shallow, fine layer to produce accurate and detailed seg- mentations. Our fully convolutional network achieves state- of-the-art segmentation of PASCAL VOC (20% relative im- provement to 62.2% mean IU on 2012), NYUDv2, and SIFT Flow, while inference takes less than one fifth of a second for a typical image.

Reference: [Fully Convolutional Networks for Semantic Segmentation - 허다운](https://www.youtube.com/watch?v=_52dopGu3Cw&t=1112s)

---

### A Fully Automated system Using A Convolutional Neural Network to predict Renal Allograft Rejection: extra-validation with Giga-pixel Immunostained slides

**Abstract:**

Pathologic diagnoses mainly depend on visual scoring by pathologists, a process that can be time- consuming, laborious, and susceptible to inter- and/or intra-observer variations. This study proposes
a novel method to enhance pathologic scoring of renal allograft rejection. A fully automated system using a convolutional neural network (CNN) was developed to identify regions of interest (RoIs) and
to detect C4d positive and negative peritubular capillaries (PTCs) in giga-pixel immunostained slides. the performance of faster R-CNN was evaluated using optimal parameters of the novel method to enlarge the size of labeled masks. Fifty and forty pixels of the enlarged size images showed the best performance in detecting C4d positive and negative PTCs, respectively. Additionally, the feasibility
of deep-learning-assisted labeling as independent dataset to enhance detection in this model was evaluated. Based on these two CNN methods, a fully automated system for renal allograft rejection was developed. This system was highly reliable, efficient, and effective, making it applicable to real clinical workflow.

Reference: [A Fully Automated system Using A Convolutional Neural Network to predict Renal Allograft Rejection](https://www.nature.com/articles/s41598-019-41479-5)

---

### CT Image Conversion among Different Reconstruction Kernels without a Sinogram by Using a Convolutional Neural Network

**Abstract:**

Objective: The aim of our study was to develop and validate a convolutional neural network (CNN) architecture to convert CT images reconstructed with one kernel to images with different reconstruction kernels without using a sinogram.
Materials and Methods: This retrospective study was approved by the Institutional Review Board. Ten chest CT scans were performed and reconstructed with the B10f, B30f, B50f, and B70f kernels. The dataset was divided into six, two, and two examinations for training, validation, and testing, respectively. We constructed a CNN architecture consisting of six convolutional layers, each with a 3 x 3 kernel with 64 filter banks. Quantitative performance was evaluated using root mean square error (RMSE) values. To validate clinical use, image conversion was conducted on 30 additional chest CT scans reconstructed with the B30f and B50f kernels. The influence of image conversion on emphysema quantification was assessed with Bland–Altman plots. Results: Our scheme rapidly generated conversion results at the rate of 0.065 s/slice. Substantial reduction in RMSE was observed in the converted images in comparison with the original images with different kernels (mean reduction, 65.7%; range, 29.5–82.2%). The mean emphysema indices for B30f, B50f, converted B30f, and converted B50f were 5.4 ± 7.2%, 15.3 ± 7.2%, 5.9 ± 7.3%, and 16.8 ± 7.5%, respectively. The 95% limits of agreement between B30f and other kernels (B50f and converted B30f) ranged from -14.1% to -2.6% (mean, -8.3%) and -2.3% to 0.7% (mean, -0.8%), respectively. Conclusion: CNN-based CT kernel conversion shows adequate performance with high accuracy and speed, indicating its potential clinical use.

---

### DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs
**Abstract:**

In this work we address the task of semantic image segmentation with Deep Learning and make three main contributions that are experimentally shown to have substantial practical merit. First, we highlight convolution with upsampled filters, or
‘atrous convolution’, as a powerful tool in dense prediction tasks. Atrous convolution allows us to explicitly control the resolution at which feature responses are computed within Deep Convolutional Neural Networks. It also allows us to effectively enlarge the field of view of filters to incorporate larger context without increasing the number of parameters or the amount of computation. Second, we propose atrous spatial pyramid pooling (ASPP) to robustly segment objects at multiple scales. ASPP probes an incoming convolutional feature layer with filters at multiple sampling rates and effective fields-of-views, thus capturing objects as well as image context at multiple scales. Third, we improve the localization of object boundaries by combining methods from DCNNs and probabilistic graphical models. The commonly deployed combination of max-pooling and downsampling in DCNNs achieves invariance but has a toll on localization accuracy. We overcome this by combining the responses at the final DCNN layer with a fully connected Conditional Random Field (CRF), which is shown both qualitatively and quantitatively to improve localization performance. Our proposed “DeepLab” system sets the new state-of-art at the PASCAL VOC-2012 semantic image segmentation task, reaching 79.7% mIOU in the test set, and advances the results on three other datasets: PASCAL-Context, PASCAL-Person-Part, and Cityscapes. All of our code is made publicly available online.

---

### U-Net: Convolutional Networks for Biomedical Image Segmentation 

* contracting path -> expanding path = encoding -> decoding
* Biomedical image Segmentation에서 쓰이기 위해서 사용
* 단순한 이미지를 분류하는 문제를 넘어서 이미지의 특정 영역을 label로 표현하는 image Segementation에 목적이 있다.
* Sliding window를 쓰지않고 patch방식을 채택. 이미지 전체를 격자로 잘라서 한번에 인식
* 기존은 Context 와 localization의 tradeoff였지만 Patch를 사용하여 해결
* Patch - 이미지 인식 단위
  <img width="657" alt="스크린샷 2020-02-04 오후 7 26 23" src="https://user-images.githubusercontent.com/46750574/73739746-2d57a900-478a-11ea-9145-720a3e80d71d.png">
* 매 contracting 마다 각 max pool 하기 전 레이어의 결과값을 우측의 같은 대응되는 크기의 output 필터에 concat시킴. 왼쪽 이미지가 더 크므로 resize해줌. Mirroring padding을 진행할때 손실되는 path를 살리기 위해서 보상처리 해줌
* contracting path에서 padding이 없었기 때문에 점점 이미지 외곽 부분이 없어짐
* 이미지가 단순 작아진게 아니라 외곽 부분이 잘려나감. 그래서 mirroring으로 이미지 복구
* 내려갈때는 1/2 down sampling , 활성화 함수는 ReLU

<img width="557" alt="스크린샷 2020-02-04 오후 7 26 51" src="https://user-images.githubusercontent.com/46750574/73739753-2df03f80-478a-11ea-9ed5-0f6e7117257a.png">

* w라는 가중치를 정답 픽셀에 대한 cross entropy loss에 추가
* d1은 가장 가까운 셀, d2는 두번째로 가까운 셀
* 두 세포사이의 간격이 좁을 수록 weight를 큰 값으로 두 세포 사이가 넓을 수록 weight를 작은 값으로 갖게 된다. 
* 간격이 넓어질 수록 loss 가 작아진다.
* 각 cell들이 만나는 경계부분에서의 가중치를 더 높게 주어서 경계부분을 확실하게 구분해 내겠다는 것.
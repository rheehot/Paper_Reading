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


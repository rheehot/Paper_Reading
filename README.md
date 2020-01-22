# Paper_Reading
## 👨‍🎓 I 'll delete, if it has a problem!

---

### S^3 DNN
Supervised Streaming and Scheduling for GPU - Accelerated Real-Time DNN Workloads

**Abstract:**

###### Deep Neural Networks (DNNs) are being widely applied in many advanced embedded systems that require autonomous decision making, e.g., autonomous driving and robotics. To handle resource-demanding DNN workloads, graphic processing units (GPUs) have been used as the main acceleration engine. Although much research has been conducted to algorithmically optimize the efficiency of applying DNN to applications such as object recognition, limited attention has been given to optimizing the execution of GPU-accelerated DNN workloads at the system level. In this paper, we propose S^3DNN, a system solution that optimizes the execution of DNN workloads on GPU in a real-time multi-tasking environment, which simultaneously optimizes the two (sometimes) conflicting goals of real-time correctness and throughput. S^3DNN contains a governor that selectively gathers system-wide DNN requests to perform smart data fusion, and a novel supervised streaming and scheduling framework that combines a deadline-aware scheduler with the concurrency-enabled CUDA stream technique. To simultaneously maximize concurrency-induced benefits and real-time performance, S^3DNN explores a rather interesting and unique characteristic of DNN workloads, where multiple layers of a DNN instance often exhibit a gradually decreased GPU resource utilization pattern. We have fully implemented S^3DNN in a GPU-accelerated system and have conducted extensive sets of experiments evaluating the efficacy of S^3DNN under a wide range of system and workload scenarios. The results show that S^3DNN significantly improves upon state-of-the-art GPU-accelerated DNN processing frameworks, e.g., up to 37% and over 40% improvements in real-time performance and throughput, respectively.

---

### YOLOv3
An Incremental Improvement 

**Abstract:**

###### We present some updates to YOLO! We made a bunch of little design changes to make it better. We also trained this new network that’s pretty swell. It’s a little bigger than last time but more accurate. It’s still fast though, don’t worry. At 320 x 320 YOLOv3 runs in 22 ms at 28.2 mAP, as accurate as SSD but three times faster. When we look at the old .5 IOU mAP detection metric YOLOv3 is quite good. It achieves 57.9 AP50 in 51 ms on a Titan X, com- pared to 57.5 AP50 in 198 ms by RetinaNet, similar perfor- mance but 3.8⇥ faster. As always, all the code is online at https://pjreddie.com/yolo/.

Reference : [PR-207: YOLOv3: An Incremental Improvement](https://www.youtube.com/watch?v=HMgcvgRrDcA&list=PLqAFpvtCnrySi60YxMXf45YAyY9X24hLO&index=2)

---

### Mask R-CNN

**Abstract:**

*We present a conceptually simple, flexible, and general framework for object instance segmentation. Our approach* 
*efficiently detects objects in an image while simultaneously generating a high-quality segmentation mask for each in- stance. The method, called Mask R-CNN, extends Faster R-CNN by adding a branch for predicting an object mask in* parallel *with the existing branch for bounding box recogni- tion. Mask R-CNN is simple to train and adds only a small overhead to Faster R-CNN, running at 5 fps. Moreover, Mask R-CNN is easy to generalize to other tasks,* e.g*., al- lowing us to estimate human poses in the same framework. We show top results in all three tracks of the COCO suite of challenges, including instance segmentation, bounding-box object detection, and person keypoint detection. Without tricks, Mask R-CNN outperforms all existing, single-model entries on every task, including the COCO 2016 challenge winners. We hope our simple and effective approach will serve as a solid baseline and help ease future research in instance-level recognition. Code will be made available.* 

Reference : [PR-057: Mask R-CNN](https://www.youtube.com/watch?v=RtSZALC9DlU&t=881s)
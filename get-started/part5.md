---
title: "Prediction, Image"
keywords: classification, detection, segmentation, extraction, image, deep,  keras, concepts
description: Learn how to create a multi-container application that uses all the machines in a cluster.
---

{% include_relative nav.html selected="5" %}
#### General Packages

* [Open CV](http://opencv.org/)
* [mexopencv](http://kyamagu.github.io/mexopencv/)
* [SimpleCV](http://simplecv.org/)

  ​

#### Feature Detection and Extraction
* [VLFeat](http://www.vlfeat.org/)
* [SIFT](http://www.cs.ubc.ca/~lowe/keypoints/)
  * David G. Lowe, "Distinctive image features from scale-invariant keypoints," International Journal of Computer Vision, 60, 2 (2004), pp. 91-110.
* [SIFT++](http://www.robots.ox.ac.uk/~vedaldi/code/siftpp.html)
* [BRISK](http://www.asl.ethz.ch/people/lestefan/personal/BRISK)
  * Stefan Leutenegger, Margarita Chli and Roland Siegwart, "BRISK: Binary Robust Invariant Scalable Keypoints", ICCV 2011
* [SURF](http://www.vision.ee.ethz.ch/~surf/)
  * Herbert Bay, Andreas Ess, Tinne Tuytelaars, Luc Van Gool, "SURF: Speeded Up Robust Features", Computer Vision and Image Understanding (CVIU), Vol. 110, No. 3, pp. 346--359, 2008
* [FREAK](http://www.ivpe.com/freak.htm)
  * A. Alahi, R. Ortiz, and P. Vandergheynst, "FREAK: Fast Retina Keypoint", CVPR 2012
* [AKAZE](http://www.robesafe.com/personal/pablo.alcantarilla/kaze.html)
  * Pablo F. Alcantarilla, Adrien Bartoli and Andrew J. Davison, "KAZE Features", ECCV 2012
* [Local Binary Patterns](https://github.com/nourani/LBP)

#### High Dynamic Range Imaging
* [HDR_Toolbox](https://github.com/banterle/HDR_Toolbox)

#### Semantic Segmentation
* [List of Semantic Segmentation algorithms](http://www.it-caesar.com/list-of-contemporary-semantic-segmentation-datasets/)

#### Low-level Vision

###### Stereo Vision
 * [Middlebury Stereo Vision](http://vision.middlebury.edu/stereo/)
 * [The KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stero)
 * [LIBELAS: Library for Efficient Large-scale Stereo Matching](http://www.cvlibs.net/software/libelas/)
 * [Ground Truth Stixel Dataset](http://www.6d-vision.com/ground-truth-stixel-dataset)

###### Optical Flow
 * [Middlebury Optical Flow Evaluation](http://vision.middlebury.edu/flow/)
 * [MPI-Sintel Optical Flow Dataset and Evaluation](http://sintel.is.tue.mpg.de/)
 * [The KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=flow)
 * [HCI Challenge](http://hci.iwr.uni-heidelberg.de/Benchmarks/document/Challenging_Data_for_Stereo_and_Optical_Flow/)
 * [Coarse2Fine Optical Flow](http://people.csail.mit.edu/celiu/OpticalFlow/) - Ce Liu (MIT)
 * [Secrets of Optical Flow Estimation and Their Principles](http://cs.brown.edu/~dqsun/code/cvpr10_flow_code.zip)
 * [C++/MatLab Optical Flow by C. Liu (based on Brox et al. and Bruhn et al.)](http://people.csail.mit.edu/celiu/OpticalFlow/)
 * [Parallel Robust Optical Flow by Sánchez Pérez et al.](http://www.ctim.es/research_works/parallel_robust_optical_flow/)

###### Image Denoising
BM3D, KSVD,

#### Material Recognition
 * [OpenSurface](http://opensurfaces.cs.cornell.edu/)
 * [Flickr Material Database](http://people.csail.mit.edu/celiu/CVPR2010/)
 * [Materials in Context Dataset](http://opensurfaces.cs.cornell.edu/publications/minc/)



#### Visual Tracking
 * [Visual Tracker Benchmark](https://sites.google.com/site/trackerbenchmark/benchmarks/v10)
 * [Visual Tracker Benchmark v1.1](https://sites.google.com/site/benchmarkpami/)
 * [VOT Challenge](http://www.votchallenge.net/)
 * [Princeton Tracking Benchmark](http://tracking.cs.princeton.edu/)
 * [Tracking Manipulation Tasks (TMT)](http://webdocs.cs.ualberta.ca/~vis/trackDB/)

#### Visual Surveillance
 * [VIRAT](http://www.viratdata.org/)
 * [CAM2](https://cam2.ecn.purdue.edu/)

#### Saliency Detection

#### Change detection
 * [ChangeDetection.net](http://changedetection.net/)

#### Visual Recognition

###### Image Classification
 * [The PASCAL Visual Object Classes](http://pascallin.ecs.soton.ac.uk/challenges/VOC/)
 * [ImageNet Large Scale Visual Recognition Challenge](http://www.image-net.org/challenges/LSVRC/2014/)

###### Scene Recognition
 * [SUN Database](http://groups.csail.mit.edu/vision/SUN/)
 * [Place Dataset](http://places.csail.mit.edu/)

###### Object Detection
 * [The PASCAL Visual Object Classes](http://pascallin.ecs.soton.ac.uk/challenges/VOC/)
 * [ImageNet Object Detection Challenge](http://www.image-net.org/challenges/LSVRC/2014/)
 * [Microsoft COCO](http://mscoco.org/)

###### Semantic labeling
 * [Stanford background dataset](http://dags.stanford.edu/projects/scenedataset.html)
 * [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)
 * [Barcelona Dataset](http://www.cs.unc.edu/~jtighe/Papers/ECCV10/)
 * [SIFT Flow Dataset](http://www.cs.unc.edu/~jtighe/Papers/ECCV10/siftflow/SiftFlowDataset.zip)

###### Multi-view Object Detection
 * [3D Object Dataset](http://cvgl.stanford.edu/resources.html)
 * [EPFL Car Dataset](http://cvlab.epfl.ch/data/pose)
 * [KTTI Dection Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php)
 * [SUN 3D Dataset](http://sun3d.cs.princeton.edu/)
 * [PASCAL 3D+](http://cvgl.stanford.edu/projects/pascal3d.html)
 * [NYU Car Dataset](http://nyc3d.cs.cornell.edu/)

###### Fine-grained Visual Recognition
 * [Fine-grained Classification Challenge](https://sites.google.com/site/fgcomp2013/)
 * [Caltech-UCSD Birds 200](http://www.vision.caltech.edu/visipedia/CUB-200.html)

###### Pedestrian Detection
 * [Caltech Pedestrian Detection Benchmark](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/)
 * [ETHZ Pedestrian Detection](https://data.vision.ee.ethz.ch/cvl/aess/dataset/)

[On to Part 6 >>](part6.md){: class="button outline-btn"}

# Optical Flow Regularized Dynamic Gaussian Splatting

## CVPR 2025

<!-- [Project Page](https://oppo-us-research.github.io/SpacetimeGaussians-website/) | [Paper](https://arxiv.org/abs/2312.16812) | [Video](https://youtu.be/YsPPmf-E6Lg) | [Viewer & Pre-Trained Models](https://huggingface.co/stack93/spacetimegaussians/tree/main) -->


This is an official implementation of the paper "Optical Flow Regularized Dynamic Gaussian Splatting".</br>
<!-- [Zhan Li](https://lizhan17.github.io/web/)<sup>1,2</sup>, 
[Zhang Chen](https://zhangchen8.github.io/)<sup>1,&dagger;</sup>, 
[Zhong Li](https://sites.google.com/site/lizhong19900216)<sup>1,&dagger;</sup>, 
[Yi Xu](https://www.linkedin.com/in/yi-xu-42654823/)<sup>1</sup> </br>
<sup>1</sup> OPPO US Research Center, <sup>2</sup> Portland State University </br>
<sup>&dagger;</sup> Corresponding authors </br>

<img src="assets/output.gif" width="100%"/></br> -->

<!-- ## Updates and News
- `Jun 16, 2024`: Added fully fused mlp for testing ours-full models on Technicolor and Neural 3D dataset (40 FPS improvement compared to paper).
- `Jun 13, 2024`: Fixed minors for reproducity on the scenes ```coffee_martini``` and ```flame_salmon_1``` (~ 0.1 PSNR).
- `Jun 9, 2024` : Supported lazy loading and ground truth image as int8 in GPU.
- `Dec 28, 2023`: Paper and Code are released. -->



## Table of Contents
1. [Installation](#installation)
1. [Preprocess Datasets](#processing-datasets)
1. [Training](#training)
1. [Testing](#testing)
1. [License Infomration](#license-information)
1. [Acknowledgement](#acknowledgement)
1. [Citations](#citations)


## Installation
<!-- ### Windows users with WSL2 :
Please first refer to [here](./script/wsl.md) to install the WSL2 system (Windows Subsystem for Linux 2) and install dependencies inside WSL2. Then you can set up our repo inside the Linux sub-system same as other Linux users.  -->
### Linux users :
Clone the source code of this repo.
<!-- ```
git clone https://github.com/oppo-us-research/SpacetimeGaussians.git
cd SpacetimeGaussians
``` -->
```
```

Then run the following command to install the environments with conda.
Note we will create two environments, one for preprocessing with colmap (```colmap_env```) and one for training and testing (```flowreg_dgs```). Training, testing and preprocessing have been tested on Ubuntu 22.04. </br>
Run the following command to set up the ```flowreg_dgs``` environment. 
```
source script/setup.sh
```
Note that you may need to manually install the following packages if you encounter errors during the installation of the above command. </br>

```
conda activate flowreg_dgs
pip install thirdparty/gaussian_splatting/submodules/gaussian_rasterization_ch3
pip install thirdparty/gaussian_splatting/submodules/forward_lite
```

## Processing Datasets
Note, the baseline paper uses the sparse points that follow 3DGS. It's per frame SfM points only use ```point_triangulator``` in Colmap instead of dense points.  
### Neural 3D Dataset
Download the dataset from [here](https://github.com/facebookresearch/Neural_3D_Video.git).
After downloading the dataset, you can run the following command to preprocess the dataset. </br>
```
conda activate colmap_env
python script/pre_n3d.py --videopath <location>/<scene>
```
```<location>``` is the path to the dataset root folder, and ```<scene>``` is the name of a scene in the dataset. </br>

- For example if you put the dataset at ```/data/dynerf```, and want to preprocess the ```cook_spinach``` scene, you can run the following command
```
conda activate colmap_env
python script/pre_n3d.py --videopath /data/dynerf/cook_spinach/
```

Our codebase expects the following directory structure for the Neural 3D (DyNeRF) Dataset after preprocessing:
```

<location>
|---cook_spinach
|   |---colmap_<0>
|   |---colmap_<...>
|   |---colmap_<50>
|---flame_salmon1

```
<!-- ### Technicolor Dataset
Please reach out to the authors of the paper "Dataset and Pipeline for Multi-View Light-Field Video" for access to the Technicolor dataset. </br>
Our codebase expects the following directory structure for this dataset before preprocessing:
```

<location>
|---Fabien
|   |---Fabien_undist_<00257>_<08>.png
|   |---Fabien_undist_<.....>_<..>.png
|---Birthday

```
Then run the following command to preprocess the dataset. </br>
```
conda activate colmapenv
python script/pre_technicolor.py --videopath <location>/<scene>
```
### Google Immersive Dataset 
Download the dataset from [here](https://github.com/augmentedperception/deepview_video_dataset).
After downloading and unzip the dataset, you can run the following command to preprocess the dataset. </br>
```
conda activate colmapenv
python script/pre_immersive_distorted.py --videopath <location>/<scene>
python script/pre_immersive_undistorted.py --videopath <location>/<scene>
```
```<location>``` is the path to the dataset root folder, and ```<scene>``` is the name of a scene in the dataset. Please rename the orginal file to the name list ```Immersiveseven```in [here](./script/pre_immersive_distorted.py) 

- For example if you put the dataset at ```/home/immersive```, and want to preprocess the ```02_Flames``` scene, you can run the following command
```
conda activate colmapenv
python script/pre_immersive_distorted.py --videopath /home/immersive/02_Flames/
```



1. Our codebase expects the following directory structure for immersive dataset before preprocessing
```
<location>
|---02_Flames
|   |---camera_0001.mp4
|   |---camera_0002.mp4
|---09_Alexa
```

2. Our codebase expects the following directory structure for immersive dataset (raw video, decoded images, distorted and undistorted) after preprocessing:

```
<location>
|---02_Flames
|   |---camera_0001
|   |---camera_0001.mp4
|   |---camera_<...>
|---02_Flames_dist
|   |---colmap_<0>
|   |---colmap_<...>
|   |---colmap_<299>
|---02_Flames_undist
|   |---colmap_<0>
|   |---colmap_<...>
|   |---colmap_<299>
|---09_Alexa
|---09_Alexa_dist
|---09_Alexa_undist
```

3. Copy the picked views files to the scene dir. The views is generated by inferencing our model initialized with ```duration=1``` points without training. We provide generated views in pkl for reproducity and simplicity. 
- For example, for the scene ```09_Alexa``` with distortion model.
copy ```configs/im_view/09_Alexa/pickview.pkl``` to ```<location>/09_Alexa_dist/pickview.pkl```
 -->


## Training
You can train our model by running the following command: </br>

```
conda activate flowreg_dgs
python train_ours.py --quiet --eval --config configs/n3d_lite/<scene>.json --model_path <path to save model> --source_path data/dynerf/<scene>/colmap_0
```

You need 24GB GPU memory to train on the Neural 3D Dataset.
In each iteration, the images selected are loaded into memory and flushed after their usage, although memory still accumulates overtime.</br>
- For example, if you want to train the **lite** model on the first 50 frames of the ```cook_spinach``` scene in the Neural 3D Dataset, you can run the following command: </br>
```
python train_ours.py --quiet --eval --config configs/n3d_lite/cook_spinach.json --model_path output/dynerf/cook_spinach_lite --source_path data/dynerf/cook_spinach/colmap_0 
```

<!-- - If you want to train the **full** model, you can run the following command </br>

```
python train.py --quiet --eval --config configs/n3d_full/cook_spinach.json --model_path log/cook_spinach/colmap_0 --source_path <location>/cook_spinach/colmap_0 
``` -->
Please refer to the .json config files for more options. We trained each model on 12000 iterations on first 50-frames of DyNeRF dataset. Make sure the resolution of each image in dataset is set to 1352 x 1014 before training. In the config files we set ```resolution = 1``` while ```--downscale``` is set to 2 in ```script/pre_n3d.py``` for preprocessing DyNeRF data. If you keep our settings, your images should have scale scale 1352 x 1014.

<!-- 
- If you want to train the **full** model with **distorted** immersive dataset, you can run the following command </br>

```
PYTHONDONTWRITEBYTECODE=1 python train_imdist.py --quiet --eval --config configs/im_distort_full/02_Flames.json --model_path log/02_Flames/colmap_0 --source_path <location>/02_Flames_dist/colmap_0 
```

Note, sometimes pycache file somehow affects the results. Please remove every pycache file and retrain the model without generating BYTECODE by ```PYTHONDONTWRITEBYTECODE=1```.



- If you want to train the **lite** model with **undistorted** immersive dataset.   
Note, we remove the ```--eval``` to reuse the loader of technicolor and also to train with all cameras.  ```maskgt 1``` is specially for training with undistorted fisheye images that have black pixels.

```
python train.py --quiet --maskgt 1 --config configs/im_undistort_lite/02_Flames.json --model_path log/02_Flames/colmap_0 --source_path <location>/02_Flames_undist/colmap_0 
```

Please refer to the .json config files for more options. -->


## Testing

- Test model on Neural 3D Dataset

```
python test_ours.py --quiet --eval --skip_train --valloader colmapvalid --configpath configs/n3d_lite/<scene>.json --model_path <path to model> --source_path data/dynerf/<scene>/colmap_0
```

- For testing ```cook_spinach``` scene of DyNeRF dataset on both train and test data, please use the following command:
```
python test_ours.py --quiet --eval --valloader colmapvalid --configpath configs/n3d_lite/cook_spinach.json --model_path output/dynerf/cook_spinach_lite --source_path data/dynerf/cook_spinach/colmap_0
```

Make sure ```test_iteration = 12000``` in .json files for testing over 12000 iterations. The testing data for DyNeRF is camera viewpoint ```cam00``` while the rest of viewpointset belongs to train data. In our case, inference over train data is done only on ```cam09```. If you want to do inference over all train/test data kindly comment out the line ```if cam.image_name == 'cam09':``` in ```test_ours.py```.

<!-- - Test model on Technicolor Dataset
```
python test.py --quiet --eval --skip_train --valloader technicolorvalid --configpath config/techni_<lite|full>/<scene>.json --model_path <path to model> --source_path <location>/<scenename>/colmap_0
```
- Test on Google Immersive Dataset with distortion camera model 

Fist Install fused mlp layer.
```
pip install thirdparty/gaussian_splatting/submodules/forward_full
```

```
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 python test.py --quiet --eval --skip_train --valloader immersivevalidss --configpath config/im_distort_<lite|full>/<scene>.json --model_path <path to model> --source_path <location>/<scenename>/colmap_0
``` -->


<!-- ## Real-Time Viewer 
The viewer is based on [SIBR](https://sibr.gitlabpages.inria.fr/) and [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting). 
### Pre-built Windows Binary
Download the viewer binary from [this link](https://huggingface.co/stack93/spacetimegaussians/tree/main) and unzip it. The binary works for Windows with CUDA >= 11.0.
We also provide pre-trained models in the link. For example, [n3d_sear_steak_lite_allcam.zip](https://huggingface.co/stack93/spacetimegaussians/blob/main/n3d_sear_steak_lite_allcam.zip) contains the lite model that uses all views during training for the sear_steak scene in the Neural 3D Dataset.
### Installation from Source 
please see bottom commented text [this link](./script/setup.sh)
### Running the Real-Time Viewer
After downloading the pre-built binary or installing from source, you can use the following command to run the real-time viewer. Adjust ```--iteration``` to match the training iterations of model. </br>
```
./<SIBR install dir>/bin/SIBR_gaussianViewer_app_rwdi.exe --iteration 25000 -m <path to trained model> 
``` 
The above command has beed tested on Nvidia RTX 3050 Laptop GPU + Windows 10.
- For 8K rendering, you can use the following command. </br>
```
./<SIBR install dir>/bin/SIBR_gaussianViewer_app_rwdi.exe --iteration 25000 --rendering-size 8000 4000 --force-aspect-ratio 1 -m <path to trained model> 
``` 
8K rendering has been tested on Nvidia RTX 4090 + Windows 11. 

### Third Party Implemented Web Viewer 
We thank Kevin Kwok (Antimatter15) for the amazing web viewer of our method: splaTV . The web viewer is released at [github](https://github.com/antimatter15/splaTV).
You can view one of our scene from the [web viewer](http://antimatter15.com/splaTV/).
## Create Your New Representations and Rendering Pipeline
If you want to customize our codebase for your own models, you can refer to the following steps </br>
- Step 1: Create a new Gaussian representation in this [folder](./thirdparty/gaussian_splatting/scene/). You can use ```oursfull.py``` or ```ourslite.py``` as a template. </br>
- Step 2: Create a new rendering pipeline in this [file](./thirdparty/gaussian_splatting/renderer/__init__.py). You can use the ```train_ours_full``` function as a template. </br>
- Step 3 (For new dataset, optional): Create a new dataloader in this [file](./thirdparty/gaussian_splatting/scene/__init__.py) and this [file](./thirdparty/gaussian_splatting/scene/dataset_readers.py). </br>
- Step 4: Update the intermidiate API in ```getmodel``` (for Step 1) and ```getrenderpip``` (for Step 2) functions in ```helper_train.py```.</br> -->


## License Information
The code in this repository (except the thirdparty folder) is licensed under MIT licence, see [LICENSE](LICENSE). thirdparty/gaussian_splatting is licensed under Gaussian-Splatting license, see [thirdparty/gaussian_splatting/LICENSE.md](thirdparty/gaussian_splatting/LICENSE.md). thirdparty/colmap is licensed under new BSD license, see [thirdparty/colmap/LICENSE.txt](thirdparty/colmap/LICENSE.txt).


## Acknowledgement
We sincerely thank the owners of the following source code repos, which are used by our released codes:
[Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting),
[Colmap](https://github.com/colmap/colmap).
Some parts of our code referenced the following repos:
[Gaussian Splatting with Depth](https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth),
[SIBR](https://gitlab.inria.fr/sibr/sibr_core.git), 
[fisheye-distortion](https://github.com/Synthesis-AI-Dev/fisheye-distortion).

We sincerely thank the anonymous reviewers of CVPR2024 for their helpful feedbacks. 


we also thank Michael Rubloff for his post on [radiancefields](https://radiancefields.com/splatv-dynamic-gaussian-splatting-viewer/). 
We also want to thank MrNeRF for [posts](https://x.com/janusch_patas/status/1740621964480217113?s=20) about our paper and other Guassian Splatting based papers. 


## Citations
<!-- Please cite our paper if you find it useful for your research.
```
@InProceedings{Li_STG_2024_CVPR,
    author    = {Li, Zhan and Chen, Zhang and Li, Zhong and Xu, Yi},
    title     = {Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {8508-8520}
}
```

Please also cite the following paper if you use Gaussian Splatting.
```
@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
``` -->

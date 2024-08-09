# Neural Radiance Fields for Novel View Generation
I wanted to learn about NeRF and this repo is my attempt at that. I will be giving a informal narrative of how I understood nerf and carried out a few experiments to test out the model's capability.

# Understanding Nerf and useful resources
The original paper by the authors (Mildenhall, Ben, et al) is the best resource to understand the paper. The readers will benefit from reading about camera coordinate systems and camera transformations which is heavily used in the code base. The orignal codebase by the authors is in tensorflow and there is a pytorch version that is also avavailable.  I have forked a branch of the pytorch version to understand the code base myself and then carried out a few fun experiments.

Coordinate Systems:
* [OpenGL docs for NDC]https://www.songho.ca/opengl/gl_projectionmatrix.html
* [YU Yue's wonderful blog on explanation of coordinates systems used in graphics](https://yconquesty.github.io/blog/ml/nerf/nerf_ndc.html#background)

## COLMAP
COLMAP is an essential part of NeRF data-setup. NeRF requires cameras to be computed for each image using classical bundle adjustment algorithm. COLMAP provides a complete suite of tools for bundle adjustment, SfM and Stereo Matching. Installing COLMAP from source can be tricky for a new user. It is easier to install COLMAP using instructions provided by LLFF authors.

LLFF's tf_colmap docker is the easiest way to install colmap. The cameras can be computed using this docker along with LLFF's imgs2poses.py script. 

---
# Experiments
I collected images for four different object scenes across Purdue University's campus. Here are the output gifs of novel view synthesised by NERF

## Sculpture
![sculpture](./nerf-pytorch/logs/sculpture_test/sculpture_test_spiral_200000_rgb_rot.gif)

## Forklift
![forklift](./nerf-pytorch/logs/forklift_test/forklift_test_spiral_200000_rgb_rot.gif)

## Scissor Lift
![lift](./nerf-pytorch/logs/lift_test/lift_test_spiral_200000_rgb_rot.gif)

## Lion Fountain
![lion fountain](./nerf-pytorch/logs/lion_fountain_test/lion_fountain_test_spiral_200000_rgb.gif)
---

# Citations
- NeRF paper [Mildenhall, Ben, et al. "Nerf: Representing scenes as neural radiance fields for view synthesis." Communications of the ACM 65.1 (2021): 99-106.](https://arxiv.org/abs/2003.08934)
- nerf-pytorch [Yen-Chen, Lin "NeRF-pytorch"](https://github.com/yenchenlin/nerf-pytorch/)
- LLFF repo [Ben Mildenhall and Pratul P. Srinivasan and Rodrigo Ortiz-Cayon and Nima Khademi Kalantari and Ravi Ramamoorthi and Ren Ng and Abhishek Kar "Local Light Field Fusion: Practical View Synthesis with Prescriptive Sampling Guidelines"](https://github.com/Fyusion/LLFF/tree/master)
- COLMAP project page [COLMAP](https://colmap.github.io/)

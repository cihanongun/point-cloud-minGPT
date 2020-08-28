# Point Cloud minGPT
This is a modified version of [minGPT](https://github.com/karpathy/minGPT) for point cloud processing. Please look at the original source for more detailed information about [minGPT](https://github.com/karpathy/minGPT). The Jupyter Notebook `play_PC.ipynb` is used to train [minGPT](https://github.com/karpathy/minGPT) model with point clouds. **All details are in the notebook with comments and markdowns.**

### Data
The "chair" class from [ShapeNet Part](http://web.stanford.edu/~ericyi/project_page/part_annotation/index.html) dataset is used after randomly downsampling 3746 models to 1024 points. Any point cloud dataset in `(Models x Points x 3)` format can be used.

### Training
The code is tested with PyTorch v1.5.0. Training takes around 3 hours on a single RTX 2070.

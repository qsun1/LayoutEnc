## Scene Image Synthesis
![teaser](assets/scene_images_samples.svg)
Scene image generation based on bounding box conditionals as done in our CVPR2021 AI4CC workshop paper [High-Resolution Complex Scene Synthesis with Transformers](https://arxiv.org/abs/2105.06458) (see talk on [workshop page](https://visual.cs.brown.edu/workshops/aicc2021/#awards)). Supporting the datasets COCO and Open Images.

### Training
Download first-stage models [COCO-8k-VQGAN](https://heibox.uni-heidelberg.de/f/78dea9589974474c97c1/) for COCO or [COCO/Open-Images-8k-VQGAN](https://heibox.uni-heidelberg.de/f/461d9a9f4fcf48ab84f4/) for Open Images.
Change `ckpt_path` in `data/coco_scene_images_transformer.yaml` and `data/open_images_scene_images_transformer.yaml` to point to the downloaded first-stage models.
Download the full COCO/OI datasets and adapt `data_path` in the same files, unless working with the 100 files provided for training and validation suits your needs already.

Code can be run with
`python main.py --base configs/coco_scene_images_transformer.yaml -t True --gpus 0,`

`python main.py --base configs/coco_pretrain_stage.yaml -t True --gpus 0,`

### Sampling 
Train a model as described above or download a pre-trained model:
 - [COCO 30 epochs](https://heibox.uni-heidelberg.de/f/0d0b2594e9074c7e9a33/)
 - [COCO 60 epochs](https://drive.google.com/file/d/1bInd49g2YulTJBjU32Awyt5qnzxxG5U9/) (find model statistics for both COCO versions in `assets/coco_scene_images_training.svg`)

When downloading a pre-trained model, remember to change `ckpt_path` in `configs/*project.yaml` to point to your downloaded first-stage model (see ->Training).

Scene image generation can be run with
`python scripts/make_scene_samples.py --outdir=/some/outdir -r /path/to/pretrained/model --resolution=512,512`


## Training on custom data

Training on your own dataset can be beneficial to get better tokens and hence better images for your domain.
Those are the steps to follow to make this work:
1. install the repo with `conda env create -f environment.yaml`, `conda activate taming` and `pip install -e .`
1. put your .jpg files in a folder `your_folder`
2. create 2 text files a `xx_train.txt` and `xx_test.txt` that point to the files in your training and test set respectively (for example `find $(pwd)/your_folder -name "*.jpg" > train.txt`)
3. adapt `configs/custom_vqgan.yaml` to point to these 2 files
4. run `python main.py --base configs/custom_vqgan.yaml -t True --gpus 0,1` to
   train on two GPUs. Use `--gpus 0,` (with a trailing comma) to train on a single GPU.

### COCO
Create a symlink `data/coco` containing the images from the 2017 split in
`train2017` and `val2017`, and their annotations in `annotations`. Files can be
obtained from the [COCO webpage](https://cocodataset.org/). In addition, we use
the [Stuff+thing PNG-style annotations on COCO 2017
trainval](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip)
annotations from [COCO-Stuff](https://github.com/nightrome/cocostuff), which
should be placed under `data/cocostuffthings`.


## BibTeX

```
@misc{esser2020taming,
      title={Taming Transformers for High-Resolution Image Synthesis}, 
      author={Patrick Esser and Robin Rombach and Björn Ommer},
      year={2020},
      eprint={2012.09841},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

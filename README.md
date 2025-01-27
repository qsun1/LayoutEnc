# LayoutEnc: Leveraging Enhanced Layout Representations for Transformer-based Complex Scene Synthesis (TOMM 2025)

### Environment

Following [Taming Transformers](https://github.com/CompVis/taming-transformers), you should create such environment named `layoutenc`

```
conda env create -f environment.yaml
conda activate layoutenc
```

### Training
Download first-stage models [COCO-8k-VQGAN](https://heibox.uni-heidelberg.de/f/78dea9589974474c97c1/).
Change `ckpt_path` in `configs/coco.yaml` to point to the downloaded first-stage models.
Download the full COCO datasets and adapt `data_path` in the same files, unless working with the 100 files provided for training and validation suits your needs already.

Code can be run with
`python main.py --base configs/coco.yaml -t True --gpus 0,`


### Demo (Local)
You only need to run such script, have fun!
```
python launch_gradio_app.py
```
![](assets/demo_snapshot.png)

### Acknowledgements
Our repo is built open [*Frido*](https://github.com/davidhalladay/Frido) and [Taming Transformers](https://github.com/CompVis/taming-transformers), thanks for your opensourcing!
<!--
**LayoutEnc/LayoutEnc** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->

### Citation

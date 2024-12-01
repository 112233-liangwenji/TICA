# :star:TICA

<b><a href='https://arxiv.org/pdf/2410.07695'>Test-Time Intensity Consistency Adaptation for Shadow Detection.</a> </b>
<div>
<span class="author-block">
Leyi Zhu</a><sup> 👨‍💻‍ </sup>
</span>,
  <span class="author-block">
    <a href='https://github.com/NiFangBaAGe'> Weihuang Liu</a><sup>‍ 👨‍💻‍ </sup>
  </span>,
    <span class="author-block">
   Xinyi Chen</a><sup>‍ </sup>
  </span>,
    <span class="author-block">
      Zimeng Li</a><sup>‍ </sup>
  </span>,
    <span class="author-block">
      <a href='https://cxh.netlify.app/'> Xuhang Chen</a><sup>‍ </sup>
  </span>,
  <span class="author-block">
   Zhen Wang</a><sup> 
  </span> and
  <span class="author-block">
  <a href="https://cmpun.github.io/" >Chi-Man Pun</a><sup> 📮</sup>
</span>
  (👨‍💻‍ Equal contributions, 📮 Corresponding )
</div>

<b>University of Macau</b>

In <b>_International Conference on Neural Information Processing 2024 (ICONIP 2024)_</b>


# 📋 Abstract
Shadow detection is crucial for accurate scene understanding in computer vision, yet it is challenged by the diverse appearances of shadows caused by variations in illumination, object geometry, and scene context. Deep learning models often struggle to generalize to realworld images due to the limited size and diversity of training datasets. To address this, we introduce TICA, a novel framework that leverages lightintensity information during test-time adaptation to enhance shadow detection accuracy. TICA exploits the inherent inconsistencies in light intensity across shadow regions to guide the model toward a more consistent prediction. A basic encoder-decoder model is initially trained on a labeled dataset for shadow detection. Then, during the testing phase, the network is adjusted for each test sample by enforcing consistent intensity predictions between two augmented input image versions. This consistency training specifically targets both foreground and background intersection regions to identify shadow regions within images accurately for robust adaptation. Extensive evaluations on the ISTD and SBU shadow detection datasets reveal that TICA significantly demonstrates that TICA outperforms existing state-of-the-art methods, achieving superior results in balanced error rate (BER).

# 🔮 Overview 
<p align="center">
  <img width="80%" alt="teaser" src="teaser/overview_1.png">
</p>

**Overview of the proposed TICA.** By leveraging light consistency training, the TICA framework enhances the model’s capabilities in shadow detection. Initially, the model is trained with a publicly accessible shadow detection dataset. We then apply random data augmentation techniques—horizontal flipping, resizing, and cropping—to the test set. This facilitates model refinement by enforcing consistent intensity predictions between the two augmented images. The consistency loss is backpropagated to update the encoder.

# 🎶 Dataset
- **ISTD**: https://github.com/DeepInsight-PCALab/ST-CGAN
- **SBU**: https://www3.cs.stonybrook.edu/~cvl/projects/shadow_noisy_label/index.html

# ⚙️ Quick Start
1. Download the dataset and put it in ./load.
2. Download the pre-trained backbone (ResNet-50, Swin-Tiny and Hrnet-18).
3. Training:
```bash
python train.py --config configs/train_{%backbone%}_shadow_{%dataset%}.yaml 
```
4. TTT:
```bash
python ttt_new.py --config configs/train_{%backbone%}_shadow_{%dataset%}.yaml --model ./save/your_pretrained_model/model_epoch_last.pth --gpu 0 --eval_type ber --name shadow_{%backbone%}_ttt_new_{%consistency%} --bg_cons True --fg_cons True
```

# 🛎 Citation
If you find our work useful in your research, please consider citing:
```bib
@article{zhu2024test,
  title={Test-Time Intensity Consistency Adaptation for Shadow Detection},
  author={Zhu, Leyi and Liu, Weihuang and Chen, Xinyi and Li, Zimeng and Chen, Xuhang and Wang, Zhen and Pun, Chi-Man},
  journal={arXiv preprint arXiv:2410.07695},
  year={2024}
}
```

# 💗 Acknowledgements
- This repo is derived from <a href="https://github.com/NiFangBaAGe/Explicit-Visual-Prompt">EVP</a>, which is an exellent work, helps us to quickly implement our ideas.


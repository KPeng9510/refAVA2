## RefAtomNet++: Advancing Referring Atomic Video Action Recognition using Semantic Retrieval based Multi-Trajectory Mamba

>This work introduces RefAtomNet++, a novel framework for Referring Atomic Video Action Recognition (RAVAR), which aims to recognize fine-grained human actions of a specific individual described by natural language.
Building upon the authors’ earlier RefAtomNet, the new model integrates a multi-trajectory semantic-retrieval Mamba and a multi-hierarchical semantic-aligned cross-attention mechanism to achieve precise visual–language alignment and efficient spatio-temporal reasoning.
To support this study, the authors present RefAVA++, an extended large-scale dataset comprising over 2.9 million frames and 75k annotated persons, designed for language-guided action recognition in complex multi-person scenes.
Extensive experiments on both RefAVA and RefAVA++ benchmarks demonstrate that RefAtomNet++ establishes new state-of-the-art results across localization and recognition metrics while maintaining high computational efficiency.
Overall, this work advances the frontier of language-guided human action understanding, bridging the gap between video-language grounding and fine-grained action recognition.

## 🎨 Training & Testing

### Training
Please use train_pami.py

### Datasets

The refAVA++ dataset is available here https://drive.google.com/drive/folders/13Rz83pdchOe5D6ZiadNm9ENZm3pPF7sx?usp=sharing. Please modify the corresponding paths in the dataset file.

## 📕 Installation

- Python >= 3.8
- PyTorch >= 1.9.0
- PyYAML, tqdm, tensorboardX

##  Trained model

Some readers reached out for the model weight, due to the limited storage space of the online storage tools, the authors are facing with issues to upload it successfully. We will update the link once the uploading issue is solved. thanks.


## 🤝 Cite:
Please consider citing this paper if you use the ```code``` or ```data``` from our work.
Thanks a lot :)

@article{peng2025refatomnet++,
  title={RefAtomNet++: Advancing Referring Atomic Video Action Recognition using Semantic Retrieval based Multi-Trajectory Mamba},
  author={Peng, Kunyu and Wen, Di and Fu, Jia and Wu, Jiamin and Yang, Kailun and Zheng, Junwei and Liu, Ruiping and Chen, Yufan and Fu, Yuqian and Paudel, Danda Pani and others},
  journal={arXiv preprint arXiv:2510.16444},
  year={2025}
}

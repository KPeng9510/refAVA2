## RefAtomNet++: Advancing Referring Atomic Video Action Recognition using Semantic Retrieval based Multi-Trajectory Mamba

>This work introduces RefAtomNet++, a novel framework for Referring Atomic Video Action Recognition (RAVAR), which aims to recognize fine-grained human actions of a specific individual described by natural language.
Building upon the authors’ earlier RefAtomNet, the new model integrates a multi-trajectory semantic-retrieval Mamba and a multi-hierarchical semantic-aligned cross-attention mechanism to achieve precise visual–language alignment and efficient spatio-temporal reasoning.
To support this study, the authors present RefAVA++, an extended large-scale dataset comprising over 2.9 million frames and 75k annotated persons, designed for language-guided action recognition in complex multi-person scenes.
Extensive experiments on both RefAVA and RefAVA++ benchmarks demonstrate that RefAtomNet++ establishes new state-of-the-art results across localization and recognition metrics while maintaining high computational efficiency.
Overall, this work advances the frontier of language-guided human action understanding, bridging the gap between video-language grounding and fine-grained action recognition.

## 🎨 Training & Testing

### Training
Please use train_pami.py

## 📕 Installation

- Python >= 3.8
- PyTorch >= 1.9.0
- PyYAML, tqdm, tensorboardX


## 🤝 Cite:
Please consider citing this paper if you use the ```code``` or ```data``` from our work.
Thanks a lot :)

bibtex will be provided soon.

# VisualAiD -- Comparison of 3D CNN Architectures for Detecting Alzheimer's Disease using Relevance Maps

Alzheimer's disease (AD) is a neurodegenerative disorder which causes gradual and irreversible damage to the brain. 
Many recent developments in deep learning (DL) and specifically in convolutional neural networks (CNN), have achieved promising results in a broad range of computer-vision tasks.
The success of CNNs have made them a common state-of-the-art tool for image classification and object localisation tasks. 
Though CNNs and DL models at large remain difficult to understand and explain, and thus are categorised as black-box models.
Feature attribution methods such as layer-wise relevance propagation (LRP) allow tracing back the information flow in CNNs. 
This enables creation of relevance heatmaps, which approximate the contribution of the input image regions on the model decision. 

**In this project, we addressed the open question which of the most common CNN architectures is best suited for AD classification based on MRI data.**


## Included 3D CNN Model Architectures

We chose to study some of the most cited CNN architectures - AlexNet, VGG, ResNet, and DenseNet.
See [Source/util.py] for implementation details.


## Key Results




## Citation

```bibtex
@incollection{Singh.2023,
 title = {Comparison of CNN Architectures for Detecting Alzheimer's Disease using Relevance Maps},
 author = {Singh, Devesh and Dyrba, Martin},
 booktitle = {Bildverarbeitung f{\"u}r die Medizin 2023},
 year = {2023},
 publisher = {{Springer Fachmedien Wiesbaden}},
 note = {accepted}
}
```

# VisualAiD -- Comparison of 3D CNN Architectures for Detecting Alzheimer's Disease using Relevance Maps

![python](https://img.shields.io/badge/python-v3.7-green)
![tf](https://img.shields.io/badge/tensorflow-v1.15-orange)
![keras](https://img.shields.io/badge/keras-v.2.2.4-red)
![innvestigate](https://img.shields.io/badge/innvestigate-v.1.0.9-blue)


Alzheimer's disease (AD) is a neurodegenerative disorder which causes gradual and irreversible damage to the brain. 
Many recent developments in deep learning (DL) and specifically in convolutional neural networks (CNN), have achieved promising results in a broad range of computer-vision tasks.
The success of CNNs have made them a common state-of-the-art tool for image classification and object localisation tasks. 
Though CNNs and DL models at large remain difficult to understand and explain, and thus are categorised as black-box models.
Feature attribution methods such as layer-wise relevance propagation (LRP) allow tracing back the information flow in CNNs. 
This enables creation of relevance heatmaps, which approximate the contribution of the input image regions on the model decision. 

Further details on the addressed open question, i.e., which of the most common CNN architectures is best suited for AD classification based on MRI data, were published in the procedings:

Singh & Dyrba (2023) Comparison of CNN Architectures for Detecting Alzheimer’s Disease using Relevance Maps. 
Bildverarbeitung für die Medizin 2023. BVM 2023. DOI: [10.1007/978-3-658-41657-7_51](https://doi.org/10.1007/978-3-658-41657-7_51)

## Included 3D CNN Model Architectures

We chose to study some of the most cited CNN architectures - AlexNet, VGG, ResNet, and DenseNet.
See [Source/util.py](Source/util.py) for implementation details.

<!---TODO: add model architecture images here or below--->

Following image illusrates the DenseNet model architecture that was utilised in this study.
![DenseNet model architecture](/Images/densenet_architecture-1.png)


## Key Results

<!---TODO: add short results summary and relevance images here--->


Follwing is the mean relevance map for the MCI group of the ADNI3 dataset obtained using the
LRP relevance propagation method, for a trained DenseNet model. Coronal slices Y=[-10,-20,-30] mm in MNI reference space are shown. 
Bright yellow represent the most relevant regions.

<p align="center">
  <img src="/Images/MeanRelevanceMap_DenseNet.png">
</p>

This study shows that DenseNet, a complex model with dense-skip connections, utilises an efficient information flow at various scales,
and generates relevance maps which focused on clinically relevant features (ex. the hippocampal region).
This study also demonstrates the added value of a holistic evaluation of models, where relevance maps are being used in combination with classical performance metrics.

## Citation

```bibtex
@incollection{Singh.2023,
 title = {Comparison of CNN Architectures for Detecting Alzheimer's Disease using Relevance Maps},
 author = {Singh, Devesh and Dyrba, Martin},
 booktitle = {Bildverarbeitung f{\"u}r die Medizin 2023},
 year = {2023},
 publisher = {{Springer Fachmedien Wiesbaden}},
 note = {accepted},
doi = {https://doi.org/10.1007/978-3-658-41657-7_51},
}
```

# Official repo for **Diversity as a Reward: Fine-Tuning LLMs on a Mixture of Domain-Undetermined Data**

We release **DaaR**,  A data diversity-driven reward method for high-quality data selection across mixed domains to enhance LLM capabilities. See more details in our [paper](https://www.arxiv.org/abs/2502.04380).

> 
>
> **Abstract:** High-performance Multimodal Large Language Models (MLLMs) rely heavily on data quality. This study introduces a novel dataset named Img-Diff, designed to enhance fine-grained image recognition in MLLMs by leveraging insights from contrastive learning and image difference captioning. By analyzing object differences between similar images, we challenge models to identify both matching and distinct components. We utilize the Stable-Diffusion-XL model and advanced image editing techniques to create pairs of similar images that highlight object replacements. Our methodology includes a Difference Area Generator for object differences identifying, followed by a Difference Captions Generator for detailed difference descriptions. The result is a relatively small but high-quality dataset of "object replacement" samples. We use the the proposed dataset to finetune state-of-the-art (SOTA) MLLMs such as MGM-7B, yielding comprehensive improvements of performance scores over SOTA models that trained with larger-scale datasets, in numerous image difference and Visual Question Answering tasks. For instance, our trained models notably surpass the SOTA models GPT-4V and Gemini on the MMVP benchmark. Besides, we investigate alternative methods for generating image difference data through "object removal" and conduct a thorough evaluation to confirm the dataset's diversity, quality, and robustness, presenting several insights on the synthesis of such a contrastive dataset. We release our codes and dataset, to encourage further research and advance the field of multimodal data synthesis and enhancement of MLLMs' fundamental capabilities for image understanding.


## Codes and Data Recipes

- The original codes are organized and presented in [DaaR](https://github.com/lingzhq/data-juicer/tree/DaaR/DaaR).
- We wil develop a series of data-juicer operators related to DaaR.


## Citation

If you find our work useful for your research, please consider citing our paper.

```
@article{ling2025diversity,
  title={Diversity as a Reward: Fine-Tuning LLMs on a Mixture of Domain-Undetermined Data},
  author={Ling, Zhenqing and Chen, Daoyuan and Yao, Liuyi and Li, Yaliang and Shen, Ying},
  journal={arXiv preprint arXiv:2502.04380},
  year={2025}
}
```

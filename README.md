<div align="center"><h2>Controllable Text Generation for Large Language Models: A Survey</h2></div>

<p align="center">
    <!-- arxiv badges -->
    <!-- <a href="https://arxiv.org/abs/2407.14507">
        <img src="https://img.shields.io/badge/Paper-red?style=flat&logo=arxiv">
    </a> -->
    <!-- Chinese Version -->
    <a href="./CTG_Survey_Chinese.pdf">
        <img src="https://img.shields.io/badge/Chinese--Version-white?style=flat&logo=google-docs">
    </a>
    <!-- Github -->
    <a href="https://github.com/IAAR-Shanghai/CTGSurvey">
        <img src="https://img.shields.io/badge/Code-black?style=flat&logo=github">
    </a>
    <!-- HuggingFace -->
    <!-- <a href="https://huggingface.co/papers/2407.14507">
        <img src="https://img.shields.io/badge/-%F0%9F%A4%97%20Page-orange?style=flat"/>
    </a> -->
    <!-- Yuque -->
    <a href="https://www.yuque.com/matong-an7ob/qf04ed/yzs6n19swv6pipri">
        <img src="https://img.shields.io/badge/Paper--List-white?style=flat&logo=googlesheets">
    </a>
</p>


<div align="center">
    <p>
        <a href="https://scholar.google.com/citations?user=d0E7YlcAAAAJ">Xun Liang</a><sup>1*</sup>, 
        <a href="https://scholar.google.com/citations?user=5LrD2HoAAAAJ">Hanyu Wang</a><sup>1*</sup>, 
        <a href="https://scholar.google.com/citations?user=EzKQkhwAAAAJ">Yezhaohui Wang</a><sup>2*</sup>, <br>
        <a href="https://ki-seki.github.io/">Shichao Song</a><sup>1</sup>, 
        <a href="https://github.com/J1awei-Yang">Jiawei Yang</a><sup>1</sup>, 
        <a href="https://github.com/siminniu">Simin Niu</a><sup>1</sup>, 
        Jie Hu<sup>3</sup>, 
        Dan Liu<sup>3</sup>, 
        Shunyu Yao<sup>3</sup>, 
        <a href="https://scholar.google.com/citations?user=GOKgLdQAAAAJ">Feiyu Xiong</a><sup>2</sup>, 
        <a href="https://www.semanticscholar.org/author/Zhiyu-Li/2268429641">Zhiyu Li</a><sup>2†</sup>
    </p>
    <p>
        <sup>1</sup><a href="https://en.ruc.edu.cn/">Renmin University of China</a> <br>
        <sup>2</sup><a href="https://www.iaar.ac.cn/">Institute for Advanced Algorithms Research, Shanghai</a> <br>
        <sup>3</sup><a href="https://www.chinatelecom.com.cn/">China Telecom Research Institute</a>
    </p>
</div>

<div align="center"><small><sup>*</sup>Equal contribution.</small></div>
<div align="center"><small><sup>†</sup>Corresponding author: Zhiyu Li (<a href="mailto:lizy@iaar.ac.cn">lizy@iaar.ac.cn</a>).</small></div>

<!-- ## News

- **[2024/07/21]** Our paper is published on the arXiv platform: https://arxiv.org/abs/2407.14507. -->

## Introduction

Welcome to the GitHub repository for our survey paper titled *"Controllable Text Generation for Large Language Models: A Survey."* This repository includes all the resources, code, and references related to the paper. Our objective is to provide a thorough overview of the techniques and methodologies used to control text generation in large language models (LLMs), with an emphasis on both theoretical underpinnings and practical implementations.

![Survey Framework](figures/framework.png)

Our survey explores the following key areas:

### 1. Demands of Controllable Text Generation

Controllable Text Generation (CTG) must meet two main requirements:

1. **Meeting Predefined Control Conditions**: 
   Ensuring that the generated text adheres to specified criteria, such as thematic consistency, safety, and stylistic adherence.
   
2. **Maintaining Text Quality**: 
   Ensuring that the text produced is fluent, helpful, and diverse while balancing control with overall quality.

### 2. Formal Definition of Controllable Text Generation

We define CTG as follows:

1. **Relationship with LLM Capabilities**:
   CTG is an ability dimension that is orthogonal to the objective knowledge capabilities of LLMs, focusing on how information is presented to meet specific needs, such as style or sentiment.

2. **Injection of Control Conditions**:
   Control conditions can be integrated into the text generation process at various stages using resources like text corpora, graphs, or databases.

3. **Quality of CTG**:
   High-quality CTG strikes a balance between adherence to control conditions and maintaining fluency, coherence, and helpfulness in the generated text.

### 3. Classification of Controllable Text Generation Tasks

CTG tasks are categorized into two main types:

1. **Content Control (Linguistic Control/Hard Control)**: 
   Focuses on managing content structure, such as format and vocabulary.

2. **Attribute Control (Semantic Control/Soft Control)**: 
   Focuses on managing attributes like sentiment, style, and safety.

### 4. Controllable Text Generation Method Classification

CTG methods are systematically categorized into two stages:

1. **Training-Stage Methods**: 
   Techniques such as model retraining, fine-tuning, and reinforcement learning that occur during the training phase.
   
2. **Inference-Stage Methods**: 
   Techniques such as prompt engineering, latent space manipulation, and decoding-time intervention applied during inference.

### 5. Evaluation Methods and Applications

We review the evaluation methods and their applications in CTG:

1. **Evaluation Methods**: 
   We introduce a range of automatic and human-based evaluation metrics, along with benchmarks that assess the effectiveness of CTG techniques, focusing on how well they balance control and text quality.

2. **Applications**: 
   We explore CTG applications across both specialized vertical domains and general tasks.

### 6. Challenges and Future Directions

This survey addresses key challenges in CTG research and suggests future directions:

1. **Key Challenges**: 
   Issues such as achieving precise control, maintaining fluency and coherence, and handling multi-attribute control in complex scenarios.

2. **Proposed Appeals**: 
   We advocate for a greater focus on real-world applications and the development of robust evaluation frameworks to advance CTG techniques.

This paper aims to provide valuable insights and guidance for researchers and developers working in the field of Controllable Text Generation. All references, along with a Chinese version of this survey, are open-sourced and available at [https://github.com/IAAR-Shanghai/CTGSurvey](https://github.com/IAAR-Shanghai/CTGSurvey).

## Project Structure

- **`figures/`**: Contains all the figures used in the repository.
- **`latex/`**: Includes the LaTeX source files for the survey paper.
- **`CTG_Survey_Chinese.pdf`**: The Chinese version of the survey paper.
- **`README.md`**: This file, providing an overview of the repository.

## Paper List

We provide a spreadsheet containing all the papers we reviewed: [Literature](https://www.yuque.com/matong-an7ob/qf04ed/yzs6n19swv6pipri). A more readable table format is under development.

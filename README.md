<div align="center"><h2>Controllable Text Generation for Large Language Models: A Survey</h2></div>

<p align="center">
    <!-- arxiv badges -->
    <a href="https://arxiv.org/abs/2408.12599">
        <img src="https://img.shields.io/badge/Paper-red?style=flat&logo=arxiv">
    </a>
    <!-- Chinese Version -->
    <a href="./CTG_Survey_Chinese.pdf">
        <img src="https://img.shields.io/badge/Chinese--Version-white?style=flat&logo=google-docs">
    </a>
    <!-- Github -->
    <a href="https://github.com/IAAR-Shanghai/CTGSurvey">
        <img src="https://img.shields.io/badge/Code-black?style=flat&logo=github">
    </a>
    <!-- HuggingFace -->
    <a href="https://huggingface.co/papers/2408.12599">
        <img src="https://img.shields.io/badge/-%F0%9F%A4%97%20Page-orange?style=flat"/>
    </a>
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


> \[!IMPORTANT\]
>
> 🌟 **Star Us!** If you find our work helpful, please consider staring our GitHub to stay updated with the latest in Controllable Text Generation!

## 📰 News

- **[2024/08/23]** Our paper is published on the arXiv platform: https://arxiv.org/abs/2408.12599.
- **[2024/08/23]** Our paper secured the second position on Hugging Face's Daily Papers module: https://huggingface.co/papers/2408.12599.
- **[2024/08/26]** We have updated our paper list, which can now be accessed on our [GitHub page](https://github.com/IAAR-Shanghai/CTGSurvey).

## 🔗 Introduction

Welcome to the GitHub repository for our survey paper titled *"Controllable Text Generation for Large Language Models: A Survey."* This repository includes all the resources, code, and references related to the paper. Our objective is to provide a thorough overview of the techniques and methodologies used to control text generation in large language models (LLMs), with an emphasis on both theoretical underpinnings and practical implementations.

<div align="center">
    <img src="figures/framework.png" alt="Survey Framework" width="70%">
</div>

Our survey explores the following key areas:

### 🎯 Demands of Controllable Text Generation

Controllable Text Generation (CTG) must meet two main requirements:

1. **Meeting Predefined Control Conditions**: 
   Ensuring that the generated text adheres to specified criteria, such as thematic consistency, safety, and stylistic adherence.
   
2. **Maintaining Text Quality**: 
   Ensuring that the text produced is fluent, helpful, and diverse while balancing control with overall quality.

### 📜 Formal Definition of Controllable Text Generation

We define CTG as follows:

1. **Relationship with LLM Capabilities**:
   CTG is an ability dimension that is orthogonal to the objective knowledge capabilities of LLMs, focusing on how information is presented to meet specific needs, such as style or sentiment.

2. **Injection of Control Conditions**:
   Control conditions can be integrated into the text generation process at various stages using resources like text corpora, graphs, or databases.

3. **Quality of CTG**:
   High-quality CTG strikes a balance between adherence to control conditions and maintaining fluency, coherence, and helpfulness in the generated text.

### 🗂️ Classification of Controllable Text Generation Tasks

CTG tasks are categorized into two main types:

1. **Content Control (Linguistic Control/Hard Control)**: 
   Focuses on managing content structure, such as format and vocabulary.

2. **Attribute Control (Semantic Control/Soft Control)**: 
   Focuses on managing attributes like sentiment, style, and safety.

### 🔧 Controllable Text Generation Method Classification

CTG methods are systematically categorized into two stages:

1. **Training-Stage Methods**: 
   Techniques such as model retraining, fine-tuning, and reinforcement learning that occur during the training phase.
   
2. **Inference-Stage Methods**: 
   Techniques such as prompt engineering, latent space manipulation, and decoding-time intervention applied during inference.

### 📊 Evaluation Methods and Applications

We review the evaluation methods and their applications in CTG:

1. **Evaluation Methods**: 
   We introduce a range of automatic and human-based evaluation metrics, along with benchmarks that assess the effectiveness of CTG techniques, focusing on how well they balance control and text quality.

2. **Applications**: 
   We explore CTG applications across both specialized vertical domains and general tasks.

### 🚀 Challenges and Future Directions

This survey addresses key challenges in CTG research and suggests future directions:

1. **Key Challenges**: 
   Issues such as achieving precise control, maintaining fluency and coherence, and handling multi-attribute control in complex scenarios.

2. **Proposed Appeals**: 
   We advocate for a greater focus on real-world applications and the development of robust evaluation frameworks to advance CTG techniques.

This paper aims to provide valuable insights and guidance for researchers and developers working in the field of Controllable Text Generation. All references, along with a Chinese version of this survey, are open-sourced and available at [https://github.com/IAAR-Shanghai/CTGSurvey](https://github.com/IAAR-Shanghai/CTGSurvey).

## 🧩 Project Structure

- **`figures/`**: Contains all the figures used in the repository.
- **`latex/`**: Includes the LaTeX source files for the survey paper.
- **`CTG_Survey_Chinese.pdf`**: The Chinese version of the survey paper.
- **`README.md`**: This file, providing an overview of the repository.

## 📚 Paper List

We’ve compiled a comprehensive spreadsheet of all the papers we reviewed, accessible [here](https://www.yuque.com/matong-an7ob/qf04ed/yzs6n19swv6pipri). A more user-friendly table format is in progress.

Below, you'll find a categorized list of papers from 2023 and 2024, organized by Type, Phase, and Classification.

### Type: Method

#### Training Stage

##### Retraining

- **Fine-Grained Sentiment-Controlled Text Generation Approach Based on Pre-Trained Language Model**  
  Zhejiang University of Technology, Appl. Sci., 2023 [[Paper](https://www.mdpi.com/2076-3417/13/1/264)]
- **Lexical Complexity Controlled Sentence Generation for Language Learning**  
  BLCU, CCL'23, 2023 [[Paper](https://aclanthology.org/2023.ccl-1.56/)]
- **Semantic Space Grounded Weighted Decoding for Multi-Attribute Controllable Dialogue Generation**  
  Shanghai Jiao Tong University, EMNLP'23, 2023 [[Paper](https://aclanthology.org/2023.emnlp-main.817/)]
- **SweCTRL-Mini: a data-transparent Transformer-based large language model for controllable text generation in Swedish**  
  KTH Royal Institute of Technology, arxiv'23, 2023 [[Paper](https://arxiv.org/pdf/2304.13994)]

##### Fine-Tuning

- **Language Detoxification with Attribute-Discriminative Latent Space**  
  KAIST, ACL'23, 2023 [[Paper](https://aclanthology.org/2023.acl-long.565/)]
- **Controlled Text Generation with Hidden Representation Transformations**  
  UCLA, ACL'23_findings, 2023 [[Paper](https://aclanthology.org/2023.findings-acl.602/)]
- **CLICK: Controllable Text Generation with Sequence Likelihood Contrastive Learning**  
  THU, ACL'24_findings, 2023 [[Paper](https://aclanthology.org/2023.findings-acl.65/)]
- **Seen to Unseen: Exploring Compositional Generalization of Multi-Attribute Controllable Dialogue Generation**  
  BUPT, ACL'23, 2023 [[Paper](https://aclanthology.org/2023.acl-long.793/)]
- **DeepPress: guided press release topic-aware text generation using ensemble transformers**  
  Universite´ de Moncton,, Neural Computing and Applications, 2023 [[Paper](https://link.springer.com/article/10.1007/s00521-023-08393-4)]
- **DuNST: Dual Noisy Self Training for Semi-Supervised Controllable Text Generation**  
  The University of British Columbia, ACL'23, 2023 [[Paper](https://aclanthology.org/2023.acl-long.488/)]
- **Controlled text generation with natural language instructions**  
  ETH Zürich, ICML'23, 2023 [[Paper](https://dl.acm.org/doi/10.5555/3618408.3620203)]
- **Controlling keywords and their positions in text generation**  
  Hitachi, Ltd. Research and Development Group, INLG'23, 2023 [[Paper](https://aclanthology.org/2023.inlg-main.29/)]
- **Toward Unified Controllable Text Generation via Regular Expression Instruction**  
  ISCAS, IJCNLP-AACL'23, 2023 [[Paper](https://aclanthology.org/2023.ijcnlp-main.1/)]
- **Controllable Text Generation with Residual Memory Transformer**  
  BIT, arxiv'23, 2023 [[Paper](https://arxiv.org/abs/2309.16231)]
- **Continuous Language Model Interpolation for Dynamic and Controllable Text Generation**  
  Harvard University, arxiv'24, 2024 [[Paper](https://arxiv.org/abs/2404.07117)]
- **CoDa: Constrained Generation based Data Augmentation for Low-Resource NLP**  
  UMD, arxiv'24, 2024 [[Paper](https://arxiv.org/abs/2404.00415)]
- **Contrastive Perplexity for Controlled Generation: An Application in Detoxifying Large Language Models**  
  SAP, arxiv'24, 2024 [[Paper](https://arxiv.org/abs/2401.08491)]
- **CTGGAN: Controllable Text Generation with Generative Adversarial Network**  
  JIUTIAN Team, China Mobile Research, Appl. Sci., 2024 [[Paper](https://www.mdpi.com/2076-3417/14/7/3106)]
- **ECCRG: A Emotion- and Content-Controllable Response Generation Model**  
  TJU, Lecture Notes of the Institute for Computer Sciences, Social Informatics and Telecommunications Engineering, 2024 [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-54528-3_7)]
- **LiFi: Lightweight Controlled Text Generation with Fine-Grained Control Codes**  
  THU, arxiv'24, 2024 [[Paper](https://arxiv.org/abs/2402.06930)]

##### Reinforcement Learning

- **STEER: Unified Style Transfer with Expert Reinforcement**  
  University of Washington, EMNLP'23_findings, 2023 [[Paper](https://aclanthology.org/2023.findings-emnlp.506/)]
- **Prompt-Based Length Controlled Generation with Multiple Control Types**  
  NWPU, ACL'24_findings, 2024 [[Paper](https://aclanthology.org/2024.findings-acl.63/)]
- **Reinforcement Learning with Dynamic Multi-Reward Weighting for Multi-Style Controllable Generation**  
  University of Minnesota, arxiv'24, 2024 [[Paper](https://arxiv.org/abs/2402.14146)]
- **Safe RLHF: Safe Reinforcement Learning from Human Feedback**  
  PKU, ICLR'24_spotlight, 2024 [[Paper](https://openreview.net/forum?id=TyFrPOKYXw)]
- **Token-level Direct Preference Optimization**  
  IACAS, arxiv'24, 2024 [[Paper](https://arxiv.org/abs/2404.11999)]
- **Reinforcement Learning with Token-level Feedback for Controllable Text Generation**  
  HUST, NAACL'24, 2024 [[Paper](https://arxiv.org/abs/2403.11558)]

#### Inference Stage

##### Prompt Engineering

- **Controllable Generation of Dialogue Acts for Dialogue Systems via Few-Shot Response Generation and Ranking**  
  University of California Santa Cruz, SIGDIAL'23, 2023 [[Paper](https://aclanthology.org/2023.sigdial-1.32/)]
- **PCFG-based Natural Language Interface Improves Generalization for Controlled Text Generation**  
  Johns Hopkins University, SEM'23, 2023 [[Paper](https://aclanthology.org/2023.starsem-1.27/)]
- **Harnessing the Plug-and-Play Controller by Prompting**  
  BUAA, GEM'23, 2023 [[Paper](https://aclanthology.org/2023.gem-1.14/)]
- **An Extensible Plug-and-Play Method for Multi-Aspect Controllable Text Generation**  
  THU&Meituan, ACL'23, 2023 [[Paper](https://aclanthology.org/2023.acl-long.849/)]
- **Tailor: A Soft-Prompt-Based Approach to Attribute-Based Controlled Text Generation**  
  Alibaba, ACL'23, 2023 [[Paper](https://aclanthology.org/2023.acl-long.25/)]
- **InstructCMP: Length Control in Sentence Compression through Instruction-based Large Language Models**  
  CNU, ACL'24_findings, 2024 [[Paper](https://aclanthology.org/2024.findings-acl.532/)]
- **Topic-Oriented Controlled Text Generation for Social Networks**  
  WHU, Journal of Signal Processing Systems, 2024 [[Paper](https://link.springer.com/article/10.1007/s11265-023-01907-2)]
- **Plug and Play with Prompts: A Prompt Tuning Approach for Controlling Text Generation**  
  University of Toronto, AAAI'24_workshop, 2024 [[Paper](https://arxiv.org/abs/2404.05143)]
- **TrustAgent: Towards Safe and Trustworthy LLM-based Agents through Agent Constitution**  
  UCSB, arxiv'24, 2024 [[Paper](https://arxiv.org/abs/2402.01586)]

##### Latent Space Manipulation

- **Activation Addition: Steering Language Models Without Optimization**  
  UC Berkeley, arxiv'23, 2023 [[Paper](https://arxiv.org/abs/2308.10248)]
- **Evaluating, Understanding, and Improving Constrained Text Generation
  for Large Language Models**  
  PKU, arxiv'23, 2023 [[Paper](https://arxiv.org/pdf/2310.16343)]
- **In-context Vectors: Making In Context Learning More Effective and Controllable Through Latent Space Steering**  
  Stanford University, arxiv'23, 2023 [[Paper](https://arxiv.org/abs/2311.06668)]
- **MacLaSa: Multi-Aspect Controllable Text Generation via Efficient Sampling from Compact Latent Space**  
  ICT CAS, EMNLP'23_findings, 2023 [[Paper](https://aclanthology.org/2023.findings-emnlp.292/)]
- **Miracle: Towards Personalized Dialogue Generation with Latent-Space Multiple Personal Attribute Control**  
  HUST, EMNLP'23_findings, 2023 [[Paper](https://aclanthology.org/2023.findings-emnlp.395/)]
- **Controllable Text Generation via Probability Density Estimation in the Latent Space**  
  HIT, EMNLP'23, 2023 [[Paper](https://aclanthology.org/2023.acl-long.704/)]
- **Self-Detoxifying Language Models via Toxification Reversal**  
  The Hong Kong Polytechnic University, EMNLP'23, 2023 [[Paper](https://aclanthology.org/2023.emnlp-main.269/)]
- **DESTEIN: Navigating Detoxification of Language Models via Universal Steering Pairs and Head-wise Activation Fusion**  
  Tongji University, arxiv'24, 2024 [[Paper](https://arxiv.org/abs/2404.10464)]
- **FreeCtrl: Constructing Control Centers with Feedforward Layers for Learning-Free Controllable Text Generation**  
  NTU, ACL'24, 2024 [[Paper](https://aclanthology.org/2024.acl-long.412/)]
- **InferAligner: Inference-Time Alignment for Harmlessness through Cross-Model Guidance**  
  FuDan, arxiv'24, 2024 [[Paper](https://arxiv.org/abs/2401.11206)]
- **Multi-Aspect Controllable Text Generation with Disentangled Counterfactual Augmentation**  
  NJU, ACL'24, 2024 [[Paper](https://aclanthology.org/2024.acl-long.500/)]
- **Style Vectors for Steering Generative Large Language Models**  
  German Aerospace Center (DLR), EACL'24_findings, 2024 [[Paper](https://aclanthology.org/2024.findings-eacl.52/)]

##### Decoding-time Intervention

- **Air-Decoding: Attribute Distribution Reconstruction for Decoding-Time Controllable Text Generation**  
  USTC, EMNLP'23, 2023 [[Paper](https://aclanthology.org/2023.emnlp-main.512/)]
- **A Block Metropolis-Hastings Sampler for Controllable Energy-based Text Generation**  
  UCSD, CoNLL'23, 2023 [[Paper](https://aclanthology.org/2023.conll-1.26/)]
- **BOLT: Fast Energy-based Controlled Text Generation with Tunable Biases**  
  University of Michigan, ACL'23_short, 2023 [[Paper](https://aclanthology.org/2023.acl-short.18/)]
- **Controlled Decoding from Language Models**  
  Google, NeurIPS_SoLaR'23, 2023 [[Paper](https://openreview.net/forum?id=jo57H1CpD8)]
- **Focused Prefix Tuning for Controllable Text Generation**  
  Tokyo Institute of Technology, ACL'23_short, 2023 [[Paper](https://aclanthology.org/2023.findings-acl.281/)]
- **Focused Prefix Tuning for Controllable Text Generation**  
  Tokyo Tech, ACL'23_short, 2023 [[Paper](https://aclanthology.org/2023.acl-short.96/)]
- **Goodtriever: Adaptive Toxicity Mitigation with Retrieval-augmented Models**  
  Cohere For AI, EMNLP'23_findings, 2023 [[Paper](https://aclanthology.org/2023.findings-emnlp.339/)]
- **GRACE: Gradient-guided Controllable Retrieval for Augmenting Attribute-based Text Generation**  
  National University of Defense Technology, ACL'23_findings, 2023 [[Paper](https://aclanthology.org/2023.findings-acl.530/)]
- **An Invariant Learning Characterization of Controlled Text Generation**  
  Columbia University, ACL'23, 2023 [[Paper](https://aclanthology.org/2023.acl-long.179/)]
- **Style Locality for Controllable Generation with kNN Language Models**  
  University of Marburg, SIGDIAL'23_TamingLLM workshop, 2023 [[Paper](https://aclanthology.org/2023.tllm-1.7/)]
- **Detoxifying Text with MaRCo: Controllable Revision with Experts and Anti-Experts**  
  University of Washington&CMU, ACL'23_short, 2023 [[Paper](https://aclanthology.org/2023.acl-short.21/)]
- **MIL-Decoding: Detoxifying Language Models at Token-Level via Multiple Instance Learning**  
  PKU, ACL'23, 2023 [[Paper](https://doi.org/10.18653/v1%2F2023.acl-long.11)]
- **Controllable Story Generation Based on Perplexity Minimization**  
  Vyatka State University, AIST 2023, 2023 [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-54534-4_11)]
- **PREADD: Prefix-Adaptive Decoding for Controlled Text Generation**  
  UC Berkeley, ACL'23_findings, 2023 [[Paper](https://aclanthology.org/2023.findings-acl.636/)]
- **Reward-Augmented Decoding: Efficient Controlled Text Generation With a Unidirectional Reward Model**  
  UNC-Chapel Hill, EMNLP'23_short, 2023 [[Paper](https://aclanthology.org/2023.emnlp-main.721/)]
- **Controlled Text Generation for Black-box Language Models via Score-based Progressive Editor**  
  Seoul National University, ACL'24, 2023 [[Paper](https://aclanthology.org/2024.acl-long.767/)]
- **Successor Features for Efficient Multisubject Controlled Text Generation**  
  Microsoft, arxiv'23, 2023 [[Paper](https://arxiv.org/abs/2311.04921)]
- **Controlled Text Generation via Language Model Arithmetic**  
  ETH Zurich, ICLR'24_spotlight, 2024 [[Paper](https://openreview.net/forum?id=SLw9fp4yI6)]
- **COLD-Attack: Jailbreaking LLMs with Stealthiness and Controllability**  
  UIUC, arxiv'24, 2024 [[Paper](https://arxiv.org/abs/2402.08679)]
- **Controlled Text Generation for Large Language Model with Dynamic Attribute Graphs**  
  RUC, arxiv'24, 2024 [[Paper](https://aclanthology.org/2024.findings-acl.345/)]
- **DECIDER: A Rule-Controllable Decoding Strategy for Language Generation by Imitating Dual-System Cognitive Theory**  
  BIT, TKDE_submitted, 2024 [[Paper](https://arxiv.org/abs/2403.01954)]
- **Word Embeddings Are Steers for Language Models**  
  UIUC, ACL'24, 2024 [[Paper](https://arxiv.org/abs/2305.12798v2)]
- **RAIN: Your Language Models Can Align Themselves without Finetuning**  
  PKU, ICLR'24, 2024 [[Paper](https://openreview.net/forum?id=pETSfWMUzy)]
- **ROSE Doesn't Do That: Boosting the Safety of Instruction-Tuned Large Language Models with Reverse Prompt Contrastive Decoding**  
  WHU, arxiv'24, 2024 [[Paper](https://arxiv.org/abs/2402.11889)]
- **Uncertainty is Fragile: Manipulating Uncertainty in Large Language Models**  
  Rutgers, arxiv'24, 2024 [[Paper](https://arxiv.org/abs/2407.11282)]

### Type: Benchmark

- **Causal ATE Mitigates Unintended Bias in Controlled Text Generation**  
  IISc, Bangalore, arxiv'23, 2023 [[Paper](https://arxiv.org/pdf/2311.11229)]
- **Evaluating Large Language Models on Controlled Generation Tasks**  
  University of Southern California, EMNLP'23, 2023 [[Paper](https://aclanthology.org/2023.emnlp-main.190.pdf)]
- **Benchmarking Large Language Models on Controllable Generation under Diversified Instructions**  
  USTC, AAAI'24, 2024 [[Paper](https://arxiv.org/abs/2401.00690)]
- **Controllable Text Generation in the Instruction-Tuning Era**  
  CMU, arxiv'24, 2024 [[Paper](https://arxiv.org/abs/2405.01490)]
- **FOFO: A Benchmark to Evaluate LLMs’ Format-Following Capability**  
  Salesforce Research, arxiv'24, 2024 [[Paper](https://arxiv.org/pdf/2402.18667)]
- **Benchmarking Generation and Evaluation Capabilities of Large Language
  Models for Instruction Controllable Summarization**  
  Yale University, NAACL'24, 2024 [[Paper](https://aclanthology.org/2024.findings-naacl.280/)]

### Type: Survey

- **How to Control Sentiment in Text Generation: A Survey of the State-of-the-Art in Sentiment-Control Techniques**  
  DCU, nan, 2023 [[Paper](https://aclanthology.org/2023.wassa-1.30/)]
- **A Survey of Controllable Text Generation using Transformer-based Pre-trained Language Models**  
  BIT, nan, 2023 [[Paper](https://dl.acm.org/doi/10.1145/3617680)]
- **A recent survey on controllable text generation: A causal perspective**  
  Tongji, nan, 2024 [[Paper](https://www.sciencedirect.com/science/article/pii/S2667325824000062)]

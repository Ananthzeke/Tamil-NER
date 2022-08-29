Multi Task learning for NER
- [Multi-task cross-lingual sequence tagging from scratch - 2016](https://doi.org/10.48550/arXiv.1603.06270)
    - This paper employs a recurrent neural network on both character and word levels to encode morphology and context information and applies CRF layer to preidict the tags.
    - This model is task independent, language independent and feature engineering free.
- [Semi-supervised multitask learning for sequence labeling - 2017](http://dx.doi.org/10.18653/v1/P17-1194)
    - This paper has a secondary training approach to predict the surrounding words for every word in the dataset.
    - This model learns general-putpose patterns of semantic and syntactic composition.
    - One half of a bidirectional LSTM is trained as a forward-moving language model, whereas the other half is trained as a backward-moving language model.
    - At the same time, both of these are also combined, in order to predict the most probable label for each word.
- [Multi-lingual Multi-task Architecture for Low-resource Sequence Labeling - 2018](http://dx.doi.org/10.18653/v1/P18-1074)
    - This paper proposes a multi-lingual multi-task architecture to develop supervised models with a minimal amount of labeled data for sequence labeling.
    - In this new architecture, we combine various transfer models using two layers of parameter sharing.
    - On the first layer, A basis of the architecture is constructed to provide universal word representation and feature extraction capability for all models.
    - On the second level, different parameter sharing strategies are adopted for different transfer schemes.
    - This architecture proves to be particularly effective for low-resource settings.

Transfer Learning for NER
- The traditional approach is through bootstrapping algorithms.
    - [Transfer learning for sequence tagging with hierarchical recurrent networks - 2017 ](https://doi.org/10.48550/arXiv.1703.06345)
        -  This paper focus to improve the performance of transfer learning in low resource setting.
        -  In this model, three neural network architectures are designed for the settings of cross-domain, cross-application, and cross-lingual transfer.
        -  The following factors are observed:
            - (a) label abundance for the target task
            - (b) relatedness between source and target tasks
            - (c) Number of parameters that can be shared
    - [A little annotation does a lot of good: A study in bootstrapping low-resource named entity recognizers - 2019](https://doi.org/10.48550/arXiv.1908.08983)
      -  using both simulated and real human annotation, we find a dual strategy approach best, starting with a cross lingual transferred model, then performing targeted annotation of only uncertain entity spans in the target language, minimizing annotator effort.
      -  (1)crosslingual transfer learning 
      -  (2) active learning to select relevant parts of the data to annotate
      -  (3) f ine-tuning of the model on these annotated segments
    

- Deep learning approach
    - [Named entity recognition for novel types by transfer learning - 2016](http://dx.doi.org/10.18653/v1/D16-1087)
        - This paper trains transfer learning in a training data in a related domain with similar named entity types and a small amount of in-domain training data.  

    - [Transfer learning for named-entity recognition with neural networks - 2017](https://doi.org/10.48550/arXiv.1705.06273)
        - This paper transfers am ANN model trained on a large labeled dataset to another dataset with a limited number of labels.
    - [Improve neural entity recognition via multi-task data selection and constrained decoding -2018](http://dx.doi.org/10.18653/v1/N18-2056)
        - This model improves the performance of neural network in NER using two techniques:
            - (i) Multi-Task Data Selection, which ensures the consistency of data distribution and labeling guidelines between source and target datasets. This technique removes noise from training data.
            - (ii) Constrained decoding using knowledge base. The decoder of the model operates at the document level, and leverages global and external information sources. 
    - [Neural adaptation layers for cross-domain named entity recognition - 2018](https://doi.org/10.48550/arXiv.1810.06368)
        -  This paper empirically investigate effective methods for conveniently adapting an existing, well-trained neural NER model for a new domain.
        -  This paper introduce e adaptation layers on top of existing neural architectures, where no re-training using the source domain data is required.
    - [Cross-Domain NER using Cross-Domain Language Modeling - 2019](http://dx.doi.org/10.18653/v1/P19-1236)
      - Transferring NER knowledge from the news domain to the target domain by contrasting large raw data in both domains through cross-domain LM(Language Modelling) training. 
      - cross-task links between NER and LM
      - cross-domain links
      - Introducing a new parameter
      - using BiLSTM and CRF and a shared embedding layer
      - This method has a modular control over cross-domain and cross-task knowledge transfer. 
    - [Low-resource name tagging learned with weakly labeled data - 2019](https://doi.org/10.48550/arXiv.1908.09659)
      - weakly labeled data is more practical to obtain but it is partially labeled sequence and massive noise in data.
      - Name tagging model that consists of two modules of sequence labeling and classification, which are combined via shared parameters.
      - splitting weakly labeled data from wikipedia anchors into high quality and noisy portions for traininch each module. 
      - sequence labeling module focuses on high-quality data and is costly due to the partial-CRFs layer.
      - classification module focuses on the annotated words in noisy data to pretrain the tag classifier efficiently
    - [Learning A unified named entity tagger from multiple partially annotated corpora for efficient adaptation - 2019](http://dx.doi.org/10.18653/v1/K19-1048)
      - a unified model that learns from multiple partially annotated datasets to make joint predictions on the union of entity types appearing in any training dataset.
      - Base model - BiLSTM-CRF with character level modelling.


Deep Active Learning for NER
- Active learning algorithm can query a user interactively to label data with the desired outputs
- [Towards robust linguistic analysis using ontonotes - 2013](https://aclanthology.org/W13-3516.pdf)
  - OntoNotes v5.0 corpus. This should set the benchmark for future development of various NLPcomponents in syntax and semantics, and possibly encourage research towards an integrated system that makes use of the various layers jointly to improve overall performance.

Reinforcement Learning for NER
- [Distantly supervised NER with partial annotation learning and reinforcement learning - 2018](https://aclanthology.org/C18-1183.pdf) 
    - To handle the problem of incomplete and noisy annotations.The proposed model consists of two parts NE tagger and instance selector( a policy network).The instance selector is trained based on the reward provided by NE Tagger.
    - NE tagger
      - Embedding(actions from instance selector)
      - Encoder(BiLSTM and MLP layer(Embedding))
      - Decoder (CRF(Encoder))
    - Instance selector
      - Reward(from CRF,action)
      - MLP(Encoder,reward)
      - action(MLP) 

Deep Adversarial learning for NER
- [Dual adversarial neural transfer for low-resource named entity recognition - 2019](http://dx.doi.org/10.18653/v1/P19-1336)
  - A transfer learning model DATNet( Dual Adversarial Neural Transfer ) for low-resource NER,which aims at addressing representation difference and resource data imbalance problems.
  - There are two variants DATNet-F (cross-language transfer) and DATNet-P (cross-domain transfer).
  - To handle Represntation difference - Bi-LSTM
  - To handle Resource data imbalance  - GRAD( Generalized Resource-Adversarial Discriminator)
   
- [Cross-lingual multi-level adversarial transfer to enhance low-resource name tagging - 2019](http://dx.doi.org/10.18653/v1/N19-1383)
  - Improving name tagging for low resource languages using annotations from related languages 
  - Integrates multi-level adversarial transfer into a BiLSTM-CRF to improve low-resource name tagging.
  - With word-level adversarial training, it can automatically project the source language into a shared semantic space with the target language without requiring any comparable data or bilingual gazetteers.
  - a sentence-level adversarial transfer to encourage the sequence encoder to extract language-agnostic features.

Neural attention for NER
- [Attending to characters in neural sequence labeling models - 2016](https://doi.org/10.48550/arXiv.1611.04361)
- [Neural named entity recognition using a self attention mechanism - 2017](https://sci-hub.se/https://doi.org/10.1109/ICTAI.2017.00104)

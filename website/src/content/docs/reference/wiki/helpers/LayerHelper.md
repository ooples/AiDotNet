---
title: "LayerHelper<T>"
description: "Provides helper methods for creating various neural network layer configurations."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Helpers`

Provides helper methods for creating various neural network layer configurations.

## How It Works

This class contains factory methods that create pre-configured sets of neural network layers
for common architectures like standard feed-forward networks, CNNs, ResNets, and more.

## Methods

| Method | Summary |
|:-----|:--------|
| `BlockFFN(Int32,IActivationFunction<>,IActivationFunction<>,Double)` | Emits the paper-faithful FFN block: `Conv1×1 expand 4× → BN → LeakyReLU → Conv1×1 contract → BN → Dropout`. |
| `CalculateVAEGroupCount(Int32,Int32)` | Calculates the optimal GroupNorm group count for VAE residual blocks. |
| `ChainResolveLazyLayers(IList<ILayer<>>,Int32[])` | Walks a sequential layer list and resolves each lazy layer's input shape from the previous layer's output (or `rootInputShape` for the first layer). |
| `ChooseDivisibleHeadConfig(Int32,Int32,Int32)` | Picks a head-count + head-dim pair for a MultiHeadAttention layer such that `heads × headDim == embedDim` exactly. |
| `ChooseGroupCount(Int32)` | Picks a GroupNormalization group count that evenly divides `numChannels`. |
| `CreateAudioVAEDecoderLayers(Int32,Int32,Int32,Int32[])` | Creates decoder layers for the AudioVAE. |
| `CreateAudioVAEEncoderLayers(Int32,Int32,Int32,Int32[])` | Creates encoder layers for the AudioVAE. |
| `CreateAudioVisualEventLocalizationLayers(Int32,Int32,Int32,Int32)` | Creates layers for the Audio-Visual Event Localization network. |
| `CreateBasicVSRPlusPlusLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates layers for the BasicVSRPlusPlus video super-resolution model. |
| `CreateBiomedParseDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the BiomedParse model. |
| `CreateBiomedParseEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the BiomedParse model. |
| `CreateBlip2Layers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates layers for the BLIP-2 native mode. |
| `CreateBlipLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates layers for the BLIP native mode. |
| `CreateCATSegDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the CATSeg model. |
| `CreateCATSegEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the CATSeg model. |
| `CreateCUPSDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the CUPS model. |
| `CreateCUPSEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the CUPS model. |
| `CreateConcertoDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the Concerto model. |
| `CreateConcertoEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the Concerto model. |
| `CreateControlNetEncoderLayers(Int32,Int32,Int32[],Int32)` | Creates layers for the ControlNet encoder. |
| `CreateDCCRNLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates DCCRN encoder, LSTM, and decoder layers. |
| `CreateDEVADecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the DEVA model. |
| `CreateDEVAEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the DEVA model. |
| `CreateDeepFilterNetLayers(Int32,Int32,Int32,Int32,Int32)` | Creates DeepFilterNet encoder, GRU, deep filtering, and decoder layers. |
| `CreateDeepONetBranchLayers(Int32,Int32,Int32,Int32)` | Creates branch network layers for DeepONet using yield pattern. |
| `CreateDeepONetTrunkLayers(Int32,Int32,Int32,Int32)` | Creates trunk network layers for DeepONet using yield pattern. |
| `CreateDefaultABINetLayers(Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default ABINet (Autonomous, Bidirectional, Iterative) layers. |
| `CreateDefaultACEStepLayers(Int32,Int32,Int32,Int32,Double,Int32)` | Creates default layers for ACE-Step accelerated music generation. |
| `CreateDefaultALIGNLayers(Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for the ALIGN model (EfficientNet CNN vision encoder + text transformer). |
| `CreateDefaultASTLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates the default audio classifier layers for AST (Gong et al. |
| `CreateDefaultASTLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default AST (Audio Spectrogram Transformer) layers following the paper architecture. |
| `CreateDefaultAcousticModelLayers(Int32,Int32,Int32,Int32,Int32,Int32,Double,Int32)` | Creates default layers for acoustic TTS models (Tacotron 2, FastSpeech 2, Grad-TTS, etc.). |
| `CreateDefaultAlphaFactorLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Double)` | Creates default layers for an AlphaFactorModel. |
| `CreateDefaultAnimateDiffLayers(Int32,Int32,Int32,Int32,Int32)` | Creates layers for an AnimateDiff motion module that adds temporal coherence. |
| `CreateDefaultAttentionAllocationLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for an AttentionAllocation model. |
| `CreateDefaultAttentionLayers(NeuralNetworkArchitecture<>)` | Creates a default set of attention-based layers for transformer-style architectures. |
| `CreateDefaultAudioDiffusionLayers(Int32,Int32,Int32)` | Creates default audio diffusion layers for mel spectrogram-based audio generation. |
| `CreateDefaultAudioFlamingo2Layers(Int32,Int32,Int32,Int32,Double)` | Creates default layers for Audio Flamingo 2 multimodal audio-language model. |
| `CreateDefaultAudioGenLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default AudioGen layers for text-to-audio generation. |
| `CreateDefaultAudioLDMClassifierLayers(Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for AudioLDM Classifier using diffusion features. |
| `CreateDefaultAudioLDMLayers(Int32,Int32,Int32,Int32,Int32[],Int32,Int32,Int32,Double)` | Creates default AudioLDM layers for text-to-audio generation using latent diffusion. |
| `CreateDefaultAudioLMLayers(Int32,Int32,Int32,Int32,Double)` | Creates default layers for AudioLM (semantic stage). |
| `CreateDefaultAudioMAELayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default Audio-MAE (Masked Autoencoders for Audio) layers for classification. |
| `CreateDefaultAudioSepLayers(Int32,Int32,Int32,Int32,Int32[],Int32,Int32,Double)` | Creates default layers for the AudioSep model. |
| `CreateDefaultAudioSuperResolutionLayers(Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for the Audio Super-Resolution model. |
| `CreateDefaultAutoDiffTabDenoiserLayers(Int32,Int32,Int32[],Int32,Double)` | Creates default denoiser layers for AutoDiffTab automated diffusion. |
| `CreateDefaultAutoEncoderLayers(NeuralNetworkArchitecture<>)` | Creates a default autoencoder neural network architecture. |
| `CreateDefaultAutoIntLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for an AutoInt (Automatic Feature Interaction) model. |
| `CreateDefaultAutoRegressiveVocoderLayers(Int32,Int32,Int32,Double)` | Creates default layers for autoregressive vocoders (WaveNet, WaveRNN). |
| `CreateDefaultAutoformerLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates the default layer configuration for an Autoformer model. |
| `CreateDefaultBASICLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for the BASIC model (CoAtNet hybrid CNN-Transformer vision + text transformer). |
| `CreateDefaultBEATsLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates the default layer configuration for a BEATs (Audio Pre-Training with Acoustic Tokenizers) audio event detection and classification model. |
| `CreateDefaultBGELayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default layers for a BGE (BAAI General Embedding) model. |
| `CreateDefaultBSRoFormerLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default BS-RoFormer layers for band-split music source separation. |
| `CreateDefaultBandSplitRNNEnhancerLayers(Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for Band-Split RNN enhancer (Luo and Yu, 2023). |
| `CreateDefaultBandSplitRNNSeparationLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for the BandSplitRNN source separation model. |
| `CreateDefaultBasicPitchLayers(Int32,Int32,Int32,Int32,Double)` | Creates default Basic Pitch multi-pitch detection layers. |
| `CreateDefaultBayesianNeuralNetworkLayers(NeuralNetworkArchitecture<>)` | Creates a default configuration of layers for a Bayesian neural network (Bayes-by-Backprop style). |
| `CreateDefaultBiLSTMCRFLayers(Int32,Int32,Int32,Int32,Int32,Double,Boolean,Int32,Int32,Boolean)` | Creates the default layer stack for a BiLSTM-CRF sequence labeling NER model. |
| `CreateDefaultBigVGANLayers(Int32,Int32,Int32[],Int32,Double)` | Creates default layers for BigVGAN universal vocoder. |
| `CreateDefaultBlackLittermanNeuralLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Double)` | Creates default layers for a BlackLittermanNeural model. |
| `CreateDefaultBlip2Layers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default layers for a BLIP-2 neural network. |
| `CreateDefaultBloombergGPTLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Double)` | Creates the default layer configuration for BloombergGPT. |
| `CreateDefaultBloombergGPTLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates the default layer configuration for BloombergGPT-style model. |
| `CreateDefaultBranchformerLayers(Int32,Int32,Int32,Int32,Int32,Int32,Double,Int32)` | Creates default layers for a Branchformer encoder with parallel attention + cgMLP branches. |
| `CreateDefaultBridgeFusionLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for BridgeTower with bridge connections between vision and text encoder layers. |
| `CreateDefaultByteTrackLayers(Int32,Int32,Int32,Int32,Int32)` | Creates default layers for ByteTrack multi-object tracking. |
| `CreateDefaultCAMPlusPlusLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for the CAM++ speaker verification model. |
| `CreateDefaultCCDMLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for the CCDM conditional continuous diffusion model. |
| `CreateDefaultCLAPAudioEncoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates the default audio-encoder layers for CLAP (Wu et al. |
| `CreateDefaultCLAPLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default CLAP (Contrastive Language-Audio Pre-training) audio encoder layers. |
| `CreateDefaultCLAPTextEncoderLayers(Int32,Int32,Int32,Int32,Int32,Int32)` | Creates the default text-encoder layers for CLAP (Wu et al. |
| `CreateDefaultCLIPTextLayers(Int32,Int32,Int32,Int32,Int32)` | Paper-faithful CLIP text encoder stack (Radford et al., ICML 2021). |
| `CreateDefaultCMGANLayers(Int32,Int32,Int32,Int32,Double)` | Creates default CMGAN layers for conformer-based speech enhancement. |
| `CreateDefaultCNNBiLSTMCRFLayers(Int32,Int32,Int32,Int32,Int32,Double,Int32,Int32,Int32)` | Creates the default layer stack for a CNN-BiLSTM-CRF Named Entity Recognition model. |
| `CreateDefaultCNNLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates a Convolutional Neural Network (CNN) with configurable layers. |
| `CreateDefaultCRAFTLayers(Int32,Int32,Int32)` | Creates default CRAFT layers for character-level text detection. |
| `CreateDefaultCREPELayers(Int32,Int32,Int32,Double)` | Creates default CREPE pitch detection layers. |
| `CreateDefaultCRNNEventDetectorLayers(Int32[],Int32,Int32,Int32,Int32,Double)` | Creates default layers for the CRNN Sound Event Detector. |
| `CreateDefaultCRNNLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default CRNN layers for sequence text recognition. |
| `CreateDefaultCSDILayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for the CSDI score-based diffusion model. |
| `CreateDefaultCSDILayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates the default layer configuration for CSDI (Conditional Score-based Diffusion for Imputation). |
| `CreateDefaultCTABGANPlusDiscriminatorLayers(Int32,Int32[],Double)` | Creates default discriminator layers for a CTAB-GAN+ discriminator. |
| `CreateDefaultCTABGANPlusGeneratorLayers(Int32,Int32,Int32[])` | Creates default generator layers for a CTAB-GAN+ generator. |
| `CreateDefaultCTCDecoderLayers(Int32,Int32,Int32,Int32,Int32,Int32,Double,Int32)` | Creates default layers for a CTC decoder ASR model. |
| `CreateDefaultCTGANDiscriminatorLayers(Int32,Int32[],Double)` | Creates default discriminator layers for a CTGAN discriminator. |
| `CreateDefaultCTGANGeneratorLayers(Int32,Int32,Int32[])` | Creates default generator layers for a CTGAN generator with residual connections. |
| `CreateDefaultCanaryLayers(Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for Canary multilingual ASR/ST model. |
| `CreateDefaultCapsuleNetworkLayers(NeuralNetworkArchitecture<>)` | Creates a default capsule network architecture. |
| `CreateDefaultCausalMultimodalLayers(Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for causal multimodal LLMs (KOSMOS-1, KOSMOS-2). |
| `CreateDefaultChatGLM3TextLayers(Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Paper-faithful ChatGLM3 text encoder stack (Zeng et al. |
| `CreateDefaultChronosBoltLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates the default layers for Chronos-Bolt (encoder-decoder with direct quantile output). |
| `CreateDefaultChronosLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layer configuration for Chronos time series foundation model. |
| `CreateDefaultClipLayers(NeuralNetworkArchitecture<>,Int32)` | Creates default layers for CLIP-style multimodal networks. |
| `CreateDefaultCodecLMLayers(Int32,Int32,Int32,Int32,Int32,Int32,Double,Int32)` | Creates default layers for codec-based LM TTS (VALL-E, CosyVoice, Fish Speech, Bark, etc.). |
| `CreateDefaultCogVideoLayers(Int32,Int32,Int32,Int32,Int32,Int32)` | Creates layers for a CogVideo text-to-video generation model. |
| `CreateDefaultColBERTLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default layers for a ColBERT (Contextualized Late Interaction over BERT) model. |
| `CreateDefaultConformerFPLayers(Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates layers for ConformerFP (Conformer-based fingerprinting). |
| `CreateDefaultConformerLayers(Int32,Int32,Int32,Int32,Int32,Int32,Double,Int32)` | Creates default layers for the Conformer ASR model (Gulati et al., 2020). |
| `CreateDefaultConformerTransducerLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double,Int32)` | Creates default layers for a Conformer with RNN-T (Transducer) decoder. |
| `CreateDefaultCopulaGANDiscriminatorLayers(Int32,Int32[],Double)` | Creates default discriminator layers for a CopulaGAN discriminator. |
| `CreateDefaultCopulaGANGeneratorLayers(Int32,Int32,Int32[])` | Creates default generator layers for a CopulaGAN generator. |
| `CreateDefaultCosyVoice2Layers(Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for CosyVoice2 streaming TTS. |
| `CreateDefaultCrossAttentionResamplerVLMLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double,Int32)` | Creates layers for Qwen-VL-style VLMs using cross-attention resampler with learned queries. |
| `CreateDefaultCrossModalFusionLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for cross-modal fusion models (LXMERT). |
| `CreateDefaultCrossformerLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for the Crossformer (Cross-Dimension Transformer) architecture. |
| `CreateDefaultCutieLayers(Int32,Int32,Int32,Int32)` | Creates layers for a Cutie video object segmentation model. |
| `CreateDefaultDACLayers(Int32,Int32[],Int32,Double)` | Creates default layers for the DAC (Descript Audio Codec) model. |
| `CreateDefaultDBNetLayers(Int32,Int32,Int32)` | Creates default layers for DBNet text detection model. |
| `CreateDefaultDCRNNLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates the default layers for a DCRNN (Diffusion Convolutional Recurrent Neural Network). |
| `CreateDefaultDIFRINTLayers(Int32,Int32,Int32,Int32,Int32)` | Creates default layers for DIFRINT video stabilization. |
| `CreateDefaultDNCLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32)` | Creates a default Differentiable Neural Computer (DNC) with pre-configured layers. |
| `CreateDefaultDPCTGANDiscriminatorLayers(Int32,Int32[],Double)` | Creates default discriminator layers for a DP-CTGAN discriminator. |
| `CreateDefaultDPCTGANGeneratorLayers(Int32,Int32,Int32[])` | Creates default generator layers for a DP-CTGAN generator. |
| `CreateDefaultDannaSepLayers(Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for Danna-Sep dual-path music source separation. |
| `CreateDefaultData2Vec2Layers(Int32,Int32,Int32,Int32,Double)` | Creates default layers for data2vec 2.0. |
| `CreateDefaultDecoderOnlyVisionLayers(Int32,Int32,Int32,Int32,Double)` | Creates layers for decoder-only vision models with no separate vision encoder. |
| `CreateDefaultDeepARLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates layers for DeepAR probabilistic autoregressive model. |
| `CreateDefaultDeepBeliefNetworkLayers(NeuralNetworkArchitecture<>)` | Creates a default Deep Belief Network (DBN) with pre-configured layers. |
| `CreateDefaultDeepBoltzmannMachineLayers(NeuralNetworkArchitecture<>)` | Creates default layers for a Deep Boltzmann Machine (DBM). |
| `CreateDefaultDeepCNNCTCLayers(Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for a deep 1D-CNN based CTC model (Jasper/QuartzNet style). |
| `CreateDefaultDeepFactorLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates the default layer configuration for a DeepFactor (Deep Factor Model). |
| `CreateDefaultDeepOperatorNetworkLayers(Int32,Int32,Int32,Int32,Int32)` | Creates default layers for a Deep Operator Network (DeepONet). |
| `CreateDefaultDeepPortfolioLayers(NeuralNetworkArchitecture<>,Int32)` | Creates the default layer configuration for a Deep Portfolio Management model. |
| `CreateDefaultDeepPortfolioLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Double)` | Creates default layers for a DeepPortfolioManager model. |
| `CreateDefaultDeepQNetworkLayers(NeuralNetworkArchitecture<>)` | Creates a default Deep Q-Network (DQN) with pre-configured layers for reinforcement learning. |
| `CreateDefaultDeepRitzLayers(NeuralNetworkArchitecture<>,Int32,Int32)` | Creates default layers for the Deep Ritz Method network. |
| `CreateDefaultDeepStateLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates the default layer configuration for a DeepState (Deep State Space) model. |
| `CreateDefaultDemucsNoiseLayers(Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for the Demucs Noise model. |
| `CreateDefaultDenseNetLayers(NeuralNetworkArchitecture<>,DenseNetConfiguration)` | Creates default layers for a DenseNet network based on the specified configuration. |
| `CreateDefaultDepthAnythingV2Layers(Int32,Int32,Int32,Int32,Int32)` | Creates default layers for Depth Anything V2 monocular depth estimation model. |
| `CreateDefaultDessurtLayers(Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default Dessurt (self-supervised document transformer) layers. |
| `CreateDefaultDiTLayers(Int32,Int32,Int32,Int32,Int32)` | Creates default DiT (Diffusion Transformer) layers with AdaLN-Zero conditioning. |
| `CreateDefaultDiTLayers(Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default DiT (Document Image Transformer) layers. |
| `CreateDefaultDiffusionTSLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates the default layer configuration for DiffusionTS (Interpretable Diffusion for Time Series). |
| `CreateDefaultDiffusionVocoderLayers(Int32,Int32,Int32,Int32,Double)` | Creates default layers for diffusion-based vocoders (DiffWave, WaveGrad, PriorGrad, FreGrad). |
| `CreateDefaultDistilledT5TextLayers(Int32,Int32,Int32,Int32,Int32,Int32)` | Paper-faithful DistilledT5 text encoder stack. |
| `CreateDefaultDocBankLayers(Int32,Int32,Int32,Int32)` | Creates default layers for DocBank page segmentation model. |
| `CreateDefaultDocFormerLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default DocFormer layers for document understanding with shared spatial encodings. |
| `CreateDefaultDocGCNLayers(Int32,Int32,Int32,Int32)` | Creates default DocGCN (Document Graph Convolutional Network) layers. |
| `CreateDefaultDocOwlLayers(Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default DocOwl (mPLUG-DocOwl) layers for document understanding. |
| `CreateDefaultDocumentOCRLayers(Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates layers for document understanding VLMs with reading-order attention. |
| `CreateDefaultDonutLayers(Int32,Int32,Int32,Int32,Int32[],Int32[],Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default Donut layers for OCR-free document understanding. |
| `CreateDefaultDualStreamFusionLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for dual-stream vision-language fusion models (ViLBERT, METER). |
| `CreateDefaultEASTLayers(Int32,Int32,Int32,String)` | Creates default EAST (Efficient and Accurate Scene Text Detector) layers. |
| `CreateDefaultEATLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default EAT (Efficient Audio Transformer) layers following the paper architecture. |
| `CreateDefaultECAPATDNNLanguageIdentifierLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32[])` | Creates default ECAPA-TDNN layers for spoken language identification. |
| `CreateDefaultECAPATDNNSpeakerLayers(Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default ECAPA-TDNN speaker embedding layers. |
| `CreateDefaultEDVRLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default layers for EDVR video restoration. |
| `CreateDefaultELMLayers(NeuralNetworkArchitecture<>,Int32)` | Creates default layers for an Extreme Learning Machine (ELM) neural network. |
| `CreateDefaultESNLayers(Int32,Int32,Int32,Double,Double)` | Creates a default Echo State Network (ESN) with pre-configured layers. |
| `CreateDefaultETSformerLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for the ETSformer (Exponential Smoothing Transformer) architecture. |
| `CreateDefaultEditingInstructionLayers(Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates layers for instruction-conditioned image editing VLMs. |
| `CreateDefaultEfficientNetLayers(NeuralNetworkArchitecture<>,EfficientNetConfiguration)` | Creates default layers for an EfficientNet network based on the specified configuration. |
| `CreateDefaultEmotion2VecLayers(Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default emotion2vec layers for speech emotion recognition. |
| `CreateDefaultEnCodecLayers(Int32[],Int32,Double)` | Creates default EnCodec encoder-decoder layers. |
| `CreateDefaultEncoderDecoderVLMLayers(Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for encoder-decoder generative VLMs (GIT, CoCa, PaLI, PaLI-X, PaLI-3). |
| `CreateDefaultFDYSEDLayers(Int32[],Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for the FDY-SED (Frequency Dynamic SED) model. |
| `CreateDefaultFEDformerLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates the default layer configuration for a FEDformer (Frequency Enhanced Decomposed Transformer) model. |
| `CreateDefaultFILMLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32)` | Creates default layers for a FILM (Frame Interpolation for Large Motion) model. |
| `CreateDefaultFLAVRLayers(Int32,Int32,Int32,Int32,Int32)` | Creates default layers for FLAVR frame interpolation. |
| `CreateDefaultFRCRNLayers(Int32,Int32,Int32,Int32,Double)` | Creates default layers for FRCRN (Zhao et al., 2022). |
| `CreateDefaultFTTransformerLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for an FT-Transformer (Feature Tokenizer + Transformer) model. |
| `CreateDefaultFactorTransformerLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for a FactorTransformer model. |
| `CreateDefaultFactorVAELayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Double)` | Creates default layers for a FactorVAE model. |
| `CreateDefaultFastConformerLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for Fast Conformer (8x downsampled Conformer). |
| `CreateDefaultFastDVDNetLayers(Int32,Int32,Int32,Int32,Int32)` | Creates default layers for FastDVDNet video denoising. |
| `CreateDefaultFastTextLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32)` | Creates default layers for a FastText model. |
| `CreateDefaultFeedForwardLayers(NeuralNetworkArchitecture<>,Int32,Int32)` | Creates default layers for a feed-forward neural network. |
| `CreateDefaultFinBERTLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for FinBERT (Financial BERT) model. |
| `CreateDefaultFinBERTToneLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates the default layer configuration for FinBERT-tone. |
| `CreateDefaultFinBERTToneLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates the default layer configuration for FinBERT-Tone (sentiment-focused FinBERT). |
| `CreateDefaultFinDiffDenoiserLayers(Int32,Int32,Int32[])` | Creates default denoiser MLP layers for a FinDiff financial diffusion generator. |
| `CreateDefaultFinDiffTimestepProjectionLayers(Int32)` | Creates default timestep projection layers for a FinDiff generator. |
| `CreateDefaultFinGPTLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Double)` | Creates the default layer configuration for FinGPT. |
| `CreateDefaultFinGPTLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates the default layer configuration for FinGPT (Financial GPT). |
| `CreateDefaultFinMALayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates the default layer configuration for FinMA. |
| `CreateDefaultFinMALayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates the default layer configuration for FinMA (Financial Multi-Agent). |
| `CreateDefaultFinancialBERTLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Double)` | Creates the default layer configuration for FinancialBERT. |
| `CreateDefaultFinancialBERTLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double,String)` | Creates the default layer configuration for FinancialBERT (domain-adapted financial BERT). |
| `CreateDefaultFishSpeechLayers(Int32,Int32,Int32,Int32,Double)` | Creates default layers for Fish Speech semantic LM. |
| `CreateDefaultFlorence2Layers(Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for Florence-2 (DaViT vision encoder + multi-task decoder). |
| `CreateDefaultFlowFormerLayers(Int32,Int32,Int32,Int32,Int32)` | Creates default layers for FlowFormer optical flow estimation. |
| `CreateDefaultFlowMatchingTTSLayers(Int32,Int32,Int32,Int32,Int32,Int32,Double,Int32)` | Creates default layers for flow-matching TTS (F5-TTS, Matcha-TTS, E2-TTS, MaskGCT). |
| `CreateDefaultFlowStateLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Double,Int32,Boolean)` | Creates the default layers for FlowState (SSM-based foundation model). |
| `CreateDefaultFoundationASRLayers(Int32,Int32,Int32,Int32,Int32,Int32,Double,Int32)` | Creates default layers for a foundation model (wav2vec2/HuBERT/WavLM) fine-tuned for ASR. |
| `CreateDefaultFoundationModelLayers(Int32,Int32,Int32,Int32,Int32,Double,Int32)` | Creates default layers for self-supervised speech foundation models (HuBERT, WavLM, wav2vec 2.0). |
| `CreateDefaultFourierNeuralOperatorLayers(NeuralNetworkArchitecture<>,Int32[],Int32,Int32,Int32)` | Creates default layers for a Fourier Neural Operator (FNO). |
| `CreateDefaultFrameInterpolationLayers(Int32,Int32,Int32,Int32)` | Creates layers for a frame interpolation model (FILM/RIFE-style). |
| `CreateDefaultFullSubNetPlusLayers(Int32,Int32,Int32,Int32,Int32,Double)` | Creates default FullSubNet+ layers for speech enhancement. |
| `CreateDefaultGANDALFLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double,Double,Boolean,Double)` | Creates default layers for a GANDALF (Gated Additive Neural Decision Forest) model. |
| `CreateDefaultGNNLayers(NeuralNetworkArchitecture<>)` | Creates default layers for a Graph Neural Network (GNN). |
| `CreateDefaultGOGGLEDecoderLayers(Int32,Int32,Int32[])` | Creates default decoder layers for GOGGLE graph-based generation. |
| `CreateDefaultGOGGLEEncoderLayers(Int32,Int32,Int32,Int32)` | Creates default GNN encoder layers for GOGGLE graph-based generation. |
| `CreateDefaultGPT4TSLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for the GPT4TS model (frozen GPT-2 backbone with task heads). |
| `CreateDefaultGRULayers(NeuralNetworkArchitecture<>)` | Creates a default Gated Recurrent Unit (GRU) neural network layer configuration. |
| `CreateDefaultGemmaTextLayers(Int32,Int32,Int32,Int32,Int32,Double)` | Paper-faithful Gemma text encoder stack (Gemma Team 2024). |
| `CreateDefaultGenreClassifierLayers(Int32,Int32,Int32,Int32,Int32,Double)` | Creates default genre classification layers. |
| `CreateDefaultGloVeLayers(NeuralNetworkArchitecture<>,Int32,Int32)` | Creates default layers for a GloVe (Global Vectors) model. |
| `CreateDefaultGraFPrintLayers(Int32,Int32,Int32,Int32,Int32,Double)` | Creates layers for GraFPrint graph neural network fingerprinting. |
| `CreateDefaultGraphAttentionLayers(NeuralNetworkArchitecture<>,Int32,Int32,Double)` | Creates default layers for a Graph Attention Network (GAT). |
| `CreateDefaultGraphClassificationLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Double)` | Creates default layers for a Graph Classification model. |
| `CreateDefaultGraphGenerationLayers(NeuralNetworkArchitecture<>,Int32,Int32)` | Creates default layers for a Graph Generation model (VGAE encoder). |
| `CreateDefaultGraphIsomorphismLayers(NeuralNetworkArchitecture<>,Int32,Int32,Boolean,Double)` | Creates default layers for a Graph Isomorphism Network (GIN). |
| `CreateDefaultGraphSAGELayers(NeuralNetworkArchitecture<>,SAGEAggregatorType,Int32,Boolean)` | Creates default layers for a GraphSAGE (Graph Sample and Aggregate) Network. |
| `CreateDefaultGraphWaveNetLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default layers for GraphWaveNet (Graph WaveNet for spatial-temporal modeling). |
| `CreateDefaultGroundingDetectionLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates layers for grounding/detection VLMs with cross-modal feature fusion. |
| `CreateDefaultHTDemucsLayers(Int32,Int32,Int32,Int32,Int32,Double)` | Creates default HTDemucs layers for hybrid transformer music source separation. |
| `CreateDefaultHTMLayers(NeuralNetworkArchitecture<>,Int32,Int32,Double)` | Creates a default Hierarchical Temporal Memory (HTM) neural network layer configuration. |
| `CreateDefaultHTSATLayers(Int32,Int32,Int32,Int32[],Int32,Int32,Double)` | Creates default HTS-AT (Hierarchical Token-Semantic Audio Transformer) layers. |
| `CreateDefaultHamiltonianLayers(NeuralNetworkArchitecture<>,Int32,Int32)` | Creates default layers for a Hamiltonian Neural Network. |
| `CreateDefaultHiFiGANLayers(Int32,Int32,Int32,Int32[],Int32[],Int32[])` | Paper-faithful HiFi-GAN generator (Kong et al. |
| `CreateDefaultHiFiGANLayers(Int32,Int32,Int32[],Int32,Double)` | Creates default layers for the HiFi-GAN vocoder (Kong et al., 2020). |
| `CreateDefaultHierarchicalRiskParityLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Double)` | Creates default layers for a HierarchicalRiskParity model. |
| `CreateDefaultHippoLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Boolean,Int32)` | Creates default layers for the HiPPO (High-order Polynomial Projection Operators) architecture. |
| `CreateDefaultHuBERTSERLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default HuBERT-SER layers for speech emotion recognition. |
| `CreateDefaultITransformerLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates the default layer configuration for an iTransformer (Inverted Transformer) model. |
| `CreateDefaultInfographicVQALayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default InfographicVQA layers for infographic understanding. |
| `CreateDefaultInformerLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates the default layer configuration for an Informer model. |
| `CreateDefaultInstructorLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default layers for an Instructor/E5 (Instruction-Tuned) embedding model. |
| `CreateDefaultInternVideo2Layers(Int32,Int32,Int32,Int32,Int32,Int32)` | Creates layers for an InternVideo2-style video understanding model. |
| `CreateDefaultInvestLMLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Double)` | Creates the default layer configuration for InvestLM. |
| `CreateDefaultInvestLMLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates the default layer configuration for InvestLM (Investment Language Model). |
| `CreateDefaultKairosLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32[],Int32,Int32,Int32,Int32,Double)` | Creates the default layers for Kairos (Mixture-of-Size Encoder). |
| `CreateDefaultKronosLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for the Kronos financial market foundation model. |
| `CreateDefaultLLMASRLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double,Int32)` | Creates default layers for an LLM-integrated ASR model. |
| `CreateDefaultLLMTimeLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for the LLM-Time model (LLM-based zero-shot forecasting). |
| `CreateDefaultLLaVAMLPProjectorLayers(Int32,Int32,Int32,Int32,Int32,Int32,Double,Int32,Int32)` | Creates layers for LLaVA-style VLMs using a 2-layer MLP cross-modal connector. |
| `CreateDefaultLSMLayers(NeuralNetworkArchitecture<>,Int32,Double,Double,Double,Double)` | Creates a default configuration of layers for a Liquid State Machine (LSM) neural network. |
| `CreateDefaultLSTMCRFLayers(Int32,Int32,Int32,Int32,Int32,Double)` | Creates the default layer stack for an LSTM-CRF Named Entity Recognition model. |
| `CreateDefaultLSTMNetworkLayers(NeuralNetworkArchitecture<>)` | Creates a default configuration of layers for a Long Short-Term Memory (LSTM) neural network. |
| `CreateDefaultLSTNetLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates the default layers for an LSTNet (Long Short-Term Time-series Network) model. |
| `CreateDefaultLagLlamaLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layer configuration for Lag-Llama time series foundation model. |
| `CreateDefaultLagrangianLayers(NeuralNetworkArchitecture<>,Int32,Int32)` | Creates default layers for a Lagrangian Neural Network. |
| `CreateDefaultLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32)` | Creates a standard feed-forward neural network with configurable hidden layers. |
| `CreateDefaultLayoutGraphLayers(Int32,Int32,Int32,Int32)` | Creates default LayoutGraph layers for graph-based layout analysis. |
| `CreateDefaultLayoutLMLayers(Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default LayoutLM (v1) layers for document understanding with layout-aware pre-training. |
| `CreateDefaultLayoutLMv2Layers(Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default LayoutLMv2 layers for document understanding with visual features. |
| `CreateDefaultLayoutLMv3Layers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default LayoutLMv3 layers for document understanding. |
| `CreateDefaultLayoutXLMLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default LayoutXLM layers for multilingual document understanding. |
| `CreateDefaultLiLTLayers(Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default LiLT (Language-Independent Layout Transformer) layers. |
| `CreateDefaultLinkPredictionLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Double)` | Creates default layers for a Link Prediction model encoder. |
| `CreateDefaultMATCHALayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default MATCHA (chart understanding) layers. |
| `CreateDefaultMERTLayers(Int32,Int32,Int32,Int32,Double)` | Creates default layers for MERT music foundation model. |
| `CreateDefaultMGTSDLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for the MG-TSD multi-granularity diffusion model. |
| `CreateDefaultMMDiTLayers(Int32,Int32,Int32,Int32)` | Creates default MMDiT (Multi-Modal Diffusion Transformer) layers with joint attention. |
| `CreateDefaultMOIRAILayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32[],Int32,Int32,Int32,Int32,Int32,Double)` | Creates the default layers for MOIRAI (Salesforce's Universal Time Series Foundation Model). |
| `CreateDefaultMOMENTLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double,Nullable<Int32>)` | Creates the default layers for MOMENT (Multi-task Optimization through Masked Encoding for Time series) foundation model. |
| `CreateDefaultMPSENetLayers(Int32,Int32,Int32,Int32,Int32,Double,Int32)` | Creates default layers for MP-SENet (Lu et al., 2023). |
| `CreateDefaultMQCNNLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates the default layer configuration for an MQCNN (Multi-Quantile CNN) model. |
| `CreateDefaultMRLLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default layers for a Matryoshka Representation Learning (MRL) model. |
| `CreateDefaultMT3Layers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates layers for MT3 multi-track music transcription (Gardner et al., 2022). |
| `CreateDefaultMTGNNLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default layers for MTGNN (Multivariate Time-series Graph Neural Network). |
| `CreateDefaultMadmomBeatTrackerLayers(Int32,Int32,Int32,Double)` | Creates layers for Madmom neural beat tracker (Bock et al., 2016). |
| `CreateDefaultMamba2Layers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double,Int32,Int32)` | Creates default layers for the Mamba-2 time series forecasting model. |
| `CreateDefaultMambaLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double,Int32)` | Creates default layers for the Mamba (Selective State Space Model). |
| `CreateDefaultMambularLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for a Mambular (State Space Model for Tabular) model. |
| `CreateDefaultMarbleNetLayers(Int32,Int32,Int32,Int32,Double)` | Creates layers for MarbleNet separable convolutional VAD (NVIDIA NeMo). |
| `CreateDefaultMarketMakingLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32)` | Creates the default layer configuration for a Market Making agent. |
| `CreateDefaultMatchaTTSLayers(Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for Matcha-TTS (flow-matching TTS). |
| `CreateDefaultMedSynthLayers(Int32,Int32,Int32[],Int32[],Double)` | Creates default MLP layers for MedSynth medical synthetic data generation. |
| `CreateDefaultMelBandRoFormerLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default MelBand-RoFormer layers for mel-band music source separation. |
| `CreateDefaultMelodyExtractorLayers(Int32,Int32,Int32,Int32,Double)` | Creates layers for neural Melody Extractor. |
| `CreateDefaultMemoryNetworkLayers(NeuralNetworkArchitecture<>,Int32,Int32)` | Creates a default Memory Network layer configuration. |
| `CreateDefaultMeshCNNLayers(NeuralNetworkArchitecture<>,Int32,Int32[],Int32[],Int32[],Int32,Boolean,Double,Boolean)` | Creates default layers for a MeshCNN architecture for mesh classification/segmentation. |
| `CreateDefaultMiDaSLayers(Int32,Int32,Int32,Int32,Int32)` | Creates default layers for MiDaS depth estimation. |
| `CreateDefaultMobileNetV2Layers(NeuralNetworkArchitecture<>,MobileNetV2Configuration)` | Creates default layers for a MobileNetV2 network based on the specified configuration. |
| `CreateDefaultMobileNetV3Layers(NeuralNetworkArchitecture<>,MobileNetV3Configuration)` | Creates default layers for a MobileNetV3 network based on the specified configuration. |
| `CreateDefaultMusicFlamingoLayers(Int32,Int32,Int32,Int32,Double)` | Creates default layers for Music Flamingo music-language model. |
| `CreateDefaultMusicGenLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default MusicGen layers for text-to-music generation. |
| `CreateDefaultMusicStructureAnalyzerLayers(Int32,Int32,Int32,Int32,Int32,Double)` | Creates layers for Music Structure Analyzer. |
| `CreateDefaultMusicTaggingTransformerLayers(Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates layers for Music Tagging Transformer (Won et al., 2021). |
| `CreateDefaultNBEATSLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates layers for N-BEATS (Neural Basis Expansion Analysis for Time Series) model. |
| `CreateDefaultNHiTSLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates layers for N-HiTS (Neural Hierarchical Interpolation for Time Series) model. |
| `CreateDefaultNODELayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Boolean,Double)` | Creates default layers for a NODE (Neural Oblivious Decision Ensembles) model. |
| `CreateDefaultNTMLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32)` | Creates a default configuration of layers for a Neural Turing Machine (NTM). |
| `CreateDefaultNeuralCVaRLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Double)` | Creates default layers for a NeuralCVaR model. |
| `CreateDefaultNeuralFPLayers(Int32,Int32,Int32,Int32,Double)` | Creates default NeuralFP audio fingerprinting layers. |
| `CreateDefaultNeuralGARCHLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Double)` | Creates default layers for a NeuralGARCH model. |
| `CreateDefaultNeuralNetworkLayers(NeuralNetworkArchitecture<>)` | Creates a default configuration of layers for a standard neural network. |
| `CreateDefaultNeuralParametricEQLayers(Int32,Int32,Int32,Int32,Double)` | Creates default layers for the Neural Parametric EQ model. |
| `CreateDefaultNeuralStressTestLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Double)` | Creates default layers for a NeuralStressTest model. |
| `CreateDefaultNeuralVaRLayers(NeuralNetworkArchitecture<>,Int32)` | Creates the default layer configuration for a Neural Value-at-Risk (VaR) model. |
| `CreateDefaultNeuralVaRLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Double)` | Creates default layers for a NeuralVaR model. |
| `CreateDefaultNodeClassificationLayers(NeuralNetworkArchitecture<>,Int32,Int32,Double)` | Creates default layers for a Node Classification model. |
| `CreateDefaultNonStationaryTransformerLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for a Non-stationary Transformer model. |
| `CreateDefaultNougatLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default Nougat layers for academic document understanding. |
| `CreateDefaultOccupancyLayers(NeuralNetworkArchitecture<>)` | Creates default layers for an occupancy detection neural network without temporal data. |
| `CreateDefaultOccupancyTemporalLayers(NeuralNetworkArchitecture<>,Int32)` | Creates default layers for an occupancy detection neural network with temporal data. |
| `CreateDefaultOnsetsAndFramesLayers(Int32,Int32,Int32,Int32,Int32,Double)` | Creates default Onsets and Frames piano transcription layers. |
| `CreateDefaultOpenCLIPLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for an OpenCLIP contrastive vision-language encoder. |
| `CreateDefaultOpenSoraLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default layers for an OpenSora video generation model. |
| `CreateDefaultOpticalFlowLayers(Int32,Int32,Int32,Int32)` | Creates layers for an optical flow estimation model (RAFT-style). |
| `CreateDefaultPANNsLayers(Int32,Int32,Double)` | Creates the default CNN14 audio classifier layers for PANNs (Kong et al. |
| `CreateDefaultPANNsLayers(Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default PANNs CNN14 layers for audio classification. |
| `CreateDefaultPATEGANDiscriminatorLayers(Int32,Int32,Int32[])` | Creates default PATE-GAN teacher/student discriminator layers. |
| `CreateDefaultPATEGANGeneratorLayers(Int32,Int32,Int32[])` | Creates default PATE-GAN generator layers with residual connections. |
| `CreateDefaultPICKLayers(Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default PICK layers for key information extraction. |
| `CreateDefaultPINNLayers(NeuralNetworkArchitecture<>,Int32,Int32)` | Creates default layers for a Physics-Informed Neural Network (PINN). |
| `CreateDefaultPSENetLayers(Int32,Int32,Int32,Int32)` | Creates default PSENet (Progressive Scale Expansion Network) layers. |
| `CreateDefaultParaformerLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double,Int32)` | Creates default layers for a Paraformer non-autoregressive ASR model. |
| `CreateDefaultPatchTSTLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates the default layers for a PatchTST (Patch Time Series Transformer) network. |
| `CreateDefaultPeakNetFPLayers(Int32,Int32,Int32,Int32,Double)` | Creates layers for PeakNetFP spectral peak-based fingerprinting. |
| `CreateDefaultPengiLayers(Int32,Int32,Int32,Double)` | Creates default layers for Pengi audio language model. |
| `CreateDefaultPerceiverResamplerLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for perceiver-resampler VLMs (OpenFlamingo, IDEFICS, IDEFICS2, IDEFICS3). |
| `CreateDefaultPix2StructLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default Pix2Struct layers for screenshot parsing. |
| `CreateDefaultPixelShuffleProjectorLayers(Int32,Int32,Int32,Int32,Int32,Int32,Double,Int32)` | Creates layers for InternVL-style VLMs using pixel shuffle spatial compression. |
| `CreateDefaultPointCloudVLMLayers(Int32,Int32,Int32,Int32,Int32,Double)` | Creates layers for 3D point cloud VLMs with point-based encoder. |
| `CreateDefaultProprietaryAPILayers(Int32,Int32,Int32,Int32,Int32,Double)` | Creates lightweight layers for proprietary API wrapper VLMs. |
| `CreateDefaultProprietaryASRLayers(Int32,Int32,Int32,Int32,Int32,Int32,Double,Int32)` | Creates default layers for a proprietary API wrapper ASR model. |
| `CreateDefaultProprietaryTTSLayers(Int32,Int32,Int32,Int32,Int32,Double,Int32)` | Creates default layers for proprietary API TTS wrappers (ElevenLabs, Azure, Google, etc.). |
| `CreateDefaultPyAnnoteLayers(Int32,Int32,Int32,Int32,Int32,Double)` | Creates default pyannote 3.x segmentation layers. |
| `CreateDefaultQFormerGenerativeLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double,Int32)` | Creates default layers for Q-Former-based generative VLMs (InstructBLIP, BLIP-3, MiniGPT-4, MiniGPT-v2). |
| `CreateDefaultQuailVadLayers(Int32,Int32,Int32,Int32,Int32)` | Creates default layers for Quail VAD lightweight voice activity detection. |
| `CreateDefaultQuantumNetworkLayers(NeuralNetworkArchitecture<>,Int32)` | Creates a default configuration of layers for a Quantum Neural Network. |
| `CreateDefaultQwen2AudioLayers(Int32,Int32,Int32,Int32,Double)` | Creates default layers for Qwen2-Audio (audio encoder + adapter). |
| `CreateDefaultQwen2TextLayers(Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Paper-faithful Qwen2 text encoder stack (Yang et al. |
| `CreateDefaultRBFNetworkLayers(NeuralNetworkArchitecture<>,Int32,IRadialBasisFunction<>)` | Creates a default Radial Basis Function (RBF) neural network layer configuration. |
| `CreateDefaultREaLTabFormerLayers(Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default autoregressive transformer layers for REaLTabFormer. |
| `CreateDefaultRNNLayers(NeuralNetworkArchitecture<>)` | Creates a default Recurrent Neural Network (RNN) layer configuration. |
| `CreateDefaultRNNTransducerLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for RNN-Transducer (encoder + prediction + joint). |
| `CreateDefaultRVMLayers(Int32,Int32,Int32,Int32)` | Creates default layers for RVM (Robust Video Matting). |
| `CreateDefaultRWKV7LanguageModelLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Double,Int32)` | Creates default layers for the RWKV-7 "Goose" language model. |
| `CreateDefaultRWKVForecastingLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for the RWKV time series forecasting model. |
| `CreateDefaultRealizedVolatilityTransformerLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for a RealizedVolatilityTransformer model. |
| `CreateDefaultRelationalGCNLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default layers for RelationalGCN (Relational Graph Convolutional Network). |
| `CreateDefaultResNet1DLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32)` | Creates ResNet layers for 1D (flat vector) input using Dense layers with residual connections. |
| `CreateDefaultResNet2DLayers(NeuralNetworkArchitecture<>,Int32[],Int32,Int32)` | Creates ResNet layers for 2D input by treating it as a single-channel image. |
| `CreateDefaultResNet3DLayers(NeuralNetworkArchitecture<>,Int32[],Int32,Int32)` | Creates ResNet layers for 3D input (standard CNN-based ResNet). |
| `CreateDefaultResNetLayers(NeuralNetworkArchitecture<>,Int32,Int32)` | Creates a Residual Neural Network (ResNet) with configurable blocks. |
| `CreateDefaultRoboticsActionLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates layers for robotics VLMs with action token output head. |
| `CreateDefaultRoomImpulseResponseLayers(Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for Neural Room Impulse Response estimation. |
| `CreateDefaultS4Layers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Boolean,Int32,Int32)` | Creates default layers for the S4 (Structured State Space Sequence) model. |
| `CreateDefaultSAINTLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for a SAINT model. |
| `CreateDefaultSALMONNLayers(Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for SALMONN (dual encoder + Q-Former). |
| `CreateDefaultSAM2Layers(Int32,Int32,Int32,Int32)` | Creates all SAM2 layers for backward compatibility. |
| `CreateDefaultSCNetLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for the SCNet (Sparse Compression Network) source separation model. |
| `CreateDefaultSD15UNetLayers(Int32,Int32,Int32[],Int32,Int32)` | Creates default U-Net layers for Stable Diffusion 1.5 architecture. |
| `CreateDefaultSDXLUNetLayers(Int32,Int32,Int32,Int32)` | Creates default U-Net layers for SDXL architecture with dual cross-attention. |
| `CreateDefaultSECBERTLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Double)` | Creates the default layer configuration for SEC-BERT. |
| `CreateDefaultSECBERTLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double,String)` | Creates the default layer configuration for SEC-BERT (Securities and Exchange Commission BERT). |
| `CreateDefaultSGPTLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default layers for an SGPT (Sentence GPT) decoder-only embedding model. |
| `CreateDefaultSPLADELayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default layers for a SPLADE (Sparse Lexical and Expansion Model) embedding model. |
| `CreateDefaultSTGNNLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default layers for the STGNN (Spatio-Temporal Graph Neural Network) model. |
| `CreateDefaultSVTRLayers(Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default SVTR (Scene Text Visual Transformer Recognizer) layers. |
| `CreateDefaultScoreGradLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32)` | Creates default layers for the ScoreGrad (Score-based Gradient) model. |
| `CreateDefaultSiameseLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32)` | Creates default layers for a Siamese neural network using a Transformer-based encoder. |
| `CreateDefaultSigLIP2Layers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Boolean,Double,Int32,Int32)` | Creates default layers for SigLIP 2 (Multilingual Vision-Language Encoders with Improved Semantic Understanding). |
| `CreateDefaultSigLIP2TextLayers(Int32,Int32,Int32,Int32,Int32)` | Paper-faithful SigLIP2 text encoder stack (Tschannen et al. |
| `CreateDefaultSigLIPLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for a SigLIP contrastive vision-language encoder. |
| `CreateDefaultSigLIPTextLayers(Int32,Int32,Int32,Int32,Int32)` | Paper-faithful SigLIP text encoder stack (Zhai et al., ICCV 2023). |
| `CreateDefaultSimCSELayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default layers for a SimCSE (Simple Contrastive Learning of Sentence Embeddings) model. |
| `CreateDefaultSimMTMLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for the SimMTM masked modeling with similarity model. |
| `CreateDefaultSingleStreamFusionLayers(Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for single-stream vision-language fusion models (VisualBERT, UNITER, Oscar, VinVL, ViLT). |
| `CreateDefaultSlowFastLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates all SlowFast layers for backward compatibility (returns only slow pathway). |
| `CreateDefaultSoundStreamLayers(Int32[],Int32,Int32,Double)` | Creates default SoundStream encoder-decoder layers. |
| `CreateDefaultSourceSeparationLayers(Int32,Int32,Int32,Int32,Double)` | Creates default music source separation layers (U-Net style). |
| `CreateDefaultSpanBasedNERLayers(Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates the default layer stack for a span-based NER model (SpERT, BiaffineNER, PURE). |
| `CreateDefaultSpeakerEmbeddingLayers(Int32,Int32,Int32,Int32,Int32,Double)` | Creates default speaker embedding layers for speaker verification and identification. |
| `CreateDefaultSpeakerLMLayers(Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for SpeakerLM language-model-based speaker recognition. |
| `CreateDefaultSpikingFullSubNetLayers(Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for Spiking-FullSubNet speech enhancement. |
| `CreateDefaultSpikingLayers(NeuralNetworkArchitecture<>,SpikingNeuronType,Double,Double,Boolean,Boolean,Int32)` | Creates default layers for a Spiking Neural Network (SNN). |
| `CreateDefaultSpiralNetLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32[],Double[],Int32[],Boolean,Double,Boolean)` | Creates the default layer sequence for a SpiralNet mesh neural network. |
| `CreateDefaultSqueezeformerLayers(Int32,Int32,Int32,Int32,Int32,Int32,Double,Int32)` | Creates default layers for a Squeezeformer encoder with temporal U-Net structure. |
| `CreateDefaultStableAudioLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default Stable Audio layers for text-to-audio generation. |
| `CreateDefaultStreamingConformerLayers(Int32,Int32,Int32,Int32,Int32,Int32,Double,Int32)` | Creates default layers for a streaming Conformer with chunked attention. |
| `CreateDefaultStyleTTS2Layers(Int32,Int32,Int32,Int32,Int32,Int32,Double,Int32)` | Creates default layers for the StyleTTS 2 TTS model (Li et al., 2023). |
| `CreateDefaultStyleTTSLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double,Int32)` | Creates default layers for style/emotion TTS (StyleTTS, StyleTTS 2, EmotiVoice). |
| `CreateDefaultSundialLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates the default layers for Sundial (decoder-only foundation model). |
| `CreateDefaultT5TextLayers(Int32,Int32,Int32,Int32,Int32,Int32)` | Paper-faithful T5 text encoder stack (Raffel et al., JMLR 2020). |
| `CreateDefaultTCNLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Double,Boolean)` | Creates the default layers for a TCN (Temporal Convolutional Network) model. |
| `CreateDefaultTESTLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for the TEST model (text-aligned time series embeddings). |
| `CreateDefaultTFCLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for the TF-C time-frequency consistency model. |
| `CreateDefaultTFGridNetLayers(Int32,Int32,Int32,Int32,Int32,Double)` | Creates default TF-GridNet layers for time-frequency grid speech enhancement. |
| `CreateDefaultTFTLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates the default layers for a Temporal Fusion Transformer (TFT) architecture. |
| `CreateDefaultTOTEMLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for the TOTEM VQ-VAE tokenization model. |
| `CreateDefaultTOTOLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for the TOTO observability foundation model. |
| `CreateDefaultTRIELayers(Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default TRIE (Text Reading and Information Extraction) layers. |
| `CreateDefaultTS2VecLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for the TS2Vec contrastive learning model. |
| `CreateDefaultTSDiffLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32)` | Creates the default layer configuration for TSDiff (Time Series Diffusion). |
| `CreateDefaultTSDiffLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for the TSDiff self-guided diffusion model. |
| `CreateDefaultTSMixerLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Double,Double)` | Creates default layers for a TSMixer model. |
| `CreateDefaultTVAEDecoderLayers(Int32,Int32,Int32[])` | Creates default TVAE decoder layers: hidden layers with ReLU, final identity output layer. |
| `CreateDefaultTVAEEncoderLayers(Int32,Int32,Int32[])` | Creates default TVAE encoder layers: hidden layers with ReLU, final layer outputs 2*latentDim (mean and logvar concatenated). |
| `CreateDefaultTabDDPMDenoiserLayers(Int32,Int32[],Double)` | Creates default TabDDPM denoiser MLP layers: SiLU hidden layers with optional dropout. |
| `CreateDefaultTabDPTLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for a TabDPT (Tabular Data Pre-Training) model. |
| `CreateDefaultTabFlowTimeProjectionLayers(Int32)` | Creates default time projection layers for a TabFlow generator. |
| `CreateDefaultTabFlowVelocityLayers(Int32,Int32,Int32[],Double)` | Creates default velocity MLP layers for a TabFlow generator. |
| `CreateDefaultTabMLayers(NeuralNetworkArchitecture<>,Int32,Int32[],Int32,Double)` | Creates default layers for a TabM (Parameter-Efficient Ensemble) model. |
| `CreateDefaultTabNetLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Double)` | Creates default layers for a TabNet model. |
| `CreateDefaultTabPFNLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for a TabPFN (Prior-Fitted Network) model. |
| `CreateDefaultTabRLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Double)` | Creates default layers for a TabR (Retrieval-Augmented) model. |
| `CreateDefaultTabSynDecoderLayers(Int32,Int32,Int32[])` | Creates default VAE decoder layers for TabSyn latent diffusion. |
| `CreateDefaultTabSynDiffusionLayers(Int32,Int32,Int32[])` | Creates default latent diffusion denoiser layers for TabSyn. |
| `CreateDefaultTabSynEncoderLayers(Int32,Int32,Int32[])` | Creates default VAE encoder layers for TabSyn latent diffusion. |
| `CreateDefaultTabTransformerLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for a TabTransformer model. |
| `CreateDefaultTableTransformerLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default layers for TableTransformer model. |
| `CreateDefaultTabularClassifierLayers(Int32,Int32,Int32[])` | Creates default classifier MLP layers for TableGAN and CTAB-GAN+ auxiliary classification. |
| `CreateDefaultTabularGANDiscriminatorLayers(Int32,Int32,Int32[],Double)` | Creates default discriminator MLP layers for GAN-based tabular generators. |
| `CreateDefaultTabularGANGeneratorLayers(Int32,Int32,Int32[],Boolean,Double)` | Creates default generator MLP layers for GAN-based tabular generators (CopulaGAN, DP-CTGAN, CTAB-GAN+, TableGAN). |
| `CreateDefaultTempogramLayers(Int32,Int32,Int32,Int32,Double)` | Creates layers for Tempogram neural tempo estimation. |
| `CreateDefaultTemporalGCNLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default layers for TemporalGCN (Temporal Graph Convolutional Network). |
| `CreateDefaultTimeBridgeLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for the TimeBridge non-stationarity foundation model. |
| `CreateDefaultTimeDiffLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for the TimeDiff non-autoregressive diffusion model. |
| `CreateDefaultTimeGANEmbedderLayers(Int32,Int32,Int32[])` | Creates default embedder layers for TimeGAN's embedding network. |
| `CreateDefaultTimeGANRecoveryLayers(Int32,Int32,Int32[])` | Creates default recovery layers for TimeGAN's recovery network. |
| `CreateDefaultTimeGANSupervisorLayers(Int32,Int32[])` | Creates default supervisor layers for TimeGAN's supervisor network. |
| `CreateDefaultTimeGPTLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for the TimeGPT-style model. |
| `CreateDefaultTimeGradLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for the TimeGrad diffusion model (RNN encoder + denoising network). |
| `CreateDefaultTimeGradLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default layers for the TimeGrad (Diffusion for Time Series) architecture. |
| `CreateDefaultTimeLLMLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates the default layers for Time-LLM (LLM Reprogramming for Time Series). |
| `CreateDefaultTimeMAELayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for the TimeMAE masked autoencoder model. |
| `CreateDefaultTimeMachineLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Boolean,Int32)` | Creates default layers for the TimeMachine (Time Series State Space Model) architecture. |
| `CreateDefaultTimeMoELayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates the default layers for Time-MoE (Mixture of Experts foundation model). |
| `CreateDefaultTimeSformerLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Boolean)` | Creates default layers for TimeSformer video classification. |
| `CreateDefaultTimerLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for the Timer (Generative Pre-Training) model. |
| `CreateDefaultTimesFMLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double,Int32)` | Creates the default layer configuration for a TimesFM (Time Series Foundation Model). |
| `CreateDefaultTimesNetLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for the TimesNet (Temporal 2D-Variation Modeling) architecture. |
| `CreateDefaultTinyTimeMixersLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates the default layers for Tiny Time Mixers (TTM) foundation model. |
| `CreateDefaultTitaNetLayers(Int32,Int32,Int32,Int32,Int32,Double)` | Creates default TitaNet speaker embedding layers. |
| `CreateDefaultTokenReductionVLMLayers(Int32,Int32,Int32,Int32,Int32,Int32,Double,Int32)` | Creates layers for VLMs with semantic-aware token reduction/downsampling. |
| `CreateDefaultTrOCRLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default layers for TrOCR text recognition model. |
| `CreateDefaultTransformerLayers(TransformerArchitecture<>)` | Creates a default Transformer neural network with pre-configured encoder and decoder layers. |
| `CreateDefaultTransformerNERLayers(Int32,Int32,Int32,Int32,Int32,Int32,Double,Boolean)` | Creates the default layer stack for a transformer-based NER model (BERT-NER, RoBERTa-NER, etc.). |
| `CreateDefaultTtsLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default TTS (Text-to-Speech) layers for speech synthesis. |
| `CreateDefaultUDOPLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default UDOP layers for unified document processing. |
| `CreateDefaultUNet3DLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32)` | Creates default layers for a 3D U-Net architecture for volumetric segmentation. |
| `CreateDefaultUniTSLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32[],Double,String,Int32)` | Creates default layers for the UniTS (Unified Time Series) model. |
| `CreateDefaultUnifiedBidirectionalLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates layers for unified bidirectional VLMs with dual decoder heads. |
| `CreateDefaultUnifiedGenerationLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for unified generation VLMs (Emu, Emu2, Emu3). |
| `CreateDefaultUnifiedMultimodalLayers(NeuralNetworkArchitecture<>,Int32,Int32)` | Creates default layers for the UnifiedMultimodalNetwork. |
| `CreateDefaultUniversalDELayers(NeuralNetworkArchitecture<>,Int32,Int32)` | Creates default layers for a Universal Differential Equation (UDE) network. |
| `CreateDefaultVAEDecoderLayers(Int32,Int32,Int32)` | Creates default VAE decoder layers for latent diffusion models. |
| `CreateDefaultVAEEncoderLayers(Int32,Int32,Int32)` | Creates default VAE encoder layers for latent diffusion models. |
| `CreateDefaultVAELayers(NeuralNetworkArchitecture<>,Int32)` | Creates a default Variational Autoencoder (VAE) with pre-configured layers. |
| `CreateDefaultVALLELayers(Int32,Int32,Int32,Int32,Double)` | Creates default layers for VALL-E AR stage. |
| `CreateDefaultVGGLayers(NeuralNetworkArchitecture<>,VGGConfiguration)` | Creates layers for a VGG network based on the specified configuration. |
| `CreateDefaultVITSLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double,Int32)` | Creates default layers for VITS-family end-to-end TTS (VITS, VITS2, YourTTS, Piper, MeloTTS). |
| `CreateDefaultVRTLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates layers for a VRT (Video Restoration Transformer) model. |
| `CreateDefaultVariationalPINNLayers(NeuralNetworkArchitecture<>,Int32,Int32)` | Creates default layers for a Variational Physics-Informed Neural Network (VPINN). |
| `CreateDefaultViTLayers(Int32,Int32,Int32,Double,Int32)` | Creates default layers for a standard Vision Transformer (ViT) encoder. |
| `CreateDefaultVideoDenoisingLayers(Int32,Int32,Int32,Int32,Int32)` | Creates layers for a video denoising model (U-Net style temporal denoiser). |
| `CreateDefaultVideoInpaintingLayers(Int32,Int32,Int32,Int32)` | Creates layers for a video inpainting model (encoder-transformer-decoder). |
| `CreateDefaultVideoMAELayers(Int32,Int32,Int32,Int32,Int32,Int32)` | Creates default layers for VideoMAE (Video Masked Autoencoder) action recognition model. |
| `CreateDefaultVideoStabilizationLayers(Int32,Int32,Int32)` | Creates layers for a video stabilization model (StabNet-style). |
| `CreateDefaultVideoSuperResolutionLayers(Int32,Int32,Int32,Int32,Int32,Int32,Boolean)` | Creates layers for a video super-resolution model (Real-ESRGAN/BasicVSR++ style). |
| `CreateDefaultVideoTemporalVLMLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double,Int32,Int32,Int32,Int32)` | Creates layers for video VLMs with temporal aggregation across frames. |
| `CreateDefaultVideoUNetLayers(Int32,Int32,Int32,Int32)` | Creates default video U-Net layers with temporal attention for video diffusion models. |
| `CreateDefaultVisionAdapterLayers(Int32,Int32,Int32,Int32,Int32,Int32,Double,Int32)` | Creates layers for VLMs with vision adapter and learnable gating. |
| `CreateDefaultVisionTSLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates the default layers for VisionTS (Visual MAE for time series). |
| `CreateDefaultVisualExpertVLMLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for visual expert VLM architecture (CogVLM pattern). |
| `CreateDefaultVocoderLayers(Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for GAN-based neural vocoders (HiFi-GAN, MelGAN, BigVGAN, etc.). |
| `CreateDefaultVocosLayers(Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for Vocos ISTFT vocoder. |
| `CreateDefaultVoiceCloningLayers(Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for voice cloning TTS (OpenVoice, MetaVoice, XTTS, Chatterbox). |
| `CreateDefaultVoiceCraftLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Double)` | Creates the default layer stack for a VoiceCraft codec language model. |
| `CreateDefaultVoxLingua107Layers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32[])` | Creates default VoxLingua107 layers for 107-language identification. |
| `CreateDefaultVoxelCNNLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32)` | Creates default layers for a Voxel-based 3D Convolutional Neural Network. |
| `CreateDefaultWav2SmallLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for the Wav2Small lightweight SER model. |
| `CreateDefaultWav2Vec2LanguageIdentifierLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default Wav2Vec2 layers for spoken language identification. |
| `CreateDefaultWavLMSERLayers(Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for the WavLM-SER emotion recognition model. |
| `CreateDefaultWavLMSpeakerLayers(Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for the WavLM Speaker verification model. |
| `CreateDefaultWaveNetLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates the default layers for a WaveNet model adapted for time series forecasting. |
| `CreateDefaultWaveNetVocoderLayers(Int32,Int32,Int32,Int32,Int32)` | Paper-faithful WaveNet-style vocoder generator (Parallel WaveGAN, Yamamoto et al. |
| `CreateDefaultWebRTCVadLayers(Int32,Int32,Int32,Double)` | Creates layers for WebRTC neural VAD. |
| `CreateDefaultWhisperEncoderDecoderLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double,Int32)` | Creates default layers for a Whisper-style encoder-decoder ASR. |
| `CreateDefaultWhisperLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for Whisper-style speech recognition models. |
| `CreateDefaultWhisperLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default Whisper layers for automatic speech recognition. |
| `CreateDefaultWord2VecLayers(NeuralNetworkArchitecture<>,Int32,Int32)` | Creates default layers for a Word2Vec model (Skip-Gram or CBOW). |
| `CreateDefaultXMemLayers(Int32,Int32,Int32,Int32)` | Creates layers for an XMem long-term video object segmentation model. |
| `CreateDefaultYingLongLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates default layers for the YingLong enterprise foundation model. |
| `CreateDefaultYuELayers(Int32,Int32,Int32,Int32,Double)` | Creates default layers for YuE semantic stage. |
| `CreateDefaultZipformerLayers(Int32[],Int32[],Int32[],Int32,Int32,Double)` | Creates default layers for Zipformer (U-Net-style multi-resolution encoder). |
| `CreateDiTNoisePredictorLayers(Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates layers for the DiT (Diffusion Transformer) noise predictor. |
| `CreateDiffCutDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates the decoder layers for DiffCut. |
| `CreateDiffCutEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates the DiffCut encoder layers using diffusion UNet features. |
| `CreateDiffCutSegmentationDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the DiffCutSegmentation model. |
| `CreateDiffCutSegmentationEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the DiffCutSegmentation model. |
| `CreateDiffSegDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates the decoder layers for DiffSeg. |
| `CreateDiffSegEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates the DiffSeg encoder layers using diffusion self-attention features. |
| `CreateDiffWaveLayers(Int32,Int32,Int32,Int32)` | Creates layers for the DiffWave audio diffusion network. |
| `CreateDonutDecoderLayers(Int32,Int32,Int32,Int32,Int32,Int32)` | Creates Donut decoder layers (BART-style) using yield pattern. |
| `CreateDonutEncoderLayers(Int32,Int32,Int32,Int32,Int32[],Int32[],Int32,Int32,Int32)` | Creates Donut encoder layers (Swin Transformer-B) using yield pattern. |
| `CreateDreamFusionNeRFLayers(Int32,Int32,Int32,Int32)` | Creates layers for the DreamFusion NeRF network. |
| `CreateE2FGVILayers(Int32,Int32,Int32,Int32,Int32)` | Creates layers for the E2FGVI video inpainting model. |
| `CreateEagleLayers(Int32,Int32,Int32,Int32,Int32)` | Creates layers for the Eagle (RWKV-5) language model architecture. |
| `CreateEdgeSAMDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the EdgeSAM model. |
| `CreateEdgeSAMEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the EdgeSAM model. |
| `CreateEfficientSAMDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the EfficientSAM model. |
| `CreateEfficientSAMEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the EfficientSAM model. |
| `CreateEfficientTAMDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the EfficientTAM model. |
| `CreateEfficientTAMEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the EfficientTAM model. |
| `CreateEoMTDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates the lightweight mask head layers for EoMT. |
| `CreateEoMTEncoderLayers(Int32,Int32,Int32,Int32,Int32[],Double)` | Creates the DINOv2 ViT encoder layers for EoMT (encoder-only architecture). |
| `CreateFILMLayers(Int32,Int32,Int32,Int32,Int32)` | Creates layers for the FILM frame interpolation model. |
| `CreateFalconMambaLayers(Int32,Int32,Int32,Int32,Int32,Int32)` | Creates layers for the FalconMamba language model architecture. |
| `CreateFastSAMDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the FastSAM model. |
| `CreateFastSAMEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the FastSAM model. |
| `CreateFinchLayers(Int32,Int32,Int32,Int32,Int32)` | Creates layers for the Finch (RWKV-6) language model architecture. |
| `CreateFlamingoLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates layers for the Flamingo native mode. |
| `CreateGLALayers(Int32,Int32,Int32,Int32,Int32)` | Creates layers for the GLA (Gated Linear Attention) language model architecture. |
| `CreateGLaMMDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the GLaMM model. |
| `CreateGLaMMEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the GLaMM model. |
| `CreateGMFlowLayers(Int32,Int32,Int32,Int32,Int32)` | Creates layers for the GMFlow optical flow model. |
| `CreateGatedDeltaNetLayers(Int32,Int32,Int32,Int32,Int32)` | Creates layers for the GatedDeltaNet language model architecture. |
| `CreateGpt4VisionLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates layers for the GPT-4 Vision native mode. |
| `CreateGriffinLayers(Int32,Int32,Int32,Int32)` | Creates layers for the Griffin language model architecture. |
| `CreateGroundedSAM2DecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the GroundedSAM2 model. |
| `CreateGroundedSAM2EncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the GroundedSAM2 model. |
| `CreateHawkLayers(Int32,Int32,Int32,Int32)` | Creates layers for the Hawk language model architecture. |
| `CreateHopeNetworkLayers(Int32,Int32,Int32,Int32)` | Creates layers for the HopeNetwork (self-modifying recurrent neural network). |
| `CreateIPAdapterLayers(Int32,Int32,Int32,Int32,Int32,Int32)` | Creates layers for the IP-Adapter image encoder. |
| `CreateImageBindLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates layers for the ImageBind native mode. |
| `CreateInstantNGPLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates layers for the Instant-NGP (Neural Graphics Primitives) model. |
| `CreateInternImageDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates the UPerNet decoder layers for InternImage. |
| `CreateInternImageEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates the InternImage encoder layers using deformable convolution approximation. |
| `CreateJambaLayers(Int32,Int32,Int32,Int32,Int32,Int32)` | Creates layers for the Jamba hybrid language model architecture. |
| `CreateKMaXDeepLabDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the KMaXDeepLab model. |
| `CreateKMaXDeepLabEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the KMaXDeepLab model. |
| `CreateLISADecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the LISA model. |
| `CreateLISAEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the LISA model. |
| `CreateLLaVALayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates layers for the LLaVA (Large Language and Vision Assistant) native mode. |
| `CreateMMDiTNoisePredictorLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Creates layers for the MMDiT (Multi-Modal Diffusion Transformer) noise predictor. |
| `CreateMOMENTClassificationHead(Int32,Int32,Int32)` | Builds MOMENT's classification head per Goswami et al. |
| `CreateMOMENTReconstructionHead(Int32,Int32,Int32)` | Builds MOMENT's per-patch reconstruction head per Goswami et al. |
| `CreateMamba2Layers(Int32,Int32,Int32,Int32,Int32,Int32)` | Creates layers for the Mamba-2 language model architecture. |
| `CreateMambaLayers(Int32,Int32,Int32,Int32,Int32,Int32)` | Creates layers for the Mamba language model architecture. |
| `CreateMask2FormerDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for Mask2Former's transformer decoder with masked cross-attention. |
| `CreateMask2FormerEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for Mask2Former's backbone (Swin/ResNet + pixel decoder). |
| `CreateMaskAdapterDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the MaskAdapter model. |
| `CreateMaskAdapterEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the MaskAdapter model. |
| `CreateMaskDINODecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates the mask decoder layers for Mask DINO. |
| `CreateMaskDINOEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates the backbone encoder layers for Mask DINO. |
| `CreateMedNeXtDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the MedNeXt model. |
| `CreateMedNeXtEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the MedNeXt model. |
| `CreateMedSAM2DecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the MedSAM2 model. |
| `CreateMedSAM2EncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the MedSAM2 model. |
| `CreateMedSAMDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the MedSAM model. |
| `CreateMedSAMEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the MedSAM model. |
| `CreateMedSegDiffV2DecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the MedSegDiffV2 model. |
| `CreateMedSegDiffV2EncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the MedSegDiffV2 model. |
| `CreateMedSegDiffV2SegmentationDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the MedSegDiffV2Segmentation model. |
| `CreateMedSegDiffV2SegmentationEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the MedSegDiffV2Segmentation model. |
| `CreateMobileSAMDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the MobileSAM model. |
| `CreateMobileSAMEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the MobileSAM model. |
| `CreateNeRFLayers(Int32,Int32,Int32,Int32,Int32,Int32)` | Creates layers for the NeRF (Neural Radiance Field) model. |
| `CreateNeuralNoiseReducerLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates NeuralNoiseReducer encoder, bottleneck, and decoder layers. |
| `CreateNnUNetDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the NnUNet model. |
| `CreateNnUNetEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the NnUNet model. |
| `CreateODISEDecoderLayers(Int32[],Int32,Int32)` | Builds the ODISE decoder: an SD-U-Net-style transposed-conv upsampling ladder that reverses the encoder's stride schedule to restore the spatial bottleneck back to full input resolution for dense per-pixel prediction (Xu et al. |
| `CreateODISEEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the ODISE model. |
| `CreateODISESegmentationDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the ODISESegmentation model. |
| `CreateODISESegmentationEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the ODISESegmentation model. |
| `CreateOMGLLaVADecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the OMGLLaVA model. |
| `CreateOMGLLaVAEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the OMGLLaVA model. |
| `CreateOMGSegDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates the task-query decoder layers for OMG-Seg. |
| `CreateOMGSegEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates the shared backbone encoder layers for OMG-Seg. |
| `CreateOneFormerDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for OneFormer. |
| `CreateOneFormerEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for OneFormer's backbone with text-conditioned features. |
| `CreateOpenVocabSAMDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the OpenVocabSAM model. |
| `CreateOpenVocabSAMEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the OpenVocabSAM model. |
| `CreatePIDNetDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the PIDNet model. |
| `CreatePIDNetEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the PIDNet model. |
| `CreatePixelLMDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the PixelLM model. |
| `CreatePixelLMEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the PixelLM model. |
| `CreatePointTransformerV3DecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the PointTransformerV3 model. |
| `CreatePointTransformerV3EncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the PointTransformerV3 model. |
| `CreateProPainterLayers(Int32,Int32,Int32,Int32,Int32,Int32)` | Creates layers for the ProPainter video inpainting model. |
| `CreateQueryMeldNetDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates the query-meld decoder layers for QueryMeldNet. |
| `CreateQueryMeldNetEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates the backbone encoder layers for QueryMeldNet. |
| `CreateRAFTLayers(Int32,Int32,Int32,Int32,Int32,Int32)` | Creates layers for the RAFT optical flow model. |
| `CreateRIFELayers(Int32,Int32,Int32,Int32,Int32)` | Creates layers for the RIFE frame interpolation model. |
| `CreateRWKV4Layers(Int32,Int32,Int32,Int32)` | Creates layers for the RWKV-4 language model architecture. |
| `CreateRWKV7Layers(Int32,Int32,Int32,Int32,Double,Int32)` | Creates layers for the RWKV-7 language model architecture. |
| `CreateRecurrentGemmaLayers(Int32,Int32,Int32,Int32)` | Creates layers for the RecurrentGemma language model architecture. |
| `CreateRepViTSAMDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the RepViTSAM model. |
| `CreateRepViTSAMEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the RepViTSAM model. |
| `CreateResNetConvLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32)` | Creates convolutional ResNet layers for 2D/3D image-like input. |
| `CreateResidualBlock(Int32,Int32,Int32,Int32,Boolean)` | Creates a residual block for ResNet-style architectures. |
| `CreateSAM21DecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates the mask decoder layers for SAM 2.1. |
| `CreateSAM21EncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates the Hiera backbone encoder layers for SAM 2.1. |
| `CreateSAM2ImageEncoderLayers(Int32,Int32,Int32,Int32)` | Creates the image encoder layers for SAM2 (Segment Anything Model 2). |
| `CreateSAM2IoUHead(Int32,Int32,Int32,Int32)` | Creates the IoU (Intersection over Union) prediction head for SAM2. |
| `CreateSAM2MaskDecoderLayers(Int32,Int32,Int32)` | Creates the shared mask decoder refinement layers for SAM2. |
| `CreateSAM2MaskHead(Int32,Int32,Int32,Int32)` | Creates the mask prediction head for SAM2. |
| `CreateSAM2MemoryLayers(Int32,Int32,Int32)` | Creates the memory attention layers for SAM2 temporal consistency. |
| `CreateSAM2OcclusionHead(Int32,Int32,Int32)` | Creates the occlusion prediction head for SAM2. |
| `CreateSAM2PromptEncoderLayers(Int32,Int32,Int32)` | Creates the prompt encoder layers for SAM2 (point, box, and mask prompts). |
| `CreateSAMDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates the lightweight mask decoder layers for SAM. |
| `CreateSAMEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates the ViT encoder layers for SAM (Segment Anything Model). |
| `CreateSAMHQDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates the HQ mask decoder layers for SAM-HQ. |
| `CreateSAMHQEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates the ViT encoder layers for SAM-HQ (High-Quality Segment Anything). |
| `CreateSANDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the SAN model. |
| `CreateSANEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the SAN model. |
| `CreateSEDDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the SED model. |
| `CreateSEDEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the SED model. |
| `CreateSEEMDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the SEEM model. |
| `CreateSEEMEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the SEEM model. |
| `CreateSambaLayers(Int32,Int32,Int32,Int32,Int32,Int32)` | Creates layers for the Samba hybrid language model architecture. |
| `CreateSegFormerDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates the MLP decode head layers for SegFormer. |
| `CreateSegFormerEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Int32[],Double)` | Creates the Mix Transformer (MiT) encoder layers for SegFormer. |
| `CreateSegGPTDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the SegGPT model. |
| `CreateSegGPTEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the SegGPT model. |
| `CreateSegMambaDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the SegMamba model. |
| `CreateSegMambaEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the SegMamba model. |
| `CreateSegNeXtDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates the Hamburger decoder layers for SegNeXt. |
| `CreateSegNeXtEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates the MSCAN (Multi-Scale Convolutional Attention Network) encoder layers for SegNeXt. |
| `CreateSileroVadLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates SileroVad convolutional, LSTM, and output layers. |
| `CreateSimpleVideoSuperResolutionLayers(Int32,Int32,Int32,Int32)` | Creates a simple super-resolution architecture for testing and lightweight use. |
| `CreateSlimSAMDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the SlimSAM model. |
| `CreateSlimSAMEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the SlimSAM model. |
| `CreateSlowFastFastPathwayLayers(Int32,Int32,Int32,Int32)` | Creates the fast pathway layers for SlowFast video recognition. |
| `CreateSlowFastFusionLayers(Int32,Int32,Int32,Int32,Int32)` | Creates the fusion and classification layers for SlowFast. |
| `CreateSlowFastSlowPathwayLayers(Int32,Int32,Int32,Int32)` | Creates the slow pathway layers for SlowFast video recognition. |
| `CreateSonataDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the Sonata model. |
| `CreateSonataEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the Sonata model. |
| `CreateSpeechEmotionRecognizerLayers(Int32,Int32,Int32,Int32,Int32,Double,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates SpeechEmotionRecognizer convolutional, dense, and output layers. |
| `CreateStableVideoDiffusionLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates layers for the Stable Video Diffusion model. |
| `CreateStandardVAEDecoderLayers(Int32,Int32,Int32,Int32[],Int32,Int32,Int32)` | Creates decoder layers for the StandardVAE (Stable Diffusion VAE architecture). |
| `CreateStandardVAEEncoderLayers(Int32,Int32,Int32,Int32[],Int32,Int32,Int32)` | Creates encoder layers for the StandardVAE (Stable Diffusion VAE architecture). |
| `CreateSwinUNETRDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the SwinUNETR model. |
| `CreateSwinUNETREncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the SwinUNETR model. |
| `CreateTacotron2Layers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates Tacotron2 encoder, attention, decoder, and post-net layers. |
| `CreateTemporalVAEDecoderLayers(Int32,Int32,Int32,Int32[],Int32,Int32,Int32)` | Creates decoder layers for the TemporalVAE (Stable Video Diffusion VAE architecture). |
| `CreateTemporalVAEEncoderLayers(Int32,Int32,Int32,Int32[],Int32,Int32,Int32)` | Creates encoder layers for the TemporalVAE (Stable Video Diffusion VAE architecture). |
| `CreateTransUNetDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the TransUNet model. |
| `CreateTransUNetEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the TransUNet model. |
| `CreateTransformerEmbeddingLayers(Int32,Int32,Int32,Int32,Int32,Int32)` | Creates layers for the TransformerEmbeddingNetwork. |
| `CreateU2SegDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates the mask decoder layers for U2Seg. |
| `CreateU2SegEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates the Swin-T encoder layers for U2Seg (unsupervised segmentation). |
| `CreateUMambaDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the UMamba model. |
| `CreateUMambaEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the UMamba model. |
| `CreateUNINEXTDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates the query-based decoder layers for UNINEXT. |
| `CreateUNINEXTEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates the backbone encoder layers for UNINEXT. |
| `CreateUNetNoisePredictorDecoderLayers(Int32,Int32,Int32[],Int32,Int32,Int32)` | Creates decoder layers for the UNet noise predictor (Stable Diffusion architecture). |
| `CreateUNetNoisePredictorEncoderLayers(Int32,Int32,Int32[],Int32,Int32,Int32,Int32,Int32)` | Creates encoder layers for the UNet noise predictor (Stable Diffusion architecture). |
| `CreateUViTNoisePredictorLayers(Int32,Int32,Int32,Int32,Int32)` | Creates layers for the UViT (U-ViT) noise predictor with skip connections. |
| `CreateUniVSDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the UniVS model. |
| `CreateUniVSEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the UniVS model. |
| `CreateUniverSegDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the UniverSeg model. |
| `CreateUniverSegEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the UniverSeg model. |
| `CreateVITSLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32[])` | Creates VITS text encoder, duration predictor, flow, and decoder layers. |
| `CreateVMambaDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the VMamba model. |
| `CreateVMambaEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the VMamba model. |
| `CreateViMUNetDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the ViMUNet model. |
| `CreateViMUNetEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the ViMUNet model. |
| `CreateViTAdapterDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates the decoder layers for ViT-Adapter. |
| `CreateViTAdapterEncoderLayers(Int32,Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates the ViT-Adapter encoder layers. |
| `CreateViTCoMerDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates the decoder layers for ViT-CoMer. |
| `CreateViTCoMerEncoderLayers(Int32,Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates the ViT-CoMer hybrid encoder layers combining CNN and transformer features. |
| `CreateVideoCLIPLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates layers for the VideoCLIP video-text understanding model. |
| `CreateVideoCLIPLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates layers for the VideoCLIP native mode. |
| `CreateVideoLISADecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the VideoLISA model. |
| `CreateVideoLISAEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the VideoLISA model. |
| `CreateVideoUNetPredictorDecoderLayers(Int32,Int32,Int32[],Int32,Int32,Int32)` | Creates decoder layers for the VideoUNet noise predictor. |
| `CreateVideoUNetPredictorEncoderLayers(Int32,Int32,Int32[],Int32,Int32,Int32,Int32,Int32,Int32)` | Creates layers for the VideoUNet noise predictor (Stable Video Diffusion architecture). |
| `CreateVisionMambaClassifierLayers(Int32,Int32,Int32,Int32)` | Creates MambaBlock layers for the Vision Mamba classifier model. |
| `CreateVisionMambaDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the VisionMamba model. |
| `CreateVisionMambaEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the VisionMamba model. |
| `CreateVisionMha(Int32,Int32,IInitializationStrategy<>)` | Builds the vision-encoder `MultiHeadAttentionLayer` for a VLM factory, with head count adjusted to divide `visionDim` exactly. |
| `CreateVisionTransformerLayers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates layers for the Vision Transformer (ViT) architecture. |
| `CreateWav2Vec2Layers(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32[],Int32[],Int32[])` | Creates Wav2Vec2 feature encoder, transformer, and CTC layers. |
| `CreateXDecoderDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates the dual-path decoder layers for X-Decoder (pixel + token paths). |
| `CreateXDecoderEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates the Focal transformer encoder layers for X-Decoder. |
| `CreateXLSTMLayers(Int32,Int32,Int32,Int32,Int32)` | Creates layers for the xLSTM language model architecture. |
| `CreateYOLO11SegDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the YOLO11Seg model. |
| `CreateYOLO11SegEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the YOLO11Seg model. |
| `CreateYOLO26SegDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the YOLO26Seg model. |
| `CreateYOLO26SegEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the YOLO26Seg model. |
| `CreateYOLOv12SegDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the YOLOv12Seg model. |
| `CreateYOLOv12SegEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the YOLOv12Seg model. |
| `CreateYOLOv8SegDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the YOLOv8-Seg model. |
| `CreateYOLOv8SegEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the YOLOv8-Seg model. |
| `CreateYOLOv9SegDecoderLayers(Int32,Int32,Int32,Int32,Int32)` | Creates decoder layers for the YOLOv9Seg model. |
| `CreateYOLOv9SegEncoderLayers(Int32,Int32,Int32,Int32[],Int32[],Double)` | Creates encoder layers for the YOLOv9Seg model. |
| `CreateZamba2Layers(Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Creates layers for the Zamba2 hybrid language model architecture. |
| `CreateZambaLayers(Int32,Int32,Int32,Int32,Int32,Int32)` | Creates layers for the Zamba hybrid language model architecture. |
| `GetMobileNetV3LargeBlocks(Double)` | Gets MobileNetV3-Large block configurations. |
| `GetMobileNetV3SmallBlocks(Double)` | Gets MobileNetV3-Small block configurations. |
| `GetTabularOutputActivation(NeuralNetworkArchitecture<>)` | Selects the final-layer activation for a tabular prediction head from the task type. |
| `MakeScaledChannels(Int32,Double)` | Scales channel count by the width coefficient for EfficientNet/MobileNet architectures. |
| `MakeScaledDepth(Int32,Double)` | Scales layer repeat count by the depth coefficient for EfficientNet. |
| `ResolveAndYield(IList<ILayer<>>,Int32[])` | Convenience: chain-resolve lazy shapes and yield each layer in order. |
| `ValidateLayerParameters(Int32,Int32,Int32)` | Validates the parameters used for creating neural network layers. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Provides operations for the numeric type T. |
| `ODISEDecoderStageLayerCount` | Number of layers emitted per decoder upsampling stage by `Int32)` (Deconv + GroupNorm + SiLU + Residual + GroupNorm). |


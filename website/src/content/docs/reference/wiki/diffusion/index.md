---
title: "Diffusion"
description: "All 458 public types in the AiDotNet.diffusion namespace, organized by kind."
section: "API Reference"
---

**458** public types in this namespace, organized by kind.

## Models & Types (424)

| Type | Summary |
|:-----|:--------|
| [`ARDiffusionModel<T>`](/docs/reference/wiki/diffusion/ardiffusionmodel/) | AR-Diffusion model combining autoregressive and diffusion generation in latent space. |
| [`ActivationPoolStats`](/docs/reference/wiki/diffusion/activationpoolstats/) | Statistics about activation pool usage. |
| [`ActivationPool<T>`](/docs/reference/wiki/diffusion/activationpool/) | Memory pool for tensor activations during diffusion model forward/backward passes. |
| [`AdaptiveProjectedGuidance<T>`](/docs/reference/wiki/diffusion/adaptiveprojectedguidance/) | Adaptive Projected Guidance (APG) for diffusion model inference. |
| [`AdversarialDistillationTrainer<T>`](/docs/reference/wiki/diffusion/adversarialdistillationtrainer/) | Trainer for Adversarial Diffusion Distillation (ADD) as used in SD/SDXL Turbo. |
| [`AllegroModel<T>`](/docs/reference/wiki/diffusion/allegromodel/) | Allegro efficient DiT-based video generation model. |
| [`AlphaCompositor<T>`](/docs/reference/wiki/diffusion/alphacompositor/) | Alpha compositing for layered diffusion outputs with transparency support. |
| [`AnimateDiffModel<T>`](/docs/reference/wiki/diffusion/animatediffmodel/) | AnimateDiff model for text-to-video and image-to-video generation. |
| [`AnyEditModel<T>`](/docs/reference/wiki/diffusion/anyeditmodel/) | AnyEdit model for handling diverse editing types through unified instruction understanding. |
| [`AsymmDiTPredictor<T>`](/docs/reference/wiki/diffusion/asymmditpredictor/) | Asymmetric Diffusion Transformer (AsymmDiT) noise predictor for video generation (Genmo Mochi 1 architecture). |
| [`AsyncOnlineDPO<T>`](/docs/reference/wiki/diffusion/asynconlinedpo/) | Asynchronous Online DPO for diffusion model alignment with on-policy generation. |
| [`AudioLDM2Model<T>`](/docs/reference/wiki/diffusion/audioldm2model/) | AudioLDM 2 - Enhanced Audio Latent Diffusion Model with dual text encoders. |
| [`AudioLDMModel<T>`](/docs/reference/wiki/diffusion/audioldmmodel/) | Audio Latent Diffusion Model (AudioLDM) for text-to-audio generation. |
| [`AudioProcessor<T>`](/docs/reference/wiki/diffusion/audioprocessor/) | Complete audio processing pipeline for diffusion-based audio generation. |
| [`AudioVAE<T>`](/docs/reference/wiki/diffusion/audiovae/) | Variational Autoencoder for audio mel-spectrogram encoding and decoding. |
| [`AuraFlowModel<T>`](/docs/reference/wiki/diffusion/auraflowmodel/) | AuraFlow model — open-source flow-matching text-to-image model. |
| [`AutoRegressiveMaskedDiffusion<T>`](/docs/reference/wiki/diffusion/autoregressivemaskeddiffusion/) | Autoregressive Masked Diffusion for hybrid discrete-continuous image generation. |
| [`AutoencoderKL<T>`](/docs/reference/wiki/diffusion/autoencoderkl/) | KL-regularized Variational Autoencoder for latent diffusion models. |
| [`BarkModel<T>`](/docs/reference/wiki/diffusion/barkmodel/) | Bark model for transformer-based text-to-audio generation with multi-lingual speech, music, and sound effects. |
| [`BlendedDiffusionModel<T>`](/docs/reference/wiki/diffusion/blendeddiffusionmodel/) | Blended Diffusion model for text-guided local image editing within user-specified masks. |
| [`BrushEditModel<T>`](/docs/reference/wiki/diffusion/brusheditmodel/) | BrushEdit model combining multimodal LLM understanding with BrushNet inpainting. |
| [`BrushNetModel<T>`](/docs/reference/wiki/diffusion/brushnetmodel/) | BrushNet model for plug-and-play dual-branch inpainting with any diffusion backbone. |
| [`BrushNetXModel<T>`](/docs/reference/wiki/diffusion/brushnetxmodel/) | BrushNet-X model extending BrushNet to SDXL architecture for high-resolution inpainting. |
| [`CACTIModel<T>`](/docs/reference/wiki/diffusion/cactimodel/) | CACTI model for content-aware controllable text-to-image style transfer. |
| [`CATDMModel<T>`](/docs/reference/wiki/diffusion/catdmmodel/) | CATDM model for controllable appearance transfer diffusion for virtual try-on. |
| [`CCSRModel<T>`](/docs/reference/wiki/diffusion/ccsrmodel/) | CCSR: Content-Consistent Super-Resolution with diffusion-based controllable restoration. |
| [`CLIPTextConditioner<T>`](/docs/reference/wiki/diffusion/cliptextconditioner/) | CLIP text encoder conditioning module (Radford et al., ICML 2021). |
| [`CameraEmbedding<T>`](/docs/reference/wiki/diffusion/cameraembedding/) | Camera position embedding for view conditioning. |
| [`CameraPose`](/docs/reference/wiki/diffusion/camerapose/) | Camera pose for rendering. |
| [`CameraPoseEncoder<T>`](/docs/reference/wiki/diffusion/cameraposeencoder/) | Encodes camera pose (polar, azimuth, radius) into embeddings. |
| [`CannyEdgePreprocessor<T>`](/docs/reference/wiki/diffusion/cannyedgepreprocessor/) | Canny edge detection preprocessor for ControlNet conditioning. |
| [`CatVTONModel<T>`](/docs/reference/wiki/diffusion/catvtonmodel/) | CatVTON model for concatenation-based virtual try-on without warping modules. |
| [`CauchyNoiseSchedule<T>`](/docs/reference/wiki/diffusion/cauchynoiseschedule/) | Cauchy noise schedule using extremely heavy-tailed Cauchy distribution for noise sampling. |
| [`Causal3DVAE<T>`](/docs/reference/wiki/diffusion/causal3dvae/) | Causal 3D VAE for video with temporal causal convolutions. |
| [`CausalTemporalAttention<T>`](/docs/reference/wiki/diffusion/causaltemporalattention/) | Causal temporal attention for autoregressive video generation. |
| [`ChatGLM3TextConditioner<T>`](/docs/reference/wiki/diffusion/chatglm3textconditioner/) | ChatGLM3 text encoder conditioning module (Zeng et al. |
| [`CogVideoModel<T>`](/docs/reference/wiki/diffusion/cogvideomodel/) | CogVideo/CogVideoX model for text-to-video and image-to-video generation. |
| [`CogVideoX15Model<T>`](/docs/reference/wiki/diffusion/cogvideox15model/) | CogVideoX 1.5 model with 10-second any-resolution video generation. |
| [`CogView4Model<T>`](/docs/reference/wiki/diffusion/cogview4model/) | CogView-4 model for bilingual text-to-image generation by Zhipu AI. |
| [`ColorPalettePreprocessor<T>`](/docs/reference/wiki/diffusion/colorpalettepreprocessor/) | Color palette extraction preprocessor for ControlNet conditioning. |
| [`ConceptEraser<T>`](/docs/reference/wiki/diffusion/concepteraser/) | Concept erasure for removing unwanted concepts from diffusion model representations. |
| [`ConsisLoRAModel<T>`](/docs/reference/wiki/diffusion/consisloramodel/) | ConsisLoRA model for consistent style transfer using LoRA with content-style disentanglement. |
| [`ConsistencyDistillationSampling<T>`](/docs/reference/wiki/diffusion/consistencydistillationsampling/) | Consistency Distillation Sampling (CSD) for 3D generation with consistency constraints. |
| [`ConsistencyDistillationTrainer<T>`](/docs/reference/wiki/diffusion/consistencydistillationtrainer/) | Trainer for consistency distillation from a pretrained diffusion model. |
| [`ConsistencyModelScheduler<T>`](/docs/reference/wiki/diffusion/consistencymodelscheduler/) | Consistency Model scheduler for single-step or few-step diffusion sampling. |
| [`ConsistencyModel<T>`](/docs/reference/wiki/diffusion/consistencymodel/) | Consistency Model for single-step or few-step image generation. |
| [`ConsistencyTrainingTrainer<T>`](/docs/reference/wiki/diffusion/consistencytrainingtrainer/) | Trainer for consistency training from scratch without a pretrained teacher model. |
| [`ContentShufflePreprocessor<T>`](/docs/reference/wiki/diffusion/contentshufflepreprocessor/) | Content shuffle preprocessor for ControlNet conditioning. |
| [`ControlARModel<T>`](/docs/reference/wiki/diffusion/controlarmodel/) | ControlAR model combining autoregressive generation with spatial control. |
| [`ControlNeXtModel<T>`](/docs/reference/wiki/diffusion/controlnextmodel/) | ControlNeXt model with improved efficiency and generalization. |
| [`ControlNetCondition<T>`](/docs/reference/wiki/diffusion/controlnetcondition/) | Represents a single control condition input for multi-control ControlNet composition. |
| [`ControlNetEncoder<T>`](/docs/reference/wiki/diffusion/controlnetencoder/) | ControlNet encoder per Zhang et al. |
| [`ControlNetFluxModel<T>`](/docs/reference/wiki/diffusion/controlnetfluxmodel/) | ControlNet adapted for the FLUX.1 architecture with flow matching. |
| [`ControlNetInpaintingModel<T>`](/docs/reference/wiki/diffusion/controlnetinpaintingmodel/) | ControlNet Inpainting model with mask-aware conditioning. |
| [`ControlNetLiteModel<T>`](/docs/reference/wiki/diffusion/controlnetlitemodel/) | Lightweight ControlNet model with reduced parameter count for faster inference. |
| [`ControlNetModel<T>`](/docs/reference/wiki/diffusion/controlnetmodel/) | ControlNet model for adding spatial conditioning to diffusion models. |
| [`ControlNetPlusPlusFluxModel<T>`](/docs/reference/wiki/diffusion/controlnetplusplusfluxmodel/) | ControlNet++ adapted for FLUX architecture with reward-guided training. |
| [`ControlNetPlusPlusModel<T>`](/docs/reference/wiki/diffusion/controlnetplusplusmodel/) | ControlNet++ model with improved conditioning via reward-guided training. |
| [`ControlNetQRModel<T>`](/docs/reference/wiki/diffusion/controlnetqrmodel/) | ControlNet QR model specialized for embedding QR codes in generated artwork. |
| [`ControlNetSD3Model<T>`](/docs/reference/wiki/diffusion/controlnetsd3model/) | ControlNet adapted for Stable Diffusion 3's MMDiT architecture. |
| [`ControlNetTileModel<T>`](/docs/reference/wiki/diffusion/controlnettilemodel/) | ControlNet Tile model for upscaling and detail enhancement. |
| [`ControlNetUnionModel<T>`](/docs/reference/wiki/diffusion/controlnetunionmodel/) | ControlNet Union model for unified multi-condition image generation with a single ControlNet. |
| [`ControlNetUnionProModel<T>`](/docs/reference/wiki/diffusion/controlnetunionpromodel/) | ControlNet Union Pro model that supports multiple control types in a single model. |
| [`ControlNetXSModel<T>`](/docs/reference/wiki/diffusion/controlnetxsmodel/) | ControlNet-XS model — lightweight control network with minimal parameters. |
| [`CosineShiftedSchedule<T>`](/docs/reference/wiki/diffusion/cosineshiftedschedule/) | Cosine-shifted noise schedule for resolution-adapted diffusion training. |
| [`CosmosModel<T>`](/docs/reference/wiki/diffusion/cosmosmodel/) | NVIDIA Cosmos physics-aware world generation model. |
| [`CubeDiffModel<T>`](/docs/reference/wiki/diffusion/cubediffmodel/) | CubeDiff model for cubemap-based panoramic generation with cross-face consistency. |
| [`CycleGANTurboModel<T>`](/docs/reference/wiki/diffusion/cycleganturbomodel/) | CycleGAN-Turbo model combining CycleGAN unpaired translation with a diffusion backbone. |
| [`D3PO<T>`](/docs/reference/wiki/diffusion/d3po/) | Direct Preference for Denoising Diffusion Policy Optimization (D3PO). |
| [`DDIMScheduler<T>`](/docs/reference/wiki/diffusion/ddimscheduler/) | DDIM (Denoising Diffusion Implicit Models) scheduler implementation. |
| [`DDPMModel<T>`](/docs/reference/wiki/diffusion/ddpmmodel/) | DDPM (Denoising Diffusion Probabilistic Models) implementation. |
| [`DDPMScheduler<T>`](/docs/reference/wiki/diffusion/ddpmscheduler/) | DDPM (Denoising Diffusion Probabilistic Models) scheduler implementation. |
| [`DEISMultistepScheduler<T>`](/docs/reference/wiki/diffusion/deismultistepscheduler/) | Diffusion Exponential Integrator Sampler (DEIS) for fast diffusion model sampling. |
| [`DIAMONDModel<T>`](/docs/reference/wiki/diffusion/diamondmodel/) | DIAMOND diffusion-based game engine from video with action conditioning. |
| [`DMD2Model<T>`](/docs/reference/wiki/diffusion/dmd2model/) | Distribution Matching Distillation v2 (DMD2) for single-step high-fidelity generation. |
| [`DPMSolverMultistepScheduler<T>`](/docs/reference/wiki/diffusion/dpmsolvermultistepscheduler/) | DPM-Solver++ multistep scheduler for fast diffusion model sampling. |
| [`DPMSolverSDEScheduler<T>`](/docs/reference/wiki/diffusion/dpmsolversdescheduler/) | DPM++ 2M SDE scheduler — stochastic variant of DPM-Solver++ multistep. |
| [`DPMSolverSinglestepScheduler<T>`](/docs/reference/wiki/diffusion/dpmsolversinglestepscheduler/) | DPM++ 2S Ancestral scheduler — single-step DPM-Solver++ with ancestral sampling. |
| [`DPMSolverV3Scheduler<T>`](/docs/reference/wiki/diffusion/dpmsolverv3scheduler/) | DPM-Solver v3 scheduler with empirical model statistics for improved convergence. |
| [`DWPosePreprocessor<T>`](/docs/reference/wiki/diffusion/dwposepreprocessor/) | DWPose whole-body keypoint detection preprocessor for ControlNet conditioning. |
| [`DallE2Model<T>`](/docs/reference/wiki/diffusion/dalle2model/) | DALL-E 2 (unCLIP) model for text-to-image generation via CLIP-guided diffusion. |
| [`DallE3Model<T>`](/docs/reference/wiki/diffusion/dalle3model/) | DALL-E 3 style text-to-image generation model with advanced prompt understanding and high-fidelity image generation capabilities. |
| [`DeepCompressionVAE<T>`](/docs/reference/wiki/diffusion/deepcompressionvae/) | Deep Compression Autoencoder (DC-AE) for extremely high spatial compression. |
| [`DeepFloydIFModel<T>`](/docs/reference/wiki/diffusion/deepfloydifmodel/) | DeepFloyd IF model for cascaded text-to-image generation in pixel space. |
| [`DenoisedScoreDistillation<T>`](/docs/reference/wiki/diffusion/denoisedscoredistillation/) | Denoised Score Distillation (DSD) for artifact-free 3D generation. |
| [`DepthEstimationPreprocessor<T>`](/docs/reference/wiki/diffusion/depthestimationpreprocessor/) | Depth estimation preprocessor for ControlNet conditioning. |
| [`DiTBlock<T>`](/docs/reference/wiki/diffusion/ditblock/) | Block structure for DiT transformer layers containing attention, MLP, and conditioning layers. |
| [`DiTNoisePredictor<T>`](/docs/reference/wiki/diffusion/ditnoisepredictor/) | Diffusion Transformer (DiT) noise predictor for diffusion models. |
| [`DiffBIRModel<T>`](/docs/reference/wiki/diffusion/diffbirmodel/) | DiffBIR model for blind image restoration with generative diffusion prior. |
| [`DiffEditModel<T>`](/docs/reference/wiki/diffusion/diffeditmodel/) | DiffEdit model for automatic mask generation and text-guided image editing. |
| [`DiffPanoModel<T>`](/docs/reference/wiki/diffusion/diffpanomodel/) | DiffPano model for scalable panorama generation with spherical epipolar-aware diffusion. |
| [`DiffWaveModel<T>`](/docs/reference/wiki/diffusion/diffwavemodel/) | DiffWave model for high-quality audio waveform synthesis using diffusion. |
| [`DiffWaveNetwork<T>`](/docs/reference/wiki/diffusion/diffwavenetwork/) | DiffWave neural network with dilated convolutions. |
| [`DiffWaveResidualBlock<T>`](/docs/reference/wiki/diffusion/diffwaveresidualblock/) | Residual block for DiffWave with dilated convolution. |
| [`DiffusionAttention<T>`](/docs/reference/wiki/diffusion/diffusionattention/) | Memory-efficient attention layer for diffusion models using Flash Attention. |
| [`DiffusionCrossAttention<T>`](/docs/reference/wiki/diffusion/diffusioncrossattention/) | Cross-attention layer for diffusion models with Flash Attention optimization. |
| [`DiffusionDPO<T>`](/docs/reference/wiki/diffusion/diffusiondpo/) | Direct Preference Optimization (DPO) adapted for diffusion models. |
| [`DiffusionMemoryManager<T>`](/docs/reference/wiki/diffusion/diffusionmemorymanager/) | Memory management utilities for diffusion models including gradient checkpointing, activation pooling, and model sharding integration. |
| [`DiffusionRLHF<T>`](/docs/reference/wiki/diffusion/diffusionrlhf/) | Reinforcement Learning from Human Feedback (RLHF) adapted for diffusion models. |
| [`DiffusionResBlock<T>`](/docs/reference/wiki/diffusion/diffusionresblock/) | Implements a residual block per the DDPM (Ho et al. |
| [`DiscretizedRFScheduler<T>`](/docs/reference/wiki/diffusion/discretizedrfscheduler/) | Discretized Rectified Flow scheduler with optimized timestep selection. |
| [`DistilledT5TextConditioner<T>`](/docs/reference/wiki/diffusion/distilledt5textconditioner/) | Distilled T5 text encoder conditioning module — same architecture as T5 but half the layer count, per the DistilBERT-style knowledge-distillation recipe (Sanh et al., 2019). |
| [`DistributionMatchingDistiller<T>`](/docs/reference/wiki/diffusion/distributionmatchingdistiller/) | Distribution Matching Distillation (DMD) trainer for single-step generation via distribution alignment. |
| [`DownBlock<T>`](/docs/reference/wiki/diffusion/downblock/) | Downsampling block for VAE encoder with multiple ResBlocks and strided convolution. |
| [`DreamFusionModel<T>`](/docs/reference/wiki/diffusion/dreamfusionmodel/) | DreamFusion model for text-to-3D generation via Score Distillation Sampling (SDS). |
| [`DreamGaussianModel<T>`](/docs/reference/wiki/diffusion/dreamgaussianmodel/) | DreamGaussian model for fast 3D Gaussian splatting generation with Score Distillation Sampling. |
| [`DreamMesh<T>`](/docs/reference/wiki/diffusion/dreammesh/) | Simple mesh representation for DreamFusion. |
| [`DualTextConditioner<T>`](/docs/reference/wiki/diffusion/dualtextconditioner/) | Dual text encoder conditioning module combining CLIP and T5 encoders. |
| [`DynamicCFGScheduler<T>`](/docs/reference/wiki/diffusion/dynamiccfgscheduler/) | Dynamic Classifier-Free Guidance that adjusts scale per timestep. |
| [`EDiffIModel<T>`](/docs/reference/wiki/diffusion/ediffimodel/) | eDiff-I model — ensemble of expert denoisers for text-to-image diffusion. |
| [`ELLAAdapter<T>`](/docs/reference/wiki/diffusion/ellaadapter/) | ELLA (Efficient Large Language Model Adapter) guidance adapter for enhanced text understanding. |
| [`EMMDiTPredictor<T>`](/docs/reference/wiki/diffusion/emmditpredictor/) | E-MMDiT (Efficient Multimodal Diffusion Transformer) noise predictor — a compact configuration of the MMDiT architecture (Stable Diffusion 3, Esser et al. |
| [`EQVAEModel<T>`](/docs/reference/wiki/diffusion/eqvaemodel/) | Equivariance-preserving VAE (EQ-VAE) with improved latent regularity. |
| [`EasyConsistencyModel<T>`](/docs/reference/wiki/diffusion/easyconsistencymodel/) | Easy Consistency Tuning (ECT) model for simple, stable consistency model training. |
| [`EmuVideo2Model<T>`](/docs/reference/wiki/diffusion/emuvideo2model/) | Emu Video 2 with improved generation quality and motion. |
| [`EmuVideoModel<T>`](/docs/reference/wiki/diffusion/emuvideomodel/) | Emu Video high-quality video generation with temporal consistency. |
| [`EulerAncestralDiscreteScheduler<T>`](/docs/reference/wiki/diffusion/eulerancestraldiscretescheduler/) | Euler Ancestral discrete scheduler for diffusion model sampling. |
| [`EulerDiscreteScheduler<T>`](/docs/reference/wiki/diffusion/eulerdiscretescheduler/) | Euler discrete scheduler for diffusion model sampling. |
| [`ExFMScheduler<T>`](/docs/reference/wiki/diffusion/exfmscheduler/) | ExFM (Exponential Flow Matching) scheduler with exponential time discretization. |
| [`FactorizedSpatioTemporalAttention<T>`](/docs/reference/wiki/diffusion/factorizedspatiotemporalattention/) | Factorized spatio-temporal attention that applies spatial and temporal attention separately. |
| [`FashionVDMModel<T>`](/docs/reference/wiki/diffusion/fashionvdmmodel/) | FashionVDM model for video-based virtual try-on with temporal consistency. |
| [`FateZeroModel<T>`](/docs/reference/wiki/diffusion/fatezeromodel/) | FateZero zero-shot video editing via attention blending. |
| [`FlagDiTPredictor<T>`](/docs/reference/wiki/diffusion/flagditpredictor/) | Flag-DiT noise predictor for the Lumina-T2X image-generation architecture (Gao et al. |
| [`FlashDiffusionModel<T>`](/docs/reference/wiki/diffusion/flashdiffusionmodel/) | Flash Diffusion model for rapid few-step generation via progressive attention distillation. |
| [`FlowDPMSolverScheduler<T>`](/docs/reference/wiki/diffusion/flowdpmsolverscheduler/) | Flow DPM-Solver scheduler applying DPM-Solver acceleration to rectified flow models. |
| [`FlowEditModel<T>`](/docs/reference/wiki/diffusion/floweditmodel/) | FlowEdit model for image editing via rectified flow inversion and re-generation. |
| [`FlowMapModel<T>`](/docs/reference/wiki/diffusion/flowmapmodel/) | FlowMap model for one-step generation via flow-map learning. |
| [`FlowMatchingScheduler<T>`](/docs/reference/wiki/diffusion/flowmatchingscheduler/) | Flow matching scheduler implementing rectified flow ODE sampling. |
| [`FlowVidModel<T>`](/docs/reference/wiki/diffusion/flowvidmodel/) | FlowVid optical flow guided video-to-video synthesis. |
| [`Flux1Model<T>`](/docs/reference/wiki/diffusion/flux1model/) | FLUX.1 model for high-quality text-to-image generation by Black Forest Labs. |
| [`Flux2Model<T>`](/docs/reference/wiki/diffusion/flux2model/) | FLUX.2 model for next-generation text-to-image generation by Black Forest Labs. |
| [`Flux2SchnellModel<T>`](/docs/reference/wiki/diffusion/flux2schnellmodel/) | FLUX.2 Schnell for next-generation ultra-fast 1-4 step image generation. |
| [`FluxDoubleStreamPredictor<T>`](/docs/reference/wiki/diffusion/fluxdoublestreampredictor/) | FLUX.1 double-stream transformer noise predictor (Black Forest Labs). |
| [`FluxInpaintingModel<T>`](/docs/reference/wiki/diffusion/fluxinpaintingmodel/) | FLUX Fill model for mask-guided inpainting using rectified flow transformers. |
| [`FluxSchnellModel<T>`](/docs/reference/wiki/diffusion/fluxschnellmodel/) | FLUX.1 Schnell for ultra-fast 1-4 step generation from the FLUX architecture. |
| [`FreeInpaintModel<T>`](/docs/reference/wiki/diffusion/freeinpaintmodel/) | Free-form inpainting model using masked diffusion with irregular mask support. |
| [`FreeNoiseModule<T>`](/docs/reference/wiki/diffusion/freenoisemodule/) | FreeNoise module for tuning-free longer video generation via noise rescheduling. |
| [`FreeNoiseVideoModel<T>`](/docs/reference/wiki/diffusion/freenoisevideomodel/) | FreeNoise extended video generation through noise rescheduling. |
| [`Full3DAttention<T>`](/docs/reference/wiki/diffusion/full3dattention/) | Full 3D attention across all spatio-temporal positions simultaneously. |
| [`GameGenXModel<T>`](/docs/reference/wiki/diffusion/gamegenxmodel/) | GameGen-X open-world game video generation with interactive control. |
| [`GemmaTextConditioner<T>`](/docs/reference/wiki/diffusion/gemmatextconditioner/) | Gemma text encoder conditioning module (Gemma Team 2024). |
| [`Genie2Model<T>`](/docs/reference/wiki/diffusion/genie2model/) | Genie 2 real-time interactive 3D environment generation. |
| [`GriffinLim<T>`](/docs/reference/wiki/diffusion/griffinlim/) | Griffin-Lim algorithm for audio reconstruction from magnitude spectrograms. |
| [`HDPainterModel<T>`](/docs/reference/wiki/diffusion/hdpaintermodel/) | HD-Painter model for high-resolution inpainting with Prompt-Aware Introverted Attention (PAIntA). |
| [`HeunDiscreteScheduler<T>`](/docs/reference/wiki/diffusion/heundiscretescheduler/) | Heun discrete scheduler for diffusion model sampling using second-order Heun's method. |
| [`HiDreamModel<T>`](/docs/reference/wiki/diffusion/hidreammodel/) | HiDream-I1 model for high-quality imaginative text-to-image generation. |
| [`HunyuanDiTModel<T>`](/docs/reference/wiki/diffusion/hunyuanditmodel/) | Hunyuan-DiT model — bilingual (Chinese-English) DiT text-to-image model by Tencent. |
| [`HunyuanVideo15Model<T>`](/docs/reference/wiki/diffusion/hunyuanvideo15model/) | HunyuanVideo 1.5 efficient video generation model for consumer GPUs. |
| [`HunyuanVideoModel<T>`](/docs/reference/wiki/diffusion/hunyuanvideomodel/) | HunyuanVideo model for dual-stream DiT video generation with unified image-video capability. |
| [`HybridODESDEScheduler<T>`](/docs/reference/wiki/diffusion/hybridodesdescheduler/) | Hybrid ODE/SDE scheduler that transitions between deterministic and stochastic sampling. |
| [`HyperSDModel<T>`](/docs/reference/wiki/diffusion/hypersdmodel/) | Hyper-SD model for unified 1-8 step generation via trajectory-segmented distillation. |
| [`ICEditModel<T>`](/docs/reference/wiki/diffusion/iceditmodel/) | ICEdit model for in-context learning based image editing without per-task fine-tuning. |
| [`IDMVTONModel<T>`](/docs/reference/wiki/diffusion/idmvtonmodel/) | IDM-VTON model for image-based virtual try-on with high-fidelity garment transfer. |
| [`IPAdapterFaceIDModel<T>`](/docs/reference/wiki/diffusion/ipadapterfaceidmodel/) | IP-Adapter FaceID model for face-specific identity preservation using facial recognition embeddings. |
| [`IPAdapterFaceIDPlusModel<T>`](/docs/reference/wiki/diffusion/ipadapterfaceidplusmodel/) | IP-Adapter FaceID Plus model for face-identity-preserving generation. |
| [`IPAdapterModel<T>`](/docs/reference/wiki/diffusion/ipadaptermodel/) | IP-Adapter model for image-based prompt conditioning in diffusion models. |
| [`IPAdapterPlusModel<T>`](/docs/reference/wiki/diffusion/ipadapterplusmodel/) | IP-Adapter Plus model for image prompt conditioning in diffusion models. |
| [`Ideogram3Model<T>`](/docs/reference/wiki/diffusion/ideogram3model/) | Ideogram 3 model for text-to-image generation with superior text rendering. |
| [`ImageEncoder<T>`](/docs/reference/wiki/diffusion/imageencoder/) | Image encoder for extracting features from reference images. |
| [`ImageProjector<T>`](/docs/reference/wiki/diffusion/imageprojector/) | Projects image features to text embedding space. |
| [`Imagen2Model<T>`](/docs/reference/wiki/diffusion/imagen2model/) | Imagen 2 model for improved cascaded text-to-image generation. |
| [`Imagen3Model<T>`](/docs/reference/wiki/diffusion/imagen3model/) | Imagen 3 model for text-to-image generation by Google DeepMind. |
| [`ImagenModel<T>`](/docs/reference/wiki/diffusion/imagenmodel/) | Imagen model for cascaded text-to-image generation with T5 text encoding. |
| [`ImagicModel<T>`](/docs/reference/wiki/diffusion/imagicmodel/) | Imagic model for text-aligned real image editing via embedding optimization and model fine-tuning. |
| [`ImprovedConsistencyModel<T>`](/docs/reference/wiki/diffusion/improvedconsistencymodel/) | Improved Consistency Training (iCT) model for single-step image generation. |
| [`ImprovedVideoVAE<T>`](/docs/reference/wiki/diffusion/improvedvideovae/) | Improved Video VAE with temporal-aware compression and motion consistency. |
| [`InpaintingMaskPreprocessor<T>`](/docs/reference/wiki/diffusion/inpaintingmaskpreprocessor/) | Inpainting mask preprocessor for ControlNet conditioning. |
| [`InstaFlowModel<T>`](/docs/reference/wiki/diffusion/instaflowmodel/) | InstaFlow model for one-step text-to-image via rectified flow straightening. |
| [`Instant3DModel<T>`](/docs/reference/wiki/diffusion/instant3dmodel/) | Instant3D model -- fast text-to-3D with feed-forward generation. |
| [`InstantIDModel<T>`](/docs/reference/wiki/diffusion/instantidmodel/) | InstantID model — zero-shot identity-preserving image generation. |
| [`InstantStyleModel<T>`](/docs/reference/wiki/diffusion/instantstylemodel/) | InstantStyle model for zero-shot style transfer using IP-Adapter with style-content separation. |
| [`InstructPix2PixModel<T>`](/docs/reference/wiki/diffusion/instructpix2pixmodel/) | InstructPix2Pix model for instruction-based image editing via natural language text prompts. |
| [`InstructVid2VidModel<T>`](/docs/reference/wiki/diffusion/instructvid2vidmodel/) | InstructVid2Vid natural language instruction-based video editing. |
| [`IntervalScoreMatching<T>`](/docs/reference/wiki/diffusion/intervalscorematching/) | Interval Score Matching (ISM) for improved 3D score distillation. |
| [`JEN1Model<T>`](/docs/reference/wiki/diffusion/jen1model/) | JEN-1 model for high-fidelity text-to-music generation. |
| [`KLoRAStyleModel<T>`](/docs/reference/wiki/diffusion/klorastylemodel/) | K-LoRA Style model for composable style transfer by merging multiple LoRA style adapters. |
| [`KandinskyModel<T>`](/docs/reference/wiki/diffusion/kandinskymodel/) | Kandinsky 2.2/3.0 model for text-to-image generation. |
| [`Kling26Model<T>`](/docs/reference/wiki/diffusion/kling26model/) | Kling 2.6 with simultaneous audio-visual generation. |
| [`KlingModel<T>`](/docs/reference/wiki/diffusion/klingmodel/) | Kling model — 3D spatiotemporal attention video generation by Kuaishou. |
| [`KolorsModel<T>`](/docs/reference/wiki/diffusion/kolorsmodel/) | Kolors model — ChatGLM3-powered bilingual text-to-image model by Kwai. |
| [`LCMScheduler<T>`](/docs/reference/wiki/diffusion/lcmscheduler/) | LCM (Latent Consistency Model) scheduler for ultra-fast diffusion sampling. |
| [`LEDITSPPModel<T>`](/docs/reference/wiki/diffusion/leditsppmodel/) | LEDITS++ model for precise multi-concept editing of real images with automatic masking. |
| [`LGMModel<T>`](/docs/reference/wiki/diffusion/lgmmodel/) | LGM (Large Gaussian Model) for feed-forward 3D Gaussian generation from multi-view images. |
| [`LMSDiscreteScheduler<T>`](/docs/reference/wiki/diffusion/lmsdiscretescheduler/) | Linear Multi-Step (LMS) discrete scheduler for diffusion model sampling. |
| [`LTXVideoModel<T>`](/docs/reference/wiki/diffusion/ltxvideomodel/) | LTX-Video model for lightweight real-time video generation with extreme latent compression. |
| [`LaplaceNoiseSchedule<T>`](/docs/reference/wiki/diffusion/laplacenoiseschedule/) | Laplace noise schedule using heavy-tailed Laplace distribution for noise sampling. |
| [`LatentConsistencyModel<T>`](/docs/reference/wiki/diffusion/latentconsistencymodel/) | Latent Consistency Model (LCM) for fast few-step image generation. |
| [`LatentInitializer<T>`](/docs/reference/wiki/diffusion/latentinitializer/) | Initializes latent tensors for diffusion generation with various noise strategies. |
| [`LatentMaskBlender<T>`](/docs/reference/wiki/diffusion/latentmaskblender/) | Blends latent representations using a mask for seamless inpainting and region editing. |
| [`LatteModel<T>`](/docs/reference/wiki/diffusion/lattemodel/) | Latte model for Latent Diffusion Transformer video generation with factorized spatial-temporal attention. |
| [`LayerCheckpointState<T>`](/docs/reference/wiki/diffusion/layercheckpointstate/) | State for layer-based gradient checkpointing. |
| [`LineArtPreprocessor<T>`](/docs/reference/wiki/diffusion/lineartpreprocessor/) | Line art extraction preprocessor for ControlNet conditioning. |
| [`LiteVAEModel<T>`](/docs/reference/wiki/diffusion/litevaemodel/) | Lightweight VAE optimized for fast encoding/decoding on edge and mobile devices. |
| [`LogSNRImportanceSampling<T>`](/docs/reference/wiki/diffusion/logsnrimportancesampling/) | Log-SNR importance sampling for efficient timestep selection during diffusion training. |
| [`LoongModel<T>`](/docs/reference/wiki/diffusion/loongmodel/) | Loong autoregressive LLM-based minute-long video generator. |
| [`LumaRay2Model<T>`](/docs/reference/wiki/diffusion/lumaray2model/) | Luma Ray 2 video model with fast natural motion and better physics. |
| [`LumaRay3Model<T>`](/docs/reference/wiki/diffusion/lumaray3model/) | Luma Ray 3 with Hi-Fi Diffusion for 4K HDR video. |
| [`LumiereModel<T>`](/docs/reference/wiki/diffusion/lumieremodel/) | Lumiere Space-Time UNet for single-pass 80-frame video generation. |
| [`LuminaImage2Model<T>`](/docs/reference/wiki/diffusion/luminaimage2model/) | Lumina Image 2.0 model for high-resolution text-to-image generation. |
| [`LuminaT2XModel<T>`](/docs/reference/wiki/diffusion/luminat2xmodel/) | Lumina-T2X unified framework for transforming text into any modality. |
| [`LuminaT2XModel<T>`](/docs/reference/wiki/diffusion/luminat2xmodel-2/) | Lumina-T2X model — transformer-based text-to-any generation (image, video, 3D, audio). |
| [`MAGI1Model<T>`](/docs/reference/wiki/diffusion/magi1model/) | MAGI-1 video model with strong temporal coherence and multi-task support. |
| [`MARModel<T>`](/docs/reference/wiki/diffusion/marmodel/) | Masked Autoregressive (MAR) model for image generation via masked token prediction. |
| [`MLSDPreprocessor<T>`](/docs/reference/wiki/diffusion/mlsdpreprocessor/) | MLSD (Mobile Line Segment Detection) preprocessor for ControlNet conditioning. |
| [`MMDiTBlock<T>`](/docs/reference/wiki/diffusion/mmditblock/) | Joint (double-stream) MMDiT block with separate image and text streams sharing attention but with independent MLPs and AdaLN. |
| [`MMDiTNoisePredictor<T>`](/docs/reference/wiki/diffusion/mmditnoisepredictor/) | Multi-Modal Diffusion Transformer (MMDiT) noise predictor for SD3 and FLUX architectures. |
| [`MMDiTSingleBlock<T>`](/docs/reference/wiki/diffusion/mmditsingleblock/) | Single-stream MMDiT block (FLUX-style) where text and image tokens are processed together through shared self-attention and parallel MLP. |
| [`MMDiTXNoisePredictor<T>`](/docs/reference/wiki/diffusion/mmditxnoisepredictor/) | Extended Multimodal Diffusion Transformer (MMDiT-X) noise predictor for the Stable Diffusion 3.5 architecture. |
| [`MVDreamModel<T>`](/docs/reference/wiki/diffusion/mvdreammodel/) | MVDream - Multi-View Diffusion Model for 3D-consistent image generation. |
| [`Magic3DModel<T>`](/docs/reference/wiki/diffusion/magic3dmodel/) | Magic3D model for high-quality text-to-3D generation using score distillation. |
| [`MagicBrushModel<T>`](/docs/reference/wiki/diffusion/magicbrushmodel/) | MagicBrush model for instruction-based image editing with visual brush stroke guidance. |
| [`MakeAVideoModel<T>`](/docs/reference/wiki/diffusion/makeavideomodel/) | Make-A-Video model — text-to-video generation without paired text-video data. |
| [`MaskBinarizer<T>`](/docs/reference/wiki/diffusion/maskbinarizer/) | Converts a soft mask (continuous values) into a binary mask (0 or 1) using a threshold. |
| [`MaskBlur<T>`](/docs/reference/wiki/diffusion/maskblur/) | Applies Gaussian blur to a mask for smooth transitions. |
| [`MaskDilation<T>`](/docs/reference/wiki/diffusion/maskdilation/) | Dilates a mask by expanding masked regions, filling small holes and connecting nearby regions. |
| [`MaskErosion<T>`](/docs/reference/wiki/diffusion/maskerosion/) | Erodes a mask by shrinking masked regions, removing thin protrusions and small artifacts. |
| [`MaskFeatherer<T>`](/docs/reference/wiki/diffusion/maskfeatherer/) | Applies feathering (soft blurring) to mask edges for smooth transitions in inpainting. |
| [`MaskFromBoundingBox<T>`](/docs/reference/wiki/diffusion/maskfromboundingbox/) | Generates a mask from one or more bounding box regions. |
| [`MaskFromPoints<T>`](/docs/reference/wiki/diffusion/maskfrompoints/) | Generates a mask from point prompts with circular regions, similar to SAM-style point selection. |
| [`MaskFromSegmentation<T>`](/docs/reference/wiki/diffusion/maskfromsegmentation/) | Generates a binary mask from a segmentation map by selecting specific class labels. |
| [`MaskInverter<T>`](/docs/reference/wiki/diffusion/maskinverter/) | Inverts a mask so masked regions become unmasked and vice versa. |
| [`MediaPipeFacePreprocessor<T>`](/docs/reference/wiki/diffusion/mediapipefacepreprocessor/) | MediaPipe Face Mesh preprocessor for ControlNet conditioning. |
| [`MeissonicModel<T>`](/docs/reference/wiki/diffusion/meissonicmodel/) | Meissonic model for non-autoregressive masked image modeling (MIM) generation. |
| [`MelSpectrogram<T>`](/docs/reference/wiki/diffusion/melspectrogram/) | Computes Mel spectrograms from audio signals. |
| [`MelodyEncoder<T>`](/docs/reference/wiki/diffusion/melodyencoder/) | Melody encoder for extracting melodic features from audio. |
| [`MemoryEstimate`](/docs/reference/wiki/diffusion/memoryestimate/) | Memory usage estimate. |
| [`MeshyModel<T>`](/docs/reference/wiki/diffusion/meshymodel/) | Meshy model for production-grade text/image to 3D generation with PBR texturing. |
| [`MidJourneyV7Model<T>`](/docs/reference/wiki/diffusion/midjourneyv7model/) | MidJourney V7-style model architecture for artistic text-to-image generation. |
| [`MinimaxVideoModel<T>`](/docs/reference/wiki/diffusion/minimaxvideomodel/) | MiniMax Hailuo video model with strong image-to-video generation. |
| [`MoMaskModel<T>`](/docs/reference/wiki/diffusion/momaskmodel/) | MoMask model for masked generative modeling of 3D human motion sequences. |
| [`Mochi1Model<T>`](/docs/reference/wiki/diffusion/mochi1model/) | Mochi 1 model for asymmetric DiT video generation with joint text-video attention. |
| [`Mochi1PreviewModel<T>`](/docs/reference/wiki/diffusion/mochi1previewmodel/) | Mochi 1 Preview with Asymmetric Diffusion Transformer (AsymmDiT). |
| [`ModelScopeT2VModel<T>`](/docs/reference/wiki/diffusion/modelscopet2vmodel/) | ModelScope Text-to-Video model with temporal U-Net for short video clip generation. |
| [`ModelShard<T>`](/docs/reference/wiki/diffusion/modelshard/) | Enables model sharding across multiple devices for large model inference. |
| [`MotionDiffuseModel<T>`](/docs/reference/wiki/diffusion/motiondiffusemodel/) | MotionDiffuse model for fine-grained text-driven motion generation with body part control. |
| [`MotionDiffusionModel<T>`](/docs/reference/wiki/diffusion/motiondiffusionmodel/) | Motion Diffusion Model (MDM) for text-to-motion generation of human body movements. |
| [`MotionModule<T>`](/docs/reference/wiki/diffusion/motionmodule/) | AnimateDiff motion module for injecting temporal awareness into image diffusion models. |
| [`MovieGenModel<T>`](/docs/reference/wiki/diffusion/moviegenmodel/) | MovieGen 30B foundation model for video, audio, and editing. |
| [`MultiDiffusionModel<T>`](/docs/reference/wiki/diffusion/multidiffusionmodel/) | MultiDiffusion model for generating seamless panoramic and ultra-wide images. |
| [`MultiStepConsistencyModel<T>`](/docs/reference/wiki/diffusion/multistepconsistencymodel/) | Multi-Step Consistency Model that bridges single-step and multi-step generation. |
| [`MultiViewAttention<T>`](/docs/reference/wiki/diffusion/multiviewattention/) | Multi-view attention module for cross-view consistency. |
| [`MultiViewUNet<T>`](/docs/reference/wiki/diffusion/multiviewunet/) | Multi-view aware U-Net for MVDream. |
| [`MultistepLCModel<T>`](/docs/reference/wiki/diffusion/multisteplcmodel/) | Multistep Latent Consistency Model (MLCM) for high-quality few-step generation. |
| [`MusicGenModel<T>`](/docs/reference/wiki/diffusion/musicgenmodel/) | MusicGen - Diffusion-based music generation model with advanced musical controls. |
| [`NeRFNetwork<T>`](/docs/reference/wiki/diffusion/nerfnetwork/) | Neural Radiance Field network for 3D representation. |
| [`NeRFResult<T>`](/docs/reference/wiki/diffusion/nerfresult/) | Result from DreamFusion generation. |
| [`NormalMapPreprocessor<T>`](/docs/reference/wiki/diffusion/normalmappreprocessor/) | Normal map estimation preprocessor for ControlNet conditioning. |
| [`NullTextInversionModel<T>`](/docs/reference/wiki/diffusion/nulltextinversionmodel/) | Null-text Inversion model for editing real images by optimizing unconditional embeddings. |
| [`OSDSModel<T>`](/docs/reference/wiki/diffusion/osdsmodel/) | One-Step Diffusion via Shortcut (OSDS) model for single-step high-quality generation. |
| [`OasisModel<T>`](/docs/reference/wiki/diffusion/oasismodel/) | Oasis playable AI game via next-frame prediction. |
| [`OffsetNoiseSchedule<T>`](/docs/reference/wiki/diffusion/offsetnoiseschedule/) | Offset noise schedule that adds a global offset to noise for improved dark/bright image generation. |
| [`OmniGen2Model<T>`](/docs/reference/wiki/diffusion/omnigen2model/) | OmniGen-2 unified model for multi-task image generation and editing with a single architecture. |
| [`OmniGenModel<T>`](/docs/reference/wiki/diffusion/omnigenmodel/) | OmniGen model — unified image generation model handling multiple tasks in one architecture. |
| [`One2345Model<T>`](/docs/reference/wiki/diffusion/one2345model/) | One-2-3-45 model for single-image to 3D mesh generation in 45 seconds. |
| [`OpenPosePreprocessor<T>`](/docs/reference/wiki/diffusion/openposepreprocessor/) | OpenPose body keypoint detection preprocessor for ControlNet conditioning. |
| [`OpenSora13Model<T>`](/docs/reference/wiki/diffusion/opensora13model/) | Open-Sora 1.3 with upgraded 3D-VAE and rectified flow. |
| [`OpenSora2Model<T>`](/docs/reference/wiki/diffusion/opensora2model/) | Open-Sora 2.0 commercial-level video generation model. |
| [`OpenSoraModel<T>`](/docs/reference/wiki/diffusion/opensoramodel/) | Open-Sora model for open-source Sora-like video generation with STDiT architecture. |
| [`OutpaintingMaskGenerator<T>`](/docs/reference/wiki/diffusion/outpaintingmaskgenerator/) | Generates masks for outpainting by marking regions outside the original image bounds. |
| [`PABCache<T>`](/docs/reference/wiki/diffusion/pabcache/) | Pyramid Attention Broadcast (PAB) cache for accelerating video diffusion inference. |
| [`PASDModel<T>`](/docs/reference/wiki/diffusion/pasdmodel/) | PASD: Pixel-Aware Stable Diffusion for real-world image super-resolution. |
| [`PCMModel<T>`](/docs/reference/wiki/diffusion/pcmmodel/) | Phased Consistency Model (PCM) for flexible-step generation with phase-based training. |
| [`PNDMScheduler<T>`](/docs/reference/wiki/diffusion/pndmscheduler/) | PNDM (Pseudo Numerical Methods for Diffusion Models) scheduler implementation. |
| [`PaintByExampleModel<T>`](/docs/reference/wiki/diffusion/paintbyexamplemodel/) | Paint-by-Example model for exemplar-based inpainting using reference images. |
| [`PeRFlowModel<T>`](/docs/reference/wiki/diffusion/perflowmodel/) | PeRFlow (Piecewise Rectified Flow) model for few-step generation via flow straightening. |
| [`PeRFlowScheduler<T>`](/docs/reference/wiki/diffusion/perflowscheduler/) | PeRFlow (Piecewise Rectified Flow) scheduler for accelerated multi-segment flow sampling. |
| [`PerturbedAttentionGuidance<T>`](/docs/reference/wiki/diffusion/perturbedattentionguidance/) | Perturbed Attention Guidance (PAG) for diffusion model inference. |
| [`PhotoMakerModel<T>`](/docs/reference/wiki/diffusion/photomakermodel/) | PhotoMaker model — identity-customized photo generation with stacked ID embedding. |
| [`Pika21Model<T>`](/docs/reference/wiki/diffusion/pika21model/) | Pika 2.1 short-form video with creative effects. |
| [`Pix2PixZeroModel<T>`](/docs/reference/wiki/diffusion/pix2pixzeromodel/) | Pix2Pix-Zero model for zero-shot image-to-image translation without paired training data. |
| [`PixArtDeltaLCMModel<T>`](/docs/reference/wiki/diffusion/pixartdeltalcmmodel/) | PixArt-Delta LCM for few-step generation from the efficient PixArt-Delta DiT architecture. |
| [`PixArtDeltaModel<T>`](/docs/reference/wiki/diffusion/pixartdeltamodel/) | PixArt-Delta model — LCM-distilled PixArt for fast 2-8 step generation. |
| [`PixArtModel<T>`](/docs/reference/wiki/diffusion/pixartmodel/) | PixArt-α model for efficient high-quality text-to-image generation using DiT architecture. |
| [`PixArtSigmaModel<T>`](/docs/reference/wiki/diffusion/pixartsigmamodel/) | PixArt-Sigma model for high-resolution text-to-image generation with improved quality. |
| [`PlaygroundV25Model<T>`](/docs/reference/wiki/diffusion/playgroundv25model/) | Playground v2.5 model for aesthetically-focused text-to-image generation. |
| [`PlaygroundV3Model<T>`](/docs/reference/wiki/diffusion/playgroundv3model/) | Playground v3 model for aesthetically optimized text-to-image generation. |
| [`PointEModel<T>`](/docs/reference/wiki/diffusion/pointemodel/) | Point-E model for text-to-3D point cloud generation. |
| [`PowerPaintModel<T>`](/docs/reference/wiki/diffusion/powerpaintmodel/) | PowerPaint v2 model for versatile task-aware image inpainting with learnable task prompts. |
| [`ProgressiveDistillationTrainer<T>`](/docs/reference/wiki/diffusion/progressivedistillationtrainer/) | Trainer for progressive distillation that halves the number of steps in each round. |
| [`PromptToPromptModel<T>`](/docs/reference/wiki/diffusion/prompttopromptmodel/) | Prompt-to-Prompt model for attention-based image editing by manipulating cross-attention maps. |
| [`PyramidFlowModel<T>`](/docs/reference/wiki/diffusion/pyramidflowmodel/) | Pyramid Flow multi-resolution video generation via pyramid flow matching. |
| [`QRCodePreprocessor<T>`](/docs/reference/wiki/diffusion/qrcodepreprocessor/) | QR code pattern preprocessor for ControlNet conditioning. |
| [`Qwen2TextConditioner<T>`](/docs/reference/wiki/diffusion/qwen2textconditioner/) | Qwen2 text encoder conditioning module (Yang et al. |
| [`RACEEraser<T>`](/docs/reference/wiki/diffusion/raceeraser/) | RACE: Robust Adversarial Concept Erasure for removing concepts resilient to red-teaming attacks. |
| [`RADModel<T>`](/docs/reference/wiki/diffusion/radmodel/) | Region-Aware Diffusion (RAD) model for spatially controlled inpainting and editing. |
| [`RAPHAELModel<T>`](/docs/reference/wiki/diffusion/raphaelmodel/) | RAPHAEL model — Mixture-of-Experts text-to-image diffusion model. |
| [`RBModulationModel<T>`](/docs/reference/wiki/diffusion/rbmodulationmodel/) | RB-Modulation model for training-free style transfer via reference-based attention modulation. |
| [`RealESRGANModel<T>`](/docs/reference/wiki/diffusion/realesrganmodel/) | Real-ESRGAN model for practical blind image super-resolution with degradation-aware training. |
| [`RecraftV3Model<T>`](/docs/reference/wiki/diffusion/recraftv3model/) | Recraft V3 model for professional-grade text-to-image generation. |
| [`RectifiedFlowScheduler<T>`](/docs/reference/wiki/diffusion/rectifiedflowscheduler/) | Rectified flow scheduler for straight-path ODE sampling with velocity prediction. |
| [`ReferenceOnlyModel<T>`](/docs/reference/wiki/diffusion/referenceonlymodel/) | Reference-Only model that uses a reference image's self-attention features for conditioning. |
| [`RefinerStage<T>`](/docs/reference/wiki/diffusion/refinerstage/) | Refiner stage for late-stage noise-add-then-denoise detail improvement. |
| [`ReplaceAnythingModel<T>`](/docs/reference/wiki/diffusion/replaceanythingmodel/) | ReplaceAnything model for text-guided object replacement within SAM-segmented regions. |
| [`RescaledCFG<T>`](/docs/reference/wiki/diffusion/rescaledcfg/) | Rescaled Classifier-Free Guidance to prevent over-saturation. |
| [`RewardGuidance<T>`](/docs/reference/wiki/diffusion/rewardguidance/) | Reward-guided sampling for inference-time alignment of diffusion models. |
| [`RewardScoreDistillation<T>`](/docs/reference/wiki/diffusion/rewardscoredistillation/) | Reward-weighted Score Distillation Sampling (RewardSDS) for preference-aligned 3D generation. |
| [`RhythmEncoder<T>`](/docs/reference/wiki/diffusion/rhythmencoder/) | Rhythm encoder for extracting beat/rhythm features from audio. |
| [`RiffusionModel<T>`](/docs/reference/wiki/diffusion/riffusionmodel/) | Riffusion model for music generation via spectrogram diffusion. |
| [`RunwayGen4Model<T>`](/docs/reference/wiki/diffusion/runwaygen4model/) | Runway Gen-4 multi-modal understanding and generation model. |
| [`RunwayGenModel<T>`](/docs/reference/wiki/diffusion/runwaygenmodel/) | Runway Gen model for multi-modal video generation with structure and content disentanglement. |
| [`SAMPreprocessor<T>`](/docs/reference/wiki/diffusion/sampreprocessor/) | SAM (Segment Anything Model) preprocessor for ControlNet conditioning. |
| [`SANAModel<T>`](/docs/reference/wiki/diffusion/sanamodel/) | SANA model for efficient high-resolution text-to-image generation. |
| [`SANASprintModel<T>`](/docs/reference/wiki/diffusion/sanasprintmodel/) | SANA Sprint model for ultra-fast 1-step generation from the SANA architecture. |
| [`SASTDModel<T>`](/docs/reference/wiki/diffusion/sastdmodel/) | SASTD model for structure-aware style transfer via diffusion with edge-guided generation. |
| [`SASolverScheduler<T>`](/docs/reference/wiki/diffusion/sasolverscheduler/) | SA-Solver (Stochastic Adams) scheduler using Adams-Bashforth/Moulton methods for SDE sampling. |
| [`SCOREFramework<T>`](/docs/reference/wiki/diffusion/scoreframework/) | SCORE: Selective Concept Obliteration for Responsible Editing in diffusion models. |
| [`SCottModel<T>`](/docs/reference/wiki/diffusion/scottmodel/) | SCott (Score Consistency via Optimal Transport) for efficient single/few-step generation. |
| [`SD3FlashModel<T>`](/docs/reference/wiki/diffusion/sd3flashmodel/) | SD3 Flash for ultra-fast 1-4 step generation from SD3 architecture. |
| [`SD3InpaintingModel<T>`](/docs/reference/wiki/diffusion/sd3inpaintingmodel/) | Stable Diffusion 3 inpainting model using MMDiT architecture for mask-guided generation. |
| [`SD3TurboModel<T>`](/docs/reference/wiki/diffusion/sd3turbomodel/) | SD3 Turbo for few-step generation from Stable Diffusion 3 via distillation. |
| [`SDEditModel<T>`](/docs/reference/wiki/diffusion/sdeditmodel/) | SDEdit model for image-guided synthesis via partial noise injection and guided denoising. |
| [`SDTurboModel<T>`](/docs/reference/wiki/diffusion/sdturbomodel/) | SD Turbo / SDXL Turbo model for real-time single-step image generation. |
| [`SDUpscalerModel<T>`](/docs/reference/wiki/diffusion/sdupscalermodel/) | Stable Diffusion x4 Upscaler model for text-guided latent super-resolution. |
| [`SDXLInpaintingModel<T>`](/docs/reference/wiki/diffusion/sdxlinpaintingmodel/) | SDXL Inpainting model for high-resolution 1024x1024 mask-based image inpainting. |
| [`SDXLLightningModel<T>`](/docs/reference/wiki/diffusion/sdxllightningmodel/) | SDXL Lightning model for 2-8 step high-quality generation via progressive distillation. |
| [`SDXLModel<T>`](/docs/reference/wiki/diffusion/sdxlmodel/) | Stable Diffusion XL (SDXL) model for high-resolution image generation. |
| [`SDXLRefiner<T>`](/docs/reference/wiki/diffusion/sdxlrefiner/) | SDXL Refiner model for enhancing generated images. |
| [`SDXLTurboModel<T>`](/docs/reference/wiki/diffusion/sdxlturbomodel/) | SDXL Turbo model for real-time single-step high-resolution image generation. |
| [`SDXLVAEModel<T>`](/docs/reference/wiki/diffusion/sdxlvaemodel/) | SDXL-optimized VAE with improved decoder fidelity for 1024x1024 generation. |
| [`SGRACEEraser<T>`](/docs/reference/wiki/diffusion/sgraceeraser/) | S-GRACE: Style-aware GRACE for erasing artistic styles from diffusion models. |
| [`STDiTBlock<T>`](/docs/reference/wiki/diffusion/stditblock/) | Spatial-Temporal DiT (STDiT) block for efficient video generation transformers. |
| [`SUPIRModel<T>`](/docs/reference/wiki/diffusion/supirmodel/) | SUPIR model for scaling up image restoration with SDXL for photo-realistic results. |
| [`ScoreDistillationSampling<T>`](/docs/reference/wiki/diffusion/scoredistillationsampling/) | Score Distillation Sampling (SDS) for text-to-3D and generator optimization. |
| [`ScoreDistillationTrainer<T>`](/docs/reference/wiki/diffusion/scoredistillationtrainer/) | Trainer for Score Distillation Sampling (SDS) and its variants for generator training. |
| [`ScribblePreprocessor<T>`](/docs/reference/wiki/diffusion/scribblepreprocessor/) | Scribble/sketch preprocessor for ControlNet conditioning. |
| [`SeamlessBlender<T>`](/docs/reference/wiki/diffusion/seamlessblender/) | Seamless blending for panoramic and tiled diffusion generation with overlap regions. |
| [`SeeSRModel<T>`](/docs/reference/wiki/diffusion/seesrmodel/) | SeeSR: Semantics-aware super-resolution using diffusion-based image upscaling. |
| [`SeedEdit3Model<T>`](/docs/reference/wiki/diffusion/seededit3model/) | SeedEdit 3 model for high-fidelity instruction-based editing with structure preservation. |
| [`Seedance1Model<T>`](/docs/reference/wiki/diffusion/seedance1model/) | Seedance 1 ranked #1 on T2V and I2V leaderboards. |
| [`SelfAttentionGuidance<T>`](/docs/reference/wiki/diffusion/selfattentionguidance/) | Self-Attention Guidance (SAG) for diffusion model inference. |
| [`SemanticScoreDistillation<T>`](/docs/reference/wiki/diffusion/semanticscoredistillation/) | Semantic Score Distillation (SemanticSDS) for semantically-guided 3D generation. |
| [`SemanticSegPreprocessor<T>`](/docs/reference/wiki/diffusion/semanticsegpreprocessor/) | Semantic segmentation preprocessor for ControlNet conditioning. |
| [`SenseFlowModel<T>`](/docs/reference/wiki/diffusion/senseflowmodel/) | SenseFlow model for accelerated flow-matching generation via knowledge distillation. |
| [`ShapEModel<T>`](/docs/reference/wiki/diffusion/shapemodel/) | Shap-E model for text-to-3D and image-to-3D generation with implicit neural representations. |
| [`ShortTimeFourierTransform<T>`](/docs/reference/wiki/diffusion/shorttimefouriertransform/) | Short-Time Fourier Transform (STFT) for analyzing audio signals over time. |
| [`Show1Model<T>`](/docs/reference/wiki/diffusion/show1model/) | Show-1 marrying pixel and latent diffusion for text-to-video. |
| [`ShufflePreprocessor<T>`](/docs/reference/wiki/diffusion/shufflepreprocessor/) | Shuffle preprocessor for ControlNet conditioning. |
| [`SiDDiTModel<T>`](/docs/reference/wiki/diffusion/sidditmodel/) | SiD-DiT: Score Identity Distillation applied to Diffusion Transformer (DiT) architectures. |
| [`SiDModel<T>`](/docs/reference/wiki/diffusion/sidmodel/) | Score Identity Distillation (SiD) model for single-step generation via score identity. |
| [`SiTPredictor<T>`](/docs/reference/wiki/diffusion/sitpredictor/) | Scalable Interpolant Transformer (SiT) noise predictor for flow-based diffusion. |
| [`SigLIP2TextConditioner<T>`](/docs/reference/wiki/diffusion/siglip2textconditioner/) | SigLIP 2 text encoder conditioning module (Tschannen et al. |
| [`SigLIPTextConditioner<T>`](/docs/reference/wiki/diffusion/sigliptextconditioner/) | SigLIP text encoder conditioning module (Zhai et al., ICCV 2023). |
| [`SkyReelsV1Model<T>`](/docs/reference/wiki/diffusion/skyreelsv1model/) | SkyReels V1 human-centric video generation model. |
| [`SnapVideoModel<T>`](/docs/reference/wiki/diffusion/snapvideomodel/) | Snap Video scaled spatiotemporal transformer for text-to-video. |
| [`SoftEdgePreprocessor<T>`](/docs/reference/wiki/diffusion/softedgepreprocessor/) | Soft edge detection preprocessor for ControlNet conditioning. |
| [`Sora2Model<T>`](/docs/reference/wiki/diffusion/sora2model/) | Sora 2 cinematic video generation with physics simulation and synced audio. |
| [`SoraModel<T>`](/docs/reference/wiki/diffusion/soramodel/) | Sora-architecture model for DiT-based video generation with native spatiotemporal patches. |
| [`SoundStormModel<T>`](/docs/reference/wiki/diffusion/soundstormmodel/) | SoundStorm model for parallel masked audio token generation with conformer architecture. |
| [`SparseVideoGen<T>`](/docs/reference/wiki/diffusion/sparsevideogen/) | Sparse video generation with selective frame denoising for faster inference. |
| [`SpotDiffusionModel<T>`](/docs/reference/wiki/diffusion/spotdiffusionmodel/) | SpotDiffusion model for spatially-organized text-guided panorama generation. |
| [`StableAudioModel<T>`](/docs/reference/wiki/diffusion/stableaudiomodel/) | Stable Audio Open model — DiT-based latent diffusion for long-form audio generation. |
| [`StableCascadeModel<T>`](/docs/reference/wiki/diffusion/stablecascademodel/) | Stable Cascade (Würstchen v3) model for high-resolution text-to-image generation. |
| [`StableDiffusion15Model<T>`](/docs/reference/wiki/diffusion/stablediffusion15model/) | Stable Diffusion 1.5 model for text-to-image generation. |
| [`StableDiffusion2Model<T>`](/docs/reference/wiki/diffusion/stablediffusion2model/) | Stable Diffusion 2.0/2.1 model for text-to-image generation. |
| [`StableDiffusion35Model<T>`](/docs/reference/wiki/diffusion/stablediffusion35model/) | Stable Diffusion 3.5 model with improved MMDiT-X architecture by Stability AI. |
| [`StableDiffusion3Model<T>`](/docs/reference/wiki/diffusion/stablediffusion3model/) | Stable Diffusion 3 / SD 3.5 model for text-to-image generation using rectified flow and MMDiT. |
| [`StableSRModel<T>`](/docs/reference/wiki/diffusion/stablesrmodel/) | StableSR model for exploiting diffusion prior for real-world image super-resolution. |
| [`StableVITONModel<T>`](/docs/reference/wiki/diffusion/stablevitonmodel/) | StableVITON model for learning semantic correspondence with Stable Diffusion for virtual try-on. |
| [`StableVideoDiffusion<T>`](/docs/reference/wiki/diffusion/stablevideodiffusion/) | Stable Video Diffusion (SVD) model for image-to-video generation. |
| [`StandardVAE<T>`](/docs/reference/wiki/diffusion/standardvae/) | Standard Variational Autoencoder for latent diffusion models. |
| [`Step1XEditModel<T>`](/docs/reference/wiki/diffusion/step1xeditmodel/) | Step1X-Edit model for one-step image editing using consistency distillation. |
| [`StepVideoModel<T>`](/docs/reference/wiki/diffusion/stepvideomodel/) | StepVideo text-to-video model with benchmark-leading quality. |
| [`StitchDiffusionModel<T>`](/docs/reference/wiki/diffusion/stitchdiffusionmodel/) | StitchDiffusion model for seamless 360-degree panorama generation with wrap-around consistency. |
| [`StochasticInterpolantScheduler<T>`](/docs/reference/wiki/diffusion/stochasticinterpolantscheduler/) | Stochastic Interpolant scheduler for generalized flow-based sampling with noise injection. |
| [`StreamingT2VModel<T>`](/docs/reference/wiki/diffusion/streamingt2vmodel/) | StreamingT2V autoregressive long video generation up to 1200 frames. |
| [`StrengthBasedScheduling<T>`](/docs/reference/wiki/diffusion/strengthbasedscheduling/) | Strength-based scheduling for img2img and inpainting denoising control. |
| [`StudentTeacherFramework<T>`](/docs/reference/wiki/diffusion/studentteacherframework/) | Generic student-teacher framework for knowledge distillation in diffusion models. |
| [`StyDiffModel<T>`](/docs/reference/wiki/diffusion/stydiffmodel/) | StyDiff model for diffusion-based artistic style transfer with content preservation. |
| [`StyleAlignedEditModel<T>`](/docs/reference/wiki/diffusion/stylealignededitmodel/) | StyleAligned-Edit model for consistent multi-image style editing via shared self-attention. |
| [`StyleAlignedModel<T>`](/docs/reference/wiki/diffusion/stylealignedmodel/) | Style-Aligned model for consistent style across multiple generated images. |
| [`StyleStudioModel<T>`](/docs/reference/wiki/diffusion/stylestudiomodel/) | StyleStudio model for text-driven style generation and transfer with disentangled control. |
| [`SwiftBrushModel<T>`](/docs/reference/wiki/diffusion/swiftbrushmodel/) | SwiftBrush model for image-free one-step text-to-image distillation. |
| [`SyncDiffusionModel<T>`](/docs/reference/wiki/diffusion/syncdiffusionmodel/) | SyncDiffusion model for coherent panorama generation with synchronized denoising. |
| [`SyncDreamerModel<T>`](/docs/reference/wiki/diffusion/syncdreamermodel/) | SyncDreamer model for synchronized multi-view diffusion with 3D-consistent generation. |
| [`T2IAdapterModel<T>`](/docs/reference/wiki/diffusion/t2iadaptermodel/) | T2I-Adapter model for adding spatial control to text-to-image diffusion models. |
| [`T5TextConditioner<T>`](/docs/reference/wiki/diffusion/t5textconditioner/) | T5 text encoder conditioning module (Raffel et al., JMLR 2020). |
| [`TCDModel<T>`](/docs/reference/wiki/diffusion/tcdmodel/) | Trajectory Consistency Distillation (TCD) model for high-quality few-step generation. |
| [`TLoRAModel<T>`](/docs/reference/wiki/diffusion/tloramodel/) | T-LoRA model for temporal LoRA-based style transfer with video-consistent stylization. |
| [`TSDSRModel<T>`](/docs/reference/wiki/diffusion/tsdsrmodel/) | TSD-SR: Timestep-Shifted Diffusion for fast and high-quality super-resolution. |
| [`TeaCache<T>`](/docs/reference/wiki/diffusion/teacache/) | Timestep Embedding Aware Cache (TeaCache) for accelerating video diffusion inference. |
| [`TemporalConvolution<T>`](/docs/reference/wiki/diffusion/temporalconvolution/) | 1D temporal convolution layer for video diffusion models. |
| [`TemporalInterpolationVAE<T>`](/docs/reference/wiki/diffusion/temporalinterpolationvae/) | Temporal interpolation VAE that generates intermediate frames in latent space. |
| [`TemporalSelfAttention<T>`](/docs/reference/wiki/diffusion/temporalselfattention/) | Temporal self-attention layer for video diffusion models. |
| [`TemporalVAE<T>`](/docs/reference/wiki/diffusion/temporalvae/) | Temporal-aware Variational Autoencoder for video diffusion models. |
| [`TilePreprocessor<T>`](/docs/reference/wiki/diffusion/tilepreprocessor/) | Tile preprocessor for ControlNet conditioning. |
| [`TokenFlowModel<T>`](/docs/reference/wiki/diffusion/tokenflowmodel/) | TokenFlow consistent video editing via token flow propagation. |
| [`TrainingEfficientLCM<T>`](/docs/reference/wiki/diffusion/trainingefficientlcm/) | Training-Efficient Latent Consistency Model for resource-constrained LCM distillation. |
| [`TrajectoryConsistencyDistiller<T>`](/docs/reference/wiki/diffusion/trajectoryconsistencydistiller/) | Trainer for Trajectory Consistency Distillation (TCD) with trajectory-aware loss. |
| [`TransfusionModel<T>`](/docs/reference/wiki/diffusion/transfusionmodel/) | Transfusion model combining autoregressive language modeling with diffusion generation. |
| [`TripleTextConditioner<T>`](/docs/reference/wiki/diffusion/tripletextconditioner/) | Triple text encoder conditioning module combining two CLIP encoders and a T5 encoder. |
| [`TripoSRModel<T>`](/docs/reference/wiki/diffusion/triposrmodel/) | TripoSR model for ultra-fast feed-forward single-image 3D reconstruction using LRM transformer. |
| [`TurboEditModel<T>`](/docs/reference/wiki/diffusion/turboeditmodel/) | TurboEdit model for fast few-step image editing using distilled SDXL Turbo. |
| [`TurboFillModel<T>`](/docs/reference/wiki/diffusion/turbofillmodel/) | TurboFill model for fast few-step inpainting using adversarial distillation on SDXL. |
| [`UNetBlock<T>`](/docs/reference/wiki/diffusion/unetblock/) | Structure for U-Net blocks containing residual, attention, and sampling layers. |
| [`UNetNoisePredictor<T>`](/docs/reference/wiki/diffusion/unetnoisepredictor/) | U-Net architecture for noise prediction in diffusion models. |
| [`UViTBlock<T>`](/docs/reference/wiki/diffusion/uvitblock/) | Block structure for U-ViT transformer layers. |
| [`UViTNoisePredictor<T>`](/docs/reference/wiki/diffusion/uvitnoisepredictor/) | U-shaped Vision Transformer (U-ViT) noise predictor for diffusion models. |
| [`UdioModel<T>`](/docs/reference/wiki/diffusion/udiomodel/) | Udio/Suno architecture model for full-song music generation with structural awareness. |
| [`UltraEditModel<T>`](/docs/reference/wiki/diffusion/ultraeditmodel/) | UltraEdit model for fine-grained instruction-based image editing with region awareness. |
| [`UniControlNetModel<T>`](/docs/reference/wiki/diffusion/unicontrolnetmodel/) | Uni-ControlNet model for simultaneous multi-condition control with condition-specific adapters. |
| [`UniPCScheduler<T>`](/docs/reference/wiki/diffusion/unipcscheduler/) | UniPC (Unified Predictor-Corrector) scheduler for fast, high-quality diffusion sampling. |
| [`UniSimModel<T>`](/docs/reference/wiki/diffusion/unisimmodel/) | UniSim universal simulator from video and action pairs. |
| [`UnifiedDistillationSampling<T>`](/docs/reference/wiki/diffusion/unifieddistillationsampling/) | Unified Distillation Sampling (UDS) framework unifying SDS, VSD, CSD, and ISM variants. |
| [`UpBlock<T>`](/docs/reference/wiki/diffusion/upblock/) | Upsampling block for VAE decoder with transposed convolution and multiple ResBlocks. |
| [`UpscaleAVideoModel<T>`](/docs/reference/wiki/diffusion/upscaleavideomodel/) | Upscale-A-Video model for temporally consistent video super-resolution with diffusion. |
| [`VAEDecoder<T>`](/docs/reference/wiki/diffusion/vaedecoder/) | Convolutional decoder for VAE that reconstructs images from latent space. |
| [`VAEEncoder<T>`](/docs/reference/wiki/diffusion/vaeencoder/) | Convolutional encoder for VAE that compresses images to latent space. |
| [`VAEResBlock<T>`](/docs/reference/wiki/diffusion/vaeresblock/) | Residual block for VAE encoder/decoder with GroupNorm and skip connections. |
| [`VariationalRFScheduler<T>`](/docs/reference/wiki/diffusion/variationalrfscheduler/) | Variational Rectified Flow scheduler with learned time-dependent noise injection. |
| [`VariationalScoreDistillation<T>`](/docs/reference/wiki/diffusion/variationalscoredistillation/) | Variational Score Distillation (VSD) for high-fidelity text-to-3D generation. |
| [`Veo3Model<T>`](/docs/reference/wiki/diffusion/veo3model/) | Veo 3 with native audio generation and dialogue synchronization. |
| [`VeoModel<T>`](/docs/reference/wiki/diffusion/veomodel/) | Veo model for Google's high-fidelity cascaded video generation with temporal super-resolution. |
| [`VideoCrafter2Model<T>`](/docs/reference/wiki/diffusion/videocrafter2model/) | VideoCrafter 2 with improved quality and style fusion. |
| [`VideoCrafterModel<T>`](/docs/reference/wiki/diffusion/videocraftermodel/) | VideoCrafter model for high-quality text-to-video and image-to-video generation. |
| [`VideoP2PModel<T>`](/docs/reference/wiki/diffusion/videop2pmodel/) | VideoP2P prompt-to-prompt cross-attention control for video editing. |
| [`VideoPoetModel<T>`](/docs/reference/wiki/diffusion/videopoetmodel/) | VideoPoet LLM-based zero-shot video generation. |
| [`VideoUNetPredictor<T>`](/docs/reference/wiki/diffusion/videounetpredictor/) | 3D U-Net architecture for video noise prediction in diffusion models. |
| [`VoiceCraftModel<T>`](/docs/reference/wiki/diffusion/voicecraftmodel/) | VoiceCraft model for zero-shot speech editing and text-to-speech with neural codec language modeling. |
| [`Wan21Model<T>`](/docs/reference/wiki/diffusion/wan21model/) | Wan 2.1 video model with MoE denoising and full 3D attention. |
| [`Wan22Model<T>`](/docs/reference/wiki/diffusion/wan22model/) | Wan 2.2 video model with timestep-specialized MoE experts. |
| [`WanVideoModel<T>`](/docs/reference/wiki/diffusion/wanvideomodel/) | Wan video model for Alibaba's scalable DiT video generation with full 3D attention. |
| [`Wonder3DModel<T>`](/docs/reference/wiki/diffusion/wonder3dmodel/) | Wonder3D model for multi-view cross-domain diffusion with simultaneous RGB and normal map generation. |
| [`Zero123Model<T>`](/docs/reference/wiki/diffusion/zero123model/) | Zero-1-to-3 model for novel view synthesis from a single image. |
| [`ZeroTerminalSNRSchedule<T>`](/docs/reference/wiki/diffusion/zeroterminalsnrschedule/) | Zero Terminal SNR noise schedule ensuring signal-to-noise ratio reaches exactly zero at the final timestep. |

## Base Classes (11)

| Type | Summary |
|:-----|:--------|
| [`AudioDiffusionModelBase<T>`](/docs/reference/wiki/diffusion/audiodiffusionmodelbase/) | Base class for audio diffusion models that generate sound and music. |
| [`CompositeConditioningBase<T>`](/docs/reference/wiki/diffusion/compositeconditioningbase/) | Base class for conditioning modules that compose other conditioners rather than owning their own learnable weights. |
| [`DiffusionModelBase<T>`](/docs/reference/wiki/diffusion/diffusionmodelbase/) | Base class for diffusion-based generative models providing common functionality. |
| [`DiffusionPreprocessorBase<T>`](/docs/reference/wiki/diffusion/diffusionpreprocessorbase/) | Base class for diffusion model condition preprocessors that convert input images into control signals (edge maps, depth maps, pose skeletons, etc.). |
| [`LatentDiffusionModelBase<T>`](/docs/reference/wiki/diffusion/latentdiffusionmodelbase/) | Base class for latent diffusion models that operate in a compressed latent space. |
| [`NoisePredictorBase<T>`](/docs/reference/wiki/diffusion/noisepredictorbase/) | Base class for noise prediction networks used in diffusion models. |
| [`NoiseSchedulerBase<T>`](/docs/reference/wiki/diffusion/noiseschedulerbase/) | Base class for diffusion model noise schedulers providing common functionality. |
| [`TextConditioningBase<T>`](/docs/reference/wiki/diffusion/textconditioningbase/) | Base class for text conditioning modules used in diffusion models. |
| [`ThreeDDiffusionModelBase<T>`](/docs/reference/wiki/diffusion/threeddiffusionmodelbase/) | Base class for 3D diffusion models that generate 3D content like point clouds, meshes, and scenes. |
| [`VAEModelBase<T>`](/docs/reference/wiki/diffusion/vaemodelbase/) | Base class for Variational Autoencoder (VAE) models used in latent diffusion. |
| [`VideoDiffusionModelBase<T>`](/docs/reference/wiki/diffusion/videodiffusionmodelbase/) | Base class for video diffusion models that generate temporal sequences. |

## Interfaces (3)

| Type | Summary |
|:-----|:--------|
| [`IContextualLayer<T>`](/docs/reference/wiki/diffusion/icontextuallayer/) | Interface for layers that accept context (conditioning). |
| [`IGuidanceMethod<T>`](/docs/reference/wiki/diffusion/iguidancemethod/) | Interface for guidance methods that modify noise predictions during diffusion sampling. |
| [`IPoseExtractor<T>`](/docs/reference/wiki/diffusion/iposeextractor/) | Pluggable keypoint extractor interface. |

## Enums (9)

| Type | Summary |
|:-----|:--------|
| [`AudioLDM2Variant`](/docs/reference/wiki/diffusion/audioldm2variant/) | AudioLDM 2 model variant. |
| [`BlendProfile`](/docs/reference/wiki/diffusion/blendprofile/) | Specifies the blending profile for overlap region transitions. |
| [`BlendingMode`](/docs/reference/wiki/diffusion/blendingmode/) | Blending modes for overlapping window regions. |
| [`ControlNetConditionType`](/docs/reference/wiki/diffusion/controlnetconditiontype/) | Specifies the type of conditioning input for multi-control ControlNet composition. |
| [`ControlType`](/docs/reference/wiki/diffusion/controltype/) | Types of control signals supported by ControlNet. |
| [`MusicGenSize`](/docs/reference/wiki/diffusion/musicgensize/) | MusicGen model size variants. |
| [`PaddingMode`](/docs/reference/wiki/diffusion/paddingmode/) | Padding mode for STFT centering. |
| [`ShardingStrategy`](/docs/reference/wiki/diffusion/shardingstrategy/) | Strategy for distributing layers across devices. |
| [`SparsityStrategy`](/docs/reference/wiki/diffusion/sparsitystrategy/) | Strategy for selecting which transformer blocks to skip in sparse computation. |

## Structs (1)

| Type | Summary |
|:-----|:--------|
| [`DreamVector3<T>`](/docs/reference/wiki/diffusion/dreamvector3/) | 3D vector type for DreamFusion. |

## Options & Configuration (8)

| Type | Summary |
|:-----|:--------|
| [`DiffusionMemoryConfig`](/docs/reference/wiki/diffusion/diffusionmemoryconfig/) | Configuration for diffusion model memory management. |
| [`DreamFusionConfig`](/docs/reference/wiki/diffusion/dreamfusionconfig/) | Configuration for DreamFusion model. |
| [`MVDreamConfig`](/docs/reference/wiki/diffusion/mvdreamconfig/) | Configuration for MVDream model. |
| [`MotionModuleConfig`](/docs/reference/wiki/diffusion/motionmoduleconfig/) | Configuration for AnimateDiff motion modules. |
| [`PixArtOptions<T>`](/docs/reference/wiki/diffusion/pixartoptions/) | Options for PixArt-α model configuration. |
| [`SchedulerConfig<T>`](/docs/reference/wiki/diffusion/schedulerconfig/) | Configuration options for diffusion model step schedulers. |
| [`ShardingConfig`](/docs/reference/wiki/diffusion/shardingconfig/) | Configuration for model sharding. |
| [`SpectrogramConfig`](/docs/reference/wiki/diffusion/spectrogramconfig/) | Configuration for spectrogram generation. |

## Helpers & Utilities (2)

| Type | Summary |
|:-----|:--------|
| [`ModelSizes<T>`](/docs/reference/wiki/diffusion/modelsizes/) | Standard DiT model sizes. |
| [`PointCounts<T>`](/docs/reference/wiki/diffusion/pointcounts/) | Standard Point-E point counts. |


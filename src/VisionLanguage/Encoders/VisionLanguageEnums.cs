namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// Specifies the precision mode for model inference and training.
/// </summary>
public enum ModelPrecision
{
    /// <summary>32-bit floating point (default, highest precision).</summary>
    Float32,
    /// <summary>16-bit floating point (faster, lower memory).</summary>
    Float16,
    /// <summary>Brain floating point 16 (faster on modern GPUs).</summary>
    BFloat16
}

/// <summary>
/// Specifies the Vision Transformer (ViT) variant used as the image encoder.
/// </summary>
public enum ViTVariant
{
    /// <summary>ViT-B/32: Base model with 32x32 patch size (86M params, fastest).</summary>
    ViTB32,
    /// <summary>ViT-B/16: Base model with 16x16 patch size (86M params, better than B/32).</summary>
    ViTB16,
    /// <summary>ViT-L/14: Large model with 14x14 patch size (304M params).</summary>
    ViTL14,
    /// <summary>ViT-L/14@336: Large model at 336px resolution.</summary>
    ViTL14At336,
    /// <summary>ViT-H/14: Huge model with 14x14 patch size (632M params).</summary>
    ViTH14,
    /// <summary>ViT-G/14: Giant model with 14x14 patch size (1.01B params).</summary>
    ViTG14,
    /// <summary>ViT-bigG/14: Biggest model with 14x14 patch size (2.54B params).</summary>
    ViTBigG14,
    /// <summary>ViT-e/14: EVA-CLIP extra-large model (4.4B params).</summary>
    ViTE14,
    /// <summary>ViT-SO400M/14: SigLIP Shape-Optimized 400M param model.</summary>
    ViTSO400M14,
    /// <summary>EfficientNet-B7: CNN backbone used by ALIGN.</summary>
    EfficientNetB7,
    /// <summary>CoAtNet: Hybrid CNN-Transformer used by BASIC.</summary>
    CoAtNet
}

/// <summary>
/// Specifies the text encoder architecture.
/// </summary>
public enum TextEncoderVariant
{
    /// <summary>GPT-2-style transformer with causal attention (used by CLIP).</summary>
    Transformer,
    /// <summary>RoBERTa-based bidirectional encoder (used by some variants).</summary>
    RoBERTa,
    /// <summary>BERT-based bidirectional encoder.</summary>
    BERT,
    /// <summary>mT5-based multilingual text encoder (used by PaLI).</summary>
    MT5,
    /// <summary>LLM-enhanced text encoder (used by LLM2CLIP).</summary>
    LLMEnhanced
}

/// <summary>
/// Specifies the contrastive loss function type.
/// </summary>
public enum ContrastiveLossType
{
    /// <summary>Standard InfoNCE (softmax) loss used by CLIP.</summary>
    InfoNCE,
    /// <summary>Sigmoid loss used by SigLIP for pairwise computation (no global normalization).</summary>
    Sigmoid,
    /// <summary>Symmetric cross-entropy used by OpenCLIP.</summary>
    SymmetricCrossEntropy
}

/// <summary>
/// Specifies the pre-training dataset used for the model.
/// </summary>
public enum PretrainingDataset
{
    /// <summary>OpenAI's proprietary WIT (WebImageText) 400M dataset.</summary>
    WIT400M,
    /// <summary>LAION-400M open dataset.</summary>
    LAION400M,
    /// <summary>LAION-2B open dataset (2 billion image-text pairs).</summary>
    LAION2B,
    /// <summary>LAION-5B open dataset (5 billion pairs).</summary>
    LAION5B,
    /// <summary>DataComp-1B curated dataset.</summary>
    DataComp1B,
    /// <summary>DataComp-12B dataset.</summary>
    DataComp12B,
    /// <summary>CommonPool filtered dataset.</summary>
    CommonPool,
    /// <summary>WebLI (Web Language-Image) dataset used by Google models.</summary>
    WebLI,
    /// <summary>JFT-300M Google internal dataset.</summary>
    JFT300M,
    /// <summary>MetaCLIP metadata-curated dataset.</summary>
    MetaCLIPCurated,
    /// <summary>DFN-filtered high-quality dataset.</summary>
    DFNFiltered,
    /// <summary>PMC-15M biomedical dataset.</summary>
    PMC15M,
    /// <summary>Custom/other dataset.</summary>
    Custom
}

/// <summary>
/// Specifies the domain specialization for a CLIP-family model.
/// </summary>
public enum DomainSpecialization
{
    /// <summary>General-purpose model (default).</summary>
    General,
    /// <summary>Biomedical image-text pairs.</summary>
    Biomedical,
    /// <summary>Medical imaging (radiology, pathology).</summary>
    Medical,
    /// <summary>Remote sensing / satellite imagery.</summary>
    RemoteSensing,
    /// <summary>Region-level (object-centric) alignment.</summary>
    RegionLevel
}

/// <summary>
/// Specifies the positional embedding type for vision transformers.
/// </summary>
public enum PositionalEmbeddingType
{
    /// <summary>Learned absolute positional embeddings (standard ViT).</summary>
    Learned,
    /// <summary>Sinusoidal fixed positional embeddings.</summary>
    Sinusoidal,
    /// <summary>Rotary positional embeddings (RoPE).</summary>
    RoPE,
    /// <summary>2D relative positional bias.</summary>
    Relative2D,
    /// <summary>No positional embedding.</summary>
    None
}

/// <summary>
/// Specifies the global feature pooling strategy for vision encoders.
/// </summary>
public enum PoolingStrategy
{
    /// <summary>Use the [CLS] token output as the global feature.</summary>
    ClsToken,
    /// <summary>Average pool all patch token outputs.</summary>
    MeanPool,
    /// <summary>Use the last token output.</summary>
    LastToken,
    /// <summary>Concatenate CLS and mean-pooled features.</summary>
    ClsPlusMean
}

/// <summary>
/// Specifies the Florence-2 model size variant.
/// </summary>
public enum Florence2ModelSize
{
    /// <summary>Florence-2 Base (0.23B parameters).</summary>
    Base,
    /// <summary>Florence-2 Large (0.77B parameters).</summary>
    Large
}

/// <summary>
/// Specifies how vision and language features are fused in a VLM.
/// </summary>
public enum FusionType
{
    /// <summary>Single stream: visual and text tokens concatenated in one transformer.</summary>
    SingleStream,
    /// <summary>Dual stream: separate encoders with cross-attention bridges.</summary>
    DualStream,
    /// <summary>Co-attention: parallel streams with co-attention layers.</summary>
    CoAttention,
    /// <summary>Cross-modal encoder: dedicated cross-modal transformer layers.</summary>
    CrossModal,
    /// <summary>Bridge layers: explicit bridge connections between encoders.</summary>
    BridgeLayers
}

/// <summary>
/// Specifies the visual feature extraction method for foundational VLMs.
/// </summary>
public enum VisualFeatureType
{
    /// <summary>Object-level features from a detection model (e.g., Faster R-CNN).</summary>
    RegionFeatures,
    /// <summary>Raw image patches linearly embedded (e.g., ViLT, ViT-based).</summary>
    PatchEmbeddings,
    /// <summary>Grid features from a CNN backbone.</summary>
    GridFeatures
}

/// <summary>
/// Specifies the architecture type for generative vision-language models.
/// </summary>
public enum GenerativeArchitectureType
{
    /// <summary>Q-Former bridge: learnable queries cross-attend to vision, then feed decoder (InstructBLIP, BLIP-3).</summary>
    QFormerBridge,
    /// <summary>Encoder-decoder: ViT encoder + autoregressive text decoder with cross-attention (GIT, CoCa, PaLI).</summary>
    EncoderDecoder,
    /// <summary>Perceiver resampler: latent queries cross-attend to vision, gated cross-attention into LLM (Flamingo, IDEFICS).</summary>
    PerceiverResampler,
    /// <summary>Causal multimodal: visual tokens embedded directly in causal language model (KOSMOS).</summary>
    CausalMultimodal,
    /// <summary>Unified generation: single model for both understanding and image/text generation (Emu).</summary>
    UnifiedGeneration
}

/// <summary>
/// Specifies the architecture type for instruction-tuned vision-language models.
/// </summary>
public enum InstructionTunedArchitectureType
{
    /// <summary>MLP projection: vision encoder -> MLP connector -> LLM (LLaVA, InternVL, DeepSeek-VL, Phi-3-Vision).</summary>
    MLPProjection,
    /// <summary>Q-Former projection: vision encoder -> Q-Former -> linear projection -> LLM (MiniGPT-4, MiniGPT-v2).</summary>
    QFormerProjection,
    /// <summary>Cross-attention resampler: vision encoder -> cross-attention resampler -> LLM (Qwen-VL series).</summary>
    CrossAttentionResampler,
    /// <summary>Visual expert: vision encoder -> visual expert modules interleaved in every LLM layer (CogVLM).</summary>
    VisualExpert,
    /// <summary>Visual abstractor: vision encoder -> learnable visual abstractor module -> LLM (mPLUG-Owl series).</summary>
    VisualAbstractor,
    /// <summary>Direct patch embedding: raw image patches go directly into the language model without a separate vision encoder (Fuyu).</summary>
    DirectPatch
}

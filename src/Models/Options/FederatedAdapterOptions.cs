namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies the federated adapter type for parameter-efficient fine-tuning.
/// </summary>
public enum FederatedAdapterType
{
    /// <summary>No adapter — standard full-model aggregation.</summary>
    None,
    /// <summary>LoRA — Low-Rank Adaptation with uniform rank across clients.</summary>
    LoRA,
    /// <summary>Heterogeneous LoRA — different ranks per client with SVD aggregation.</summary>
    HeterogeneousLoRA,
    /// <summary>Prompt Tuning — soft prompt token aggregation.</summary>
    PromptTuning,
    /// <summary>FedPETuning — unified PEFT framework (LoRA, adapters, prefix, BitFit).</summary>
    FedPETuning,
    /// <summary>FedAdapter — bottleneck adapter layers inserted into transformer blocks.</summary>
    FedAdapter,
    /// <summary>FLoRA — stacked lossless LoRA aggregation via SVD.</summary>
    FLoRA,
    /// <summary>HierFedLoRA — hierarchical LoRA for edge-cloud topologies.</summary>
    HierFedLoRA,
    /// <summary>SLoRA — sparse LoRA that only communicates non-zero adapter elements.</summary>
    SLoRA,
    /// <summary>DP-FedLoRA — differentially private LoRA with per-layer noise calibration.</summary>
    DPFedLoRA,
    /// <summary>FedMeZO — memory-efficient zeroth-order optimization for LLM fine-tuning.</summary>
    FedMeZO
}

/// <summary>
/// Configuration options for federated parameter-efficient fine-tuning (PEFT).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When fine-tuning large models (like LLMs) in a federated setting,
/// sending all model parameters between clients and server is too expensive. PEFT adapters only
/// train and communicate a tiny fraction of parameters (often &lt;1%), making federated LLM
/// fine-tuning practical.</para>
/// </remarks>
public class FederatedAdapterOptions : ModelOptions
{
    /// <summary>
    /// Gets or sets the adapter type. Default: None.
    /// </summary>
    public FederatedAdapterType AdapterType { get; set; } = FederatedAdapterType.None;

    /// <summary>
    /// Gets or sets the LoRA rank. Lower values = more compression. Default: 8.
    /// </summary>
    /// <remarks>
    /// Typical values: 4 (aggressive compression) to 64 (high fidelity).
    /// Production LLM fine-tuning commonly uses rank 8–16.
    /// </remarks>
    public int LoRARank { get; set; } = 8;

    /// <summary>
    /// Gets or sets the LoRA alpha scaling factor. Default: 16.
    /// </summary>
    /// <remarks>
    /// The effective scaling is alpha/rank. Higher alpha relative to rank amplifies the adaptation.
    /// </remarks>
    public double LoRAAlpha { get; set; } = 16.0;

    /// <summary>
    /// Gets or sets the number of layers to apply LoRA adapters to. Default: 4.
    /// </summary>
    public int NumAdaptedLayers { get; set; } = 4;

    /// <summary>
    /// Gets or sets the layer input dimension for LoRA. Default: 768.
    /// </summary>
    public int LayerInputDimension { get; set; } = 768;

    /// <summary>
    /// Gets or sets the layer output dimension for LoRA. Default: 768.
    /// </summary>
    public int LayerOutputDimension { get; set; } = 768;

    /// <summary>
    /// Gets or sets the maximum LoRA rank for heterogeneous LoRA. Default: 64.
    /// </summary>
    public int MaxHeterogeneousRank { get; set; } = 64;

    /// <summary>
    /// Gets or sets the number of soft prompt tokens for prompt tuning. Default: 20.
    /// </summary>
    public int NumPromptTokens { get; set; } = 20;

    /// <summary>
    /// Gets or sets the embedding dimension for prompt tuning. Default: 768.
    /// </summary>
    public int EmbeddingDimension { get; set; } = 768;

    /// <summary>
    /// Gets or sets whether to apply adapter-level differential privacy. Default: false.
    /// </summary>
    /// <remarks>
    /// When enabled, DP noise is applied only to adapter parameters rather than the full model,
    /// giving better privacy-utility tradeoffs for PEFT.
    /// </remarks>
    public bool AdapterLevelDP { get; set; } = false;

    /// <summary>
    /// Gets or sets the DP noise multiplier for adapter-level privacy. Default: 1.0.
    /// </summary>
    public double AdapterDPNoiseMultiplier { get; set; } = 1.0;
}

namespace AiDotNet.Enums
{
    /// <summary>
    /// Represents different strategies for fusing multiple data modalities in multimodal AI systems.
    /// Strategies for fusing multiple modalities in multimodal models
    /// </summary>
    public enum ModalityFusionStrategy
    {
        /// <summary>
        /// Fuses modalities at the input level before processing.
        /// Early fusion - combine modalities at input level
        /// </summary>
        EarlyFusion,

        /// <summary>
        /// Processes each modality independently and combines results at the end.
        /// Late fusion - combine modalities at output/decision level
        /// </summary>
        LateFusion,

        /// <summary>
        /// Uses cross-attention mechanisms to fuse modalities during processing.
        /// Cross-attention mechanism for modality fusion
        /// </summary>
        CrossAttention,

        /// <summary>
        /// Uses hierarchical fusion combining multiple fusion strategies.
        /// Hierarchical fusion with multiple levels
        /// </summary>
        Hierarchical
        Hierarchical,

        /// <summary>
        /// Transformer-based fusion
        /// </summary>
        Transformer,

        /// <summary>
        /// Gated fusion with learnable gates
        /// </summary>
        Gated,

        /// <summary>
        /// Tensor fusion network
        /// </summary>
        TensorFusion,

        /// <summary>
        /// Bilinear pooling fusion
        /// </summary>
        BilinearPooling,

        /// <summary>
        /// Attention-weighted averaging
        /// </summary>
        AttentionWeighted,

        /// <summary>
        /// Simple concatenation
        /// </summary>
        Concatenation
    }
}
namespace AiDotNet.Enums
{
    /// <summary>
    /// Represents different strategies for fusing multiple data modalities in multimodal AI systems.
    /// </summary>
    public enum ModalityFusionStrategy
    {
        /// <summary>
        /// Fuses modalities at the input level before processing.
        /// </summary>
        EarlyFusion,

        /// <summary>
        /// Processes each modality independently and combines results at the end.
        /// </summary>
        LateFusion,

        /// <summary>
        /// Uses cross-attention mechanisms to fuse modalities during processing.
        /// </summary>
        CrossAttention,

        /// <summary>
        /// Uses hierarchical fusion combining multiple fusion strategies.
        /// </summary>
        Hierarchical
    }
}

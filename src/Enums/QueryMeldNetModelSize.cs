namespace AiDotNet.Enums;

/// <summary>
/// Defines the backbone size variants for QueryMeldNet (MQ-Former).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> QueryMeldNet uses dynamic query melding to scale across diverse
/// datasets. Instance queries and stuff queries are fused via cross-attention, enabling
/// strong generalization across multiple segmentation benchmarks.
/// </para>
/// <para>
/// <b>Technical Details:</b> Extends mask-based segmentation with a dynamic query melding
/// mechanism. Instance and stuff queries interact through cross-attention layers, improving
/// both panoptic and instance segmentation quality.
/// </para>
/// <para>
/// <b>Reference:</b> "QueryMeldNet: Dynamic Query Melding for Multi-Dataset Segmentation", CVPR 2025.
/// </para>
/// </remarks>
public enum QueryMeldNetModelSize
{
    /// <summary>
    /// ResNet-50 backbone. Efficient baseline variant.
    /// </summary>
    R50,

    /// <summary>
    /// Swin-L backbone. High-accuracy variant with Swin Transformer.
    /// </summary>
    SwinLarge
}

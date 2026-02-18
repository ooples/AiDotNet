namespace AiDotNet.Enums;

/// <summary>
/// Defines the backbone size variants for UNINEXT.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> UNINEXT (Universal INstance pErception through neXt-generation learning)
/// reformulates 10+ instance perception tasks as a unified object discovery and retrieval problem.
/// It achieves state-of-the-art on over 20 benchmarks across object detection, instance
/// segmentation, referring expression comprehension, and more.
/// </para>
/// <para>
/// <b>Technical Details:</b> Uses a shared backbone with task-specific prompt embeddings.
/// Tasks include detection, instance segmentation, SOT, MOT, VIS, R-VOS, and more.
/// All are reformulated as retrieve-then-segment with unified query representations.
/// </para>
/// <para>
/// <b>Reference:</b> Yan et al., "Universal Instance Perception as Object Discovery and Retrieval",
/// CVPR 2023.
/// </para>
/// </remarks>
public enum UNINEXTModelSize
{
    /// <summary>
    /// ResNet-50 backbone. Fast and efficient.
    /// </summary>
    R50,

    /// <summary>
    /// Swin-L backbone. Highest accuracy.
    /// </summary>
    SwinLarge,

    /// <summary>
    /// ViT-H backbone (Huge). Maximum capacity for multi-task perception.
    /// </summary>
    ViTHuge
}

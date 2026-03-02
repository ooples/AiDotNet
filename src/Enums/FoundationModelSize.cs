namespace AiDotNet.Enums;

/// <summary>
/// Defines the size variants available for foundation models.
/// </summary>
/// <remarks>
/// <para>
/// Foundation models typically come in multiple sizes, trading off between accuracy and
/// computational cost. This enum replaces error-prone string-based size selection
/// (e.g., <c>"base"</c>, <c>"large"</c>) with compile-time type safety.
/// </para>
/// <para>
/// <b>For Beginners:</b> Larger models generally produce better predictions but require
/// more memory and computation time. Choose a size based on your resources:
/// <list type="bullet">
/// <item><b>Tiny/Mini:</b> Fast experiments, limited hardware, edge deployment</item>
/// <item><b>Small/Base:</b> Good balance of quality and speed for most use cases</item>
/// <item><b>Large/XLarge:</b> Best accuracy when compute resources are available</item>
/// </list>
/// </para>
/// </remarks>
public enum FoundationModelSize
{
    /// <summary>
    /// Tiny variant with minimal parameters (~1-5M). Fastest inference, lowest memory.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The smallest available model. Use when speed is critical
    /// or hardware is very limited (e.g., edge devices, real-time applications).
    /// May sacrifice some accuracy for efficiency.
    /// </para>
    /// </remarks>
    Tiny,

    /// <summary>
    /// Mini variant (~5-20M parameters). Lightweight but capable.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A step up from Tiny. Good for quick experiments and
    /// prototyping where you want reasonable quality without heavy compute.
    /// </para>
    /// </remarks>
    Mini,

    /// <summary>
    /// Small variant (~14-50M parameters). Good efficiency-quality tradeoff.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A solid choice when you need better quality than Mini
    /// but still want relatively fast inference. Works well on consumer GPUs.
    /// </para>
    /// </remarks>
    Small,

    /// <summary>
    /// Base variant (~50-200M parameters). The default recommended size for most use cases.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The recommended starting point for most applications.
    /// Provides strong accuracy with reasonable computational requirements.
    /// This is typically the default size for foundation models.
    /// </para>
    /// </remarks>
    Base,

    /// <summary>
    /// Large variant (~200-700M parameters). Higher capacity for demanding tasks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use when you need the best possible accuracy and have
    /// access to good hardware (e.g., dedicated GPU). Slower inference but captures
    /// more complex patterns in the data.
    /// </para>
    /// </remarks>
    Large,

    /// <summary>
    /// Extra-large variant (700M+ parameters). Maximum capacity.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The largest available variant. Use only when maximum
    /// accuracy is needed and compute resources are not a constraint. Requires
    /// significant GPU memory and processing time.
    /// </para>
    /// </remarks>
    XLarge
}

namespace AiDotNet.ModelLoading.Pretrained;

/// <summary>
/// The numeric precision a pretrained checkpoint's weights are decoded into when loaded.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Model files store their weights in a specific precision
/// (32-bit float, 16-bit half, or 16-bit "brain float"). This picks what precision AiDotNet
/// decodes them <em>to</em>. The default, <see cref="Auto"/>, keeps whatever the file used —
/// which is almost always what you want.
/// </para>
/// </remarks>
public enum PretrainedDType
{
    /// <summary>Keep the checkpoint's on-disk precision (recommended default).</summary>
    Auto,

    /// <summary>Decode weights to 32-bit single precision.</summary>
    Float32,

    /// <summary>Decode weights to IEEE 754 half precision (F16).</summary>
    Float16,

    /// <summary>Decode weights to bfloat16 (BF16) — the dtype most modern LLMs ship in.</summary>
    BFloat16,
}

namespace AiDotNet.Enums;

/// <summary>
/// How an attention layer initializes the bias of its output projection.
/// </summary>
/// <remarks>
/// <para>
/// Reference attention modules build the output projection as a plain <c>nn.Linear(inner, dim)</c>, so its bias starts
/// at PyTorch's default, uniform in <c>±1/sqrt(fan_in)</c> (LDM / MVDream / ImageDream <c>CrossAttention.to_out</c>,
/// LGM's <c>MemEffAttention.proj</c>). Some code paths in this library start it at zero instead.
/// </para>
/// <para><b>For Beginners:</b> this only changes the starting value of one small vector. Pretrained weights overwrite
/// it either way; it matters when the layer is trained from scratch and must match the reference run.</para>
/// </remarks>
public enum AttentionBiasInitialization
{
    /// <summary>The bias starts at zero.</summary>
    Zero = 0,

    /// <summary>
    /// PyTorch's <c>nn.Linear</c> default: uniform in <c>±1/sqrt(fan_in)</c>, with <c>fan_in</c> the projection's input
    /// width.
    /// </summary>
    TorchDefault = 1
}

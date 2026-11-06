namespace AiDotNet.Enums;

/// <summary>
/// How an Adam-family optimizer stores its first and second moments (m, v): the bulk of a large model's training
/// memory after the weights themselves.
/// </summary>
/// <remarks>
/// <para>The update math is fp32 in every mode; only the stored moments are narrower, as in bitsandbytes' 8-bit
/// optimizers. Device and host steps keep the same state, so a run can move between them and checkpoints stay in one
/// format.</para>
/// <para><b>For Beginners:</b> AdamW remembers two numbers per weight. Storing them in fewer bits saves memory with
/// little or no change in how the model trains: <see cref="BFloat16"/> halves that memory and <see cref="Int8Blockwise"/>
/// quarters it.</para>
/// </remarks>
public enum OptimizerMomentStorage
{
    /// <summary>32-bit moments: 8 bytes per parameter for m and v together.</summary>
    Float32 = 0,

    /// <summary>bfloat16 moments, round-to-nearest-even: 4 bytes per parameter.</summary>
    BFloat16 = 1,

    /// <summary>
    /// 8-bit block-wise dynamic quantization (Dettmers et al., "8-bit Optimizers via Block-wise Quantization"): one
    /// byte per moment element plus one scale per block, about 2 bytes per parameter. Tensors shorter than the
    /// configured minimum keep 32-bit moments, as bitsandbytes does.
    /// </summary>
    Int8Blockwise = 2,
}

internal static class OptimizerMomentDefaults
{
    internal const int BlockSize = 256;
    internal const int MinQuantizedLength = 4096;
}

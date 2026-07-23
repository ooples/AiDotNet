namespace AiDotNet.Enums;

/// <summary>
/// Defines the distributed training strategy to use.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b>
/// These strategies determine how your model and data are split across multiple GPUs or machines.
/// DDP (Distributed Data Parallel) is the most common and works for 90% of use cases.
/// </para>
/// </remarks>
public enum DistributedStrategy
{
    /// <summary>
    /// No distributed training - Single device training (default).
    /// </summary>
    /// <remarks>
    /// <para><b>Use when:</b> Training on a single GPU or CPU.</para>
    /// <para><b>Memory:</b> Full model on one device.</para>
    /// <para><b>Communication:</b> None.</para>
    /// </remarks>
    None = 0,

    /// <summary>
    /// Distributed Data Parallel (DDP) - Most common strategy.
    /// Parameters are replicated on each GPU, gradients are averaged.
    /// </summary>
    /// <remarks>
    /// <para><b>Use when:</b> Your model fits on one GPU but you want faster training.</para>
    /// <para><b>Memory:</b> Each GPU has full model copy.</para>
    /// <para><b>Communication:</b> Low (only gradients).</para>
    /// </remarks>
    DDP,

    /// <summary>
    /// Fully Sharded Data Parallel (FSDP) - PyTorch style, full parameter sharding.
    /// </summary>
    /// <remarks>
    /// <para><b>Use when:</b> Your model is too large to fit on one GPU.</para>
    /// <para><b>Memory:</b> Excellent (parameters sharded).</para>
    /// <para><b>Communication:</b> High (AllGather for parameters).</para>
    /// </remarks>
    FSDP,

    /// <summary>
    /// ZeRO Stage 1 - Only optimizer states are sharded.
    /// </summary>
    /// <remarks>
    /// <para><b>Use when:</b> Optimizer state memory is the bottleneck (e.g., Adam).</para>
    /// <para><b>Memory:</b> Good (4-8x reduction in optimizer memory).</para>
    /// </remarks>
    ZeRO1,

    /// <summary>
    /// ZeRO Stage 2 - Optimizer states + gradients are sharded.
    /// </summary>
    /// <remarks>
    /// <para><b>Use when:</b> Both optimizer and gradient memory are issues.</para>
    /// <para><b>Memory:</b> Very good (optimizer + gradient savings).</para>
    /// </remarks>
    ZeRO2,

    /// <summary>
    /// ZeRO Stage 3 - Full sharding (equivalent to FSDP).
    /// </summary>
    /// <remarks>
    /// <para><b>Use when:</b> You prefer ZeRO terminology over FSDP.</para>
    /// <para><b>Memory:</b> Excellent (everything sharded).</para>
    /// </remarks>
    ZeRO3,

    /// <summary>
    /// Pipeline Parallel - Model split into stages across GPUs.
    /// </summary>
    /// <remarks>
    /// <para><b>Use when:</b> You have a very deep model with many layers.</para>
    /// <para><b>Memory:</b> Excellent for deep models.</para>
    /// </remarks>
    PipelineParallel,

    /// <summary>
    /// Tensor Parallel - Individual layers split across GPUs (Megatron-LM style).
    /// </summary>
    /// <remarks>
    /// <para><b>Use when:</b> You have very wide layers (large hidden dimensions).</para>
    /// <para><b>Communication:</b> High (requires fast interconnect like NVLink).</para>
    /// </remarks>
    TensorParallel,

    /// <summary>
    /// Hybrid - 3D parallelism combining data + tensor + pipeline.
    /// </summary>
    /// <remarks>
    /// <para><b>Use when:</b> Training frontier models (100B+ parameters) on 100s of GPUs.</para>
    /// <para><b>Complexity:</b> Very high, requires expert knowledge.</para>
    /// </remarks>
    Hybrid
}

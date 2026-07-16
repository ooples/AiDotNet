namespace AiDotNet.DistributedTraining;

/// <summary>
/// Configuration for parameter sharding in distributed training.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b>
/// This configuration tells the sharding system how to divide up parameters
/// and how to handle communication. Think of it as the "rules" for how the
/// team collaborates.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type</typeparam>
[AiDotNet.Configuration.YamlConfigurable("ShardingConfiguration")]
public interface IShardingConfiguration<T>
{
    /// <summary>
    /// Gets whether to automatically synchronize gradients after backward pass.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// When true, gradients are automatically shared across all processes after
    /// each training step. This is usually what you want for standard training.
    /// You might set it to false if you want manual control over synchronization.
    /// </para>
    /// <para>Default: true</para>
    /// </remarks>
    bool AutoSyncGradients { get; }

    /// <summary>
    /// Gets the communication backend to use for distributed operations.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This is the "communication system" that processes use to talk to each other.
    /// It could be an in-memory backend for testing or an MPI backend for real
    /// distributed training across multiple machines.
    /// </para>
    /// </remarks>
    ICommunicationBackend<T> CommunicationBackend { get; }

    /// <summary>
    /// Gets the minimum parameter group size for sharding.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Parameters smaller than this might be grouped together to reduce communication overhead.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// Sending many tiny messages is inefficient. This setting groups small
    /// parameters together into larger chunks before communicating them.
    /// Think of it like sending one big box instead of 100 tiny envelopes.
    /// </para>
    /// <para>Default: 1024</para>
    /// </remarks>
    int MinimumParameterGroupSize { get; }

    /// <summary>
    /// Gets whether to enable gradient compression to reduce communication costs.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// Gradient compression reduces the size of data that needs to be sent
    /// between processes. It's like zipping a file before sending it - faster
    /// to send, but requires a tiny bit of extra work to compress/decompress.
    /// This can significantly speed up training on slower networks.
    /// </para>
    /// <para>Default: false</para>
    /// </remarks>
    bool EnableGradientCompression { get; }

    /// <summary>
    /// Gets the learning rate for gradient application during training.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// The learning rate controls how much to update model parameters based on
    /// computed gradients. A typical default is 0.01. Lower values mean slower
    /// but more stable learning; higher values mean faster but potentially unstable learning.
    /// </para>
    /// <para>Default: 0.01</para>
    /// </remarks>
    T LearningRate { get; }

    // ───────────────────── CPU offload (ZeRO-Offload equivalent) ─────────────────────
    // The three flags below mirror DeepSpeed ZeRO-Offload / PyTorch FSDP CPUOffload.
    // They let a train step keep its optimizer state, its gradients, and/or its
    // sharded parameters in CPU RAM instead of GPU VRAM, trading PCIe traffic for
    // the ability to train models whose combined resident footprint would
    // otherwise exceed the GPU. Any combination is legal; the three degrees of
    // freedom compose into DeepSpeed's Stage-1 (optimizer), Stage-2 (+ gradients),
    // and Stage-3 (+ parameters) equivalents when the strategy is FSDP/ZeRO2/ZeRO3.
    // DDP is compatible with CpuOffloadOptimizer only — its non-sharded gradient
    // reduction still needs the grads present locally, and DDP doesn't shard
    // params at all.

    /// <summary>
    /// Gets whether to keep optimizer state (Adam's m/v moments, momentum buffers,
    /// etc.) on the CPU and run the optimizer step on the CPU too.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The "80% memory win" of DeepSpeed's ZeRO-Offload. For an Adam-family
    /// optimizer, m and v each mirror the parameter tensor's shape, so full-
    /// precision optimizer state costs ~2× the parameter memory. Moving that
    /// to CPU frees GPU VRAM for larger batch sizes, longer sequences, or a
    /// larger model — at the cost of a PCIe round-trip per step for gradients
    /// (in) and updated parameters (out).
    /// </para>
    /// <para><b>For Beginners:</b>
    /// Turn this on when you're running out of GPU memory and the optimizer's
    /// bookkeeping (m and v arrays) is a significant chunk of what you're
    /// holding. Training math still happens on the GPU; only the parameter
    /// update step runs on the CPU.
    /// </para>
    /// <para>Default: false</para>
    /// </remarks>
    bool CpuOffloadOptimizer { get; }

    /// <summary>
    /// Gets whether to page gradient tensors to CPU RAM before the sharded
    /// reduction (all-reduce for DDP, reduce-scatter for FSDP/ZeRO). Composes
    /// with <see cref="CpuOffloadOptimizer"/> so the whole optimizer step
    /// (grads + m + v + params) can run on the CPU when both are enabled.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Standalone <c>CpuOffloadGradients</c> is DeepSpeed ZeRO Stage-2's
    /// equivalent: gradients are freed from GPU as soon as the backward pass
    /// produces them, reduced across ranks on CPU, then discarded (each rank
    /// only keeps its own shard).
    /// </para>
    /// <para><b>For Beginners:</b>
    /// After the backward pass computes gradients on the GPU, this option
    /// moves them to CPU RAM immediately. Frees more GPU VRAM but adds a
    /// download per parameter tensor per step.
    /// </para>
    /// <para>Default: false</para>
    /// </remarks>
    bool CpuOffloadGradients { get; }

    /// <summary>
    /// Gets whether to page the (sharded) parameter tensors to CPU RAM between
    /// training steps. Equivalent to PyTorch FSDP's <c>cpu_offload_params=True</c>
    /// / DeepSpeed ZeRO Stage-3's "offload_param" — parameters live on CPU,
    /// are gathered to GPU for the current forward/backward, then freed.
    /// </summary>
    /// <remarks>
    /// <para><b>Semantics depend on how the model is built:</b></para>
    /// <para>
    /// • <b>Black-box wrapped model (ShardedModelBase / ShardedOptimizerBase):</b> this flag performs
    /// GPU-parameter-CACHE EVICTION between steps — after the update, the GPU-side parameter buffer is
    /// materialized to CPU and the GPU cache entry is dropped, so the next step re-uploads from the
    /// CPU-resident values. It does NOT reduce peak resident parameters, because those paths still
    /// gather the full parameter vector for the forward/backward (the wrapped model's compute is opaque
    /// and cannot be streamed layer-by-layer).
    /// </para>
    /// <para>
    /// • <b>True Stage-3 residency</b> (gather only the active layer, release after — PyTorch FSDP
    /// <c>cpu_offload_params</c> / DeepSpeed ZeRO Stage-3 <c>offload_param</c>) is provided by building
    /// the model from <see cref="AiDotNet.DistributedTraining.Layers.Stage3ShardedLinear{T}"/>, which
    /// keeps only each rank's parameter shard resident and AllGathers the full weight just-in-time per
    /// layer via a tape unshard/reshard op. Use those layers when peak parameter residency (not just GPU
    /// cache pressure) must scale with the layer rather than the model. NVMe offload is a further
    /// extension (not implemented).
    /// </para>
    /// <para>Only meaningful for sharded strategies (FSDP, ZeRO2, ZeRO3); DDP replicates full parameters
    /// and has no shard to offload. Default: false.</para>
    /// </remarks>
    bool CpuOffloadParams { get; }
}

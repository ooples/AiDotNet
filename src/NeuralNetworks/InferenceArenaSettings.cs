using System;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Global opt-in switch for the inference forward-caching allocator (#1661 / Tensors #661).
/// When enabled, <see cref="NeuralNetworkBase{T}.Predict"/> runs the forward inside a recycled
/// <c>TensorArena</c> so intermediate-tensor buffers are reused per call (~98% intermediate
/// allocation reduction, bit-identical output — proven in Tensors #661).
/// </summary>
/// <remarks>
/// <para>
/// Default <c>ON</c> — per the facade pattern, this is an industry-standard zero-config default
/// (PyTorch's caching allocator is always on for inference): callers get the allocation win without
/// opting in. Every <c>NeuralNetworkBase{T}</c> model routes through the <c>PredictCore</c> funnel,
/// and the forward is bit-identical with the arena on or off. Set <c>AIDOTNET_INFERENCE_ARENA=0</c>
/// to disable (escape hatch), or set <see cref="Enabled"/> in code (e.g. for A/B alloc tests).
/// </para>
/// <para>
/// Process-wide by design: <c>TensorArena.Current</c> is <c>[ThreadStatic]</c>, so concurrent
/// <c>Predict</c> on a single model instance already requires external serialization — the same
/// contract as the existing eval-mode flip in <see cref="NeuralNetworkBase{T}.Predict"/>.
/// </para>
/// </remarks>
public static class InferenceArenaSettings
{
    /// <summary>
    /// Whether <see cref="NeuralNetworkBase{T}.Predict"/> opens a per-call <c>TensorArena</c>
    /// around the forward. Default <c>true</c>; set <c>AIDOTNET_INFERENCE_ARENA=0</c> to disable.
    /// </summary>
    public static bool Enabled { get; set; } =
        !string.Equals(
            Environment.GetEnvironmentVariable("AIDOTNET_INFERENCE_ARENA"),
            "0",
            StringComparison.Ordinal);

    /// <summary>
    /// Whether the multi-step diffusion denoise loop (<c>DiffusionModelBase.Generate</c>) opens a
    /// per-step <c>TensorArena</c> (recycling each step's noise-predictor + scheduler intermediates
    /// instead of GC-churning them). Default <c>ON</c>; set
    /// <c>AIDOTNET_INFERENCE_ARENA_DIFFUSION=0</c> to disable (escape hatch).
    /// </summary>
    /// <remarks>
    /// This was OFF historically (issue #1668): the per-step <c>arena.Reset()</c> recycled scratch
    /// that diffusion forward layers still referenced via per-forward backward-activation caches
    /// (<c>_lastInput</c>, conv <c>_preAllocatedOutput</c>, ...), so a later step's allocation
    /// aliased a still-referenced buffer and corrupted it. The arena cannot auto-detect this (its
    /// pool strong-refs every buffer it hands out), so the fix is layer-side: the denoise loop runs
    /// inside an <c>InferenceMode</c> scope (the <c>torch.inference_mode()</c> analog) under which
    /// every layer's <c>ShouldCacheForBackward</c> guard is false, so no backward-activation cache is
    /// populated — nothing references scratch across a <c>Reset</c> — and the convolution output
    /// buffer is re-rented per forward instead of being reused across the recycle boundary. Verified
    /// bit-identical (arena on vs off) on DDPM <c>Generate</c>; the diffusion model-family and
    /// <c>DiffusionGenerateArenaIntegrationTests</c> exercise the arena across model types in CI.
    /// </remarks>
    public static bool DiffusionDenoiseEnabled { get; set; } =
        !string.Equals(
            Environment.GetEnvironmentVariable("AIDOTNET_INFERENCE_ARENA_DIFFUSION"),
            "0",
            StringComparison.Ordinal);
}

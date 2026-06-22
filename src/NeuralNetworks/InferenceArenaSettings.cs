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
}

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
/// Default <c>false</c> — this is rolled out incrementally (the <c>PredictCore</c> funnel + a
/// representative cross-section first; flip default-on later once a broad set is validated
/// bit-identical). Seeded from the environment variable <c>AIDOTNET_INFERENCE_ARENA=1</c> (matching
/// the existing <c>AIDOTNET_*</c> toggle convention) and settable in code for tests.
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
    /// around the forward. Default <c>false</c>; set <c>AIDOTNET_INFERENCE_ARENA=1</c> to opt in.
    /// </summary>
    public static bool Enabled { get; set; } =
        string.Equals(
            Environment.GetEnvironmentVariable("AIDOTNET_INFERENCE_ARENA"),
            "1",
            StringComparison.Ordinal);
}

using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.LoRA;

/// <summary>
/// Bakes LoRA adapter layers into plain layers across a whole network, producing a model that serves the
/// fine-tuned behavior with no per-layer LoRA overhead — the standard way to deploy a LoRA-fine-tuned model
/// as its own servable variant.
/// </summary>
/// <remarks>
/// <para>
/// Each <see cref="ILoRAAdapter{T}"/> layer in the model is replaced by the result of
/// <see cref="ILoRAAdapter{T}.MergeToOriginalLayer"/> (<c>W' = W + scaling · B·A</c>), which preserves the
/// adapted behavior exactly while collapsing "base + adapter" into a single layer. Adapters whose type does
/// not support merging are left in place (they still compute correctly, just without the speed/simplicity
/// win). The network's forward iterates its <see cref="NeuralNetworkBase{T}.Layers"/> list directly, so the
/// in-place replacement takes effect immediately.
/// </para>
/// <para><b>For Beginners:</b> after fine-tuning a model with LoRA you have "original weights + small
/// changes". Merging bakes the changes into the weights so you can serve one clean model — like accepting all
/// tracked changes in a document.</para>
/// </remarks>
public static class LoRAAdapterMerger
{
    /// <summary>
    /// Merges every mergeable LoRA adapter layer in <paramref name="model"/> into a plain layer, in place.
    /// </summary>
    /// <param name="model">The network whose LoRA adapter layers should be baked in.</param>
    /// <returns>The number of adapter layers that were merged.</returns>
    public static int MergeInPlace<T>(NeuralNetworkBase<T> model)
    {
        Guard.NotNull(model);

        int merged = 0;
        var layers = model.Layers;
        for (int i = 0; i < layers.Count; i++)
        {
            if (layers[i] is ILoRAAdapter<T> adapter)
            {
                try
                {
                    layers[i] = adapter.MergeToOriginalLayer();
                    merged++;
                }
                catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or NotImplementedException)
                {
                    // This adapter variant does not support weight merging; leave it as an adapter layer —
                    // it still produces correct output, just without the merged-inference speedup.
                    InferenceDiagnostics.RecordException(
                        area: "LoRA.LoRAAdapterMerger",
                        feature: "MergeInPlace",
                        ex: ex,
                        reason: $"Adapter layer at index {i} ({adapter.GetType().Name}) does not support merging; left in place.");
                }
            }
        }
        return merged;
    }

    /// <summary>
    /// Gets a value indicating whether <paramref name="model"/> still contains any LoRA adapter layers.
    /// </summary>
    public static bool HasAdapters<T>(NeuralNetworkBase<T> model)
    {
        Guard.NotNull(model);
        foreach (var layer in model.Layers)
        {
            if (layer is ILoRAAdapter<T>)
            {
                return true;
            }
        }
        return false;
    }
}

using AiDotNet.Helpers;
using AiDotNet.LoRA.Adapters;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.LoRA;

/// <summary>
/// Shared helper for selecting the active multi-LoRA task across a network's <see cref="MultiLoRAAdapter{T}"/>
/// layers. Centralizes the "switch every adapter layer to task X" primitive so the inference-session path
/// (<c>AiModelResult</c>) and the serving batcher use identical logic — one shared base, many small adapters.
/// </summary>
public static class LoRAAdapterSelection
{
    /// <summary>
    /// Switches every <see cref="MultiLoRAAdapter{T}"/> layer in <paramref name="model"/> to the given task.
    /// </summary>
    /// <param name="model">The network whose adapter layers should be switched.</param>
    /// <param name="task">The task/adapter name to activate (must already be registered on the adapters).</param>
    /// <returns>The number of adapter layers that were switched.</returns>
    public static int SelectTask<T>(NeuralNetworkBase<T> model, string task)
    {
        Guard.NotNull(model);
        Guard.NotNullOrWhiteSpace(task);

        int applied = 0;
        foreach (var multi in model.Layers.OfType<MultiLoRAAdapter<T>>())
        {
            multi.SetCurrentTask(task);
            applied++;
        }
        return applied;
    }

    /// <summary>Gets a value indicating whether <paramref name="model"/> has any multi-LoRA adapter layers.</summary>
    public static bool HasMultiLoRAAdapters<T>(NeuralNetworkBase<T> model)
    {
        Guard.NotNull(model);
        foreach (var _ in model.Layers.OfType<MultiLoRAAdapter<T>>())
        {
            return true;
        }
        return false;
    }
}

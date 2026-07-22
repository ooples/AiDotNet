using AiDotNet.Interfaces;
using AiDotNet.ModelLoading.Pretrained;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet;

/// <summary>
/// Pretrained-model loading partial of <see cref="AiModelBuilder{T, TInput, TOutput}"/>: the
/// <c>ConfigureModel(PretrainedSource)</c> facade entry that resolves a Hugging Face / safetensors
/// checkpoint to a ready model and configures it like any hand-built one.
/// </summary>
public partial class AiModelBuilder<T, TInput, TOutput>
{
    /// <inheritdoc/>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureModel(PretrainedSource source)
    {
        Guard.NotNull(source);

        // Resolve the descriptor to a concrete model (download + reconstruct + weight-load). Decoder
        // language models are Tensor<T> -> Tensor<T>; only then does the produced model satisfy the
        // builder's IFullModel<T, TInput, TOutput> — surface a clear error otherwise.
        var model = PretrainedLoader<T>.Load(source);
        if (model is not IFullModel<T, TInput, TOutput> typed)
            throw new InvalidOperationException(
                $"Pretrained source '{source}' produces a Tensor<{typeof(T).Name}> -> Tensor<{typeof(T).Name}> " +
                $"model, but this builder is configured for {typeof(TInput).Name} -> {typeof(TOutput).Name}. " +
                "Configure the builder as AiModelBuilder<T, Tensor<T>, Tensor<T>> to load decoder models.");

        return ConfigureModel(typed);
    }

    /// <inheritdoc/>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureModel(
        PretrainedSource source, AiDotNet.Tensors.DeviceInfo device)
    {
        Guard.NotNull(source);

        var model = PretrainedLoader<T>.Load(source);
        if (model is not IFullModel<T, TInput, TOutput> typed)
            throw new InvalidOperationException(
                $"Pretrained source '{source}' produces a Tensor<{typeof(T).Name}> -> Tensor<{typeof(T).Name}> " +
                $"model, but this builder is configured for {typeof(TInput).Name} -> {typeof(TOutput).Name}. " +
                "Configure the builder as AiModelBuilder<T, Tensor<T>, Tensor<T>> to load decoder models.");

        // Place the whole model (weights + buffers) on the requested device so its forward runs there.
        if (typed is AiDotNet.NeuralNetworks.NeuralNetworkBase<T> network)
            network.To(device);

        return ConfigureModel(typed);
    }
}

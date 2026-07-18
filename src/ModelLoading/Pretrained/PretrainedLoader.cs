using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ModelLoading.Pretrained;

/// <summary>
/// Resolves a <see cref="PretrainedSource"/> to a ready-to-use model: it downloads (for hub sources),
/// reads <c>config.json</c>, looks the architecture up in <see cref="PretrainedArchitectures{T}"/>,
/// reconstructs the network, and loads its pretrained weights. This is the engine behind the
/// <c>AiModelBuilder.ConfigureModel(PretrainedSource)</c> facade overload.
/// </summary>
/// <typeparam name="T">The numeric type used for computation.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Give it a recipe card (<see cref="PretrainedSource"/>) and it hands back a
/// finished model — downloading, unpacking, and wiring the weights for you.
/// </para>
/// </remarks>
public static class PretrainedLoader<T>
{
    /// <summary>
    /// Loads the model described by <paramref name="source"/>. Blocks on any required download.
    /// </summary>
    /// <param name="source">The pretrained-source descriptor.</param>
    /// <returns>A weight-loaded <see cref="NeuralNetworkBase{T}"/> (which is an
    /// <see cref="IFullModel{T, TInput, TOutput}"/> over <see cref="Tensor{T}"/>).</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="source"/> is null.</exception>
    /// <exception cref="NotSupportedException">Thrown for source kinds not yet wired through the facade
    /// (ONNX, GGUF) or for an unrecognized architecture.</exception>
    public static NeuralNetworkBase<T> Load(PretrainedSource source) =>
        LoadAsync(source, CancellationToken.None).GetAwaiter().GetResult();

    /// <summary>Asynchronous variant of <see cref="Load"/>.</summary>
    /// <param name="source">The pretrained-source descriptor.</param>
    /// <param name="cancellationToken">Cancellation for the download phase.</param>
    public static async Task<NeuralNetworkBase<T>> LoadAsync(PretrainedSource source, CancellationToken cancellationToken = default)
    {
        Guard.NotNull(source);

        switch (source.Kind)
        {
            case PretrainedModelKind.Safetensors:
                return LoadFromDirectory(source.Locator, source);

            case PretrainedModelKind.HuggingFace:
            {
                string directory = await DownloadHubModelAsync(source, cancellationToken).ConfigureAwait(false);
                return LoadFromDirectory(directory, source);
            }

            case PretrainedModelKind.Onnx:
                throw new NotSupportedException(
                    "ONNX models run through the ONNX runtime (OnnxModel<T>) rather than the decoder builder; " +
                    "facade wiring for PretrainedSource.Onnx is not yet available. Construct OnnxModel<T> directly " +
                    "and pass it to ConfigureModel(model) in the meantime.");

            case PretrainedModelKind.Gguf:
                throw new NotSupportedException(
                    "GGUF import (metadata->config + GGUF tensor-name mapping) is not yet wired through the facade; " +
                    "use PretrainedSource.HuggingFace or PretrainedSource.Safetensors.");

            default:
                throw new NotSupportedException($"Unknown pretrained source kind '{source.Kind}'.");
        }
    }

    private static NeuralNetworkBase<T> LoadFromDirectory(string directory, PretrainedSource source)
    {
        if (string.IsNullOrWhiteSpace(directory) || !Directory.Exists(directory))
            throw new DirectoryNotFoundException($"Model directory not found: {directory}");

        string configPath = Path.Combine(directory, "config.json");
        var config = HuggingFaceConfig.FromFile(configPath);

        if (!PretrainedArchitectures<T>.TryResolve(config, source.ArchitectureOverride, out var factory))
        {
            string declared = config.Architectures.Count > 0
                ? string.Join(", ", config.Architectures)
                : $"model_type={config.ModelType}";
            throw new NotSupportedException(
                $"Architecture [{declared}] is not supported. Registered architectures: " +
                $"{string.Join(", ", PretrainedArchitectures<T>.RegisteredNames)}.");
        }

        // The tensor source is only needed during construction (Build copies every weight into the
        // model's own tensors), so it is safe to dispose immediately afterwards.
        using var weights = ShardedSafetensorsSource.Open(directory);
        return factory(config, weights);
    }

    private static async Task<string> DownloadHubModelAsync(PretrainedSource source, CancellationToken cancellationToken)
    {
        var loader = new HuggingFaceModelLoader<T>(cacheDir: source.CacheDirectory);
        // Default patterns already fetch *.safetensors + *.json (config.json + optional index).
        await loader.DownloadModelAsync(source.Locator, source.Revision, cancellationToken: cancellationToken)
            .ConfigureAwait(false);
        return loader.GetCachePath(source.Locator, source.Revision);
    }
}

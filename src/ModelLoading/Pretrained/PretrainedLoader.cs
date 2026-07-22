using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tokenization.Interfaces;

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
    /// <returns>A ready model as an <see cref="IFullModel{T, TInput, TOutput}"/> over <see cref="Tensor{T}"/>:
    /// a weight-loaded decoder for safetensors/hub sources, or an ONNX-graph adapter for ONNX sources.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="source"/> is null.</exception>
    /// <exception cref="NotSupportedException">Thrown for source kinds not yet wired through the facade
    /// (GGUF) or for an unrecognized architecture.</exception>
    public static IFullModel<T, Tensor<T>, Tensor<T>> Load(PretrainedSource source) =>
        LoadAsync(source, CancellationToken.None).GetAwaiter().GetResult();

    /// <summary>Asynchronous variant of <see cref="Load"/>.</summary>
    /// <param name="source">The pretrained-source descriptor.</param>
    /// <param name="cancellationToken">Cancellation for the download phase.</param>
    public static async Task<IFullModel<T, Tensor<T>, Tensor<T>>> LoadAsync(PretrainedSource source, CancellationToken cancellationToken = default)
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
                // Run the ONNX graph through the runtime, adapted to the IFullModel contract.
                return OnnxFullModelAdapter<T>.FromFile(source.Locator);

            case PretrainedModelKind.Gguf:
                return LoadFromGguf(source.Locator, source);

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

    private static NeuralNetworkBase<T> LoadFromGguf(string path, PretrainedSource source)
    {
        // GGUF is self-describing: its metadata yields the config and its tensors are exposed under Hugging
        // Face names, so the same architecture registry + builder reconstruct and weight-load the decoder.
        using var gguf = GgufModelSource.Open(path);
        return BuildGgufModel(gguf, source);
    }

    /// <summary>
    /// Loads a GGUF checkpoint as both a decoder model and its matching tokenizer from a single file open,
    /// so a server has everything it needs to run the model at parity with llama.cpp. The weights are read
    /// once; the tokenizer comes from the same file's <c>tokenizer.ggml.*</c> metadata.
    /// </summary>
    /// <param name="path">Path to the <c>.gguf</c> file.</param>
    /// <param name="source">Optional descriptor (for an architecture override); defaults to a plain GGUF source.</param>
    /// <returns>The weight-loaded decoder and the tokenizer it was trained with.</returns>
    public static (NeuralNetworkBase<T> Model, ITokenizer Tokenizer) LoadGgufWithTokenizer(
        string path, PretrainedSource? source = null)
    {
        source ??= PretrainedSource.Gguf(path);
        using var gguf = GgufModelSource.Open(path);
        var model = BuildGgufModel(gguf, source);
        var tokenizer = gguf.BuildTokenizer();
        return (model, tokenizer);
    }

    private static NeuralNetworkBase<T> BuildGgufModel(GgufModelSource gguf, PretrainedSource source)
    {
        if (!PretrainedArchitectures<T>.TryResolve(gguf.Config, source.ArchitectureOverride, out var factory))
            throw new NotSupportedException(
                $"GGUF architecture '{gguf.Config.ModelType}' is not supported. Registered architectures: " +
                $"{string.Join(", ", PretrainedArchitectures<T>.RegisteredNames)}.");
        return factory(gguf.Config, gguf);
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

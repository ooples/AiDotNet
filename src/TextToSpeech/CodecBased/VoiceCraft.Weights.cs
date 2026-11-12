using AiDotNet.LinearAlgebra;

namespace AiDotNet.TextToSpeech.CodecBased;

public partial class VoiceCraft<T>
{
    /// <summary>Loads the authors' named Transformer and GigaSpeech codec state dictionaries.</summary>
    public void LoadAuthorWeights(string transformerPath, string codecPath, string codecPrefix = "")
    {
        RequireConvertedCheckpointTarget();
        using (var file = AiDotNet.ModelLoading.TorchCheckpointFile.Open(transformerPath, strictAxes: true))
            _network!.LoadTorchWeights(AuthorTransformer(file));
        _codecNetwork!.LoadAuthorWeights(codecPath, codecPrefix);
    }

    /// <summary>The released Transformer's state, under its <c>model.</c> wrapper when it has one.</summary>
    internal static AiDotNet.ModelLoading.TorchTensorSource AuthorTransformer(AiDotNet.ModelLoading.TorchTensorSource state) =>
        state.WithPrefix(state.Contains("model.text_embedding.word_embeddings.weight") ? "model." : string.Empty);

    /// <summary><see cref="AuthorTransformer"/> over an already loaded state dict, read through the allocating form with
    /// each tensor's axes checked.</summary>
    internal static Func<string, int[], double[]> AuthorTransformerReader(IReadOnlyDictionary<string, Tensor<float>> state)
    {
        var source = AuthorTransformer(AiDotNet.ModelLoading.TorchTensorSource.From(state, "the VoiceCraft checkpoint"));
        return (name, shape) =>
        {
            var axes = source.ShapeOf(name) ?? throw new InvalidDataException($"VoiceCraft checkpoint is missing '{name}'.");
            if (!axes.SequenceEqual(shape.Select(d => (long)d)))
                throw new InvalidDataException($"VoiceCraft tensor '{name}' has an incompatible shape.");
            return source.Read(name, shape);
        };
    }

    private const string ConvertedParameterTensor = "voicecraft.parameters";

    /// <summary>Loads a strictly converted official VoiceCraft language-model and GigaSpeech-EnCodec checkpoint.</summary>
    /// <remarks>
    /// The upstream release stores the Transformer and its custom 16 kHz EnCodec separately. The converter maps both
    /// state dictionaries into this native graph's serialization order and writes their concatenated values under
    /// <c>voicecraft.parameters</c>. Exact length matching prevents loading an 830M checkpoint into a 330M graph, mixing
    /// the paper's 12-head description with the released 16-head layout, or silently omitting the codec.
    /// </remarks>
    public void LoadConvertedWeights(string path)
    {
        RequireConvertedCheckpointTarget();
        if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("Checkpoint path required.", nameof(path));
        if (!File.Exists(path)) throw new FileNotFoundException($"Checkpoint not found: {path}", path);
        using var file = AiDotNet.ModelLoading.TorchCheckpointFile.Open(path);
        LoadConvertedWeights(file);
    }

    internal void LoadConvertedWeights(Func<string, double[]?> read)
    {
        Validation.Guard.NotNull(read);
        LoadConvertedWeights(AiDotNet.ModelLoading.TorchTensorSource.From(read));
    }

    /// <summary>Copies <c>{ConvertedParameterTensor}</c> from <paramref name="source"/> straight into a parameter vector.</summary>
    internal void LoadConvertedWeights(AiDotNet.ModelLoading.TorchTensorSource source)
    {
        RequireConvertedCheckpointTarget();
        if (!source.Contains(ConvertedParameterTensor))
            throw new InvalidDataException($"The checkpoint has no tensor '{ConvertedParameterTensor}'.");
        int expected = GetParameters().Length;
        if (source.ShapeOf(ConvertedParameterTensor) is { } shape && shape.Aggregate(1L, (a, b) => a * b) is var count && count != expected)
            throw new InvalidDataException(
                $"'{ConvertedParameterTensor}' has {count} values; this VoiceCraft configuration expects {expected}.");
        var parameters = new Vector<T>(expected);
        source.CopyTo(ConvertedParameterTensor, new[] { expected }, new Tensor<T>(new[] { expected }, parameters));
        SetParameters(parameters);
    }

    private void RequireConvertedCheckpointTarget()
    {
        if (!_useNativeMode || !HasPaperPipeline || Architecture.Layers is { Count: > 0 })
            throw new NotSupportedException(
                "Converted VoiceCraft weights require the native GigaSpeech-EnCodec/causal-Transformer graph.");
    }
}

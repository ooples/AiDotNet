using AiDotNet.LinearAlgebra;

namespace AiDotNet.TextToSpeech.CodecBased;

public partial class VoiceCraft<T>
{
    /// <summary>Loads the authors' named Transformer and GigaSpeech codec state dictionaries.</summary>
    public void LoadAuthorWeights(string transformerPath, string codecPath, string codecPrefix = "")
    {
        RequireConvertedCheckpointTarget();
        var state = new ComputerVision.Weights.WeightLoader().LoadWeights(transformerPath);
        _network!.LoadTorchWeights(AuthorTransformerReader(state));
        _codecNetwork!.LoadAuthorWeights(codecPath, codecPrefix);
    }

    internal static Func<string, int[], double[]> AuthorTransformerReader(IReadOnlyDictionary<string, Tensor<float>> state)
    {
        string prefix = state.ContainsKey("model.text_embedding.word_embeddings.weight") ? "model." : string.Empty;
        return (name, shape) =>
        {
            if (!state.TryGetValue(prefix + name, out var tensor)) throw new InvalidDataException($"VoiceCraft checkpoint is missing '{prefix + name}'.");
            if (!tensor.Shape.ToArray().SequenceEqual(shape))
                throw new InvalidDataException($"VoiceCraft tensor '{name}' has an incompatible shape.");
            return tensor.ToArray().Select(v => (double)v).ToArray();
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
        var state = new ComputerVision.Weights.WeightLoader().LoadWeights(path);
        LoadConvertedWeights(name => state.TryGetValue(name, out var tensor)
            ? tensor.ToVector().Select(value => (double)value).ToArray()
            : null);
    }

    internal void LoadConvertedWeights(Func<string, double[]?> read)
    {
        Validation.Guard.NotNull(read);
        RequireConvertedCheckpointTarget();
        var values = read(ConvertedParameterTensor)
            ?? throw new InvalidDataException($"The checkpoint has no tensor '{ConvertedParameterTensor}'.");
        int expected = GetParameters().Length;
        if (values.Length != expected)
            throw new InvalidDataException(
                $"'{ConvertedParameterTensor}' has {values.Length} values; this VoiceCraft configuration expects {expected}.");
        SetParameters(new Vector<T>(Array.ConvertAll(values, NumOps.FromDouble)));
    }

    private void RequireConvertedCheckpointTarget()
    {
        if (!_useNativeMode || !HasPaperPipeline || Architecture.Layers is { Count: > 0 })
            throw new NotSupportedException(
                "Converted VoiceCraft weights require the native GigaSpeech-EnCodec/causal-Transformer graph.");
    }
}

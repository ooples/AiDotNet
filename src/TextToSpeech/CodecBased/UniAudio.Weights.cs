using AiDotNet.LinearAlgebra;

namespace AiDotNet.TextToSpeech.CodecBased;

public partial class UniAudio<T>
{
    /// <summary>Loads original named UniAudio multi-scale LM and universal SoundStream codec tensors.</summary>
    public void LoadAuthorWeights(string languageModelPath, string codecPath, string codecPrefix = "")
    {
        RequireConvertedCheckpointTarget();
        Func<string, int[], double[]> Read(string path, string marker)
        {
            var state = new ComputerVision.Weights.WeightLoader().LoadWeights(path, name =>
                name.StartsWith("model.", StringComparison.Ordinal) || name.StartsWith("codec_model.", StringComparison.Ordinal) ||
                name.StartsWith("g_", StringComparison.Ordinal) || name.StartsWith("l_", StringComparison.Ordinal) ||
                name.StartsWith("g2l_", StringComparison.Ordinal) || name.StartsWith("lm_head.", StringComparison.Ordinal) ||
                name.StartsWith("encoder.", StringComparison.Ordinal) || name.StartsWith("decoder.", StringComparison.Ordinal) ||
                name.StartsWith("quantizer.", StringComparison.Ordinal) ||
                (!string.IsNullOrEmpty(codecPrefix) && name.StartsWith(codecPrefix, StringComparison.Ordinal)));
            return AuthorStateReader(state, marker);
        }
        _languageModel!.LoadTorchWeights(Read(languageModelPath, "g_emb.weight"));
        _codec!.LoadTorchWeights(Read(codecPath, codecPrefix + "encoder.model.0.conv.conv.weight_v"), codecPrefix);
    }

    internal static Func<string, int[], double[]> AuthorStateReader(IReadOnlyDictionary<string, Tensor<float>> state, string marker)
    {
        string prefix = state.ContainsKey(marker) ? string.Empty :
            state.ContainsKey("model." + marker) ? "model." :
            state.ContainsKey("codec_model." + marker) ? "codec_model." : string.Empty;
        return (name, shape) =>
        {
            if (!state.TryGetValue(prefix + name, out var tensor))
                throw new InvalidDataException($"UniAudio checkpoint is missing '{prefix + name}'.");
            if (!tensor.Shape.ToArray().SequenceEqual(shape))
                throw new InvalidDataException($"UniAudio tensor '{name}' has an incompatible shape.");
            return tensor.ToArray().Select(value => (double)value).ToArray();
        };
    }

    private const string ConvertedParameterTensor = "uniaudio.parameters";

    /// <summary>Loads an AiDotNet-converted official UniAudio checkpoint.</summary>
    /// <remarks>
    /// The upstream release stores a PyTorch multi-scale LM and a separate universal-codec state dictionary. The
    /// converter maps both into this native graph's serialization order and writes their concatenated values under
    /// <c>uniaudio.parameters</c>. Loading is strict so an incompatible vocabulary, compact test model or codec preset
    /// cannot be silently partially restored.
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
                $"'{ConvertedParameterTensor}' has {values.Length} values; this UniAudio configuration expects {expected}.");
        SetParameters(new Vector<T>(Array.ConvertAll(values, NumOps.FromDouble)));
    }

    private void RequireConvertedCheckpointTarget()
    {
        if (!_useNativeMode || !HasPaperPipeline || Architecture.Layers is { Count: > 0 })
            throw new NotSupportedException(
                "Converted UniAudio weights require the native universal-codec/global/local graph.");
    }
}

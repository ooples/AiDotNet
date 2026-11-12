using AiDotNet.LinearAlgebra;

namespace AiDotNet.TextToSpeech.CodecBased;

public partial class UniAudio<T>
{
    /// <summary>Loads original named UniAudio multi-scale LM and universal SoundStream codec tensors.</summary>
    public void LoadAuthorWeights(string languageModelPath, string codecPath, string codecPrefix = "")
    {
        RequireConvertedCheckpointTarget();
        // A training checkpoint's optimizer and discriminator state is never unpickled.
        bool Include(string name) =>
            name.StartsWith("model.", StringComparison.Ordinal) || name.StartsWith("codec_model.", StringComparison.Ordinal) ||
            name.StartsWith("g_", StringComparison.Ordinal) || name.StartsWith("l_", StringComparison.Ordinal) ||
            name.StartsWith("g2l_", StringComparison.Ordinal) || name.StartsWith("lm_head.", StringComparison.Ordinal) ||
            name.StartsWith("encoder.", StringComparison.Ordinal) || name.StartsWith("decoder.", StringComparison.Ordinal) ||
            name.StartsWith("quantizer.", StringComparison.Ordinal) ||
            (!string.IsNullOrEmpty(codecPrefix) && name.StartsWith(codecPrefix, StringComparison.Ordinal));
        using (var lm = AiDotNet.ModelLoading.TorchCheckpointFile.Open(languageModelPath, strictAxes: true, Include))
            _languageModel!.LoadTorchWeights(AuthorState(lm, "g_emb.weight"));
        using (var codec = AiDotNet.ModelLoading.TorchCheckpointFile.Open(codecPath, strictAxes: true, Include))
            _codec!.LoadTorchWeights(AuthorState(codec, codecPrefix + "encoder.model.0.conv.conv.weight_v"), codecPrefix);
    }

    /// <summary>The released state under whichever wrapper (none, <c>model.</c>, <c>codec_model.</c>) holds
    /// <paramref name="marker"/>.</summary>
    internal static AiDotNet.ModelLoading.TorchTensorSource AuthorState(AiDotNet.ModelLoading.TorchTensorSource state, string marker)
    {
        string prefix = state.Contains(marker) ? string.Empty :
            state.Contains("model." + marker) ? "model." :
            state.Contains("codec_model." + marker) ? "codec_model." : string.Empty;
        return state.WithPrefix(prefix);
    }

    /// <summary><see cref="AuthorState"/> over an already loaded state dict, read through the allocating form with each
    /// tensor's axes checked.</summary>
    internal static Func<string, int[], double[]> AuthorStateReader(IReadOnlyDictionary<string, Tensor<float>> state, string marker)
    {
        var source = AuthorState(AiDotNet.ModelLoading.TorchTensorSource.From(state, "the UniAudio checkpoint"), marker);
        return (name, shape) =>
        {
            var axes = source.ShapeOf(name) ?? throw new InvalidDataException($"UniAudio checkpoint is missing '{name}'.");
            if (!axes.SequenceEqual(shape.Select(d => (long)d)))
                throw new InvalidDataException($"UniAudio tensor '{name}' has an incompatible shape.");
            return source.Read(name, shape);
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
                $"'{ConvertedParameterTensor}' has {count} values; this UniAudio configuration expects {expected}.");
        var parameters = new Vector<T>(expected);
        source.CopyTo(ConvertedParameterTensor, new[] { expected }, new Tensor<T>(new[] { expected }, parameters));
        SetParameters(parameters);
    }

    private void RequireConvertedCheckpointTarget()
    {
        if (!_useNativeMode || !HasPaperPipeline || Architecture.Layers is { Count: > 0 })
            throw new NotSupportedException(
                "Converted UniAudio weights require the native universal-codec/global/local graph.");
    }
}

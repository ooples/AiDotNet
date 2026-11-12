using AiDotNet.LinearAlgebra;
using AiDotNet.ModelLoading;

namespace AiDotNet.TextToSpeech.CodecBased;

public partial class SparkTTS<T>
{
    /// <summary>Loads all three original release components. XLSR uses its original Hugging Face parameter names.</summary>
    public void LoadAuthorWeights(string languageModelPath, string biCodecPath, string wav2VecPath)
    {
        RequireConvertedCheckpointTarget();
        LoadLanguageModelWeights(languageModelPath);
        LoadAuthorState(biCodecPath, null, source => _biCodec!.LoadTorchWeights(source));
        LoadAuthorState(wav2VecPath, "wav2vec2.", source => _biCodec!.Wav2Vec.LoadTorchWeights(source));
    }

    /// <summary>Loads the speaker_encoder subtree of the released BiCodec checkpoint, including ECAPA running statistics.</summary>
    public void LoadSpeakerTokenizerWeights(string path, string prefix = "speaker_encoder.")
    {
        RequireConvertedCheckpointTarget();
        LoadAuthorState(path, null, source => _biCodec!.Speaker.LoadTorchWeights(source, prefix));
    }

    /// <summary>Loads the original Hugging Face Qwen2.5 state dictionary, including its tied output embedding.</summary>
    public void LoadLanguageModelWeights(string path)
    {
        RequireConvertedCheckpointTarget();
        LoadAuthorState(path, null, source => _languageModel!.Transformer.LoadTorchWeights(source));
    }

    /// <summary>Opens a released component's checkpoint, each tensor's axes checked, and loads it with
    /// <paramref name="load"/>.</summary>
    private static void LoadAuthorState(string path, string? backbonePrefix, Action<TorchTensorSource> load)
    {
        using var file = TorchCheckpointFile.Open(path, strictAxes: true);
        load(AuthorState(file, backbonePrefix));
    }

    // HF distributes XLSR as Wav2Vec2ForPreTraining, whose backbone has a wav2vec2. prefix.
    // Also accept a standalone Wav2Vec2Model state; choose one namespace for the whole import.
    internal static TorchTensorSource AuthorState(TorchTensorSource state, string? backbonePrefix = null)
    {
        string prefix = backbonePrefix is not null && state.Contains(backbonePrefix + "feature_extractor.conv_layers.0.conv.weight")
            ? backbonePrefix : string.Empty;
        const string positional = "encoder.pos_conv_embed.conv.";
        bool legacyWeightNorm = backbonePrefix is not null
            && !state.Contains(prefix + positional + "parametrizations.weight.original1")
            && state.Contains(prefix + positional + "weight_v");
        // Original XLSR releases predate torch's parametrizations API. Both layouts store the same g/v tensors.
        return state.WithNames(name =>
            legacyWeightNorm && name == positional + "parametrizations.weight.original1" ? prefix + positional + "weight_v"
            : legacyWeightNorm && name == positional + "parametrizations.weight.original0" ? prefix + positional + "weight_g"
            : prefix + name);
    }

    /// <summary><see cref="AuthorState"/> over an already loaded state dict, read through the allocating form with each
    /// tensor's axes checked.</summary>
    internal static Func<string, int[], double[]> AuthorStateReader(
        IReadOnlyDictionary<string, Tensor<float>> state, string? backbonePrefix = null)
    {
        var source = AuthorState(TorchTensorSource.From(state, "the Spark checkpoint"), backbonePrefix);
        return (name, shape) =>
        {
            var axes = source.ShapeOf(name) ?? throw new InvalidDataException($"Spark checkpoint is missing '{name}'.");
            if (!axes.SequenceEqual(shape.Select(d => (long)d))) throw new InvalidDataException($"Spark tensor '{name}' has an incompatible shape.");
            return source.Read(name, shape);
        };
    }

    private const string ConvertedParameterTensor = "sparktts.parameters";

    /// <summary>Loads a legacy AiDotNet parameter-vector snapshot into an identical native configuration.</summary>
    /// <remarks>
    /// This restores parameters in GetParameters order. It does not import original component names or non-parameter
    /// buffers such as ECAPA running statistics. Use the component weight loaders for original released weights.
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
                $"'{ConvertedParameterTensor}' has {count} values; this Spark-TTS configuration expects {expected}.");
        var parameters = new Vector<T>(expected);
        source.CopyTo(ConvertedParameterTensor, new[] { expected }, new Tensor<T>(new[] { expected }, parameters));
        SetParameters(parameters);
    }

    private void RequireConvertedCheckpointTarget()
    {
        if (!_useNativeMode || !HasPaperPipeline || Architecture.Layers is { Count: > 0 })
            throw new NotSupportedException(
                "Converted Spark-TTS weights require the native Qwen/BiCodec graph, not caller-supplied layers or ONNX mode.");
    }
}

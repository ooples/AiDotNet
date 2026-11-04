using AiDotNet.LinearAlgebra;

namespace AiDotNet.TextToSpeech.CodecBased;

public partial class SparkTTS<T>
{
    /// <summary>Loads all three original release components. XLSR uses its original Hugging Face parameter names.</summary>
    public void LoadAuthorWeights(string languageModelPath, string biCodecPath, string wav2VecPath)
    {
        RequireConvertedCheckpointTarget();
        LoadLanguageModelWeights(languageModelPath);
        _biCodec!.LoadTorchWeights(ReadAuthorState(biCodecPath));
        _biCodec.Wav2Vec.LoadTorchWeights(ReadAuthorState(wav2VecPath, "wav2vec2."));
    }

    /// <summary>Loads the speaker_encoder subtree of the released BiCodec checkpoint, including ECAPA running statistics.</summary>
    public void LoadSpeakerTokenizerWeights(string path, string prefix = "speaker_encoder.")
    {
        RequireConvertedCheckpointTarget();
        _biCodec!.Speaker.LoadTorchWeights(ReadAuthorState(path), prefix);
    }

    /// <summary>Loads the original Hugging Face Qwen2.5 state dictionary, including its tied output embedding.</summary>
    public void LoadLanguageModelWeights(string path)
    {
        RequireConvertedCheckpointTarget();
        _languageModel!.Transformer.LoadTorchWeights(ReadAuthorState(path));
    }

    private static Func<string, int[], double[]> ReadAuthorState(string path, string? backbonePrefix = null)
    {
        var state = new ComputerVision.Weights.WeightLoader().LoadWeights(path);
        return AuthorStateReader(state, backbonePrefix);
    }

    // HF distributes XLSR as Wav2Vec2ForPreTraining, whose backbone has a wav2vec2. prefix.
    // Also accept a standalone Wav2Vec2Model state; choose one namespace for the whole import.
    internal static Func<string, int[], double[]> AuthorStateReader(
        IReadOnlyDictionary<string, Tensor<float>> state, string? backbonePrefix = null)
    {
        string prefix = backbonePrefix is not null && state.ContainsKey(backbonePrefix + "feature_extractor.conv_layers.0.conv.weight")
            ? backbonePrefix : string.Empty;
        const string positional = "encoder.pos_conv_embed.conv.";
        bool legacyWeightNorm = backbonePrefix is not null
            && !state.ContainsKey(prefix + positional + "parametrizations.weight.original1")
            && state.ContainsKey(prefix + positional + "weight_v");
        return (name, shape) =>
        {
            // Original XLSR releases predate torch's parametrizations API. Both layouts store the same g/v tensors.
            if (legacyWeightNorm && name == positional + "parametrizations.weight.original1") name = positional + "weight_v";
            else if (legacyWeightNorm && name == positional + "parametrizations.weight.original0") name = positional + "weight_g";
            if (!state.TryGetValue(prefix + name, out var tensor)) throw new InvalidDataException($"Spark checkpoint is missing '{prefix + name}'.");
            if (!tensor.Shape.ToArray().SequenceEqual(shape)) throw new InvalidDataException($"Spark tensor '{name}' has an incompatible shape.");
            return tensor.ToArray().Select(value => (double)value).ToArray();
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
                $"'{ConvertedParameterTensor}' has {values.Length} values; this Spark-TTS configuration expects {expected}.");
        SetParameters(new Vector<T>(Array.ConvertAll(values, NumOps.FromDouble)));
    }

    private void RequireConvertedCheckpointTarget()
    {
        if (!_useNativeMode || !HasPaperPipeline || Architecture.Layers is { Count: > 0 })
            throw new NotSupportedException(
                "Converted Spark-TTS weights require the native Qwen/BiCodec graph, not caller-supplied layers or ONNX mode.");
    }
}

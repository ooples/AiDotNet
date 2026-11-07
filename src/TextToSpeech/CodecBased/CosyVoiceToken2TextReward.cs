using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Training;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>A trainable token-to-text sequence model that supplies a differentiable ASR reward for DiffRO.</summary>
/// <remarks>This concrete native reward uses a T5 encoder/decoder and a separate speech embedding table. CosyVoice's
/// paper does not release its reward architecture or weights, so these configurable choices are not author defaults.
/// Train or load the reward before policy optimization. Text targets are explicit, including EOS when desired.</remarks>
public sealed class CosyVoiceToken2TextReward<T> : ICosyVoiceTokenReward<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly List<LayerBase<T>> _layers = new();
    private readonly TiedEmbeddingLayer<T> _speechEmbedding;
    private readonly T5Seq2Seq<T> _network;

    /// <summary>Creates a native reward model with caller-configured dimensions and vocabulary IDs.</summary>
    public CosyVoiceToken2TextReward(int speechVocabulary, int textVocabulary, int modelDim = 256,
        int feedForwardDim = 1024, int heads = 4, int encoderLayers = 4, int decoderLayers = 4,
        int decoderStartTokenId = 0, int initializationSeed = 2294)
    {
        if (speechVocabulary < 2 || textVocabulary < 2 || modelDim <= 0 || feedForwardDim <= 0
            || heads <= 0 || modelDim % heads != 0 || encoderLayers <= 0 || decoderLayers <= 0
            || decoderStartTokenId < 0 || decoderStartTokenId >= textVocabulary)
            throw new ArgumentException("Invalid token-to-text vocabulary, Transformer dimensions or decoder start ID.");
        _engine = AiDotNetEngine.Current;
        SpeechVocabularySize = speechVocabulary;
        TextVocabularySize = textVocabulary;
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(initializationSeed);
        _speechEmbedding = new TiedEmbeddingLayer<T>(speechVocabulary, modelDim);
        _layers.Add(_speechEmbedding);
        _speechEmbedding.Reinitialize(() => Math.Sqrt(-2 * Math.Log(1 - random.NextDouble())) * Math.Cos(2 * Math.PI * random.NextDouble()));
        _network = new T5Seq2Seq<T>(_engine, _layers, new T5Configuration(textVocabulary, modelDim,
            feedForwardDim, modelDim / heads, heads, encoderLayers, decoderLayers, Dropout: 0,
            DecoderStartTokenId: decoderStartTokenId));
        _network.InitializeLikeHuggingFace(random);
        foreach (var layer in _layers) layer.SetTrainingMode(false);
    }

    /// <inheritdoc />
    public int SpeechVocabularySize { get; }

    /// <summary>The number of text-token categories in the reward's decoder.</summary>
    public int TextVocabularySize { get; }

    /// <inheritdoc />
    public IReadOnlyList<LayerBase<T>> Layers => _layers;

    /// <summary>The reward's trainable parameter tensors; these are not part of the policy's parameter ownership.</summary>
    public IReadOnlyList<Tensor<T>> Parameters => TapeTrainingStep<T>.CollectParameters(_layers.Cast<ILayer<T>>().ToList(), -1);

    /// <inheritdoc />
    public Tensor<T> LogReward(Tensor<T> softSpeechTokens, IReadOnlyList<int> textTokens)
    {
        if (softSpeechTokens is null) throw new ArgumentNullException(nameof(softSpeechTokens));
        if (textTokens is null) throw new ArgumentNullException(nameof(textTokens));
        if (softSpeechTokens.Rank != 2 || softSpeechTokens.Shape[0] == 0 || softSpeechTokens.Shape[1] != SpeechVocabularySize)
            throw new ArgumentException("The reward needs nonempty [frames, speech categories] soft tokens.", nameof(softSpeechTokens));
        if (textTokens.Count == 0 || textTokens.Any(id => id < 0 || id >= TextVocabularySize))
            throw new ArgumentException("The reward transcript must contain valid text IDs.", nameof(textTokens));
        var c = _network.Configuration;
        var memory = _engine.TensorMatMul(softSpeechTokens, _speechEmbedding.Table);
        int frames = memory.Shape[0];
        var bias = T5Seq2Seq<T>.RelativeBias(_engine, c, _network.EncoderBias, frames, frames,
            bidirectional: true, causal: false);
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(0);
        foreach (var block in _network.Encoder) memory = block.Forward(memory, bias, null, null, false, random);
        memory = _network.EncoderFinalNorm.Forward(memory);
        var shifted = new Tensor<T>(new[] { textTokens.Count });
        shifted[0] = Ops.FromDouble(c.DecoderStartTokenId);
        for (int i = 1; i < textTokens.Count; i++) shifted[i] = Ops.FromDouble(textTokens[i - 1]);
        var logits = _network.Decode(shifted, memory, false, random);
        var selected = new Tensor<T>(new[] { textTokens.Count, TextVocabularySize });
        for (int i = 0; i < textTokens.Count; i++) selected[i, textTokens[i]] = Ops.One;
        return _engine.ReduceSum(_engine.TensorMultiply(_engine.TensorLogSoftmax(logits, axis: 1), selected),
            new[] { 0, 1 }, false);
    }

    /// <summary>Trains the ASR-like reward on hard speech IDs and paired text using one actual SGD step.</summary>
    public T Train(IReadOnlyList<int> speechTokens, IReadOnlyList<int> textTokens, T learningRate)
    {
        if (speechTokens is null) throw new ArgumentNullException(nameof(speechTokens));
        double rate = Ops.ToDouble(learningRate);
        if (double.IsNaN(rate) || double.IsInfinity(rate) || rate <= 0) throw new ArgumentOutOfRangeException(nameof(learningRate));
        if (speechTokens.Count == 0 || speechTokens.Any(id => id < 0 || id >= SpeechVocabularySize))
            throw new ArgumentException("Reward training needs valid speech-code IDs.", nameof(speechTokens));
        var hard = new Tensor<T>(new[] { speechTokens.Count, SpeechVocabularySize });
        for (int i = 0; i < speechTokens.Count; i++) hard[i, speechTokens[i]] = Ops.One;
        var dummy = new Tensor<T>(new[] { 1 });
        return TapeTrainingStep<T>.Step(TapeTrainingStep<T>.CollectTrainableLayers(_layers.Cast<ILayer<T>>().ToList(), -1),
            hard, dummy, learningRate, input => LogReward(input, textTokens),
            (reward, _) => _engine.TensorMultiplyScalar(reward, Ops.FromDouble(-1)));
    }

    /// <summary>Loads native converted reward weights: speech_embedding.weight plus Hugging Face T5 names.</summary>
    /// <remarks>No original CosyVoice reward checkpoint is implied by this conversion contract.</remarks>
    public void LoadWeights(Func<string, int[], double[]> read)
    {
        if (read is null) throw new ArgumentNullException(nameof(read));
        _speechEmbedding.LoadTable(read("speech_embedding.weight", new[] { SpeechVocabularySize, _network.Configuration.ModelDim }));
        _network.LoadHuggingFaceWeights(read);
    }
}

using AiDotNet.ActivationFunctions;
using AiDotNet.Audio.Foundations;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>
/// SpiritLM Expressive's released style tokenizer (<c>Wav2Vec2StyleEncoder</c>, <c>style_encoder_w2v2</c>): wav2vec 2.0
/// Base over the waveform normalized to zero mean and unit variance, its last hidden states projected to 256, averaged over
/// windows of 50 frames (one second), and classified into 100 styles; one style token per second.
/// </summary>
/// <remarks>The paper describes speechprop features pooled per second and clustered by a 100-centroid k-means after a
/// style fine-tuning; the release ships this classifier in its place, and its classes are the released language model's
/// style tokens. The encoder runs as <see cref="HubertNetwork{T}"/> with wav2vec 2.0 Base's sizes, which share HuBERT's
/// architecture.</remarks>
internal sealed class SpiritLmStyleEncoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public SpiritLmStyleEncoder(IEngine engine, List<LayerBase<T>> layers, HubertConfiguration configuration, int projection = 256, int styles = 100,
        int poolSize = 50)
    {
        _engine = engine;
        Network = new HubertNetwork<T>(engine, layers, configuration);
        ProjectionWidth = projection;
        Styles = styles;
        PoolSize = poolSize;
        DenseLayer<T> Linear(int outputs)
        {
            var layer = new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>());
            layers.Add(layer);
            return layer;
        }
        Projector = Linear(projection);
        Classifier = Linear(styles);
    }

    /// <summary>wav2vec 2.0 Base (<c>facebook/wav2vec2-base</c>): group norm on the first convolution, 768 wide, 12 post-norm
    /// layers.</summary>
    public static HubertConfiguration Wav2Vec2Base() => new()
    {
        LayerNormFeatureEncoder = false, HiddenSize = 768, Layers = 12, Heads = 12, IntermediateSize = 3072, LayerNormFirst = false,
    };

    public HubertNetwork<T> Network { get; }
    public DenseLayer<T> Projector { get; }
    public DenseLayer<T> Classifier { get; }
    public int ProjectionWidth { get; }
    public int Styles { get; }
    public int PoolSize { get; }

    /// <summary>The style logits <c>[windows, styles]</c> of 16 kHz audio.</summary>
    public Tensor<T> Logits(IReadOnlyList<double> audio)
    {
        if (audio.Count == 0) throw new ArgumentException("The audio is empty.", nameof(audio));
        // Wav2Vec2FeatureExtractor: (x − mean) / √(var + 1e-7).
        double mean = 0;
        foreach (var v in audio) mean += v;
        mean /= audio.Count;
        double variance = 0;
        foreach (var v in audio) variance += (v - mean) * (v - mean);
        variance /= audio.Count;
        double scale = 1.0 / Math.Sqrt(variance + 1e-7);
        var input = new Tensor<T>(new[] { audio.Count });
        for (int i = 0; i < audio.Count; i++) input[i] = NumOps.FromDouble((audio[i] - mean) * scale);
        var hidden = Projector.Forward(Network.Forward(input, Network.EncoderLayers.Count)[^1]);              // [frames, 256]
        int frames = hidden.Shape[0], windows = (frames + PoolSize - 1) / PoolSize;
        var pooled = new List<Tensor<T>>();
        for (int w = 0; w < windows; w++)
        {
            int start = w * PoolSize, count = Math.Min(PoolSize, frames - start);
            var chunk = _engine.TensorSlice(hidden, new[] { start, 0 }, new[] { count, ProjectionWidth });
            pooled.Add(_engine.TensorMultiplyScalar(_engine.ReduceSum(chunk, new[] { 0 }, keepDims: true), NumOps.FromDouble(1.0 / count)));
        }
        return Classifier.Forward(pooled.Count == 1 ? pooled[0] : _engine.TensorConcatenate(pooled.ToArray(), 0));
    }

    /// <summary>The style token of each second of 16 kHz audio (the first class on a tie).</summary>
    public int[] Tokens(IReadOnlyList<double> audio)
    {
        Tensor<T> logits;
        using (new NoGradScope<T>()) logits = Logits(audio);
        var tokens = new int[logits.Shape[0]];
        for (int w = 0; w < tokens.Length; w++)
        {
            int best = 0;
            for (int c = 1; c < Styles; c++) if (NumOps.ToDouble(logits[w, c]) > NumOps.ToDouble(logits[w, best])) best = c;
            tokens[w] = best;
        }
        return tokens;
    }

    /// <summary>Loads transformers' <c>Wav2Vec2ForSequenceClassification</c> state (<c>wav2vec2.*</c>, <c>projector</c>,
    /// <c>classifier</c>).</summary>
    public void LoadTorchWeights(Func<string, double[]?> take)
    {
        double[] Read(string name) => take(name) ?? throw new InvalidDataException($"The style encoder checkpoint has no tensor '{name}'.");
        Network.LoadTorchWeights(name => name.StartsWith("hubert.", StringComparison.Ordinal) ? null : take("wav2vec2." + name));
        int hidden = Network.Configuration.HiddenSize;
        TorchParameters.Linear(_engine, Projector, hidden, ProjectionWidth, Read("projector.weight"), Read("projector.bias"));
        TorchParameters.Linear(_engine, Classifier, ProjectionWidth, Styles, Read("classifier.weight"), Read("classifier.bias"));
    }
}

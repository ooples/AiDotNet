using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Audio.Foundations;

/// <summary>
/// A wav2vec 2.0 encoder with a CTC head, as transformers' <c>Wav2Vec2ForCTC</c> builds it: the shared native path of
/// every recognizer whose paper network is the wav2vec 2.0 architecture (XLS-R, MMS, HuBERT and data2vec fine-tuned
/// for CTC).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
internal sealed class Wav2Vec2CtcStack<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _vocabSize;

    /// <summary>Builds the encoder and head, initialized as <c>Wav2Vec2ForCTC</c> initializes them.</summary>
    /// <param name="engine">The engine every layer runs on.</param>
    /// <param name="configuration">The wav2vec 2.0 encoder configuration.</param>
    /// <param name="vocabSize">The CTC vocabulary size, blank included.</param>
    /// <param name="seed">The initialization seed.</param>
    public Wav2Vec2CtcStack(IEngine engine, Wav2Vec2Configuration configuration, int vocabSize, int seed)
    {
        _engine = engine;
        _vocabSize = vocabSize;
        var layers = new List<LayerBase<T>>();
        Network = new Wav2Vec2Network<T>(engine, layers, configuration);
        Head = new DenseLayer<T>(vocabSize, (IActivationFunction<T>)new IdentityActivation<T>());
        layers.Add(Head);
        Layers = layers;

        var random = Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        Network.InitializeLikeReference(random);
        // Wav2Vec2ForCTC's lm_head: N(0, 0.02) weights, zero biases.
        int hidden = configuration.HiddenSize;
        var weights = new double[vocabSize * hidden];
        for (int i = 0; i < weights.Length; i++)
            weights[i] = 0.02 * Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble());
        TorchParameters.Linear(engine, Head, hidden, vocabSize, weights, new double[vocabSize]);
    }

    /// <summary>The wav2vec 2.0 encoder.</summary>
    public Wav2Vec2Network<T> Network { get; }

    /// <summary>The CTC projection (<c>lm_head</c>).</summary>
    public DenseLayer<T> Head { get; }

    /// <summary>Every layer, in registration order, the head last.</summary>
    public IReadOnlyList<LayerBase<T>> Layers { get; }

    /// <summary>
    /// CTC logits for a waveform: <c>[frames, vocab]</c> for <c>[samples]</c>, <c>[batch, frames, vocab]</c> for
    /// <c>[batch, samples]</c>. A waveform shorter than the feature encoder's receptive field is zero-padded to it.
    /// </summary>
    public Tensor<T> Run(Tensor<T> waveform, bool training, Random random, bool freezeFeatureEncoder, double finalDropout)
    {
        if (waveform.Rank > 2) waveform = _engine.Reshape(waveform, new[] { waveform.Shape[0], waveform.Length / waveform.Shape[0] });
        int field = ReceptiveField;
        if (waveform.Shape[^1] < field)
        {
            var padded = new Tensor<T>(waveform.Rank == 1 ? new[] { field } : new[] { waveform.Shape[0], field });
            int length = waveform.Shape[^1], rows = waveform.Rank == 1 ? 1 : waveform.Shape[0];
            for (int r = 0; r < rows; r++)
                for (int i = 0; i < length; i++) padded[r * field + i] = waveform[r * length + i];
            waveform = padded;
        }

        var hidden = Network.Forward(waveform, training, random, freezeFeatureEncoder: freezeFeatureEncoder).Hidden;
        if (training && finalDropout > 0)
        {
            var mask = new Tensor<T>(hidden._shape);
            var keep = NumOps.FromDouble(1.0 / (1.0 - finalDropout));
            for (int i = 0; i < mask.Length; i++) mask[i] = random.NextDouble() < finalDropout ? NumOps.Zero : keep;
            hidden = _engine.TensorMultiply(hidden, mask);
        }

        int batch = hidden.Shape[0], frames = hidden.Shape[1];
        var logits = _engine.Reshape(Head.Forward(_engine.Reshape(hidden, new[] { batch * frames, hidden.Shape[2] })),
            new[] { batch, frames, _vocabSize });
        return waveform.Rank == 1 ? _engine.Reshape(logits, new[] { frames, _vocabSize }) : logits;
    }

    /// <summary>The fewest samples the feature encoder turns into one frame.</summary>
    public int ReceptiveField
    {
        get
        {
            int receptive = 1;
            var c = Network.Configuration;
            for (int i = c.ConvKernels.Length - 1; i >= 0; i--) receptive = (receptive - 1) * c.ConvStrides[i] + c.ConvKernels[i];
            return receptive;
        }
    }

    /// <summary>Whether a layer belongs to the convolutional feature encoder (frozen during fine-tuning).</summary>
    public bool IsFeatureEncoderLayer(object layer)
        => Network.ConvLayers.Any(c => ReferenceEquals(c.Conv, layer) || ReferenceEquals(c.Norm, layer));

    /// <summary>
    /// Loads a transformers <c>*ForCTC</c> state dict: the encoder under <paramref name="encoderPrefix"/> and the head
    /// as <c>lm_head.*</c>.
    /// </summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read, string encoderPrefix)
    {
        int hidden = Network.Configuration.HiddenSize;
        Network.LoadTorchWeights((name, shape) => read(encoderPrefix + name, shape));
        TorchParameters.Linear(_engine, Head, hidden, _vocabSize,
            read("lm_head.weight", new[] { _vocabSize, hidden }), read("lm_head.bias", new[] { _vocabSize }));
    }
}

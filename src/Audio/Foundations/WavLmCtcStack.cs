using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Audio.Foundations;

/// <summary>
/// A WavLM encoder with a CTC head, as transformers' <c>WavLMForCTC</c> builds it: the shared native path of the
/// recognizers whose paper network is WavLM.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
internal sealed class WavLmCtcStack<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _vocabSize;

    /// <summary>Builds the encoder and head, initialized as <c>WavLMForCTC</c> initializes them.</summary>
    public WavLmCtcStack(IEngine engine, WavLmConfiguration configuration, int vocabSize, int seed)
    {
        _engine = engine;
        _vocabSize = vocabSize;
        var layers = new List<LayerBase<T>>();
        Network = new WavLmNetwork<T>(engine, layers, configuration);
        Head = new DenseLayer<T>(vocabSize, (IActivationFunction<T>)new IdentityActivation<T>());
        layers.Add(Head);
        Layers = layers;

        var random = Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        Network.InitializeLikeReference(random);
        // WavLMForCTC's lm_head: N(0, 0.02) weights, zero biases.
        int hidden = configuration.HiddenSize;
        var weights = new double[vocabSize * hidden];
        for (int i = 0; i < weights.Length; i++)
            weights[i] = 0.02 * Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble());
        TorchParameters.Linear(engine, Head, hidden, vocabSize, weights, new double[vocabSize]);
    }

    /// <summary>The WavLM encoder.</summary>
    public WavLmNetwork<T> Network { get; }

    /// <summary>The CTC projection (<c>lm_head</c>).</summary>
    public DenseLayer<T> Head { get; }

    /// <summary>Every layer, in registration order, the head last.</summary>
    public IReadOnlyList<LayerBase<T>> Layers { get; }

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

    /// <summary>
    /// CTC logits: <c>[frames, vocab]</c> for <c>[samples]</c>, <c>[batch, frames, vocab]</c> for <c>[batch, samples]</c>
    /// (any further axes read as samples). A waveform shorter than the receptive field is zero-padded to it.
    /// </summary>
    /// <param name="waveform">The 16 kHz audio.</param>
    /// <param name="regularization">Train-mode dropout and LayerDrop, or null for evaluation mode.</param>
    /// <param name="finalDropout">Dropout before the CTC head in train mode.</param>
    public Tensor<T> Run(Tensor<T> waveform, WavLmRegularization? regularization, double finalDropout)
    {
        bool single = waveform.Rank == 1;
        int rows = single ? 1 : waveform.Shape[0], samples = waveform.Length / rows;
        int length = Math.Max(samples, ReceptiveField);
        var outputs = new Tensor<T>[rows];
        for (int r = 0; r < rows; r++)
        {
            Tensor<T> row;
            if (length == samples)
                row = single ? waveform
                    : _engine.Reshape(_engine.TensorSlice(_engine.Reshape(waveform, new[] { rows, samples }), new[] { r, 0 }, new[] { 1, samples }),
                        new[] { samples });
            else
            {
                row = new Tensor<T>(new[] { length });
                for (int i = 0; i < samples; i++) row[i] = waveform[r * samples + i];
            }
            var hidden = Network.HiddenStates(row, regularization)[^1];                                     // [frames, H]
            if (regularization is { } reg && finalDropout > 0)
            {
                var mask = new Tensor<T>(hidden._shape);
                var keep = NumOps.FromDouble(1.0 / (1.0 - finalDropout));
                for (int i = 0; i < mask.Length; i++) mask[i] = reg.Random.NextDouble() < finalDropout ? NumOps.Zero : keep;
                hidden = _engine.TensorMultiply(hidden, mask);
            }
            outputs[r] = Head.Forward(hidden);                                                               // [frames, V]
        }
        return single ? outputs[0] : _engine.TensorStack(outputs, 0);
    }

    /// <summary>Whether a layer belongs to the convolutional feature encoder (frozen during fine-tuning).</summary>
    public bool IsFeatureEncoderLayer(object layer)
        => Network.Convs.Any(c => ReferenceEquals(c, layer)) || ReferenceEquals(Network.FirstNorm, layer)
            || Network.ConvNorms.Any(n => ReferenceEquals(n, layer));

    /// <summary>Loads a transformers <c>WavLMForCTC</c> state dict: <c>wavlm.*</c> and <c>lm_head.*</c>.</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read, Func<string, bool> has)
    {
        int hidden = Network.Configuration.HiddenSize;
        Network.LoadTorchWeights((name, shape) => read("wavlm." + name, shape), name => has("wavlm." + name));
        TorchParameters.Linear(_engine, Head, hidden, _vocabSize,
            read("lm_head.weight", new[] { _vocabSize, hidden }), read("lm_head.bias", new[] { _vocabSize }));
    }
}

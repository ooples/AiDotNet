using AiDotNet.ActivationFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>
/// LLaMA-Omni's speech adaptor (Fang et al. 2025 §2.2, eq. 1–2; the release's <c>EncoderProjectorConcat</c>): every k
/// consecutive encoder frames concatenated (the frames past a multiple of k dropped), then a 2-layer perceptron with ReLU
/// into the language model's embedding space.
/// </summary>
internal sealed class LlamaOmniAdaptor<T>
{
    private readonly IEngine _engine;

    public LlamaOmniAdaptor(IEngine engine, List<LayerBase<T>> layers, int encoderDim, int downsample, int hidden, int outputDim, Random initialization)
    {
        _engine = engine;
        EncoderDim = encoderDim;
        Downsample = downsample;
        Hidden = hidden;
        OutputDim = outputDim;
        double Uniform(int fanIn) => (2 * initialization.NextDouble() - 1) / Math.Sqrt(fanIn);
        DenseLayer<T> Linear(int inputs, int outputs)
        {
            var layer = new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>());
            layers.Add(layer);
            TorchParameters.Linear(engine, layer, inputs, outputs, Enumerable.Range(0, inputs * outputs).Select(_ => Uniform(inputs)).ToArray(),
                Enumerable.Range(0, outputs).Select(_ => Uniform(inputs)).ToArray());
            return layer;
        }
        First = Linear(encoderDim * downsample, hidden);
        Second = Linear(hidden, outputDim);
    }

    public int EncoderDim { get; }
    public int Downsample { get; }
    public int Hidden { get; }
    public int OutputDim { get; }
    public DenseLayer<T> First { get; }
    public DenseLayer<T> Second { get; }

    /// <summary>The speech embeddings <c>[⌊N / k⌋, d]</c> of encoder states <c>[N, encoder width]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> encoder)
    {
        int frames = encoder.Shape[0] / Downsample * Downsample;
        if (frames == 0) throw new ArgumentException($"The adaptor needs at least {Downsample} encoder frames.", nameof(encoder));
        var kept = frames == encoder.Shape[0] ? encoder : _engine.TensorSlice(encoder, new[] { 0, 0 }, new[] { frames, EncoderDim });
        var stacked = _engine.Reshape(kept, new[] { frames / Downsample, EncoderDim * Downsample });
        return Second.Forward(_engine.ReLU(First.Forward(stacked)));
    }

    /// <summary>Loads <c>{prefix}linear1</c> and <c>{prefix}linear2</c>.</summary>
    public void Load(Func<string, int[], double[]> read, string prefix)
    {
        int inputs = EncoderDim * Downsample, hidden = Hidden, output = OutputDim;
        TorchParameters.Linear(_engine, First, inputs, hidden, read(prefix + "linear1.weight", new[] { hidden, inputs }), read(prefix + "linear1.bias", new[] { hidden }));
        TorchParameters.Linear(_engine, Second, hidden, output, read(prefix + "linear2.weight", new[] { output, hidden }), read(prefix + "linear2.bias", new[] { output }));
    }
}

/// <summary>The shape of LLaMA-Omni's CTC speech decoder (<c>ctc_decoder_config</c> "(layers, width, heads, ffn)",
/// <c>ctc_upsample_factor</c>, <c>unit_vocab_size</c>), its Llama layers rotating as the language model does.</summary>
internal sealed record LlamaOmniDecoderConfiguration(int LanguageModelWidth, int Layers, int Width, int Heads, int Intermediate,
    int UpsampleFactor, int Units, double RmsEpsilon, double RopeTheta, double RopeScalingFactor, double RopeLowFrequencyFactor,
    double RopeHighFrequencyFactor, int RopeOriginalMaxPositions);

/// <summary>
/// LLaMA-Omni's streaming speech decoder (§2.4; the release's <c>SpeechGeneratorCTC</c>): the language model's hidden
/// states of the text response, each repeated λ times, projected, through causal Llama decoder layers (no final norm) to
/// logits over the units and a blank; trained with CTC (eq. 5), decoded by the best alignment collapsed (repeats merged,
/// blanks dropped).
/// </summary>
internal sealed class LlamaOmniSpeechDecoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public LlamaOmniSpeechDecoder(IEngine engine, List<LayerBase<T>> layers, LlamaOmniDecoderConfiguration configuration, Random initialization)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        double Uniform(int fanIn) => (2 * initialization.NextDouble() - 1) / Math.Sqrt(fanIn);
        DenseLayer<T> Linear(int inputs, int outputs)
        {
            var layer = new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>());
            layers.Add(layer);
            TorchParameters.Linear(engine, layer, inputs, outputs, Enumerable.Range(0, inputs * outputs).Select(_ => Uniform(inputs)).ToArray(),
                Enumerable.Range(0, outputs).Select(_ => Uniform(inputs)).ToArray());
            return layer;
        }
        InputProjection = Linear(c.LanguageModelWidth, c.Width);
        Network = new LlamaNetwork<T>(engine, layers, new LlamaConfiguration
        {
            VocabSize = 0, IncludeHead = false, IncludeFinalNorm = false, Hidden = c.Width, Layers = c.Layers, Heads = c.Heads,
            KeyValueHeads = c.Heads, Intermediate = c.Intermediate, RmsEpsilon = c.RmsEpsilon, RopeTheta = c.RopeTheta,
            RopeScalingFactor = c.RopeScalingFactor, RopeLowFrequencyFactor = c.RopeLowFrequencyFactor,
            RopeHighFrequencyFactor = c.RopeHighFrequencyFactor, RopeOriginalMaxPositions = c.RopeOriginalMaxPositions,
        });
        Network.InitializeLikeReference(initialization);
        OutputProjection = Linear(c.Width, c.Units + 1);
    }

    public LlamaOmniDecoderConfiguration Configuration { get; }
    public DenseLayer<T> InputProjection { get; }
    public LlamaNetwork<T> Network { get; }
    public DenseLayer<T> OutputProjection { get; }

    /// <summary>The blank's index: after the units.</summary>
    public int Blank => Configuration.Units;

    /// <summary>
    /// <c>upsample</c>: each of M states repeated to fill max(λM, <paramref name="minimumFrames"/>) frames by the release's
    /// uniform assignment (frame j reads state ⌊j / (frames / M)⌋, the ratio in float32 as torch divides integer
    /// tensors).
    /// </summary>
    internal Tensor<T> Upsample(Tensor<T> states, int minimumFrames)
    {
        int m = states.Shape[0], frames = Math.Max(m * Configuration.UpsampleFactor, minimumFrames);
        float ratio = (float)frames / m;
        var index = new Tensor<int>(new[] { frames });
        for (int j = 0; j < frames; j++) index[j] = (int)((float)j / ratio);
        return _engine.TensorIndexSelect(states, index, 0);
    }

    /// <summary>The log-probabilities <c>[frames, units + 1]</c> of hidden states <c>[M, LM width]</c>, upsampled to at
    /// least <paramref name="minimumFrames"/>.</summary>
    public Tensor<T> LogProbabilities(Tensor<T> states, int minimumFrames = 0)
    {
        var up = InputProjection.Forward(Upsample(states, minimumFrames));
        int frames = up.Shape[0];
        var (hidden, _) = Network.Forward(_engine.Reshape(up, new[] { 1, frames, Configuration.Width }));
        var logits = OutputProjection.Forward(_engine.Reshape(hidden, new[] { frames, Configuration.Width }));
        return _engine.TensorLogSoftmax(logits, axis: 1);
    }

    /// <summary>
    /// The CTC loss of utterances (hidden states and units each): the summed negative log-likelihoods over the summed unit
    /// counts (<c>reduction="sum"</c> divided by the target lengths), an infeasible alignment counting zero
    /// (<c>zero_infinity</c>).
    /// </summary>
    public Tensor<T> Loss(IReadOnlyList<(Tensor<T> States, int[] Units)> utterances)
    {
        Tensor<T>? total = null;
        int targets = 0;
        foreach (var (states, units) in utterances)
        {
            if (units.Length == 0) throw new ArgumentException("An utterance has no units.", nameof(utterances));
            if (units.Any(u => u < 0 || u >= Configuration.Units)) throw new ArgumentOutOfRangeException(nameof(utterances), $"Units lie in [0, {Configuration.Units}).");
            var logProbabilities = LogProbabilities(states, units.Length);
            int frames = logProbabilities.Shape[0];
            var nll = _engine.TensorCTCLoss(_engine.Reshape(logProbabilities, new[] { frames, 1, Configuration.Units + 1 }),
                new Tensor<int>(new[] { units.Length }, new Vector<int>(units)), new[] { frames }, new[] { units.Length }, Blank);
            if (double.IsInfinity(NumOps.ToDouble(nll[0]))) nll = _engine.TensorMultiplyScalar(logProbabilities, NumOps.Zero);
            var sum = _engine.ReduceSum(nll, null, false);
            total = total is null ? sum : _engine.TensorAdd(total, sum);
            targets += units.Length;
        }
        return _engine.TensorMultiplyScalar(total ?? throw new ArgumentException("There are no utterances.", nameof(utterances)), NumOps.FromDouble(1.0 / targets));
    }

    /// <summary><c>predict</c>: the best alignment of hidden states <c>[M, LM width]</c> (each frame's argmax).</summary>
    public int[] Alignment(Tensor<T> states)
    {
        var logProbabilities = LogProbabilities(states);
        int frames = logProbabilities.Shape[0], classes = Configuration.Units + 1;
        var alignment = new int[frames];
        for (int f = 0; f < frames; f++)
        {
            int best = 0;
            for (int k = 1; k < classes; k++)
                if (NumOps.GreaterThan(logProbabilities[f, k], logProbabilities[f, best])) best = k;
            alignment[f] = best;
        }
        return alignment;
    }

    /// <summary>The collapse of an alignment (<c>ctc_postprocess</c>): repeats merged, then blanks dropped.</summary>
    public int[] Collapse(IReadOnlyList<int> alignment)
    {
        var units = new List<int>();
        for (int i = 0; i < alignment.Count; i++)
            if ((i == 0 || alignment[i] != alignment[i - 1]) && alignment[i] != Blank) units.Add(alignment[i]);
        return units.ToArray();
    }

    /// <summary>Loads the release's <c>{prefix}input_proj</c>, <c>{prefix}layers.{i}.*</c> and <c>{prefix}output_proj</c>.</summary>
    public void Load(Func<string, int[], double[]> read, string prefix)
    {
        var c = Configuration;
        TorchParameters.Linear(_engine, InputProjection, c.LanguageModelWidth, c.Width, read(prefix + "input_proj.weight", new[] { c.Width, c.LanguageModelWidth }),
            read(prefix + "input_proj.bias", new[] { c.Width }));
        Network.LoadTorchWeights(read, prefix);
        TorchParameters.Linear(_engine, OutputProjection, c.Width, c.Units + 1, read(prefix + "output_proj.weight", new[] { c.Units + 1, c.Width }),
            read(prefix + "output_proj.bias", new[] { c.Units + 1 }));
    }
}

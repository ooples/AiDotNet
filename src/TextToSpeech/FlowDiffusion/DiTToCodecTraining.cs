using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>Separate Mel-VAE training, CLaM-TTS equations 4–6 and DiTTo equation 2.</summary>
/// <remarks>The papers do not disclose discriminator widths, crop lengths, resolution reductions or loss
/// coefficients. These exposed defaults are implementation choices, not recovered private checkpoint settings.
/// Semantic finetuning uses a frozen ByT5-base decoder and requires its pretrained weights.</remarks>
public sealed class DiTToCodecTrainingOptions
{
    /// <summary>Crop lengths, in mel frames, of the multi-length critics (one critic per entry).</summary>
    public int[] DiscriminatorWindows { get; set; } = { 32, 64, 128 };
    /// <summary>Frequency-resolution reductions of the multi-resolution spectrogram critics (one critic per entry).</summary>
    public int[] ResolutionDivisors { get; set; } = { 1, 2, 4 };
    /// <summary>Channel width of every critic convolution.</summary>
    public int DiscriminatorChannels { get; set; } = 32;
    /// <summary>Weight of the Mel-VAE reconstruction loss.</summary>
    public double ReconstructionWeight { get; set; } = 1;
    /// <summary>Weight of the residual-quantizer commitment loss.</summary>
    public double CommitmentWeight { get; set; } = 0.25;
    /// <summary>Weight of the critics' adversarial loss on the generator.</summary>
    public double AdversarialWeight { get; set; } = 1;
    /// <summary>Weight of the critics' feature-matching loss on the generator.</summary>
    public double FeatureMatchingWeight { get; set; } = 1;
    /// <summary>Weight of the ByT5 semantic loss during semantic finetuning (DiTTo equation 2).</summary>
    public double SemanticWeight { get; set; } = 1;
    /// <summary>Adam learning rate of the codec generator and the critics.</summary>
    public double LearningRate { get; set; } = 2e-4;
    /// <summary>Model width of the frozen ByT5-base decoder used for semantic finetuning.</summary>
    public int SemanticModelDim { get; set; } = 1536;
    /// <summary>Feed-forward width of the ByT5-base decoder.</summary>
    public int SemanticFeedForwardDim { get; set; } = 3968;
    /// <summary>Per-head key/value width of the ByT5-base decoder.</summary>
    public int SemanticKeyValueDim { get; set; } = 64;
    /// <summary>Attention heads of the ByT5-base decoder.</summary>
    public int SemanticHeads { get; set; } = 12;
    /// <summary>Decoder layers of the ByT5-base decoder.</summary>
    public int SemanticDecoderLayers { get; set; } = 6;
    /// <summary>Byte vocabulary size of the ByT5 decoder.</summary>
    public int SemanticVocabulary { get; set; } = 384;

    /// <summary>Creates the options with the defaults above.</summary>
    public DiTToCodecTrainingOptions() { }
    /// <summary>Creates a deep copy of <paramref name="other"/>.</summary>
    public DiTToCodecTrainingOptions(DiTToCodecTrainingOptions other)
    {
        Validation.Guard.NotNull(other);
        DiscriminatorWindows = (int[])other.DiscriminatorWindows.Clone();
        ResolutionDivisors = (int[])other.ResolutionDivisors.Clone();
        DiscriminatorChannels = other.DiscriminatorChannels;
        ReconstructionWeight = other.ReconstructionWeight; CommitmentWeight = other.CommitmentWeight;
        AdversarialWeight = other.AdversarialWeight; FeatureMatchingWeight = other.FeatureMatchingWeight;
        SemanticWeight = other.SemanticWeight; LearningRate = other.LearningRate;
        SemanticModelDim = other.SemanticModelDim; SemanticFeedForwardDim = other.SemanticFeedForwardDim;
        SemanticKeyValueDim = other.SemanticKeyValueDim; SemanticHeads = other.SemanticHeads;
        SemanticDecoderLayers = other.SemanticDecoderLayers; SemanticVocabulary = other.SemanticVocabulary;
    }
}

/// <summary>Mel-input multi-length and multi-resolution critics. Their unpublished micro-layout is configurable.</summary>
internal sealed class DiTToMelCritics<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private sealed record Critic(int Window, int Reduction, Conv2DLayer<T>[] Convs, Conv2DLayer<T> Output);
    private readonly List<Critic> _critics = new();
    internal List<LayerBase<T>> Layers { get; } = new();

    internal DiTToMelCritics(IEngine engine, DiTToCodecTrainingOptions options)
    {
        _engine = engine;
        if (options.DiscriminatorChannels <= 0 || options.DiscriminatorWindows.Length == 0 || options.ResolutionDivisors.Length == 0
            || options.DiscriminatorWindows.Any(value => value <= 0) || options.ResolutionDivisors.Any(value => value <= 0))
            throw new ArgumentException("Mel critics need positive widths, crop lengths and resolution divisors.", nameof(options));
        void Add(int window, int reduction, bool resolution)
        {
            int width = options.DiscriminatorChannels;
            var convs = new Conv2DLayer<T>[resolution ? 5 : 3];
            for (int i = 0; i < convs.Length; i++)
            {
                // Multi-length critics reduce both axes; spectrogram critics retain frequency resolution.
                int frequencyStride = resolution ? 1 : 2;
                int timeStride = resolution && (i == 0 || i == 4) ? 1 : 2;
                int timeKernel = resolution && i < 4 ? 9 : 3;
                convs[i] = new Conv2DLayer<T>(i == 0 ? 1 : width, width, 3, timeKernel,
                    frequencyStride, timeStride, 1, timeKernel / 2, true);
                Layers.Add(convs[i]);
            }
            var output = new Conv2DLayer<T>(width, 1, 3, 3, 1, 1, 1, 1, true); Layers.Add(output);
            _critics.Add(new Critic(window, reduction, convs, output));
        }
        foreach (int window in options.DiscriminatorWindows) Add(window, 1, false);
        foreach (int reduction in options.ResolutionDivisors) Add(0, reduction, true);
    }

    internal List<(Tensor<T> Score, List<Tensor<T>> Features)> Forward(Tensor<T> mel, int[] starts)
    {
        int frames = mel.Shape[0], bins = mel.Shape[1];
        var results = new List<(Tensor<T>, List<Tensor<T>>)>() ;
        for (int i = 0; i < _critics.Count; i++)
        {
            var critic = _critics[i];
            int length = critic.Window == 0 ? frames : Math.Min(frames, critic.Window);
            int start = critic.Window == 0 ? 0 : starts[i];
            var rows = _engine.TensorSlice(mel, new[] { start, 0 }, new[] { length, bins });
            var x = _engine.Reshape(_engine.TensorTranspose(rows), new[] { 1, 1, bins, length });
            if (critic.Reduction > 1)
            {
                // Explicit mel-grid resolution choice: average complete frequency/time cells.
                int frequencyCells = Math.Max(1, bins / critic.Reduction), timeCells = Math.Max(1, length / critic.Reduction);
                int frequencyWidth = Math.Min(bins, critic.Reduction), timeWidth = Math.Min(length, critic.Reduction);
                x = _engine.TensorSlice(x, new[] { 0, 0, 0, 0 }, new[] { 1, 1, frequencyCells * frequencyWidth, timeCells * timeWidth });
                x = _engine.Reshape(x, new[] { frequencyCells, frequencyWidth, timeCells, timeWidth });
                x = _engine.ReduceMean(x, new[] { 1, 3 }, false);
                x = _engine.Reshape(x, new[] { 1, 1, frequencyCells, timeCells });
            }
            var features = new List<Tensor<T>>();
            foreach (var conv in critic.Convs)
            {
                x = _engine.LeakyReLU(conv.Forward(x), Ops.FromDouble(.2)); features.Add(x);
            }
            var score = critic.Output.Forward(x); features.Add(score);
            results.Add((score, features));
        }
        return results;
    }
}

internal sealed class DiTToCodecTrainer<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    internal DiTToCodecTrainingOptions Options { get; }
    internal DiTToMelCritics<T> Critics { get; }
    internal List<LayerBase<T>> ProjectionLayers { get; } = new();
    internal List<LayerBase<T>> DecoderLayers { get; } = new();
    internal DenseLayer<T>? Projection { get; private set; }
    internal T5Seq2Seq<T>? SemanticDecoder { get; private set; }
    internal bool DecoderWeightsLoaded { get; set; }

    internal DiTToCodecTrainer(IEngine engine, DiTToCodecTrainingOptions options)
    {
        _engine = engine; Options = options; Critics = new DiTToMelCritics<T>(engine, options);
        if (new[] { options.ReconstructionWeight, options.CommitmentWeight, options.AdversarialWeight,
                options.FeatureMatchingWeight, options.SemanticWeight }.Any(value => double.IsNaN(value) || double.IsInfinity(value) || value < 0))
            throw new ArgumentException("Codec loss weights must be finite and nonnegative.", nameof(options));
    }

    internal void InitializeSemanticDecoder(int latentDim, Random random)
    {
        if (SemanticDecoder is not null) return;
        Projection = IndexTtsOps.Linear(_engine, ProjectionLayers, random, latentDim, Options.SemanticModelDim);
        SemanticDecoder = new T5Seq2Seq<T>(_engine, DecoderLayers, new T5Configuration(Options.SemanticVocabulary,
            Options.SemanticModelDim, Options.SemanticFeedForwardDim, Options.SemanticKeyValueDim,
            Options.SemanticHeads, EncoderLayers: 0, Options.SemanticDecoderLayers, Dropout: 0, EndTokenId: 1,
            GatedGelu: true, TieWordEmbeddings: false));
    }

    internal Tensor<T> Mean(Tensor<T> value) => _engine.ReduceMean(value, Enumerable.Range(0, value.Rank).ToArray(), false);
    private Tensor<T> Square(Tensor<T> value) => Mean(_engine.TensorMultiply(value, value));
    private Tensor<T> Detached(Tensor<T> value) => _engine.StopGradient(value);

    internal Tensor<T> Generator(Tensor<T> real, Tensor<T> generated, Tensor<T> commitment, Tensor<T> variational, int[] starts)
    {
        var loss = _engine.TensorAdd(variational, _engine.TensorMultiplyScalar(commitment, Ops.FromDouble(Options.CommitmentWeight)));
        loss = _engine.TensorAdd(loss, _engine.TensorMultiplyScalar(Mean(_engine.TensorAbs(_engine.TensorSubtract(real, generated))),
            Ops.FromDouble(Options.ReconstructionWeight)));
        var positive = Critics.Forward(real, starts); var negative = Critics.Forward(generated, starts);
        for (int i = 0; i < negative.Count; i++)
        {
            var adversarial = Square(_engine.TensorAddScalar(negative[i].Score, Ops.FromDouble(-1)));
            Tensor<T>? matching = null;
            for (int j = 0; j < negative[i].Features.Count; j++)
            {
                var distance = Mean(_engine.TensorAbs(_engine.TensorSubtract(negative[i].Features[j], Detached(positive[i].Features[j]))));
                matching = matching is null ? distance : _engine.TensorAdd(matching, distance);
            }
            var term = _engine.TensorAdd(adversarial, _engine.TensorMultiplyScalar(matching!, Ops.FromDouble(Options.FeatureMatchingWeight)));
            loss = _engine.TensorAdd(loss, _engine.TensorMultiplyScalar(term, Ops.FromDouble(Options.AdversarialWeight / negative.Count)));
        }
        return loss;
    }

    internal Tensor<T> Discriminator(Tensor<T> real, Tensor<T> generated, int[] starts)
    {
        var positive = Critics.Forward(real, starts); var negative = Critics.Forward(generated, starts);
        Tensor<T>? loss = null;
        for (int i = 0; i < negative.Count; i++)
        {
            var term = _engine.TensorAdd(Square(_engine.TensorAddScalar(positive[i].Score, Ops.FromDouble(-1))), Square(negative[i].Score));
            loss = loss is null ? term : _engine.TensorAdd(loss, term);
        }
        return _engine.TensorMultiplyScalar(loss!, Ops.FromDouble(1.0 / negative.Count));
    }

    internal Tensor<T> SemanticLoss(Tensor<T> latents, string text)
    {
        if (!DecoderWeightsLoaded || SemanticDecoder is null || Projection is null)
            throw new InvalidOperationException("Load pretrained ByT5 decoder weights before semantic codec finetuning.");
        int[] labels = System.Text.Encoding.UTF8.GetBytes(text).Select(value => (int)value + 3).Concat(new[] { 1 }).ToArray();
        if (labels.Any(value => value >= Options.SemanticVocabulary)) throw new ArgumentException("The ByT5 vocabulary cannot represent this UTF-8 text.", nameof(text));
        var inputs = new int[labels.Length]; Array.Copy(labels, 0, inputs, 1, labels.Length - 1);
        // No NoGradScope here: fixed decoder parameters still transmit gradients to f(z) and the speech encoder.
        var logits = SemanticDecoder.Decode(IndexTtsOps.Ids<T>(inputs), Projection.Forward(latents), false, new Random(0));
        var targets = new Tensor<T>(logits.Shape.ToArray());
        for (int i = 0; i < labels.Length; i++) targets[i, labels[i]] = Ops.FromDouble(-1.0 / labels.Length);
        return _engine.ReduceSum(_engine.TensorMultiply(_engine.TensorLogSoftmax(logits, -1), targets), new[] { 0, 1 }, false);
    }
}

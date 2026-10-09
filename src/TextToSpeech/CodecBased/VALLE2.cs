using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.FrontEnd;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// VALL-E 2: VALL-E with grouped code modeling and repetition-aware sampling, which predicts several first-codebook codes
/// per autoregressive step and avoids the decoding loops of plain sampling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Reference: "VALL-E 2: Neural Codec Language Models are Human Parity Zero-Shot Text to Speech Synthesizers" (Chen et al.,
/// 2024). Microsoft released neither code nor a reproduction; what the paper leaves open is chosen in
/// <see cref="VALLE2Options"/> and the decision log.
/// </para>
/// <para>
/// <b>Model</b> (§3.2–3.3): the AR model reads the text, <c>&lt;eos&gt; &lt;bos&gt;</c> and one embedding per group of
/// G first-codebook codes, and predicts the next group's G codes; the NAR model reads the text, an acoustic condition
/// (all eight codebooks), the lower codebooks of the target and a code-ID token, and predicts codebook j. Both use learned
/// positions and VALL-E's Transformer layers (<see cref="VallE2Core{T}"/>).
/// </para>
/// <para>
/// <b>Training</b>: the AR model on whole utterances (clipped at the start to whole groups, §3.1); the NAR model with the
/// utterance's first frames as its acoustic condition, at most half of it, otherwise a random 3–30 seconds (§4.1.1).
/// </para>
/// <para>
/// <b>Synthesis</b> (§3.4): the prompt's transcript and the text, the prompt's first codebook as the AR prefix (clipped
/// to whole groups), repetition-aware sampling (Algorithm 1), the NAR greedy over the whole text with the prompt's eight
/// codebooks as condition, and Vocos's EnCodec model decoding the new codes.
/// </para>
/// <para><b>For Beginners:</b> VALL-E 2 writes speech tokens a few at a time instead of one by one, and when it notices
/// itself repeating it switches to a more random choice, so it does not get stuck saying the same sound.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional,
///     NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
/// var valle2 = new VALLE2&lt;float&gt;(architecture, new VALLE2Options());
/// valle2.Voice = valle2.CreateVoice(promptAudio24kHz, "the prompt's transcript");
/// var audio = valle2.Synthesize("Hello there.");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "VALL-E 2: Neural Codec Language Models are Human Parity Zero-Shot Text to Speech Synthesizers",
    "https://arxiv.org/abs/2406.05370",
    Year = 2024,
    Authors = "Chen et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 5e-4, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, WeightDecay = 0.01,
    Schedule = LearningRateSchedulerType.LinearWarmup, WarmupSteps = 32_000,
    PostWarmupDecay = LinearWarmupScheduler.DecayMode.Linear,
    Component = "autoregressive", Provenance = RecipeProvenance.DerivedFromCitedWork,
    Source = "Section 4.1.1: AdamW, warmed up over the first 32k updates to a peak learning rate, then decayed linearly; "
             + "the paper does not give the peak or the step count, so they are VALL-E's (Wang et al. 2023 §5.1: 5e-4, "
             + "800k steps), which VALL-E 2 follows, with PyTorch's AdamW defaults.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 5e-4, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, WeightDecay = 0.01,
    Schedule = LearningRateSchedulerType.LinearWarmup, WarmupSteps = 32_000,
    PostWarmupDecay = LinearWarmupScheduler.DecayMode.Linear,
    Component = "non-autoregressive", Provenance = RecipeProvenance.DerivedFromCitedWork,
    Source = "Section 4.1.1: AdamW, warmed up over the first 32k updates to a peak learning rate, then decayed linearly; "
             + "the paper does not give the peak or the step count, so they are VALL-E's (Wang et al. 2023 §5.1: 5e-4, "
             + "800k steps), which VALL-E 2 follows, with PyTorch's AdamW defaults.")]
public partial class VALLE2<T> : VallEModelBase<T>
{
    private VallE2Core<T>? _core2;
    private VocosEncodecDecoder<T>? _vocos;

    /// <summary>Creates an ONNX-backed VALL-E 2 for inference.</summary>
    public VALLE2(NeuralNetworkArchitecture<T> architecture, string modelPath, VALLE2Options? options = null)
        : base(architecture, modelPath, options ?? new VALLE2Options())
    {
    }

    /// <summary>Creates a native, trainable VALL-E 2.</summary>
    public VALLE2(NeuralNetworkArchitecture<T> architecture, VALLE2Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new VALLE2Options(), optimizer)
    {
    }

    private VALLE2Options Options2 => (VALLE2Options)Settings;

    // The networks, for tests of the paper's equations.
    internal VallE2Core<T> Core2ForTests => Core2;

    private VallE2Core<T> Core2 => _core2 ?? throw new InvalidOperationException("VALL-E 2's networks exist in native mode only.");

    /// <inheritdoc />
    public override ModelOptions GetOptions() => Settings;

    /// <inheritdoc />
    protected override IReadOnlyList<string> PhonemeTable => LibriTtsPhonemeTable.Symbols;

    /// <inheritdoc />
    protected override IReadOnlyList<string> PhonemizeText(string text) => EnglishG2P.Default.Phonemize(text);

    /// <inheritdoc />
    private protected override void BuildNetworks(List<LayerBase<T>> arLayers, List<LayerBase<T>> narLayers, Random initialization)
    {
        var o = Options2;
        if (o.GroupSize < 1) throw new ArgumentException("The group size must be at least 1.");
        _core2 = new VallE2Core<T>(Engine, arLayers, narLayers, new VallE2Configuration(
            TextTokens: o.TextTokens, AudioTokens: o.CodebookSize, Codebooks: o.NumCodebooks, GroupSize: o.GroupSize,
            ModelDim: o.HiddenDim, Heads: o.NumHeads, Layers: o.NumDecoderLayers, FeedForwardDim: o.FeedForwardDim,
            Dropout: o.DropoutRate, MaxTextPositions: 2 * o.MaxTextLength + 4, MaxCodePositions: o.MaxCodePositions));
        _core2.Initialize(initialization);
        // Vocos's bandwidth classes are 2, 4, 8 and 16 codebooks of the codec's rate.
        double perCodebook = (double)o.SampleRate / o.HopSize * Math.Log(o.CodebookSize, 2) / 1000.0;
        var bandwidths = new[] { 2, 4, 8, 16 }.Select(n => n * perCodebook).ToArray();
        if (!new[] { 2, 4, 8, 16 }.Contains(o.NumCodebooks))
            throw new ArgumentException("Vocos decodes 2, 4, 8 or 16 codebooks; set NumCodebooks to one of them.");
        _vocos = new VocosEncodecDecoder<T>(Engine, initialization, new VocosEncodecConfiguration(bandwidths,
            CodebookSize: o.CodebookSize, LatentDim: o.Codec.Dimension, FrameRate: o.SampleRate / o.HopSize, Dim: o.DecoderDim,
            IntermediateDim: o.DecoderIntermediateDim, Layers: o.DecoderLayers, FftSize: 4 * o.HopSize, HopSize: o.HopSize));
    }

    /// <inheritdoc />
    private protected override IEnumerable<LayerBase<T>> DecoderLayers => _vocos?.Layers ?? Array.Empty<LayerBase<T>>();

    /// <inheritdoc />
    /// <remarks>Vocos's EnCodec model (§4.1.1).</remarks>
    protected override Tensor<T> DecodeCodes(int[,] codes) =>
        (_vocos ?? throw new InvalidOperationException("VALL-E 2's decoder exists in native mode only.")).Decode(codes);

    /// <summary>Loads Vocos's released EnCodec model (<c>pytorch_model.bin</c> of charactr/vocos-encodec-24khz, as
    /// <c>.pt</c> or safetensors).</summary>
    public void LoadDecoderWeights(string path)
    {
        var file = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path);
        (_vocos ?? throw new InvalidOperationException("VALL-E 2's decoder exists in native mode only.")).LoadTorchWeights((name, shape) =>
        {
            if (!file.TryGetValue(name, out var tensor))
                throw new InvalidDataException($"The checkpoint has no tensor '{name}'.");
            return tensor.ToVector().Select(v => (double)v).ToArray();
        });
    }

    /// <inheritdoc />
    private protected override void LoadNetworkWeights(Func<string, int[], double[]> read) =>
        throw new NotSupportedException("VALL-E 2 has no released checkpoint whose layout could be loaded.");

    // Codes clipped at the start to whole groups (§3.1: the clipped codes are the utterance's leading silence).
    private int[] WholeGroups(int[] codes)
    {
        int g = Options2.GroupSize, clip = codes.Length % g;
        return clip == 0 ? codes : codes.Skip(clip).ToArray();
    }

    /// <inheritdoc />
    protected override Func<Tensor<T>, Tensor<T>, Tensor<T>> AutoRegressiveObjective(TtsTrainingSample<T> sample, int[] text,
        int[,] codes, bool training, Random random)
    {
        var first = WholeGroups(FirstCodebook(codes));
        if (first.Length == 0) throw new ArgumentException("The utterance is shorter than one group.", nameof(sample));
        return (_, _) => Core2.ArLoss(text, first, training, random);
    }

    /// <inheritdoc />
    /// <remarks>The acoustic condition is the utterance's first T′ frames: at most half of it, otherwise a random 3–30
    /// seconds (§4.1.1).</remarks>
    protected override Func<Tensor<T>, Tensor<T>, Tensor<T>> NonAutoRegressiveObjective(TtsTrainingSample<T> sample,
        int[] text, int[,] codes, bool training, Random random)
    {
        var o = Options2;
        int frames = codes.GetLength(0);
        double seconds = o.MinPromptSeconds + random.NextDouble() * (o.MaxPromptSeconds - o.MinPromptSeconds);
        int promptFrames = Math.Min(frames / 2, (int)Math.Round(seconds * CodecFrameRate));
        return (_, _) => Core2.NarLoss(text, codes, promptFrames, training, random);
    }

    /// <inheritdoc />
    /// <remarks>The prompt's first codebook, clipped to whole groups, is the AR prefix; the NAR reads the whole text and the
    /// prompt's eight codebooks.</remarks>
    protected override int[,] Generate(int[] target, int[] enrolled, int[,] prompt, TtsVoice<T> voice, Random random)
    {
        var o = Options2;
        var full = JoinPrompt(enrolled, target);
        var prefix = WholeGroups(FirstCodebook(prompt));
        var first = Core2.ArGenerate(full, prefix, o.TopP, o.RepetitionWindow, o.RepetitionThreshold,
            o.MaxCodesPerTextToken * full.Length, random);
        return Core2.NarGenerate(full, prompt, first.ToArray(), random);
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = IsNative ? "VALL-E-2-Native" : "VALL-E-2-ONNX",
            Description = "VALL-E 2: Neural Codec Language Models are Human Parity Zero-Shot Text to Speech Synthesizers (Chen et al., 2024)",
            FeatureCount = Settings.HiddenDim,
            Complexity = Settings.NumEncoderLayers + Settings.NumDecoderLayers,
        };
        metadata.AdditionalInfo["Architecture"] = "VALL-E 2";
        metadata.AdditionalInfo["GroupSize"] = Options2.GroupSize.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }
}

using AiDotNet.ActivationFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.FlowDiffusion;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The loss CosyVoice's flow matching minimizes.</summary>
public enum CosyVoiceFlowLoss
{
    /// <summary>
    /// The paper's eq. 10, <c>|ω_t − ν_t|</c>: the absolute error between the target vector field and the network's,
    /// averaged over the mel channels and frames.
    /// </summary>
    AbsoluteError,

    /// <summary>The official code's <c>compute_loss</c>: the squared error (<c>F.mse_loss</c>), averaged the same way.</summary>
    SquaredError,
}

/// <summary>How a CosyVoice 1 flow stretches its encoded tokens to the mel's frames.</summary>
internal enum CosyVoiceRegulator
{
    /// <summary>CosyVoice's <c>InterpolateRegulator</c>: linear interpolation; at inference the prompt's and the text's
    /// tokens are stretched separately (<see cref="CosyVoiceFlow{T}.StretchForInference"/>).</summary>
    Linear,

    /// <summary>GLM-4-Voice's fork: nearest-neighbour interpolation, and at inference the whole sequence (prompt and
    /// text) stretched at once to the frames of all its tokens.</summary>
    Nearest,

    /// <summary>Step-Audio's fork: linear interpolation, at inference the prompt and the text stretched separately with
    /// the text in one piece (no 20-token head and tail), and its frames <c>int(tokens / frame rate · sample rate /
    /// hop)</c> computed in double precision on a Python int.</summary>
    LinearWhole,
}

/// <summary>The sizes of CosyVoice 1's flow model (<c>MaskedDiffWithXvec</c> with its encoder, length regulator and
/// conditional flow matching), or with <c>Causal</c> CosyVoice 2's <c>CausalMaskedDiffWithXvec</c>: an
/// <see cref="UpsampleConformerEncoder{T}"/> giving each token <c>TokenMelRatio</c> mel frames (no length regulator) and the
/// causal estimator, both chunked in streaming by the encoder's <c>StaticChunkSize</c> (the estimator by that times the
/// ratio).</summary>
/// <param name="Regulator">The stretch of the encoded tokens (GLM-4-Voice's fork: <see cref="CosyVoiceRegulator.Nearest"/>).</param>
/// <param name="PromptRatio">The largest prompt prefix training draws, as a fraction of the mel's frames (CosyVoice 0.3,
/// GLM-4-Voice 0.8).</param>
/// <param name="CosineTrainingTime">Whether training's t ~ U(0, 1) goes through the cosine schedule 1 − cos(πt/2), as
/// GLM-4-Voice's <c>compute_loss</c> does.</param>
internal sealed record CosyVoiceFlowConfiguration(
    int VocabSize, int InputSize, int OutputSize, int SpeakerDim, double InputFrameRate, int SampleRate, int HopSize,
    WenetEncoderConfiguration Encoder, int RegulatorStages, int[] DecoderChannels, int DecoderHeads, int DecoderHeadDim,
    int DecoderBlocksPerLevel, int DecoderMidBlocks, double SigmaMin, double TrainingCfgRate, double InferenceCfgRate,
    int Timesteps, CosyVoiceFlowLoss Loss = CosyVoiceFlowLoss.AbsoluteError, bool Causal = false, int TokenMelRatio = 1,
    int Lookahead = 3, CosyVoiceRegulator Regulator = CosyVoiceRegulator.Linear, double PromptRatio = 0.3,
    bool CosineTrainingTime = false);

/// <summary>
/// CosyVoice 1's flow model (FunAudioLLM/CosyVoice <c>cosyvoice/flow</c>: <c>MaskedDiffWithXvec</c>,
/// <c>InterpolateRegulator</c>, <c>ConditionalCFM</c>): speech tokens to a mel spectrogram by conditional flow matching,
/// conditioned on a speaker x-vector and a prompt's mel.
/// </summary>
/// <remarks>
/// <para>Speech tokens are embedded and encoded by a Conformer (<see cref="WenetEncoder{T}"/>), projected to the mel's
/// channels and stretched to the mel's length by linear interpolation and the regulator's conv–GroupNorm–Mish stages.
/// The x-vector is L2-normalized and projected to the mel's channels.</para>
/// <para>Training (<c>compute_loss</c>): t ~ U(0, 1), noise z, the path y = (1 − (1 − σ)t) z + t x₁ and target
/// u = x₁ − (1 − σ) z; with probability <see cref="CosyVoiceFlowConfiguration.TrainingCfgRate"/> the encoder output,
/// the x-vector and the prompt condition are dropped; the loss is the masked mean square of the estimator's field
/// against u — by default the paper's absolute error (eq. 10), or the code's squared error
/// (<see cref="CosyVoiceFlowConfiguration.Loss"/>). Half the utterances condition on a random prefix of up to 30 % of
/// their own mel (<see cref="TrainingLoss"/> draws all of this).</para>
/// <para>Inference: Euler steps over the cosine schedule t = 1 − cos(πs/2), each estimate guided as
/// (1 + w)·v(cond) − w·v(uncond) with w = <see cref="CosyVoiceFlowConfiguration.InferenceCfgRate"/>.</para>
/// </remarks>
internal sealed class CosyVoiceFlow<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public CosyVoiceFlow(IEngine engine, List<LayerBase<T>> layers, CosyVoiceFlowConfiguration configuration, Random initialization)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T> { layers.Add(layer); return layer; }
        double Uniform(int fanIn) => (2 * initialization.NextDouble() - 1) / Math.Sqrt(fanIn);
        DenseLayer<T> Linear(int inputs, int outputs)
        {
            var layer = Add(new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>()));
            TorchParameters.Linear(engine, layer, inputs, outputs, Enumerable.Range(0, inputs * outputs).Select(_ => Uniform(inputs)).ToArray(),
                Enumerable.Range(0, outputs).Select(_ => Uniform(inputs)).ToArray());
            return layer;
        }
        InputEmbedding = Add(new TiedEmbeddingLayer<T>(c.VocabSize, c.InputSize));
        InputEmbedding.Reinitialize(() => Math.Sqrt(-2.0 * Math.Log(1.0 - initialization.NextDouble())) * Math.Cos(2.0 * Math.PI * initialization.NextDouble()));
        SpeakerProjection = Linear(c.SpeakerDim, c.OutputSize);
        if (c.Causal)
            UpsampleEncoder = new UpsampleConformerEncoder<T>(engine, layers, c.Encoder, c.Lookahead, initialization);
        else
            Encoder = new WenetEncoder<T>(engine, layers, c.Encoder, initialization);
        EncoderProjection = Linear(c.Encoder.OutputSize, c.OutputSize);
        for (int i = 0; i < (c.Causal ? 0 : c.RegulatorStages); i++)
        {
            var conv = Add(new NormedConv1DLayer<T>(c.OutputSize, c.OutputSize, 3, 1, 1, 1, 1, false, ConvolutionNormalization.None));
            conv.Reinitialize(() => Uniform(c.OutputSize * 3), bias: 0.0);
            RegulatorConvs.Add(conv);
            RegulatorNorms.Add(Add(new GroupNormalizationLayer<T>(1, c.OutputSize, 1e-5)));
        }
        if (!c.Causal)
        {
            RegulatorOut = Add(new NormedConv1DLayer<T>(c.OutputSize, c.OutputSize, 1, 1, 1, 1, 0, false, ConvolutionNormalization.None));
            RegulatorOut.Reinitialize(() => Uniform(c.OutputSize), bias: 0.0);
        }
        // in_channels: x, μ, the x-vector repeated over frames and the prompt condition, each OutputSize wide.
        Decoder = new MatchaDecoder<T>(engine, 4 * c.OutputSize, c.OutputSize, c.DecoderChannels, 0.0, c.DecoderHeadDim, c.DecoderHeads,
            c.DecoderMidBlocks, c.DecoderBlocksPerLevel, MatchaFeedForward.Gelu, biasMask: true, causal: c.Causal);
        layers.AddRange(Decoder.Layers);
    }

    public CosyVoiceFlowConfiguration Configuration { get; }
    public TiedEmbeddingLayer<T> InputEmbedding { get; }
    public DenseLayer<T> SpeakerProjection { get; }
    /// <summary>CosyVoice 1's Conformer encoder; null for the causal flow.</summary>
    public WenetEncoder<T>? Encoder { get; }

    /// <summary>CosyVoice 2's upsampling encoder; null for CosyVoice 1's flow.</summary>
    public UpsampleConformerEncoder<T>? UpsampleEncoder { get; }
    public DenseLayer<T> EncoderProjection { get; }
    public List<NormedConv1DLayer<T>> RegulatorConvs { get; } = new();
    public List<GroupNormalizationLayer<T>> RegulatorNorms { get; } = new();
    /// <summary>The regulator's output convolution; null for the causal flow, which has no regulator.</summary>
    public NormedConv1DLayer<T>? RegulatorOut { get; }
    public MatchaDecoder<T> Decoder { get; }

    /// <summary>Mel frames for a number of speech tokens: <c>int(tokens / frame rate · sample rate / hop)</c> in float32
    /// as torch computes it on an integer tensor, or <c>tokens · TokenMelRatio</c> for the causal flow.</summary>
    public int MelFrames(int tokens)
    {
        if (Configuration.Causal) return tokens * Configuration.TokenMelRatio;
        if (Configuration.Regulator == CosyVoiceRegulator.LinearWhole)
            return (int)(tokens / Configuration.InputFrameRate * Configuration.SampleRate / Configuration.HopSize);
        float seconds = (float)tokens / (float)Configuration.InputFrameRate;
        float samples = seconds * (float)Configuration.SampleRate;
        return (int)(samples / (float)Configuration.HopSize);
    }

    /// <summary>The projected x-vector <c>[OutputSize]</c>: L2-normalized, then the affine layer.</summary>
    public Tensor<T> ProjectSpeaker(Tensor<T> xvector)
    {
        double norm = 0;
        for (int i = 0; i < xvector.Length; i++) norm += Math.Pow(NumOps.ToDouble(xvector[i]), 2);
        norm = Math.Max(Math.Sqrt(norm), 1e-12);
        var normalized = _engine.TensorMultiplyScalar(_engine.Reshape(xvector, new[] { 1, xvector.Length }), NumOps.FromDouble(1.0 / norm));
        return _engine.Reshape(SpeakerProjection.Forward(normalized), new[] { Configuration.OutputSize });
    }

    /// <summary>The encoded, projected tokens <c>[tokens, OutputSize]</c> (the causal flow's <c>[2·tokens, OutputSize]</c>,
    /// chunked when <paramref name="streaming"/>).</summary>
    public Tensor<T> Encode(IReadOnlyList<int> tokens, bool training, Random random, bool streaming = false)
    {
        var ids = new Tensor<T>(new[] { tokens.Count });
        for (int i = 0; i < tokens.Count; i++) ids[i] = NumOps.FromDouble(Math.Max(0, tokens[i]));
        var embedded = InputEmbedding.Forward(ids);
        var encoded = UpsampleEncoder is not null
            ? UpsampleEncoder.Forward(embedded, streaming, training, random)
            : (Encoder ?? throw new InvalidOperationException("The flow has an encoder.")).Forward(embedded, training, random);
        return EncoderProjection.Forward(encoded);
    }

    // The estimator's attention chunk: the encoder's chunk times the token-to-mel ratio when streaming, else none.
    private int Chunk(bool streaming) => streaming ? Configuration.Encoder.StaticChunkSize * Configuration.TokenMelRatio : 0;

    /// <summary>
    /// PyTorch's <c>F.interpolate(x, size, mode="linear", align_corners=False)</c> over the frames of <c>[1, C, T]</c>:
    /// output frame i reads source position max(0, (i + 0.5)·T/size − 0.5), linearly between its neighbours.
    /// </summary>
    internal Tensor<T> Interpolate(Tensor<T> x, int size)
    {
        int channels = x.Shape[1], frames = x.Shape[2];
        var weights = new Tensor<T>(new[] { frames, size });
        double scale = frames / (double)size;
        for (int i = 0; i < size; i++)
        {
            double source = Math.Max(0, (i + 0.5) * scale - 0.5);
            int low = Math.Min((int)Math.Floor(source), frames - 1), high = Math.Min(low + 1, frames - 1);
            double lambda = source - low;
            weights[low, i] = NumOps.Add(weights[low, i], NumOps.FromDouble(1 - lambda));
            weights[high, i] = NumOps.Add(weights[high, i], NumOps.FromDouble(lambda));
        }
        return _engine.Reshape(_engine.TensorMatMul(_engine.Reshape(x, new[] { channels, frames }), weights), new[] { 1, channels, size });
    }

    /// <summary>
    /// PyTorch's <c>F.interpolate(x, size, mode="nearest")</c> over the frames of <c>[1, C, T]</c>: output frame i reads
    /// source frame min(⌊i · (T / size)⌋, T − 1), the scale and product in float32 as ATen's <c>nearest_idx</c> computes them.
    /// </summary>
    internal Tensor<T> InterpolateNearest(Tensor<T> x, int size)
    {
        int channels = x.Shape[1], frames = x.Shape[2];
        var index = new Tensor<int>(new[] { size });
        float scale = (float)frames / size;
        for (int i = 0; i < size; i++)
            index[i] = frames == size ? i : size == 2 * frames ? i >> 1 : Math.Min((int)Math.Floor((float)(i * scale)), frames - 1);
        return _engine.Reshape(_engine.TensorIndexSelect(_engine.Reshape(x, new[] { channels, frames }), index, 1), new[] { 1, channels, size });
    }

    /// <summary>The regulator's stages on <c>[1, C, frames]</c>: (conv, GroupNorm(1), Mish) per stage, then a 1×1 conv.</summary>
    internal Tensor<T> Regulate(Tensor<T> x)
    {
        for (int i = 0; i < RegulatorConvs.Count; i++)
        {
            var h = RegulatorConvs[i].Forward(x);
            int c = h.Shape[1], t = h.Shape[2];
            h = _engine.Reshape(RegulatorNorms[i].Forward(_engine.Reshape(h, new[] { 1, c, 1, t })), new[] { 1, c, t });
            x = _engine.Mish(h);
        }
        return (RegulatorOut ?? throw new InvalidOperationException("The causal flow has no length regulator.")).Forward(x);
    }

    /// <summary>μ <c>[1, OutputSize, frames]</c> of encoded tokens <c>[tokens, OutputSize]</c> stretched to
    /// <paramref name="frames"/> (<c>InterpolateRegulator.forward</c>).</summary>
    public Tensor<T> Stretch(Tensor<T> encoded, int frames)
    {
        var channels = _engine.Reshape(_engine.TensorTranspose(encoded), new[] { 1, encoded.Shape[1], encoded.Shape[0] });
        return Regulate(Configuration.Regulator == CosyVoiceRegulator.Nearest ? InterpolateNearest(channels, frames) : Interpolate(channels, frames));
    }

    /// <summary>The estimator on <c>[1, OutputSize, frames]</c> inputs: x, μ, the projected x-vector repeated over the
    /// frames, and the condition, stacked on the channels; returns <c>[1, OutputSize, frames]</c>.</summary>
    public Tensor<T> Field(Tensor<T> x, Tensor<T> mu, Tensor<T> speaker, Tensor<T> condition, double t, Tensor<T> mask,
        bool streaming = false)
    {
        int c = Configuration.OutputSize, frames = x.Shape[2];
        var repeated = _engine.TensorBroadcastTo(_engine.Reshape(speaker, new[] { 1, c, 1 }), new[] { 1, c, frames });
        var stacked = _engine.TensorConcatenate(new[] { x, mu, repeated, condition }, 1);
        var field = Decoder.EstimateChannels(stacked, t, mask, Chunk(streaming));                         // [frames, C]
        return _engine.Reshape(_engine.TensorTranspose(field), new[] { 1, c, frames });
    }

    /// <summary>
    /// The flow-matching loss (<c>ConditionalCFM.compute_loss</c>) for a target mel <c>[1, OutputSize, frames]</c> given
    /// μ, the projected x-vector and the condition, at time <paramref name="t"/> with noise <paramref name="noise"/>;
    /// <paramref name="dropConditions"/> zeroes μ, the x-vector and the condition (classifier-free guidance training).
    /// </summary>
    public Tensor<T> Loss(Tensor<T> target, Tensor<T> mu, Tensor<T> speaker, Tensor<T> condition, double t, Tensor<T> noise,
        bool dropConditions, bool streaming = false)
    {
        var c = Configuration;
        int frames = target.Shape[2];
        double sigma = c.SigmaMin;
        var y = _engine.TensorAdd(_engine.TensorMultiplyScalar(noise, NumOps.FromDouble(1 - (1 - sigma) * t)),
            _engine.TensorMultiplyScalar(target, NumOps.FromDouble(t)));
        var u = _engine.TensorSubtract(target, _engine.TensorMultiplyScalar(noise, NumOps.FromDouble(1 - sigma)));
        if (dropConditions)
        {
            mu = _engine.TensorMultiplyScalar(mu, NumOps.Zero);
            speaker = _engine.TensorMultiplyScalar(speaker, NumOps.Zero);
            condition = _engine.TensorMultiplyScalar(condition, NumOps.Zero);
        }
        var mask = Ones(frames);
        var difference = _engine.TensorSubtract(Field(y, mu, speaker, condition, t, mask, streaming), u);
        // Σ |·| or Σ (·)² over the frames, divided by (Σ mask · channels), as compute_loss normalizes.
        var pointwise = c.Loss == CosyVoiceFlowLoss.SquaredError ? _engine.TensorMultiply(difference, difference) : _engine.TensorAbs(difference);
        return _engine.TensorMultiplyScalar(_engine.ReduceSum(pointwise, new[] { 0, 1, 2 }, false),
            NumOps.FromDouble(1.0 / ((double)frames * c.OutputSize)));
    }

    /// <summary>
    /// <c>MaskedDiffWithXvec.forward</c> for one utterance with every random draw made from <paramref name="random"/>:
    /// half the time no prompt, otherwise a prefix of U{0, …, ⌊<see cref="CosyVoiceFlowConfiguration.PromptRatio"/> · frames⌋}
    /// frames of the target as the condition; t ~ U(0, 1) (through the cosine schedule when
    /// <see cref="CosyVoiceFlowConfiguration.CosineTrainingTime"/>); the noise; and the conditions dropped with probability <see cref="CosyVoiceFlowConfiguration.TrainingCfgRate"/>.
    /// </summary>
    /// <param name="tokens">The utterance's speech tokens.</param>
    /// <param name="mel">Its mel <c>[1, OutputSize, frames]</c>.</param>
    /// <param name="xvector">Its speaker x-vector.</param>
    /// <param name="training">Whether dropout is active.</param>
    /// <param name="random">The draws' source.</param>
    public Tensor<T> TrainingLoss(IReadOnlyList<int> tokens, Tensor<T> mel, Tensor<T> xvector, bool training, Random random)
    {
        var c = Configuration;
        int frames = mel.Shape[2];
        // CausalMaskedDiffWithXvec draws streaming first (unified training: half the steps chunked).
        bool streaming = c.Causal && random.NextDouble() < 0.5;
        Tensor<T> mu;
        if (c.Causal)
        {
            if (frames != MelFrames(tokens.Count))
                throw new ArgumentException($"The causal flow gives {MelFrames(tokens.Count)} mel frames for {tokens.Count} tokens; the mel has {frames}.", nameof(mel));
            var encoded = Encode(tokens, training, random, streaming);
            mu = _engine.Reshape(_engine.TensorTranspose(encoded), new[] { 1, c.OutputSize, frames });
        }
        else
        {
            mu = Stretch(Encode(tokens, training, random), frames);
        }
        int prompt = random.NextDouble() < 0.5 ? 0 : random.Next((int)(c.PromptRatio * frames) + 1);
        var condition = new Tensor<T>(new[] { 1, c.OutputSize, frames });
        for (int m = 0; m < c.OutputSize; m++)
            for (int f = 0; f < prompt; f++) condition[0, m, f] = mel[0, m, f];
        double t = random.NextDouble();
        if (c.CosineTrainingTime) t = 1 - Math.Cos(t * 0.5 * Math.PI);
        var noise = new Tensor<T>(new[] { 1, c.OutputSize, frames });
        for (int i = 0; i < noise.Length; i++)
            noise[i] = NumOps.FromDouble(Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble()));
        bool drop = !(random.NextDouble() > c.TrainingCfgRate);
        return Loss(mel, mu, ProjectSpeaker(xvector), condition, t, noise, drop, streaming);
    }

    /// <summary>
    /// μ for generation (<c>InterpolateRegulator.inference</c>): the prompt's and the text's encoded tokens
    /// <c>[tokens, OutputSize]</c> stretched separately to their frame counts — the text's first and last 20 tokens each
    /// to <c>int(20 / frame rate · sample rate / hop)</c> frames when it has more than 40 — then the regulator's stages;
    /// <c>[1, OutputSize, promptFrames + textFrames]</c>.
    /// </summary>
    public Tensor<T> StretchForInference(Tensor<T> promptEncoded, Tensor<T> textEncoded, int promptFrames, int textFrames)
    {
        var c = Configuration;
        Tensor<T> Channels(Tensor<T> encoded, int start, int count) =>
            _engine.Reshape(_engine.TensorTranspose(_engine.TensorSlice(encoded, new[] { start, 0 }, new[] { count, encoded.Shape[1] })),
                new[] { 1, encoded.Shape[1], count });
        int tokens = textEncoded.Shape[0];
        Tensor<T> text;
        if (tokens > 40 && c.Regulator != CosyVoiceRegulator.LinearWhole)
        {
            int edge = MelFrames(20);
            text = _engine.TensorConcatenate(new[]
            {
                Interpolate(Channels(textEncoded, 0, 20), edge),
                Interpolate(Channels(textEncoded, 20, tokens - 40), textFrames - 2 * edge),
                Interpolate(Channels(textEncoded, tokens - 20, 20), edge),
            }, 2);
        }
        else
        {
            text = Interpolate(Channels(textEncoded, 0, tokens), textFrames);
        }
        var x = promptEncoded.Shape[0] != 0
            ? _engine.TensorConcatenate(new[] { Interpolate(Channels(promptEncoded, 0, promptEncoded.Shape[0]), promptFrames), text }, 2)
            : text;
        return Regulate(x);
    }

    /// <summary>
    /// The mel <c>[1, OutputSize, frames]</c> of speech tokens after a prompt (<c>MaskedDiffWithXvec.inference</c>): the
    /// prompt's tokens and these encoded together, each part stretched to its frames (the prompt's to its mel's), the
    /// prompt's mel <c>[promptFrames, OutputSize]</c> as the condition, sampling from <paramref name="noise"/>
    /// <c>[1, OutputSize, promptFrames + frames]</c>, and the prompt's frames removed.
    /// </summary>
    public Tensor<T> Inference(IReadOnlyList<int> tokens, IReadOnlyList<int> promptTokens, Tensor<T>? promptMel, Tensor<T> xvector,
        Tensor<T> noise, bool streaming = false)
    {
        var c = Configuration;
        if (tokens.Count == 0) throw new ArgumentException("There are no speech tokens to decode.", nameof(tokens));
        if (promptTokens.Count > 0 && promptMel is null)
            throw new ArgumentException("A prompt's tokens need the prompt's mel.", nameof(promptMel));
        int promptFrames = promptTokens.Count > 0 && promptMel is not null ? promptMel.Shape[0] : 0;
        bool whole = c.Regulator == CosyVoiceRegulator.Nearest;
        // GLM-4-Voice sizes the whole sequence by all its tokens; the frames after the prompt's mel are the output.
        int textFrames = whole ? MelFrames(promptTokens.Count + tokens.Count) - promptFrames : MelFrames(tokens.Count);
        int total = promptFrames + textFrames;
        if (textFrames <= 0)
            throw new ArgumentException($"The prompt's mel has {promptFrames} frames; all {promptTokens.Count + tokens.Count} tokens give only {MelFrames(promptTokens.Count + tokens.Count)}.", nameof(promptMel));
        if (noise.Length != c.OutputSize * total)
            throw new ArgumentException($"The noise must be [1, {c.OutputSize}, {total}].", nameof(noise));

        var encoded = Encode(promptTokens.Concat(tokens).ToList(), training: false, new Random(0), streaming);
        int width = encoded.Shape[1];
        if (c.Causal)
        {
            // The upsampled states are the mel frames: the prompt's first, as many as its mel (the front end trims the
            // prompt to two frames per token), the text's after.
            if (encoded.Shape[0] != total)
                throw new ArgumentException($"The prompt's mel has {promptFrames} frames; its {promptTokens.Count} tokens give {MelFrames(promptTokens.Count)}.", nameof(promptMel));
            var causalCondition = new Tensor<T>(new[] { 1, c.OutputSize, total });
            if (promptMel is not null)
                for (int f = 0; f < promptFrames; f++)
                    for (int m = 0; m < c.OutputSize; m++) causalCondition[0, m, f] = promptMel[f, m];
            var causalMu = _engine.Reshape(_engine.TensorTranspose(encoded), new[] { 1, c.OutputSize, total });
            var causalMel = Sample(causalMu, ProjectSpeaker(xvector), causalCondition, _engine.Reshape(noise, new[] { 1, c.OutputSize, total }), streaming);
            return _engine.TensorSlice(causalMel, new[] { 0, 0, promptFrames }, new[] { 1, c.OutputSize, textFrames });
        }
        if (whole)
        {
            var wholeCondition = new Tensor<T>(new[] { 1, c.OutputSize, total });
            if (promptMel is not null)
                for (int f = 0; f < promptFrames; f++)
                    for (int m = 0; m < c.OutputSize; m++) wholeCondition[0, m, f] = promptMel[f, m];
            var wholeMel = Sample(Stretch(encoded, total), ProjectSpeaker(xvector), wholeCondition, _engine.Reshape(noise, new[] { 1, c.OutputSize, total }));
            return _engine.TensorSlice(wholeMel, new[] { 0, 0, promptFrames }, new[] { 1, c.OutputSize, textFrames });
        }
        var promptEncoded = promptTokens.Count > 0
            ? _engine.TensorSlice(encoded, new[] { 0, 0 }, new[] { promptTokens.Count, width })
            : new Tensor<T>(new[] { 0, width });
        var textEncoded = _engine.TensorSlice(encoded, new[] { promptTokens.Count, 0 }, new[] { tokens.Count, width });
        var mu = StretchForInference(promptEncoded, textEncoded, promptFrames, textFrames);
        var condition = new Tensor<T>(new[] { 1, c.OutputSize, total });
        if (promptMel is not null)
            for (int f = 0; f < promptFrames; f++)
                for (int m = 0; m < c.OutputSize; m++) condition[0, m, f] = promptMel[f, m];
        var mel = Sample(mu, ProjectSpeaker(xvector), condition, _engine.Reshape(noise, new[] { 1, c.OutputSize, total }));
        return _engine.TensorSlice(mel, new[] { 0, 0, promptFrames }, new[] { 1, c.OutputSize, textFrames });
    }

    /// <summary>
    /// Samples a mel <c>[1, OutputSize, frames]</c> (<c>ConditionalCFM.forward</c> with <c>solve_euler</c>): from
    /// <paramref name="noise"/>, Euler steps over the cosine schedule with classifier-free guidance.
    /// </summary>
    public Tensor<T> Sample(Tensor<T> mu, Tensor<T> speaker, Tensor<T> condition, Tensor<T> noise, bool streaming = false)
    {
        var c = Configuration;
        int frames = mu.Shape[2], steps = c.Timesteps;
        var mask = Ones(frames);
        var span = Enumerable.Range(0, steps + 1).Select(i => 1 - Math.Cos(i / (double)steps * 0.5 * Math.PI)).ToArray();
        var x = noise;
        var zeroMu = _engine.TensorMultiplyScalar(mu, NumOps.Zero);
        var zeroSpeaker = _engine.TensorMultiplyScalar(speaker, NumOps.Zero);
        var zeroCondition = _engine.TensorMultiplyScalar(condition, NumOps.Zero);
        double t = span[0], dt = span[1] - span[0];
        for (int step = 1; step <= steps; step++)
        {
            var guided = Field(x, mu, speaker, condition, t, mask, streaming);
            var free = Field(x, zeroMu, zeroSpeaker, zeroCondition, t, mask, streaming);
            var field = _engine.TensorSubtract(_engine.TensorMultiplyScalar(guided, NumOps.FromDouble(1 + c.InferenceCfgRate)),
                _engine.TensorMultiplyScalar(free, NumOps.FromDouble(c.InferenceCfgRate)));
            x = _engine.TensorAdd(x, _engine.TensorMultiplyScalar(field, NumOps.FromDouble(dt)));
            t += dt;
            if (step < steps) dt = span[step + 1] - t;
        }
        return x;
    }

    private static Tensor<T> Ones(int frames)
    {
        var mask = new Tensor<T>(new[] { frames });
        for (int i = 0; i < frames; i++) mask[i] = NumOps.One;
        return mask;
    }

    /// <summary>Loads <c>flow.pt</c>'s state: <c>input_embedding</c>, <c>spk_embed_affine_layer</c>, <c>encoder.*</c>,
    /// <c>encoder_proj</c>, <c>length_regulator.model.*</c> and <c>decoder.estimator.*</c>.</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read, string prefix)
    {
        var c = Configuration;
        InputEmbedding.LoadTable(read(prefix + "input_embedding.weight", new[] { c.VocabSize, c.InputSize }));
        TorchParameters.Linear(_engine, SpeakerProjection, c.SpeakerDim, c.OutputSize,
            read(prefix + "spk_embed_affine_layer.weight", new[] { c.OutputSize, c.SpeakerDim }), read(prefix + "spk_embed_affine_layer.bias", new[] { c.OutputSize }));
        if (UpsampleEncoder is not null) UpsampleEncoder.LoadTorchWeights(read, prefix + "encoder.");
        else (Encoder ?? throw new InvalidOperationException("The flow has an encoder.")).LoadTorchWeights(read, prefix + "encoder.");
        TorchParameters.Linear(_engine, EncoderProjection, c.Encoder.OutputSize, c.OutputSize,
            read(prefix + "encoder_proj.weight", new[] { c.OutputSize, c.Encoder.OutputSize }), read(prefix + "encoder_proj.bias", new[] { c.OutputSize }));
        for (int i = 0; i < RegulatorConvs.Count; i++)
        {
            string p = $"{prefix}length_regulator.model.{3 * i}";
            RegulatorConvs[i].LoadTorchWeights(read(p + ".weight", new[] { c.OutputSize, c.OutputSize, 3 }), null, read(p + ".bias", new[] { c.OutputSize }));
            string n = $"{prefix}length_regulator.model.{3 * i + 1}";
            RegulatorNorms[i].LoadAffine(read(n + ".weight", new[] { c.OutputSize }), read(n + ".bias", new[] { c.OutputSize }));
        }
        if (RegulatorOut is not null)
        {
            string o = $"{prefix}length_regulator.model.{3 * RegulatorConvs.Count}";
            RegulatorOut.LoadTorchWeights(read(o + ".weight", new[] { c.OutputSize, c.OutputSize, 1 }), null, read(o + ".bias", new[] { c.OutputSize }));
        }
        Decoder.LoadTorchWeights(read, prefix + "decoder.estimator.");
    }
}

/// <summary>The sizes of CosyVoice 1's language model (<c>TransformerLM</c>).</summary>
internal sealed record CosyVoiceLmConfiguration(
    int TextVocabulary, int SpeechVocabulary, int TextEncoderInputSize, int LlmSize, int SpeakerDim,
    WenetEncoderConfiguration TextEncoder, WenetEncoderConfiguration Llm, double TopP = 0.8, int TopK = 25, int RepetitionWindow = 10,
    double RepetitionRatio = 0.1, double MinTokenTextRatio = 2, double MaxTokenTextRatio = 20);

/// <summary>
/// CosyVoice 1's text-to-speech-token language model (FunAudioLLM/CosyVoice <c>cosyvoice/llm/llm.py</c>
/// <c>TransformerLM</c>): a causal Conformer text encoder, then a causal Transformer over
/// <c>[sos, x-vector, text…, task_id, speech…]</c> whose head predicts the next speech token or the end.
/// </summary>
/// <remarks>
/// <para>Training: the cross-entropy of every speech token and the end, each predicted from the position before it
/// (the first 2 + |text| positions are ignored), averaged over them (<c>LabelSmoothingLoss</c> with no smoothing,
/// normalized by length).</para>
/// <para>Inference: the end is forbidden until <c>2·|text|</c> tokens and generation stops at <c>20·|text|</c>; each
/// token is drawn by repetition-aware sampling (VALL-E 2): from the most likely tokens up to cumulative probability
/// 0.8 and at most 25, unless the draw already occurs at least <c>10 · 0.1</c> times among the last 10 tokens, when it
/// is excluded and the token is drawn from the whole distribution.</para>
/// </remarks>
internal sealed class CosyVoiceLm<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public CosyVoiceLm(IEngine engine, List<LayerBase<T>> layers, CosyVoiceLmConfiguration configuration, Random initialization)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T> { layers.Add(layer); return layer; }
        double Uniform(int fanIn) => (2 * initialization.NextDouble() - 1) / Math.Sqrt(fanIn);
        Func<double> normal = () => Math.Sqrt(-2.0 * Math.Log(1.0 - initialization.NextDouble())) * Math.Cos(2.0 * Math.PI * initialization.NextDouble());
        DenseLayer<T> Linear(int inputs, int outputs)
        {
            var layer = Add(new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>()));
            TorchParameters.Linear(engine, layer, inputs, outputs, Enumerable.Range(0, inputs * outputs).Select(_ => Uniform(inputs)).ToArray(),
                Enumerable.Range(0, outputs).Select(_ => Uniform(inputs)).ToArray());
            return layer;
        }
        TiedEmbeddingLayer<T> Table(int rows, int width)
        {
            var table = Add(new TiedEmbeddingLayer<T>(rows, width));
            table.Reinitialize(normal);
            return table;
        }
        TextEmbedding = Table(c.TextVocabulary, c.TextEncoderInputSize);
        TextEncoder = new WenetEncoder<T>(engine, layers, c.TextEncoder, initialization);
        TextAffine = Linear(c.TextEncoder.OutputSize, c.LlmSize);
        LlmEmbedding = Table(2, c.LlmSize);
        Llm = new WenetEncoder<T>(engine, layers, c.Llm, initialization);
        Head = Linear(c.Llm.OutputSize, c.SpeechVocabulary + 1);
        SpeechEmbedding = Table(c.SpeechVocabulary, c.LlmSize);
        SpeakerAffine = Linear(c.SpeakerDim, c.LlmSize);
    }

    public CosyVoiceLmConfiguration Configuration { get; }
    public TiedEmbeddingLayer<T> TextEmbedding { get; }
    public WenetEncoder<T> TextEncoder { get; }
    public DenseLayer<T> TextAffine { get; }
    public TiedEmbeddingLayer<T> LlmEmbedding { get; }
    public WenetEncoder<T> Llm { get; }
    public DenseLayer<T> Head { get; }
    public TiedEmbeddingLayer<T> SpeechEmbedding { get; }
    public DenseLayer<T> SpeakerAffine { get; }

    /// <summary>The end-of-speech id (the speech vocabulary's size).</summary>
    public int EndToken => Configuration.SpeechVocabulary;

    private Tensor<T> Ids(IReadOnlyList<int> ids)
    {
        var tensor = new Tensor<T>(new[] { ids.Count });
        for (int i = 0; i < ids.Count; i++) tensor[i] = NumOps.FromDouble(ids[i]);
        return tensor;
    }

    /// <summary>The LM's input <c>[2 + |text| + 1 + |speech|, LlmSize]</c>: sos, the projected x-vector (none when null),
    /// the encoded text, task_id, the speech tokens' embeddings.</summary>
    internal Tensor<T> Input(IReadOnlyList<int> text, IReadOnlyList<int> speech, Tensor<T>? xvector, bool training, Random random)
    {
        var c = Configuration;
        var encoded = TextAffine.Forward(TextEncoder.Forward(TextEmbedding.Forward(Ids(text)), training, random));
        var parts = new List<Tensor<T>> { LlmEmbedding.Forward(Ids(new[] { 0 })) };
        if (xvector is not null)
        {
            double norm = 0;
            for (int i = 0; i < xvector.Length; i++) norm += Math.Pow(NumOps.ToDouble(xvector[i]), 2);
            norm = Math.Max(Math.Sqrt(norm), 1e-12);
            parts.Add(SpeakerAffine.Forward(_engine.TensorMultiplyScalar(_engine.Reshape(xvector, new[] { 1, c.SpeakerDim }), NumOps.FromDouble(1.0 / norm))));
        }
        parts.Add(encoded);
        parts.Add(LlmEmbedding.Forward(Ids(new[] { 1 })));
        if (speech.Count > 0) parts.Add(SpeechEmbedding.Forward(Ids(speech)));
        return _engine.TensorConcatenate(parts.ToArray(), 0);
    }

    /// <summary>The head's log-probabilities <c>[L, SpeechVocabulary + 1]</c> of an input.</summary>
    internal Tensor<T> LogProbabilities(Tensor<T> input, bool training, Random random) =>
        _engine.TensorLogSoftmax(Head.Forward(Llm.Forward(input, training, random)), axis: -1);

    /// <summary>The training loss: the mean cross-entropy of the speech tokens and the end, each from the position before.</summary>
    public Tensor<T> Loss(IReadOnlyList<int> text, IReadOnlyList<int> speech, Tensor<T>? xvector, bool training, Random random)
    {
        var logp = LogProbabilities(Input(text, speech, xvector, training, random), training, random);
        int first = (xvector is null ? 1 : 2) + text.Count, targets = speech.Count + 1;          // task_id predicts speech[0]
        var picks = new Tensor<T>(logp._shape);
        for (int i = 0; i < targets; i++)
            picks[first + i, i < speech.Count ? speech[i] : EndToken] = NumOps.FromDouble(-1.0 / targets);
        return _engine.ReduceSum(_engine.TensorMultiply(logp, picks), new[] { 0, 1 }, false);
    }

    /// <summary>
    /// Repetition-aware sampling (<c>ras_sampling</c>): the nucleus draw, replaced by a draw from the whole distribution
    /// (without it) when it already occurs at least <c>window · ratio</c> times among the last <c>window</c> tokens.
    /// </summary>
    internal static int SampleRepetitionAware(double[] logProbabilities, IReadOnlyList<int> decoded, CosyVoiceLmConfiguration c, Random random)
        => SampleRepetitionAware(logProbabilities, decoded, c.TopP, c.TopK, c.RepetitionWindow, c.RepetitionRatio, random);

    /// <inheritdoc cref="SampleRepetitionAware(double[], IReadOnlyList{int}, CosyVoiceLmConfiguration, Random)"/>
    internal static int SampleRepetitionAware(double[] logProbabilities, IReadOnlyList<int> decoded, double topP, int topK,
        int repetitionWindow, double repetitionRatio, Random random)
    {
        var c = (TopP: topP, TopK: topK, RepetitionWindow: repetitionWindow, RepetitionRatio: repetitionRatio);
        var probabilities = logProbabilities.Select(Math.Exp).ToArray();
        var order = Enumerable.Range(0, probabilities.Length).OrderByDescending(v => probabilities[v]).ThenBy(v => v).ToList();
        var kept = new List<int>();
        double cumulative = 0;
        foreach (int v in order)
        {
            if (cumulative >= c.TopP || kept.Count >= c.TopK) break;
            cumulative += probabilities[v];
            kept.Add(v);
        }
        int draw = Draw(kept, kept.Select(v => probabilities[v]).ToArray(), random);
        int repeats = decoded.Skip(Math.Max(0, decoded.Count - c.RepetitionWindow)).Count(token => token == draw);
        if (repeats >= c.RepetitionWindow * c.RepetitionRatio)
        {
            probabilities[draw] = 0;
            var all = Enumerable.Range(0, probabilities.Length).ToList();
            draw = Draw(all, probabilities, random);
        }
        return draw;
    }

    private static int Draw(IReadOnlyList<int> ids, double[] weights, Random random)
    {
        double total = weights.Sum(), target = random.NextDouble() * total;
        for (int i = 0; i < ids.Count; i++)
        {
            target -= weights[i];
            if (target <= 0 && weights[i] > 0) return ids[i];
        }
        return ids[Array.FindLastIndex(weights, w => w > 0)];
    }

    /// <summary>
    /// Generates speech tokens (<c>TransformerLM.inference</c>) for <paramref name="text"/> after an optional prompt:
    /// the prompt's text precedes the text and its speech tokens follow task_id.
    /// </summary>
    public List<int> Generate(IReadOnlyList<int> text, IReadOnlyList<int> promptText, IReadOnlyList<int> promptSpeech, Tensor<T>? xvector,
        Random random)
    {
        var c = Configuration;
        var allText = promptText.Concat(text).ToList();
        int minLength = (int)(text.Count * c.MinTokenTextRatio), maxLength = (int)(text.Count * c.MaxTokenTextRatio);
        var generated = new List<int>();
        for (int i = 0; i < maxLength; i++)
        {
            var speech = promptSpeech.Concat(generated).ToList();
            var logp = LogProbabilities(Input(allText, speech, xvector, false, random), false, random);
            int last = logp.Shape[0] - 1, vocabulary = logp.Shape[1];
            var row = new double[vocabulary];
            for (int v = 0; v < vocabulary; v++) row[v] = NumOps.ToDouble(logp[last, v]);
            if (i < minLength) row[EndToken] = double.NegativeInfinity;
            int next = SampleRepetitionAware(row, generated, c, random);
            if (next == EndToken) break;
            generated.Add(next);
        }
        return generated;
    }

    /// <summary>Loads <c>llm.pt</c>'s state: <c>text_embedding</c>, <c>text_encoder.*</c>, <c>text_encoder_affine_layer</c>,
    /// <c>llm_embedding</c>, <c>llm.*</c>, <c>llm_decoder</c>, <c>speech_embedding</c>, <c>spk_embed_affine_layer</c>.</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read, string prefix)
    {
        var c = Configuration;
        void Dense(DenseLayer<T> layer, string name, int inputs, int outputs) =>
            TorchParameters.Linear(_engine, layer, inputs, outputs, read(name + ".weight", new[] { outputs, inputs }), read(name + ".bias", new[] { outputs }));
        TextEmbedding.LoadTable(read(prefix + "text_embedding.weight", new[] { c.TextVocabulary, c.TextEncoderInputSize }));
        TextEncoder.LoadTorchWeights(read, prefix + "text_encoder.");
        Dense(TextAffine, prefix + "text_encoder_affine_layer", c.TextEncoder.OutputSize, c.LlmSize);
        LlmEmbedding.LoadTable(read(prefix + "llm_embedding.weight", new[] { 2, c.LlmSize }));
        Llm.LoadTorchWeights(read, prefix + "llm.");
        Dense(Head, prefix + "llm_decoder", c.Llm.OutputSize, c.SpeechVocabulary + 1);
        SpeechEmbedding.LoadTable(read(prefix + "speech_embedding.weight", new[] { c.SpeechVocabulary, c.LlmSize }));
        Dense(SpeakerAffine, prefix + "spk_embed_affine_layer", c.SpeakerDim, c.LlmSize);
    }
}
using System.Collections.Generic;
using System.Linq;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.FlowDiffusion;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The loss CosyVoice's flow matching minimizes.</summary>
public enum CosyVoiceFlowLoss
{
    /// <summary>
    /// The paper's eq. 10: the absolute error between the target vector field ω_t and the network's ν_t, averaged over
    /// the mel bins and frames.
    /// </summary>
    AbsoluteError,

    /// <summary>The official code's <c>compute_loss</c>: the squared error, averaged the same way.</summary>
    SquaredError,
}

/// <summary>The sizes and settings of CosyVoice's token-to-mel flow-matching model.</summary>
internal sealed record CosyVoiceFlowConfiguration(
    int Vocabulary, int InputSize, int MelChannels, int SpeakerEmbeddingDim, WenetEncoderConfiguration Encoder,
    int RegulatorBlocks, int[] EstimatorChannels, int EstimatorHeadDim, int EstimatorHeads, int EstimatorBlocks,
    int EstimatorMidBlocks, double EstimatorDropout, double InputFrameRate, int SampleRate, int HopSize,
    double SigmaMin = 1e-6, double TrainingCfgRate = 0.2, double InferenceCfgRate = 0.7, int Timesteps = 10,
    CosyVoiceFlowLoss Loss = CosyVoiceFlowLoss.AbsoluteError);

/// <summary>
/// CosyVoice's token-to-mel model (Du et al. 2024, §2.3; the reference's <c>MaskedDiffWithXvec</c> with
/// <c>ConditionalCFM</c>): the speech tokens are embedded and encoded by a WeNet encoder, projected to the mel width and
/// interpolated to the mel length (<c>InterpolateRegulator</c>: four conv–GroupNorm–Mish blocks and a 1×1 convolution);
/// an optimal-transport conditional flow-matching U-Net (<see cref="MatchaDecoder{T}"/> as CosyVoice configures it) then
/// maps Gaussian noise to the mel spectrogram, conditioned on those features μ, the projected x-vector v and the masked
/// prompt mel X̃₁.
/// </summary>
/// <remarks>
/// Training draws t ~ U(0, 1) and noise X₀, keeps a random prefix of up to 30 % of the target mel as the prompt (half the
/// time none), drops all conditions with probability 0.2 for classifier-free guidance, and regresses ν_t on
/// ω_t = X₁ − (1 − σ)X₀ at φ_t = (1 − (1 − σ)t)X₀ + tX₁. Generation integrates the ODE with Euler steps on the cosine
/// schedule t ← 1 − cos(½πt) (eq. 13) and guidance (1 + β)ν(Ψ) − βν(∅), β = 0.7 (eq. 14).
/// </remarks>
internal sealed class CosyVoiceFlow<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public CosyVoiceFlow(IEngine engine, List<LayerBase<T>> layers, CosyVoiceFlowConfiguration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        InputEmbedding = WenetEncoderBlock<T>.Own(layers, new TiedEmbeddingLayer<T>(c.Vocabulary, c.InputSize));
        SpeakerAffine = WenetEncoderBlock<T>.Own(layers, new DenseLayer<T>(c.MelChannels));
        Encoder = new WenetEncoder<T>(engine, layers, c.Encoder);
        EncoderProjection = WenetEncoderBlock<T>.Own(layers, new DenseLayer<T>(c.MelChannels));
        for (int i = 0; i < c.RegulatorBlocks; i++)
        {
            RegulatorConvs.Add(WenetEncoderBlock<T>.Own(layers, new KaimingConv1DLayer<T>(c.MelChannels, c.MelChannels, 3, 1, 1, false)));
            RegulatorNorms.Add(WenetEncoderBlock<T>.Own(layers, new GroupNormalizationLayer<T>(1, c.MelChannels, 1e-5)));
        }
        RegulatorOutput = WenetEncoderBlock<T>.Own(layers, new KaimingConv1DLayer<T>(c.MelChannels, c.MelChannels, 1, 1, 0, false));
        Estimator = new MatchaDecoder<T>(engine, 4 * c.MelChannels, c.MelChannels, c.EstimatorChannels, c.EstimatorDropout,
            c.EstimatorHeadDim, c.EstimatorHeads, c.EstimatorMidBlocks, c.EstimatorBlocks, geluFeedForward: true);
        layers.AddRange(Estimator.Layers);
    }

    public CosyVoiceFlowConfiguration Configuration { get; }
    public TiedEmbeddingLayer<T> InputEmbedding { get; }
    public DenseLayer<T> SpeakerAffine { get; }
    public WenetEncoder<T> Encoder { get; }
    public DenseLayer<T> EncoderProjection { get; }
    public List<KaimingConv1DLayer<T>> RegulatorConvs { get; } = new();
    public List<GroupNormalizationLayer<T>> RegulatorNorms { get; } = new();
    public KaimingConv1DLayer<T> RegulatorOutput { get; }
    public MatchaDecoder<T> Estimator { get; }

    /// <summary>Mel frames for <paramref name="tokens"/> speech tokens: <c>int(tokens / rate · sample rate / hop)</c>.</summary>
    public int MelFrames(int tokens) => (int)(tokens / Configuration.InputFrameRate * Configuration.SampleRate / Configuration.HopSize);

    private Tensor<T> Ids(IReadOnlyList<int> ids)
    {
        var tensor = new Tensor<T>(new[] { ids.Count });
        for (int i = 0; i < ids.Count; i++) tensor[i] = NumOps.FromDouble(ids[i]);
        return tensor;
    }

    // The encoded tokens projected to the mel width, channels-first [mel, tokens].
    private Tensor<T> Encode(IReadOnlyList<int> tokens, bool training, Random random)
    {
        var h = EncoderProjection.Forward(Encoder.Forward(InputEmbedding.Forward(Ids(tokens)), training, random));
        return _engine.TensorTranspose(h);
    }

    /// <summary>The projected L2-normalized x-vector <c>[1, mel]</c>.</summary>
    private Tensor<T> Speaker(IReadOnlyList<double> xvector)
    {
        if (xvector.Count != Configuration.SpeakerEmbeddingDim)
            throw new ArgumentException($"Expected a {Configuration.SpeakerEmbeddingDim}-dimensional speaker embedding, got {xvector.Count}.");
        double divisor = Math.Max(Math.Sqrt(xvector.Sum(x => x * x)), 1e-12);
        var v = new Tensor<T>(new[] { 1, xvector.Count });
        for (int i = 0; i < xvector.Count; i++) v[0, i] = NumOps.FromDouble(xvector[i] / divisor);
        return SpeakerAffine.Forward(v);
    }

    /// <summary>
    /// <c>F.interpolate(mode='linear', align_corners=False)</c> along time, <c>[C, n]</c> → <c>[C, m]</c>, as a product
    /// with the fixed interpolation matrix.
    /// </summary>
    private Tensor<T> Interpolate(Tensor<T> x, int size)
    {
        int n = x.Shape[1];
        var w = new Tensor<T>(new[] { n, size });
        double scale = (double)n / size;
        for (int j = 0; j < size; j++)
        {
            double source = Math.Max(scale * (j + 0.5) - 0.5, 0.0);
            int i0 = (int)source;
            int step = i0 < n - 1 ? 1 : 0;
            double lambda1 = source - i0, lambda0 = 1 - lambda1;
            w[i0, j] = NumOps.Add(w[i0, j], NumOps.FromDouble(lambda0));
            w[i0 + step, j] = NumOps.Add(w[i0 + step, j], NumOps.FromDouble(lambda1));
        }
        return _engine.TensorMatMul(x, w);
    }

    // InterpolateRegulator's convolution stack on [mel, frames].
    private Tensor<T> Regulate(Tensor<T> x)
    {
        int channels = x.Shape[0], frames = x.Shape[1];
        var h = _engine.Reshape(x, new[] { 1, channels, frames });
        for (int i = 0; i < RegulatorConvs.Count; i++)
        {
            h = RegulatorConvs[i].Forward(h);
            h = _engine.Reshape(RegulatorNorms[i].Forward(_engine.Reshape(h, new[] { 1, channels, 1, frames })), new[] { 1, channels, frames });
            h = _engine.Mish(h);
        }
        return _engine.Reshape(RegulatorOutput.Forward(h), new[] { channels, frames });
    }

    // The regulator's inference path: prompt and text features interpolated separately, the text's first and last
    // 20 tokens to fixed lengths when it has more than 40.
    private Tensor<T> RegulateForInference(Tensor<T> prompt, Tensor<T> text, int promptFrames, int textFrames)
    {
        var c = Configuration;
        int tokens = text.Shape[1];
        Tensor<T> Columns(Tensor<T> x, int start, int count) => _engine.TensorSlice(x, new[] { 0, start }, new[] { x.Shape[0], count });
        Tensor<T> textPart;
        if (tokens > 40)
        {
            int edge = (int)(20 / c.InputFrameRate * c.SampleRate / c.HopSize);
            textPart = _engine.TensorConcatenate(new[]
            {
                Interpolate(Columns(text, 0, 20), edge),
                Interpolate(Columns(text, 20, tokens - 40), textFrames - 2 * edge),
                Interpolate(Columns(text, tokens - 20, 20), edge),
            }, 1);
        }
        else
        {
            textPart = Interpolate(text, textFrames);
        }
        var x = prompt.Shape[1] != 0 ? _engine.TensorConcatenate(new[] { Interpolate(prompt, promptFrames), textPart }, 1) : textPart;
        return Regulate(x);
    }

    // The estimator's input [1, 4·mel, frames]: x, μ, the speaker embedding repeated over time, and the prompt mel.
    private Tensor<T> EstimatorInput(Tensor<T> x, Tensor<T> mu, Tensor<T> speaker, Tensor<T> condition)
    {
        int mel = Configuration.MelChannels, frames = x.Shape[1];
        var repeated = _engine.TensorTile(_engine.Reshape(speaker, new[] { mel, 1 }), new[] { 1, frames });
        return _engine.Reshape(_engine.TensorConcatenate(new[] { x, mu, repeated, condition }, 0), new[] { 1, 4 * mel, frames });
    }

    private static Tensor<T> Ones(int frames)
    {
        var mask = new Tensor<T>(new[] { frames });
        for (int i = 0; i < frames; i++) mask[i] = NumOps.One;
        return mask;
    }

    /// <summary>
    /// The flow-matching loss for one utterance with its random draws given: speech <paramref name="tokens"/>, the target
    /// mel <paramref name="mel"/> <c>[frames, mel]</c>, the x-vector, the prompt length kept as the condition, t, the
    /// noise X₀ <c>[mel, frames]</c> and whether the conditions are kept (false: dropped for guidance).
    /// </summary>
    public Tensor<T> Loss(IReadOnlyList<int> tokens, Tensor<T> mel, IReadOnlyList<double> xvector, int promptFrames,
        double t, Tensor<T> noise, bool keepConditions, bool training, Random random)
    {
        var c = Configuration;
        int frames = mel.Shape[0];
        var target = _engine.TensorTranspose(mel);                                                            // [mel, frames]
        var mu = Regulate(Interpolate(Encode(tokens, training, random), frames));
        var speaker = Speaker(xvector);
        var condition = new Tensor<T>(new[] { c.MelChannels, frames });
        for (int m = 0; m < c.MelChannels; m++)
            for (int f = 0; f < promptFrames; f++) condition[m, f] = mel[f, m];
        if (!keepConditions)
        {
            mu = _engine.TensorMultiplyScalar(mu, NumOps.Zero);
            speaker = _engine.TensorMultiplyScalar(speaker, NumOps.Zero);
            condition = new Tensor<T>(condition._shape);
        }
        double s = 1 - (1 - c.SigmaMin) * t;
        var y = _engine.TensorAdd(_engine.TensorMultiplyScalar(noise, NumOps.FromDouble(s)), _engine.TensorMultiplyScalar(target, NumOps.FromDouble(t)));
        var u = _engine.TensorSubtract(target, _engine.TensorMultiplyScalar(noise, NumOps.FromDouble(1 - c.SigmaMin)));
        var prediction = _engine.Reshape(Estimator.Estimate(EstimatorInput(y, mu, speaker, condition), t, Ones(frames)), new[] { c.MelChannels, frames });
        var error = _engine.TensorSubtract(prediction, u);
        var pointwise = c.Loss == CosyVoiceFlowLoss.SquaredError ? _engine.TensorMultiply(error, error) : _engine.TensorAbs(error);
        return _engine.TensorMultiplyScalar(_engine.ReduceSum(pointwise, new[] { 0, 1 }, keepDims: false),
            NumOps.FromDouble(1.0 / (frames * c.MelChannels)));
    }

    /// <summary>
    /// Draws the flow-matching loss's random terms as the reference does and returns the loss: the prompt (half the time
    /// none, else a prefix of U{0, …, ⌊0.3·frames⌋} frames), t ~ U(0, 1), X₀ ~ N(0, I) and the guidance drop.
    /// </summary>
    public Tensor<T> Loss(IReadOnlyList<int> tokens, Tensor<T> mel, IReadOnlyList<double> xvector, bool training, Random random)
    {
        int frames = mel.Shape[0];
        int prompt = random.NextDouble() < 0.5 ? 0 : random.Next((int)(0.3 * frames) + 1);
        double t = random.NextDouble();
        var noise = Gaussian(new[] { Configuration.MelChannels, frames }, random, 1.0);
        bool keep = random.NextDouble() > Configuration.TrainingCfgRate;
        return Loss(tokens, mel, xvector, prompt, t, noise, keep, training, random);
    }

    private static Tensor<T> Gaussian(int[] shape, Random random, double scale)
    {
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            tensor[i] = NumOps.FromDouble(scale * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
        }
        return tensor;
    }

    /// <summary>
    /// Generates the mel <c>[frames, mel]</c> for <paramref name="tokens"/> after a prompt (its speech tokens and mel
    /// <c>[promptFrames, mel]</c>, either may be empty) in <paramref name="speaker"/>'s voice, from the noise X₀
    /// <c>[mel, prompt + generated frames]</c> (drawn when null).
    /// </summary>
    public Tensor<T> Generate(IReadOnlyList<int> promptTokens, IReadOnlyList<int> tokens, Tensor<T>? promptMel,
        IReadOnlyList<double> xvector, Random random, Tensor<T>? noise = null, double temperature = 1.0)
    {
        var c = Configuration;
        using var _ = new NoGradScope<T>();
        var encoded = Encode(promptTokens.Concat(tokens).ToList(), false, random);
        Tensor<T> Columns(Tensor<T> x, int start, int count) => _engine.TensorSlice(x, new[] { 0, start }, new[] { x.Shape[0], count });
        int promptFrames = promptMel?.Shape[0] ?? 0, textFrames = MelFrames(tokens.Count);
        int frames = promptFrames + textFrames;
        var mu = RegulateForInference(Columns(encoded, 0, promptTokens.Count), Columns(encoded, promptTokens.Count, tokens.Count),
            promptFrames, textFrames);
        var speaker = Speaker(xvector);
        var condition = new Tensor<T>(new[] { c.MelChannels, frames });
        for (int m = 0; m < c.MelChannels; m++)
            for (int f = 0; f < promptFrames; f++) condition[m, f] = promptMel![f, m];
        var x = noise ?? Gaussian(new[] { c.MelChannels, frames }, random, temperature);
        if (x.Shape[0] != c.MelChannels || x.Shape[1] != frames)
            throw new ArgumentException($"Expected noise [{c.MelChannels}, {frames}].", nameof(noise));

        // ConditionalCFM.solve_euler: t accumulates dt, and each next dt is measured from the accumulated t.
        var span = Enumerable.Range(0, c.Timesteps + 1).Select(i => 1 - Math.Cos((double)i / c.Timesteps * 0.5 * Math.PI)).ToArray();
        double time = span[0], dt = span[1] - span[0];
        var mask = Ones(frames);
        var zeroMu = new Tensor<T>(mu._shape);
        var zeroSpeaker = new Tensor<T>(speaker._shape);
        var zeroCondition = new Tensor<T>(condition._shape);
        double beta = c.InferenceCfgRate;
        for (int step = 1; step < span.Length; step++)
        {
            var conditioned = _engine.Reshape(Estimator.Estimate(EstimatorInput(x, mu, speaker, condition), time, mask), new[] { c.MelChannels, frames });
            var unconditioned = _engine.Reshape(Estimator.Estimate(EstimatorInput(x, zeroMu, zeroSpeaker, zeroCondition), time, mask), new[] { c.MelChannels, frames });
            var velocity = _engine.TensorSubtract(_engine.TensorMultiplyScalar(conditioned, NumOps.FromDouble(1.0 + beta)),
                _engine.TensorMultiplyScalar(unconditioned, NumOps.FromDouble(beta)));
            x = _engine.TensorAdd(x, _engine.TensorMultiplyScalar(velocity, NumOps.FromDouble(dt)));
            time += dt;
            if (step < span.Length - 1) dt = span[step + 1] - time;
        }
        return _engine.TensorTranspose(Columns(x, promptFrames, textFrames));
    }

    /// <summary>PyTorch's initialization of every module, as the reference constructs them.</summary>
    public void InitializeLikeTorch(Random random)
    {
        var c = Configuration;
        CosyVoiceInit.Embedding(random, InputEmbedding);
        CosyVoiceInit.Linear(_engine, random, SpeakerAffine, c.SpeakerEmbeddingDim, c.MelChannels);
        Encoder.InitializeLikeTorch(random);
        CosyVoiceInit.Linear(_engine, random, EncoderProjection, c.Encoder.OutputSize, c.MelChannels);
        foreach (var conv in RegulatorConvs.Append(RegulatorOutput))
        {
            double bound = 1.0 / Math.Sqrt(conv.InChannels * conv.KernelSize);
            conv.Load(CosyVoiceInit.Uniform(random, conv.InChannels * conv.OutChannels * conv.KernelSize, bound),
                CosyVoiceInit.Uniform(random, conv.OutChannels, bound));
        }
        Estimator.InitializeLikeReference(random);
    }

    /// <summary>Loads the reference's <c>flow.pt</c> state dict (PyTorch names, row-major).</summary>
    public void LoadTorch(IReadOnlyDictionary<string, double[]> state)
    {
        var c = Configuration;
        InputEmbedding.LoadTable(state["input_embedding.weight"]);
        TorchParameters.Linear(_engine, SpeakerAffine, c.SpeakerEmbeddingDim, c.MelChannels, state["spk_embed_affine_layer.weight"], state["spk_embed_affine_layer.bias"]);
        Encoder.LoadTorch(state, "encoder.");
        TorchParameters.Linear(_engine, EncoderProjection, c.Encoder.OutputSize, c.MelChannels, state["encoder_proj.weight"], state["encoder_proj.bias"]);
        for (int i = 0; i < RegulatorConvs.Count; i++)
        {
            RegulatorConvs[i].Load(state[$"length_regulator.model.{3 * i}.weight"], state[$"length_regulator.model.{3 * i}.bias"]);
            var affine = state[$"length_regulator.model.{3 * i + 1}.weight"].Concat(state[$"length_regulator.model.{3 * i + 1}.bias"])
                .Select(v => NumOps.FromDouble(v)).ToArray();
            RegulatorNorms[i].SetParameters(new Vector<T>(affine));
        }
        int last = 3 * RegulatorConvs.Count;
        RegulatorOutput.Load(state[$"length_regulator.model.{last}.weight"], state[$"length_regulator.model.{last}.bias"]);
        Estimator.LoadTorch(state, "decoder.estimator.");
    }
}

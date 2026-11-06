using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Initialization;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// StegaStamp: a learned image watermark. A U-Net encoder adds a residual that hides a bit string; a decoder with a
/// spatial transformer reads it back after the printing-and-photographing distortions it was trained against.
/// </summary>
/// <remarks>
/// <para>
/// Architecture and training follow Tancik, Mildenhall and Ng (2020). Encoder: the secret through a dense layer to an
/// (S/8) x (S/8) x 3 map, upsampled x8 and concatenated with the image, then a U-Net (32 at stride 1; 32, 64, 128, 256
/// at stride 2; four upsampling stages with skip connections; two 32-channel convolutions) to a 3-channel residual that
/// is added to the image. Decoder: a spatial transformer (three stride-2 convolutions 32/64/128, dense 128, dense 6
/// initialised to the identity warp) followed by convolutions 32 s2, 32, 64 s2, 64, 64 s2, 128 s2, 128 s2, dense 512
/// and one logit per secret bit. A WGAN critic (convolutions 8, 16, 32, 64 at stride 2 and a 1-channel output, weights
/// clipped) scores the encoded image.
/// </para>
/// <para>
/// Each training step encodes random secrets into the images, applies the paper's differentiable distortions, and
/// trains the encoder and decoder on the secret's cross-entropy, a border-weighted YUV L2 loss on the residual and the
/// critic's score, with the image losses held off and then ramped in as in the paper. The distortions: a random
/// perspective warp, Gaussian or motion blur, Gaussian noise, contrast, brightness, per-channel hue offsets and
/// desaturation, and an 8x8-block JPEG with the cubic rounding approximation (Shin and Song 2017), each ramped in.
/// The paper's LPIPS term needs pretrained AlexNet features, which this library does not ship, so it is not used.
/// </para>
/// <para>
/// As a keyed watermarker, <see cref="EmbedWatermark"/> hides the key's payload: 56 bits derived from the key,
/// protected by the paper's BCH code (corrects 5 bit errors), then zero padding to <see cref="StegaStampOptions.SecretBits"/>.
/// <see cref="Detect"/> correlates the decoder's per-bit logits with the key codeword's signs, a z that is exactly
/// standard normal given the logits when the image does not carry the key's payload. <see cref="EncodeSecret"/> and <see cref="DecodeSecret"/> embed and read arbitrary bits.
/// </para>
/// <para>
/// <b>For Beginners:</b> two small neural networks learn together. One hides a message in an image by changing it
/// almost invisibly; the other learns to read the message back even after the image is warped, blurred, recoloured and
/// compressed. The network has to be trained before it hides anything: call <see cref="NeuralNetworkBase{T}.Train"/>
/// with images and random bit strings, or load a trained model.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.AnomalyDetection)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("StegaStamp: Invisible Hyperlinks in Physical Photographs",
    "https://arxiv.org/abs/1904.05343",
    Year = 2020,
    Authors = "Matthew Tancik, Ben Mildenhall, Ren Ng")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input, BatchOptional = true)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
public partial class StegaStampWatermarker<T> : NeuralNetworkBase<T>, IImageWatermarker<T>, IKeyedWatermarker
{
    private const int PayloadSalt = 0x53544753; // "STGS"
    private const int BlurSize = 7;

    private readonly StegaStampOptions _options;
    private readonly WatermarkKey _key;
    private readonly BchCode? _bch;
    private readonly Random _trainingRandom;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _criticOptimizer;

    private readonly List<ILayer<T>> _encoderLayers = new();
    private readonly List<ILayer<T>> _decoderLayers = new();
    private readonly List<ILayer<T>> _criticLayers = new();

    // Encoder.
    private DenseLayer<T> _secretDense = null!;
    private ConvolutionalLayer<T>[] _down = null!;   // conv1 .. conv5
    private ConvolutionalLayer<T>[] _up = null!;     // up6 .. up9
    private ConvolutionalLayer<T>[] _merge = null!;  // conv6 .. conv9
    private ConvolutionalLayer<T> _conv10 = null!;
    private ConvolutionalLayer<T> _residual = null!;

    // Decoder.
    private ConvolutionalLayer<T>[] _stnConvs = null!;
    private DenseLayer<T> _stnDense = null!;
    private DenseLayer<T> _stnTheta = null!;
    private ConvolutionalLayer<T>[] _decoderConvs = null!;
    private DenseLayer<T> _decoderDense = null!;
    private DenseLayer<T> _decoderLogits = null!;

    // Critic.
    private ConvolutionalLayer<T>[] _criticConvs = null!;

    private int _trainingStep;

    /// <inheritdoc />
    public virtual string ModuleName => "StegaStampWatermarker";

    /// <summary>The watermark's display name in results and findings.</summary>
    private protected virtual string WatermarkName => "StegaStamp watermark";

    /// <summary>The validated options, for derived watermarkers built on these networks.</summary>
    private protected StegaStampOptions Settings => _options;

    /// <summary>The decoder's output count: one logit per secret bit.</summary>
    private protected virtual int DecoderOutputs => _options.SecretBits;

    /// <inheritdoc />
    public bool IsReady => true;

    /// <inheritdoc />
    public bool UsesPublicKey => _key.IsPublic;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>Training steps taken so far, which drive the paper's loss and distortion ramps.</summary>
    public int TrainingStep => _trainingStep;

    /// <summary>Creates a StegaStamp watermarker with the paper's 400 x 400 images and 100-bit secrets.</summary>
    public StegaStampWatermarker() : this(new StegaStampOptions()) { }

    /// <summary>Creates a StegaStamp watermarker.</summary>
    /// <param name="options">Image and secret sizes and the training schedule.</param>
    /// <param name="key">The secret key whose payload <see cref="EmbedWatermark"/> hides; null uses the published key.</param>
    public StegaStampWatermarker(StegaStampOptions options, string? key = null)
        : base(ArchitectureFor(options), new BinaryCrossEntropyLoss<T>())
    {
        _options = (options ?? throw new ArgumentNullException(nameof(options))).Copy();
        _options.Validate();
        Options = _options;
        _key = WatermarkKey.FromOptional(key);
        _bch = _options.UseErrorCorrection ? BchCode.StegaStamp() : null;
        _trainingRandom = _options.Seed is int seed ? RandomHelper.CreateSeededRandom(seed) : RandomHelper.CreateSecureRandom();
        InitializeLayers();
        // The layers infer their input sizes on first use; one forward sizes every weight before the optimizers bind.
        var image = new Tensor<T>(new[] { 1, 3, _options.ImageSize, _options.ImageSize });
        var secret = new Tensor<T>(new[] { 1, _options.SecretBits });
        var encoded = Engine.TensorAdd(image, EncodeResidual(image, secret));
        DecodeLogits(encoded);
        CriticScore(encoded);
        _optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = _options.LearningRate });
        _criticOptimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = _options.CriticLearningRate });
    }

    /// <summary>
    /// A watermarker with the same trained networks that hides and detects <paramref name="key"/>'s payload: the
    /// networks are trained once and serve every key.
    /// </summary>
    public StegaStampWatermarker<T> WithKey(string? key)
    {
        var keyed = new StegaStampWatermarker<T>(_options, key);
        keyed.CopyTrainedStateFrom(this);
        return keyed;
    }

    /// <summary>Takes <paramref name="trained"/>'s weights and training position.</summary>
    private protected void CopyTrainedStateFrom(StegaStampWatermarker<T> trained)
    {
        SetParameters(trained.GetParameters());
        _trainingStep = trained._trainingStep;
    }

    private static NeuralNetworkArchitecture<T> ArchitectureFor(StegaStampOptions? options)
    {
        var o = options ?? new StegaStampOptions();
        return new NeuralNetworkArchitecture<T>(InputType.ThreeDimensional, NeuralNetworkTaskType.MultiLabelClassification,
            inputHeight: o.ImageSize, inputWidth: o.ImageSize, inputDepth: 3, outputSize: o.SecretBits);
    }

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        int s = _options.ImageSize;
        IActivationFunction<T> relu = new ReLUActivation<T>();
        IActivationFunction<T>? none = null;
        // The reference implementation initialises every layer He-normal (Keras he_normal); the library default is
        // PyTorch's Kaiming-uniform, whose weights are about 2.4x smaller. Seeded from the options for reproducibility.
        var he = _options.Seed is int seed
            ? new HeInitializationStrategy<T>(RandomHelper.CreateSeededRandom(unchecked(seed * 31 + 7)))
            : new HeInitializationStrategy<T>();
        ConvolutionalLayer<T> Conv(int depth, int stride, int kernel = 3, bool activate = true) =>
            new(depth, kernel, stride, kernel / 2, activate ? relu : none, he);

        _secretDense = new DenseLayer<T>(3 * (s / 8) * (s / 8), relu, he);
        _down = new[] { Conv(32, 1), Conv(32, 2), Conv(64, 2), Conv(128, 2), Conv(256, 2) };
        // The paper's up-convolutions use 2x2 kernels with TensorFlow's asymmetric 'same' padding; 3x3 keeps the
        // padding symmetric with the same receptive role.
        _up = new[] { Conv(128, 1), Conv(64, 1), Conv(32, 1), Conv(32, 1) };
        _merge = new[] { Conv(128, 1), Conv(64, 1), Conv(32, 1), Conv(32, 1) };
        _conv10 = Conv(32, 1);
        _residual = Conv(3, 1, kernel: 1, activate: false);
        _encoderLayers.Clear();
        _encoderLayers.Add(_secretDense);
        _encoderLayers.AddRange(_down);
        _encoderLayers.AddRange(_up);
        _encoderLayers.AddRange(_merge);
        _encoderLayers.Add(_conv10);
        _encoderLayers.Add(_residual);

        _stnConvs = new[] { Conv(32, 2), Conv(64, 2), Conv(128, 2) };
        _stnDense = new DenseLayer<T>(128, relu, he);
        _stnTheta = new DenseLayer<T>(6, none, new ZeroInitializationStrategy<T>());
        _decoderConvs = new[] { Conv(32, 2), Conv(32, 1), Conv(64, 2), Conv(64, 1), Conv(64, 2), Conv(128, 2), Conv(128, 2) };
        _decoderDense = new DenseLayer<T>(512, relu, he);
        _decoderLogits = new DenseLayer<T>(DecoderOutputs, none, he);
        _decoderLayers.Clear();
        _decoderLayers.AddRange(_stnConvs);
        _decoderLayers.Add(_stnDense);
        _decoderLayers.Add(_stnTheta);
        _decoderLayers.AddRange(_decoderConvs);
        _decoderLayers.Add(_decoderDense);
        _decoderLayers.Add(_decoderLogits);

        _criticConvs = new[] { Conv(8, 2), Conv(16, 2), Conv(32, 2), Conv(64, 2), Conv(1, 1, activate: false) };
        _criticLayers.Clear();
        _criticLayers.AddRange(_criticConvs);

        ClearLayers();
        Layers.AddRange(_encoderLayers);
        Layers.AddRange(_decoderLayers);
        Layers.AddRange(_criticLayers);
    }

    // ---------------------------------------------------------------- networks (tape-tracked)

    /// <summary>The encoder's residual for images [B, 3, S, S] in [0, 1] and secrets [B, bits] in {0, 1}.</summary>
    private protected Tensor<T> EncodeResidual(Tensor<T> images, Tensor<T> secrets)
    {
        int b = images.Shape[0], s = _options.ImageSize;
        var secretMap = Engine.Reshape(_secretDense.Forward(Engine.TensorAddScalar(secrets, NumOps.FromDouble(-0.5))),
            new[] { b, 3, s / 8, s / 8 });
        secretMap = Engine.Upsample(secretMap, 8, 8);
        var centred = Engine.TensorAddScalar(images, NumOps.FromDouble(-0.5));
        var inputs = Engine.TensorConcatenate(new[] { secretMap, centred }, axis: 1);

        var skips = new Tensor<T>[5];
        var x = inputs;
        for (int i = 0; i < 5; i++)
        {
            x = _down[i].Forward(x);
            skips[i] = x;
        }

        for (int i = 0; i < 4; i++)
        {
            var up = _up[i].Forward(Engine.Upsample(x, 2, 2));
            var merged = i < 3
                ? Engine.TensorConcatenate(new[] { skips[3 - i], up }, axis: 1)
                : Engine.TensorConcatenate(new[] { skips[0], up, inputs }, axis: 1);
            x = _merge[i].Forward(merged);
        }

        return _residual.Forward(_conv10.Forward(x));
    }

    /// <summary>The decoder's per-bit logits for images [B, 3, S, S].</summary>
    private protected Tensor<T> DecodeLogits(Tensor<T> images)
    {
        int b = images.Shape[0], s = _options.ImageSize;
        var centred = Engine.TensorAddScalar(images, NumOps.FromDouble(-0.5));
        var h = centred;
        foreach (var conv in _stnConvs) h = conv.Forward(h);
        h = _stnDense.Forward(Engine.Reshape(h, new[] { b, h.Length / b }));
        var theta = Engine.Reshape(_stnTheta.Forward(h), new[] { b, 2, 3 });
        theta = Engine.TensorAdd(theta, Engine.TensorBroadcastTo(IdentityTheta(), new[] { b, 2, 3 }));
        var warped = Engine.GridSample(centred, Engine.AffineGrid(theta, s, s));

        var x = warped;
        foreach (var conv in _decoderConvs) x = conv.Forward(x);
        x = _decoderDense.Forward(Engine.Reshape(x, new[] { b, x.Length / b }));
        return _decoderLogits.Forward(x);
    }

    /// <summary>The WGAN critic's mean score per image, [B].</summary>
    private protected Tensor<T> CriticScore(Tensor<T> images)
    {
        var x = Engine.TensorAddScalar(images, NumOps.FromDouble(-0.5));
        foreach (var conv in _criticConvs) x = conv.Forward(x);
        return Engine.ReduceMean(x, new[] { 1, 2, 3 }, keepDims: false);
    }

    private Tensor<T> IdentityTheta()
    {
        var theta = new Tensor<T>(new[] { 1, 2, 3 });
        theta[0, 0, 0] = NumOps.One;
        theta[0, 1, 1] = NumOps.One;
        return theta;
    }

    // ---------------------------------------------------------------- public watermarking surface

    /// <summary>
    /// Decoder probabilities for each secret bit, [B, bits], for images [B, 3, S, S] (or [3, S, S]) in [0, 1].
    /// </summary>
    protected override Tensor<T> PredictCore(Tensor<T> input) => Engine.Sigmoid(DecodeLogits(AsBatch(input)));

    /// <summary>Hides <paramref name="secret"/> (one 0/1 value per bit) in <paramref name="image"/>.</summary>
    /// <param name="image">[3, H, W] or [1, 3, H, W] with values in [0, 1]; other sizes are encoded at the model's size
    /// and the residual is resized back.</param>
    /// <param name="secret">The bits, <see cref="StegaStampOptions.SecretBits"/> of them.</param>
    public Tensor<T> EncodeSecret(Tensor<T> image, IReadOnlyList<bool> secret)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        if (secret is null) throw new ArgumentNullException(nameof(secret));
        if (secret.Count != _options.SecretBits)
            throw new ArgumentException($"Expected {_options.SecretBits} secret bits, got {secret.Count}.", nameof(secret));
        var batch = AsBatch(image);
        int h = batch.Shape[2], w = batch.Shape[3], s = _options.ImageSize;
        var resized = h == s && w == s ? batch : Engine.Interpolate(batch, new[] { s, s }, InterpolateMode.Bilinear);
        var bits = new Tensor<T>(new[] { 1, secret.Count });
        for (int i = 0; i < secret.Count; i++) bits[0, i] = secret[i] ? NumOps.One : NumOps.Zero;
        var residual = EncodeResidual(resized, bits);
        if (h != s || w != s) residual = Engine.Interpolate(residual, new[] { h, w }, InterpolateMode.Bilinear);
        var encoded = Engine.TensorClip(Engine.TensorAdd(batch, residual), NumOps.Zero, NumOps.One);
        return image.Rank == 3 ? Engine.Reshape(encoded, image.Shape.ToArray()) : encoded;
    }

    /// <summary>The decoded bits of <paramref name="image"/> (resized to the model's size if needed).</summary>
    public bool[] DecodeSecret(Tensor<T> image)
    {
        var probabilities = Predict(Resized(AsBatch(image)));
        var bits = new bool[_options.SecretBits];
        for (int i = 0; i < bits.Length; i++) bits[i] = NumOps.ToDouble(probabilities[0, i]) > 0.5;
        return bits;
    }

    /// <summary>
    /// The key's payload recovered from <paramref name="image"/> by BCH decoding, with the number of bits corrected;
    /// null when error correction is off or the decoded bits hold more errors than the code corrects.
    /// </summary>
    public (bool[] Data, int Corrected)? DecodePayload(Tensor<T> image)
    {
        if (_bch is null) return null;
        var bits = DecodeSecret(image);
        return _bch.Decode(bits.Take(_bch.CodewordBits).ToArray());
    }

    /// <summary>The bit string <see cref="EmbedWatermark"/> hides for this key.</summary>
    public bool[] KeySecret()
    {
        int dataBits = _bch?.DataBits ?? _options.SecretBits;
        var pattern = _key.SignPattern(PayloadSalt, 0, dataBits);
        var data = pattern.Select(v => v > 0).ToArray();
        var secret = new bool[_options.SecretBits];
        var keyed = _bch?.Encode(data) ?? data;
        Array.Copy(keyed, secret, keyed.Length);
        return secret;
    }

    /// <inheritdoc />
    public Tensor<T> EmbedWatermark(Tensor<T> image) => EncodeSecret(image, KeySecret());

    /// <summary>
    /// Tests <paramref name="image"/> for this key's payload with a soft-decision score: the correlation of the
    /// decoder's logits with the key codeword's signs, z = sum(s_i l_i) / sqrt(sum(l_i^2)).
    /// </summary>
    /// <remarks>
    /// On an image that does not carry this key's payload, the key's signs are fair coins independent of the
    /// logits (and pairwise independent across the BCH codeword), so given the logits the score has mean 0 and
    /// variance exactly 1 whatever the decoder outputs. The paper counts hard-decided bits instead
    /// (<see cref="HardDecisionZ"/>). The soft score weights each bit by the decoder's confidence, so the same
    /// decoder gives a larger z on marked images while unmarked ones stay calibrated.
    /// </remarks>
    public virtual WatermarkResult Detect(Tensor<T> image)
    {
        var logits = DecoderLogits(image);
        var signs = KeySigns();
        double correlation = 0, energy = 0;
        for (int i = 0; i < signs.Length; i++)
        {
            correlation += signs[i] * logits[i];
            energy += logits[i] * logits[i];
        }

        return WatermarkStatistics.Result(WatermarkStatistics.CorrelationZ(correlation, energy), signs.Length,
            "image", WatermarkName, 0.3);
    }

    /// <summary>
    /// The paper's hard-decision score: decoded bits that agree with the key's codeword, as a binomial z. Kept to
    /// compare against <see cref="Detect"/>'s soft score.
    /// </summary>
    internal double HardDecisionZ(Tensor<T> image)
    {
        var logits = DecoderLogits(image);
        var signs = KeySigns();
        int hits = 0;
        for (int i = 0; i < signs.Length; i++) if ((logits[i] > 0) == (signs[i] > 0)) hits++;
        return WatermarkStatistics.CountZ(hits, signs.Length, 0.5);
    }

    // The decoder's logit per scored bit (the codeword's bits; padding excluded).
    private protected double[] DecoderLogits(Tensor<T> image)
    {
        var logits = DecodeLogits(Resized(AsBatch(image)));
        int scored = _bch?.CodewordBits ?? _options.SecretBits;
        var values = new double[scored];
        for (int i = 0; i < scored; i++) values[i] = NumOps.ToDouble(logits[0, i]);
        return values;
    }

    // +1 where the key's codeword holds a 1, -1 where it holds a 0.
    private protected double[] KeySigns()
    {
        var secret = KeySecret();
        int scored = _bch?.CodewordBits ?? _options.SecretBits;
        var signs = new double[scored];
        for (int i = 0; i < scored; i++) signs[i] = secret[i] ? 1.0 : -1.0;
        return signs;
    }

    /// <inheritdoc />
    public double DetectWatermark(Tensor<T> image) => Detect(image).Confidence;

    /// <inheritdoc />
    public IReadOnlyList<SafetyFinding> EvaluateImage(Tensor<T> image)
    {
        var findings = new List<SafetyFinding>();
        var result = Detect(image);
        if (result.WatermarkDetected)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Watermarked,
                Severity = SafetySeverity.Info,
                Confidence = result.Confidence,
                Description = $"{WatermarkName} detected (confidence: {result.Confidence:F3}, z = {result.ZScore:F2}).",
                RecommendedAction = SafetyAction.Log,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    /// <inheritdoc />
    public IReadOnlyList<SafetyFinding> Evaluate(Vector<T> content)
    {
        if (content is null) throw new ArgumentNullException(nameof(content));
        int s = _options.ImageSize;
        if (content.Length != 3 * s * s) return Array.Empty<SafetyFinding>();
        return EvaluateImage(new Tensor<T>(new[] { 3, s, s }, content));
    }

    private protected Tensor<T> AsBatch(Tensor<T> image)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        if (image.Rank == 3 && image.Shape[0] == 3) return Engine.Reshape(image, new[] { 1, 3, image.Shape[1], image.Shape[2] });
        if (image.Rank == 4 && image.Shape[1] == 3) return image;
        throw new ArgumentException($"Expected an RGB image, [3, H, W] or [B, 3, H, W]; got [{string.Join(", ", image.Shape.ToArray())}].", nameof(image));
    }

    private protected Tensor<T> Resized(Tensor<T> batch)
    {
        int s = _options.ImageSize;
        return batch.Shape[2] == s && batch.Shape[3] == s ? batch : Engine.Interpolate(batch, new[] { s, s }, InterpolateMode.Bilinear);
    }

    // ---------------------------------------------------------------- training

    /// <inheritdoc />
    /// <remarks>
    /// <paramref name="input"/> holds training images [B, 3, S, S] in [0, 1]; <paramref name="target"/> the secrets
    /// [B, bits] to hide in them, normally uniformly random. One call takes the critic's step and then the encoder and
    /// decoder's step. Every random draw of the step travels in the step's data, so a compiled plan replays this step's
    /// distortions, not the first step's.
    /// </remarks>
    protected override IReadOnlyList<Training.TrainingObjective<T>>? DefineTrainingObjectives(Tensor<T> input, Tensor<T> target)
    {
        var images = Resized(AsBatch(input));
        int b = images.Shape[0], s = _options.ImageSize;
        if (target is null || target.Length != b * _options.SecretBits)
            throw new ArgumentException($"Expected {b} x {_options.SecretBits} secret bits as the target.", nameof(target));
        var secrets = Engine.Reshape(target, new[] { b, _options.SecretBits });
        int step = _trainingStep++;

        // Critic: WGAN on the current encoder's output, which is data here, so this step cannot reach the encoder.
        var encodedNow = Engine.TensorAdd(images, EncodeResidual(images, secrets));
        var critic = new Training.TrainingObjective<T>(new[] { images, encodedNow }, data =>
            Engine.TensorSubtract(Engine.ReduceMean(CriticScore(data[0]), new[] { 0 }, keepDims: true),
                Engine.ReduceMean(CriticScore(data[1]), new[] { 0 }, keepDims: true)))
        {
            Optimizer = _criticOptimizer,
            Parameters = () => Training.TapeTrainingStep<T>.CollectParameters(_criticLayers),
            AfterStep = ClipCriticWeights,
        };

        var draw = DrawDistortions(b, s, step);
        var weights = LossWeights(step);
        var main = new Training.TrainingObjective<T>(
            new[] { images, secrets, draw.Grid, draw.Blur, draw.Noise, draw.Contrast, draw.Shift, draw.Saturation, draw.Quantization, weights },
            data => EncoderDecoderLoss(data))
        {
            Optimizer = _optimizer,
            Parameters = () => Training.TapeTrainingStep<T>.CollectParameters(_encoderLayers.Concat(_decoderLayers)),
        };

        return new[] { critic, main };
    }

    /// <summary>
    /// The encoder and decoder's loss. <paramref name="data"/>: images, secrets, the distortion draw (grid, blur,
    /// noise, contrast, shift, saturation, JPEG table) and the loss weights [secret, image, critic].
    /// </summary>
    private protected virtual Tensor<T> EncoderDecoderLoss(IReadOnlyList<Tensor<T>> data)
    {
        var images = data[0];
        var secrets = data[1];
        var residual = EncodeResidual(images, secrets);
        var encoded = Engine.TensorAdd(images, residual);
        var distorted = Distort(encoded, data[2], data[3], data[4], data[5], data[6], data[7], data[8]);
        var logits = DecodeLogits(distorted);
        var secretLoss = SecretLoss(logits, secrets);
        var imageLoss = ResidualLoss(residual);
        var criticLoss = Engine.ReduceMean(CriticScore(encoded), new[] { 0 }, keepDims: false);

        var w = data[9];
        Tensor<T> Weighted(Tensor<T> loss, int index) =>
            Engine.TensorMultiply(Engine.Reshape(loss, new[] { 1 }), Engine.TensorSlice(w, new[] { index }, new[] { 1 }));
        return Engine.TensorAdd(Engine.TensorAdd(Weighted(secretLoss, 0), Weighted(imageLoss, 1)), Weighted(criticLoss, 2));
    }

    /// <summary>Mean binary cross-entropy with logits over the secret bits, softplus(x) - y x.</summary>
    private protected Tensor<T> SecretLoss(Tensor<T> logits, Tensor<T> secrets)
    {
        int b = secrets.Shape[0], bits = _options.SecretBits;
        var bitLogits = logits.Shape[1] == bits ? logits : Engine.TensorSlice(logits, new[] { 0, 0 }, new[] { b, bits });
        return Engine.ReduceMean(Engine.TensorSubtract(Engine.Softplus(bitLogits), Engine.TensorMultiply(secrets, bitLogits)),
            new[] { 0, 1 }, keepDims: false);
    }

    // Border-weighted L2 of the residual in YUV, averaged over pixels.
    private protected Tensor<T> ResidualLoss(Tensor<T> residual)
    {
        int b = residual.Shape[0], s = _options.ImageSize;
        Tensor<T> Channel(int c) => Engine.TensorSlice(residual, new[] { 0, c, 0, 0 }, new[] { b, 1, s, s });
        Tensor<T> Mix(double r, double g, double bl) => Engine.TensorAdd(Engine.TensorAdd(
            Engine.TensorMultiplyScalar(Channel(0), NumOps.FromDouble(r)),
            Engine.TensorMultiplyScalar(Channel(1), NumOps.FromDouble(g))),
            Engine.TensorMultiplyScalar(Channel(2), NumOps.FromDouble(bl)));
        var yuv = new[] { Mix(0.299, 0.587, 0.114), Mix(-0.14713, -0.28886, 0.436), Mix(0.615, -0.51499, -0.10001) };
        var mask = Engine.TensorBroadcastTo(EdgeWeights(), new[] { b, 1, s, s });
        Tensor<T>? total = null;
        for (int c = 0; c < 3; c++)
        {
            var term = Engine.TensorMultiplyScalar(
                Engine.ReduceMean(Engine.TensorMultiply(Engine.TensorSquare(yuv[c]), mask), new[] { 0, 1, 2, 3 }, keepDims: false),
                NumOps.FromDouble(_options.YuvLossScales[c]));
            total = total is null ? term : Engine.TensorAdd(total, term);
        }

        return total!;
    }

    // 1 + gain * (1 - f), where f rises from 0 at the border to 1 a quarter of the way in (raised cosine).
    private Tensor<T> EdgeWeights()
    {
        int s = _options.ImageSize;
        double ramp = Math.Max(1.0, s / 4.0);
        var weights = new Tensor<T>(new[] { 1, 1, s, s });
        for (int y = 0; y < s; y++)
        {
            for (int x = 0; x < s; x++)
            {
                double d = Math.Min(Math.Min(y, x), Math.Min(s - 1 - y, s - 1 - x));
                double f = d >= ramp ? 1.0 : 0.5 - 0.5 * Math.Cos(Math.PI * d / ramp);
                weights[0, 0, y, x] = NumOps.FromDouble(1.0 + _options.EdgeLossGain * (1.0 - f));
            }
        }

        return weights;
    }

    private Tensor<T> LossWeights(int step)
    {
        double imageRamp = Ramp(step - _options.NoImageLossSteps, _options.ImageLossRampSteps);
        double criticRamp = Ramp(step - _options.NoImageLossSteps, _options.CriticLossRampSteps);
        bool imageOn = step >= _options.NoImageLossSteps;
        var w = new Tensor<T>(new[] { 3 });
        w[0] = NumOps.FromDouble(_options.SecretLossScale);
        w[1] = NumOps.FromDouble(imageOn ? _options.ImageLossScale * imageRamp : 0.0);
        w[2] = NumOps.FromDouble(imageOn ? _options.CriticLossScale * criticRamp : 0.0);
        return w;
    }

    private static double Ramp(int step, int length) => length <= 0 ? 1.0 : Math.Max(0.0, Math.Min(1.0, (double)step / length));

    private void ClipCriticWeights()
    {
        double c = _options.CriticWeightClip;
        foreach (var layer in _criticLayers)
        {
            var p = layer.GetParameters();
            var clipped = new Vector<T>(p.Length);
            for (int i = 0; i < p.Length; i++) clipped[i] = NumOps.FromDouble(Math.Max(-c, Math.Min(c, NumOps.ToDouble(p[i]))));
            layer.SetParameters(clipped);
        }
    }

    // ---------------------------------------------------------------- distortions

    private sealed record Distortions(Tensor<T> Grid, Tensor<T> Blur, Tensor<T> Noise, Tensor<T> Contrast,
        Tensor<T> Shift, Tensor<T> Saturation, Tensor<T> Quantization);

    // The paper's ramps: each distortion grows from none to its full range over its ramp.
    private Distortions DrawDistortions(int b, int s, int step)
    {
        var rng = _trainingRandom;
        double perspective = _options.PerspectiveJitter * Ramp(step, _options.PerspectiveRampSteps);
        double color = Ramp(step, _options.ColorRampSteps);

        var grid = new Tensor<T>(new[] { b, s, s, 2 });
        for (int n = 0; n < b; n++) WriteHomographyGrid(grid, n, s, perspective, rng);

        var blur = BlurKernel(rng);

        var noise = new Tensor<T>(new[] { b, 3, s, s });
        for (int n = 0; n < b; n++)
        {
            double sd = rng.NextDouble() * _options.NoiseStdDev * color;
            for (int c = 0; c < 3; c++)
                for (int y = 0; y < s; y++)
                    for (int x = 0; x < s; x++)
                        noise[n, c, y, x] = NumOps.FromDouble(sd * Gaussian(rng));
        }

        var contrast = new Tensor<T>(new[] { b, 1, 1, 1 });
        var shift = new Tensor<T>(new[] { b, 3, 1, 1 });
        var saturation = new Tensor<T>(new[] { b, 1, 1, 1 });
        for (int n = 0; n < b; n++)
        {
            double low = 1.0 - (1.0 - _options.ContrastLow) * color, high = 1.0 + (_options.ContrastHigh - 1.0) * color;
            contrast[n, 0, 0, 0] = NumOps.FromDouble(low + rng.NextDouble() * (high - low));
            double brightness = (2 * rng.NextDouble() - 1) * _options.BrightnessShift * color;
            for (int c = 0; c < 3; c++)
                shift[n, c, 0, 0] = NumOps.FromDouble(brightness + (2 * rng.NextDouble() - 1) * _options.HueShift * color);
            saturation[n, 0, 0, 0] = NumOps.FromDouble(rng.NextDouble() * _options.SaturationMix * color);
        }

        double quality = 100.0 - rng.NextDouble() * color * (100.0 - _options.MinimumJpegQuality);
        return new Distortions(grid, blur, noise, contrast, shift, saturation, QuantizationTable(b, s, quality));
    }

    private protected Tensor<T> Distort(Tensor<T> encoded, Tensor<T> grid, Tensor<T> blur, Tensor<T> noise, Tensor<T> contrast,
        Tensor<T> shift, Tensor<T> saturation, Tensor<T> quantization)
    {
        int b = encoded.Shape[0], s = _options.ImageSize, pad = BlurSize / 2;
        var x = Engine.GridSample(encoded, grid);
        x = Engine.DepthwiseConv2D(x, blur, new[] { 1, 1 }, new[] { pad, pad });
        x = Engine.TensorAdd(x, noise);
        x = Engine.TensorMultiply(x, contrast);
        x = Engine.TensorAdd(x, Engine.TensorBroadcastTo(shift, new[] { b, 3, s, s }));
        var luminance = Engine.TensorBroadcastTo(Engine.ReduceMean(x, new[] { 1 }, keepDims: true), new[] { b, 3, s, s });
        x = Engine.TensorAdd(x, Engine.TensorMultiply(Engine.TensorSubtract(luminance, x), saturation));
        x = Engine.TensorClip(x, NumOps.Zero, NumOps.One);
        return Engine.TensorClip(DifferentiableJpeg(x, quantization), NumOps.Zero, NumOps.One);
    }

    // Output corners (+-1, +-1) map to source corners jittered by up to 2 * jitter in normalized units; the grid holds
    // the source coordinate of every output pixel (align_corners = false centres).
    private void WriteHomographyGrid(Tensor<T> grid, int n, int s, double jitter, Random rng)
    {
        var dst = new[] { (-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0) };
        var src = dst.Select(p => (p.Item1 + 2 * jitter * (2 * rng.NextDouble() - 1), p.Item2 + 2 * jitter * (2 * rng.NextDouble() - 1))).ToArray();
        var h = Homography(dst, src);
        for (int y = 0; y < s; y++)
        {
            double v = (2.0 * y + 1) / s - 1;
            for (int x = 0; x < s; x++)
            {
                double u = (2.0 * x + 1) / s - 1;
                double w = h[6] * u + h[7] * v + 1.0;
                grid[n, y, x, 0] = NumOps.FromDouble((h[0] * u + h[1] * v + h[2]) / w);
                grid[n, y, x, 1] = NumOps.FromDouble((h[3] * u + h[4] * v + h[5]) / w);
            }
        }
    }

    // The 8-parameter homography taking each from-point to its to-point (Gaussian elimination with pivoting).
    private static double[] Homography((double X, double Y)[] from, (double X, double Y)[] to)
    {
        var a = new double[8, 9];
        for (int i = 0; i < 4; i++)
        {
            var (x, y) = from[i];
            var (u, v) = to[i];
            double[] r1 = { x, y, 1, 0, 0, 0, -u * x, -u * y, u };
            double[] r2 = { 0, 0, 0, x, y, 1, -v * x, -v * y, v };
            for (int k = 0; k < 9; k++) { a[2 * i, k] = r1[k]; a[2 * i + 1, k] = r2[k]; }
        }

        for (int col = 0; col < 8; col++)
        {
            int pivot = col;
            for (int r = col + 1; r < 8; r++) if (Math.Abs(a[r, col]) > Math.Abs(a[pivot, col])) pivot = r;
            for (int k = 0; k < 9; k++) (a[col, k], a[pivot, k]) = (a[pivot, k], a[col, k]);
            for (int r = 0; r < 8; r++)
            {
                if (r == col) continue;
                double f = a[r, col] / a[col, col];
                for (int k = col; k < 9; k++) a[r, k] -= f * a[col, k];
            }
        }

        var h = new double[8];
        for (int i = 0; i < 8; i++) h[i] = a[i, 8] / a[i, i];
        return h;
    }

    // [3, 1, k, k]: identity, a Gaussian (sigma in [1, 3]) or a motion line (length 3..7), as in the reference code.
    private Tensor<T> BlurKernel(Random rng)
    {
        var k = new double[BlurSize, BlurSize];
        double pick = rng.NextDouble();
        int c0 = BlurSize / 2;
        if (pick < _options.BlurProbability)
        {
            double sigma = 1.0 + 2.0 * rng.NextDouble();
            for (int y = 0; y < BlurSize; y++)
                for (int x = 0; x < BlurSize; x++)
                    k[y, x] = Math.Exp(-((y - c0) * (y - c0) + (x - c0) * (x - c0)) / (2 * sigma * sigma));
        }
        else if (pick < 2 * _options.BlurProbability)
        {
            int length = 3 + rng.Next(BlurSize - 2);
            double angle = Math.PI * rng.NextDouble();
            for (int i = 0; i < length; i++)
            {
                double t = i - (length - 1) / 2.0;
                int x = (int)Math.Round(c0 + t * Math.Cos(angle)), y = (int)Math.Round(c0 + t * Math.Sin(angle));
                k[y, x] += 1.0;
            }
        }
        else
        {
            k[c0, c0] = 1.0;
        }

        double sum = 0;
        foreach (double v in k) sum += v;
        var kernel = new Tensor<T>(new[] { 3, 1, BlurSize, BlurSize });
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < BlurSize; y++)
                for (int x = 0; x < BlurSize; x++)
                    kernel[c, 0, y, x] = NumOps.FromDouble(k[y, x] / sum);
        return kernel;
    }

    private static readonly int[] LuminanceTable =
    {
        16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99
    };

    private static readonly int[] ChrominanceTable =
    {
        17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
        47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99
    };

    // One row of 64 quantization steps per 8x8 block, in the block order of Blocks(): Y rows then Cb, Cr per image.
    internal Tensor<T> QuantizationTable(int b, int s, double quality)
    {
        double scale = quality < 50 ? 5000.0 / quality : 200.0 - 2.0 * quality;
        int blocksPerChannel = (s / 8) * (s / 8);
        var q = new Tensor<T>(new[] { b * 3 * blocksPerChannel, 64 });
        for (int n = 0; n < b; n++)
            for (int c = 0; c < 3; c++)
            {
                var table = c == 0 ? LuminanceTable : ChrominanceTable;
                for (int blk = 0; blk < blocksPerChannel; blk++)
                {
                    int row = (n * 3 + c) * blocksPerChannel + blk;
                    for (int k = 0; k < 64; k++)
                        q[row, k] = NumOps.FromDouble(Math.Max(1.0, Math.Floor((table[k] * scale + 50) / 100)));
                }
            }

        return q;
    }

    // JPEG in YCbCr on 8x8 blocks: DCT, divide by the quantization step, round, multiply back, inverse DCT. Training
    // rounds with the cubic approximation r + (x - r)^3 (gradient 3 (x - r)^2); exactRounding is real JPEG rounding,
    // for measuring robustness (no chroma subsampling or entropy coding, which are lossless or near it here).
    internal Tensor<T> DifferentiableJpeg(Tensor<T> rgb, Tensor<T> quantization, bool exactRounding = false)
    {
        int b = rgb.Shape[0], s = _options.ImageSize, nb = s / 8;
        var ycc = ColorTransform(Engine.TensorMultiplyScalar(rgb, NumOps.FromDouble(255.0)), RgbToYcc, -128.0, after: false);
        var blocks = Engine.Reshape(Engine.TensorPermute(Engine.Reshape(ycc, new[] { b, 3, nb, 8, nb, 8 }), new[] { 0, 1, 2, 4, 3, 5 }).Contiguous(),
            new[] { b * 3 * nb * nb, 64 });
        var dct = DctMatrix();
        var coefficients = Engine.TensorMatMul(blocks, dct);
        var scaled = Engine.TensorDivide(coefficients, quantization);
        var rounded = Engine.TensorRound(scaled);
        var diff = Engine.TensorSubtract(scaled, rounded);
        var approx = exactRounding ? rounded : Engine.TensorAdd(rounded, Engine.TensorMultiply(Engine.TensorMultiply(diff, diff), diff));
        var restored = Engine.TensorMatMul(Engine.TensorMultiply(approx, quantization), Engine.TensorTranspose(dct));
        var image = Engine.Reshape(Engine.TensorPermute(Engine.Reshape(restored, new[] { b, 3, nb, nb, 8, 8 }), new[] { 0, 1, 2, 4, 3, 5 }).Contiguous(),
            new[] { b, 3, s, s });
        return Engine.TensorMultiplyScalar(ColorTransform(image, YccToRgb, -128.0, after: true), NumOps.FromDouble(1.0 / 255.0));
    }

    private static readonly double[,] RgbToYcc =
    {
        { 0.299, 0.587, 0.114 }, { -0.168736, -0.331264, 0.5 }, { 0.5, -0.418688, -0.081312 }
    };

    private static readonly double[,] YccToRgb =
    {
        { 1.0, 0.0, 1.402 }, { 1.0, -0.344136, -0.714136 }, { 1.0, 1.772, 0.0 }
    };

    // out_c = sum_k m[c, k] x_k. JPEG's level shift: Y - 128 after the forward transform, and the same offset
    // removed (Y + 128) before the inverse; centred chroma needs none (its +128 storage offset cancels the shift).
    private Tensor<T> ColorTransform(Tensor<T> x, double[,] m, double lumaOffset, bool after)
    {
        int b = x.Shape[0], s = x.Shape[2];
        var offset = new Tensor<T>(new[] { 1, 3, 1, 1 });
        offset[0, 0, 0, 0] = NumOps.FromDouble(lumaOffset);
        var offsets = Engine.TensorBroadcastTo(offset, new[] { b, 3, s, s });
        if (after) x = Engine.TensorSubtract(x, offsets);
        var kernel = new Tensor<T>(new[] { 3, 3, 1, 1 });
        for (int c = 0; c < 3; c++)
            for (int k = 0; k < 3; k++)
                kernel[c, k, 0, 0] = NumOps.FromDouble(m[c, k]);
        var y = Engine.Conv2D(x, kernel);
        return after ? y : Engine.TensorAdd(y, offsets);
    }

    // Orthonormal 2D DCT-II on a flattened 8x8 block: coefficients = block . D, D[(y, x), (v, u)] = c(v) c(u) cos cos.
    private Tensor<T> DctMatrix()
    {
        var d = new Tensor<T>(new[] { 64, 64 });
        for (int y = 0; y < 8; y++)
            for (int x = 0; x < 8; x++)
                for (int v = 0; v < 8; v++)
                    for (int u = 0; u < 8; u++)
                    {
                        double cv = v == 0 ? Math.Sqrt(1.0 / 8) : Math.Sqrt(2.0 / 8);
                        double cu = u == 0 ? Math.Sqrt(1.0 / 8) : Math.Sqrt(2.0 / 8);
                        double value = cv * cu * Math.Cos((2 * y + 1) * v * Math.PI / 16) * Math.Cos((2 * x + 1) * u * Math.PI / 16);
                        d[y * 8 + x, v * 8 + u] = NumOps.FromDouble(value);
                    }

        return d;
    }

    private static double Gaussian(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        AdditionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "StegaStamp" }, { "ImageSize", _options.ImageSize }, { "SecretBits", _options.SecretBits },
            { "ErrorCorrection", _options.UseErrorCorrection }, { "TrainingStep", _trainingStep }
        },
        ModelDataProvider = () => this.Serialize()
    };
}

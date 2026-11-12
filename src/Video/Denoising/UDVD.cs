using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Denoising;

/// <summary>
/// UDVD: unsupervised deep video denoising with a blind-spot network (Sheth et al., ICCV 2021).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The network is the authors' "blind-video-net-4" (github.com/sreyas-mohan/udvd, models/udvd.py). Five contiguous
/// frames predict the middle one. Every convolution is <b>blind</b>: the input is shifted down one row before a
/// replicate-padded 3x3 convolution and the extra row cropped after, so each output sees only pixels above it, and
/// max-pooling is preceded by the same shift. The frames are stacked in four rotations, so the four copies together
/// see everything except the centre pixel; a final shift removes it. A first blind U-Net maps each of the frame
/// triplets (1-3, 2-4, 3-5) to features, a second maps their concatenation, the rotations are undone and stacked on
/// the channels, and three 1x1 convolutions produce, per pixel, the mean μ and a factor A of the covariance
/// Σ = AᵀA of the clean value.
/// </para>
/// <para>
/// Training maximizes the Gaussian log-likelihood of the target frame under N(μ, AᵀA + σ²I) (utils/loss_funtions.py,
/// "loglike"), with the release's -0.1σ regularizer. Passing the noisy frames as the target is the paper's
/// unsupervised training; a clean target makes it supervised. The denoised frame is the posterior mean
/// E[x|y] = (Σ⁻¹ + σ⁻²I)⁻¹(Σ⁻¹μ + σ⁻²y) (utils/post_processing.py). Adam at 1e-4 halved at epochs 20, 25 and 30.
/// </para>
/// <para><b>For Beginners:</b> UDVD learns from noisy video alone. Its network can never see the pixel it is
/// predicting, so the best it can do is predict the clean signal from the neighbourhood and nearby frames, which is
/// exactly what denoising needs; the noisy pixel is then folded back in with Bayes' rule.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.Generative,
///     inputHeight: 128, inputWidth: 128, inputDepth: 3);
/// var udvd = new UDVD&lt;double&gt;(architecture, new UDVDOptions());
/// udvd.Train(noisyFrames, noisyFrames);           // unsupervised: the noisy video is its own target
/// var clean = udvd.Denoise(noisyFrames);          // [T, 3, H, W], pixel values 0-255
/// </code>
/// </example>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Unsupervised Deep Video Denoising",
    "https://arxiv.org/abs/2011.15045",
    Year = 2021,
    Authors = "Dev Yashpal Sheth, Sreyas Mohan, Joshua L. Vincent, Ramon Manzorro, Peter A. Crozier, Mitesh M. Khapra, Eero P. Simoncelli, Carlos Fernandez-Granda")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-4, DecayRate = 0.5,
                Milestones = [20, 25, 30],
                Schedule = LearningRateSchedulerType.MultiStep,
                ScheduleStepMode = SchedulerStepMode.StepPerEpoch,
                Source = "Sheth et al. 2021, Optimization Details: Adam from a starting learning rate "
                        + "of 1e-4, halved at epochs 20, 25 and 30 over a total of 40 epochs.")]
public partial class UDVD<T> : VideoDenoisingBase<T>
{
    private const int WindowFrames = 5;
    private const double LeakySlope = 0.1;
    private const double Epsilon = 1e-5;

    private readonly UDVDOptions _options;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _usingCustomLayers;
    private bool _disposed;

    /// <summary>
    /// Creates a UDVD model for ONNX inference.
    /// </summary>
    public UDVD(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        UDVDOptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrEmpty(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        _options = options ?? new UDVDOptions();
        Options = _options;
        _useNativeMode = false;
        IsBlindDenoising = true;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a UDVD model for native training and inference.
    /// </summary>
    public UDVD(
        NeuralNetworkArchitecture<T> architecture,
        UDVDOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new UDVDOptions();
        Options = _options;
        if (_options.EncoderFeatures < 1 || _options.DecoderFeatures < 1 || _options.FirstStageFeatures < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "Feature widths must be positive.");
        _useNativeMode = true;
        // Adam at 1e-4, halved at epochs 20, 25 and 30 (the release's MultiStepLR).
        _optimizer = optimizer ?? PaperOptimizerFactory.VerifyHandBuilt(this,
            new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
                this,
                new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
                {
                    InitialLearningRate = _options.LearningRate,
                    LearningRateScheduler = new MultiStepLRScheduler(_options.LearningRate, new[] { 20, 25, 30 }, 0.5),
                    SchedulerStepMode = SchedulerStepMode.StepPerEpoch,
                }));
        IsBlindDenoising = true;
        InitializeLayers();
    }

    /// <inheritdoc/>
    public override Tensor<T> Denoise(Tensor<T> noisyFrames)
    {
        ThrowIfDisposed();
        var preprocessed = PreprocessFrames(noisyFrames);
        var output = IsOnnxMode ? RunOnnxInference(preprocessed) : Forward(preprocessed);
        return PostprocessOutput(output);
    }

    #region Layers

    private int Channels => Architecture.InputDepth > 0 ? Architecture.InputDepth : 3;

    // The network's per-pixel outputs: C mean values and the C(C+1)/2 entries of the covariance factor.
    private int OutputChannels => Channels + Channels * (Channels + 1) / 2;

    // A blind U-Net: enc1, enc2, enc3 (three convs each) and dec2, dec1 (four each) = 17 convolutions.
    private const int UNetLength = 17;
    private const int FirstUNet = 0;
    private const int SecondUNet = UNetLength;
    private const int NinStart = 2 * UNetLength;

    private ConvolutionalLayer<T> BlindConv(int outChannels)
        => new(outChannels, kernelSize: 3, stride: 1, padding: 0,
            activationFunction: new LeakyReLUActivation<T>(LeakySlope), initializationStrategy: null,
            nonlinearityForInit: null, groups: 1, biasMode: BiasMode.Never);

    private void AddUNet(int outChannels)
    {
        int e = _options.EncoderFeatures, d = _options.DecoderFeatures;
        foreach (int width in new[] { e, e, e, e, e, e, d, d, e, d, d, d, d, d, d, d, outChannels })
            Layers.Add(BlindConv(width));
    }

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            _usingCustomLayers = true;
            Layers.AddRange(Architecture.Layers);
            return;
        }

        AddUNet(_options.FirstStageFeatures);
        AddUNet(_options.DecoderFeatures);
        int stacked = 4 * _options.DecoderFeatures;
        Layers.Add(new ConvolutionalLayer<T>(stacked, 1, 1, 0, new LeakyReLUActivation<T>(LeakySlope), null, null, 1, BiasMode.Never));
        Layers.Add(new ConvolutionalLayer<T>(_options.DecoderFeatures, 1, 1, 0, new LeakyReLUActivation<T>(LeakySlope), null, null, 1, BiasMode.Never));
        Layers.Add(new ConvolutionalLayer<T>(OutputChannels, 1, 1, 0, null, null, null, 1, BiasMode.Never));
    }

    /// <inheritdoc/>
    protected override void ResolveLazyLayerShapes()
    {
        if (_usingCustomLayers || !_useNativeMode) return;
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        Network(new Tensor<T>(new[] { 1, WindowFrames * Channels, 8, 8 }));
    }

    #endregion

    #region Blind-spot network

    private Tensor<T> ShiftDown(Tensor<T> x)
        => Engine.TensorNarrow(Engine.Pad(x, 1, 0, 0, 0, NumOps.Zero), 2, 0, x.Shape[2]);

    private Tensor<T> ReplicatePad(Tensor<T> x)
    {
        int h = x.Shape[2], w = x.Shape[3];
        x = Engine.TensorConcatenate(new[] { Engine.TensorNarrow(x, 2, 0, 1), x, Engine.TensorNarrow(x, 2, h - 1, 1) }, 2);
        return Engine.TensorConcatenate(new[] { Engine.TensorNarrow(x, 3, 0, 1), x, Engine.TensorNarrow(x, 3, w - 1, 1) }, 3);
    }

    // Conv(blind=True): zero row on top, replicate pad, valid 3x3 conv with LeakyReLU(0.1), drop the last row.
    private Tensor<T> Blind(int index, Tensor<T> x)
    {
        int h = x.Shape[2];
        var padded = ReplicatePad(Engine.Pad(x, 1, 0, 0, 0, NumOps.Zero));
        return Engine.TensorNarrow(Layers[index].Forward(padded), 2, 0, h);
    }

    private Tensor<T> Pool(Tensor<T> x) => Engine.MaxPool2D(ShiftDown(x), 2, 2);

    private Tensor<T> Encode(int start, Tensor<T> x, bool reduce)
    {
        x = Blind(start + 2, Blind(start + 1, Blind(start, x)));
        return reduce ? Pool(x) : x;
    }

    private Tensor<T> Decode(int start, Tensor<T> x, Tensor<T> skip)
    {
        var up = Engine.Upsample(x, 2, 2);
        int dy = skip.Shape[2] - up.Shape[2], dx = skip.Shape[3] - up.Shape[3];
        if (dy != 0 || dx != 0) up = Engine.Pad(up, dy / 2, dy - dy / 2, dx / 2, dx - dx / 2, NumOps.Zero);
        x = Engine.TensorConcatenate(new[] { up, skip }, 1);
        return Blind(start + 3, Blind(start + 2, Blind(start + 1, Blind(start, x))));
    }

    // Blind_UNet.forward: two pooled encoders, a third without pooling, and two decoders that skip to the first
    // encoder's output and to the input.
    private Tensor<T> RunUNet(int start, Tensor<T> input)
    {
        var x1 = Encode(start, input, reduce: true);
        var x2 = Encode(start + 3, x1, reduce: true);
        var x = Encode(start + 6, x2, reduce: false);
        x = Decode(start + 9, x, x1);
        return Decode(start + 13, x, input);
    }

    private Tensor<T> Flip(Tensor<T> x, params int[] axes) => Engine.TensorFlip(x, axes);

    private Tensor<T> Transpose(Tensor<T> x) => Engine.TensorPermute(x, new[] { 0, 1, 3, 2 });

    // rotate: the input and its 90, 180 and 270 degree rotations, stacked on the batch.
    internal Tensor<T> Rotate(Tensor<T> x)
        => Engine.TensorConcatenate(new[] { x, Flip(Transpose(x), 3), Flip(x, 2, 3), Flip(Transpose(x), 2) }, 0);

    // unrotate: each quarter rotated back, stacked on the channels.
    internal Tensor<T> Unrotate(Tensor<T> x)
    {
        int n = x.Shape[0] / 4;
        var r0 = Engine.TensorNarrow(x, 0, 0, n);
        var r90 = Flip(Transpose(Engine.TensorNarrow(x, 0, n, n)), 2);
        var r180 = Flip(Engine.TensorNarrow(x, 0, 2 * n, n), 2, 3);
        var r270 = Flip(Transpose(Engine.TensorNarrow(x, 0, 3 * n, n)), 3);
        return Engine.TensorConcatenate(new[] { r0, r90, r180, r270 }, 1);
    }

    private Tensor<T> ReflectPad(Tensor<T> x, int axis, int before, int after)
    {
        var parts = new List<Tensor<T>>();
        int size = x.Shape[axis];
        if (before > 0) parts.Add(Flip(Engine.TensorNarrow(x, axis, 1, before), axis));
        parts.Add(x);
        if (after > 0) parts.Add(Flip(Engine.TensorNarrow(x, axis, size - 1 - after, after), axis));
        return parts.Count == 1 ? x : Engine.TensorConcatenate(parts.ToArray(), axis);
    }

    /// <summary>BlindVideoNet.forward over windows [N, 5C, H, W]: the per-pixel mean and covariance factor.</summary>
    private Tensor<T> Network(Tensor<T> x)
    {
        int c = Channels, h = x.Shape[2], w = x.Shape[3];
        // Square the frames by reflection, as the four rotations need.
        if (h > w) x = ReflectPad(x, 3, (h - w) / 2, h - w - (h - w) / 2);
        else if (w > h) x = ReflectPad(x, 2, (w - h) / 2, w - h - (w - h) / 2);

        var triplets = new Tensor<T>[3];
        for (int k = 0; k < 3; k++)
            triplets[k] = RunUNet(FirstUNet, Rotate(Engine.TensorNarrow(x, 1, k * c, 3 * c)));
        var y = RunUNet(SecondUNet, Engine.TensorConcatenate(triplets, 1));
        y = Unrotate(ShiftDown(y));
        y = Layers[NinStart + 2].Forward(Layers[NinStart + 1].Forward(Layers[NinStart].Forward(y)));

        if (h > w) y = Engine.TensorNarrow(y, 3, (h - w) / 2, w);
        else if (w > h) y = Engine.TensorNarrow(y, 2, (w - h) / 2, h);
        return y;
    }

    #endregion

    #region Windows, likelihood and posterior

    // The five-frame window around each frame, reflected at the ends of the clip.
    private static int Reflect(int index, int length)
    {
        if (length == 1) return 0;
        int period = 2 * (length - 1);
        int k = ((index % period) + period) % period;
        return k < length ? k : period - k;
    }

    /// <summary>Frames [T, C, H, W] or [N, T, C, H, W] as windows [N*T, 5C, H, W], centred on each frame.</summary>
    private Tensor<T> Windows(Tensor<T> frames)
    {
        var clips = frames.Rank == 5 ? frames : Engine.Reshape(frames, new[] { 1 }.Concat(frames._shape).ToArray());
        int n = clips.Shape[0], t = clips.Shape[1], c = clips.Shape[2], h = clips.Shape[3], w = clips.Shape[4];
        var windows = new List<Tensor<T>>();
        for (int b = 0; b < n; b++)
            for (int f = 0; f < t; f++)
            {
                var window = new Tensor<T>[WindowFrames];
                for (int k = 0; k < WindowFrames; k++)
                {
                    int source = Reflect(f + k - WindowFrames / 2, t);
                    window[k] = Engine.Reshape(Engine.TensorNarrow(Engine.TensorNarrow(clips, 0, b, 1), 1, source, 1), new[] { 1, c, h, w });
                }
                windows.Add(Engine.TensorConcatenate(window, 1));
            }
        return Engine.TensorConcatenate(windows.ToArray(), 0);
    }

    private Tensor<T> AsFrames(Tensor<T> perWindow, Tensor<T> like)
    {
        int c = perWindow.Shape[1], h = perWindow.Shape[2], w = perWindow.Shape[3];
        return like.Rank == 5
            ? Engine.Reshape(perWindow, new[] { like.Shape[0], like.Shape[1], c, h, w })
            : perWindow;
    }

    private double WindowNoiseSigma(Tensor<T> normalized)
        => _options.NoiseLevel ?? EstimateNoiseLevel(normalized.Rank == 5 ? normalized.Slice(0) : normalized);

    /// <summary>
    /// utils/loss_funtions.py "loglike": the mean over pixels of 0.5 (t - μ)ᵀ Σ⁻¹ (t - μ) + 0.5 log det Σ with
    /// Σ = AᵀA + (σ² + 1e-5) I, minus 0.1σ, through a Cholesky factorization of the per-pixel C×C matrix.
    /// </summary>
    internal Tensor<T> NegativeLogLikelihood(Tensor<T> output, Tensor<T> target, double sigma)
    {
        int c = Channels;
        Tensor<T> Plane(Tensor<T> t, int channel) => Engine.TensorNarrow(t, 1, channel, 1);

        // ax[i*C : i*C + C - i] holds row i of A: columns 0 .. C-i-1.
        var a = new Tensor<T>?[c, c];
        int index = c;
        for (int i = 0; i < c; i++)
            for (int j = 0; j < c - i; j++)
                a[i, j] = Plane(output, index++);

        var sigma2 = NumOps.FromDouble(sigma * sigma + Epsilon);
        var cov = new Tensor<T>[c, c];
        for (int p = 0; p < c; p++)
            for (int q = 0; q <= p; q++)
            {
                Tensor<T>? sum = null;
                for (int i = 0; i < c; i++)
                {
                    var aip = a[i, p];
                    var aiq = a[i, q];
                    if (aip is null || aiq is null) continue;
                    var term = Engine.TensorMultiply(aip, aiq);
                    sum = sum is null ? term : Engine.TensorAdd(sum, term);
                }
                var entry = sum ?? Engine.TensorMultiplyScalar(Plane(output, 0), NumOps.Zero);
                cov[p, q] = p == q ? Engine.TensorAddScalar(entry, sigma2) : entry;
            }

        var l = new Tensor<T>[c, c];
        var z = new Tensor<T>[c];
        Tensor<T>? logDet = null;
        Tensor<T>? quadratic = null;
        for (int i = 0; i < c; i++)
        {
            for (int j = 0; j < i; j++)
            {
                var value = cov[i, j];
                for (int k = 0; k < j; k++) value = Engine.TensorSubtract(value, Engine.TensorMultiply(l[i, k], l[j, k]));
                l[i, j] = Engine.TensorDivide(value, l[j, j]);
            }
            var diagonal = cov[i, i];
            for (int k = 0; k < i; k++) diagonal = Engine.TensorSubtract(diagonal, Engine.TensorMultiply(l[i, k], l[i, k]));
            l[i, i] = Engine.TensorSqrt(diagonal);

            var residual = Engine.TensorSubtract(Plane(target, i), Plane(output, i));
            for (int k = 0; k < i; k++) residual = Engine.TensorSubtract(residual, Engine.TensorMultiply(l[i, k], z[k]));
            z[i] = Engine.TensorDivide(residual, l[i, i]);

            var squared = Engine.TensorMultiply(z[i], z[i]);
            quadratic = quadratic is null ? squared : Engine.TensorAdd(quadratic, squared);
            var log = Engine.TensorLog(l[i, i]);
            logDet = logDet is null ? log : Engine.TensorAdd(logDet, log);
        }

        // 0.5 quad + 0.5 log det Σ = 0.5 quad + Σ log L_ii.
        var perPixel = Engine.TensorAdd(Engine.TensorMultiplyScalar(quadratic ?? Plane(output, 0), NumOps.FromDouble(0.5)),
            logDet ?? Plane(output, 0));
        var mean = Engine.ReduceMean(perPixel, Enumerable.Range(0, perPixel.Rank).ToArray(), keepDims: false);
        return Engine.TensorAddScalar(mean, NumOps.FromDouble(-0.1 * sigma));
    }

    /// <summary>
    /// utils/post_processing.py: per pixel, Σx = AᵀA + 1e-5 I and E[x|y] = (Σx⁻¹ + σ⁻²I)⁻¹(Σx⁻¹μ + σ⁻²y).
    /// </summary>
    internal Tensor<T> PosteriorMean(Tensor<T> output, Tensor<T> noisy, double sigma)
    {
        int n = output.Shape[0], c = Channels, h = output.Shape[2], w = output.Shape[3], plane = h * w;
        int outC = output.Shape[1];
        var result = new Tensor<T>(new[] { n, c, h, w });
        var o = output.Data.Span;
        var y = noisy.Data.Span;
        var r = result.Data.Span;
        double inv = 1.0 / (sigma * sigma + Epsilon);
        var a = new double[c, c];
        var m = new double[c, c];
        var rhs = new double[c];
        for (int b = 0; b < n; b++)
            for (int p = 0; p < plane; p++)
            {
                Array.Clear(a, 0, a.Length);
                int index = c;
                for (int i = 0; i < c; i++)
                    for (int j = 0; j < c - i; j++)
                        a[i, j] = NumOps.ToDouble(o[(b * outC + index++) * plane + p]);
                var cov = new double[c, c];
                for (int i = 0; i < c; i++)
                    for (int j = 0; j < c; j++)
                    {
                        double s = i == j ? Epsilon : 0;
                        for (int k = 0; k < c; k++) s += a[k, i] * a[k, j];
                        cov[i, j] = s;
                    }
                var precision = Invert(cov);
                for (int i = 0; i < c; i++)
                {
                    double s = inv * NumOps.ToDouble(y[(b * c + i) * plane + p]);
                    for (int j = 0; j < c; j++)
                    {
                        s += precision[i, j] * NumOps.ToDouble(o[(b * outC + j) * plane + p]);
                        m[i, j] = precision[i, j] + (i == j ? inv : 0);
                    }
                    rhs[i] = s;
                }
                var mInverse = Invert(m);
                for (int i = 0; i < c; i++)
                {
                    double s = 0;
                    for (int j = 0; j < c; j++) s += mInverse[i, j] * rhs[j];
                    r[(b * c + i) * plane + p] = NumOps.FromDouble(s);
                }
            }
        return result;
    }

    // Gauss-Jordan with partial pivoting on a small symmetric positive-definite matrix.
    private static double[,] Invert(double[,] matrix)
    {
        int size = matrix.GetLength(0);
        var work = (double[,])matrix.Clone();
        var inverse = new double[size, size];
        for (int i = 0; i < size; i++) inverse[i, i] = 1;
        for (int col = 0; col < size; col++)
        {
            int pivot = col;
            for (int row = col + 1; row < size; row++)
                if (Math.Abs(work[row, col]) > Math.Abs(work[pivot, col])) pivot = row;
            if (pivot != col)
                for (int k = 0; k < size; k++)
                {
                    (work[col, k], work[pivot, k]) = (work[pivot, k], work[col, k]);
                    (inverse[col, k], inverse[pivot, k]) = (inverse[pivot, k], inverse[col, k]);
                }
            double scale = work[col, col];
            for (int k = 0; k < size; k++)
            {
                work[col, k] /= scale;
                inverse[col, k] /= scale;
            }
            for (int row = 0; row < size; row++)
            {
                if (row == col) continue;
                double factor = work[row, col];
                for (int k = 0; k < size; k++)
                {
                    work[row, k] -= factor * work[col, k];
                    inverse[row, k] -= factor * inverse[col, k];
                }
            }
        }
        return inverse;
    }

    /// <summary>Test hook: the network's per-pixel outputs for each frame's window.</summary>
    internal Tensor<T> NetworkOutputForTest(Tensor<T> normalizedFrames) => Network(Windows(normalizedFrames));

    #endregion

    #region Forward and training

    /// <inheritdoc/>
    /// <remarks>The posterior mean of every frame given its five-frame window.</remarks>
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        if (_usingCustomLayers) return base.Forward(input);
        var output = Network(Windows(input));
        var noisy = input.Rank == 5
            ? Engine.Reshape(input, new[] { input.Shape[0] * input.Shape[1], input.Shape[2], input.Shape[3], input.Shape[4] })
            : input;
        return AsFrames(PosteriorMean(output, noisy, WindowNoiseSigma(input)), input);
    }

    /// <inheritdoc/>
    /// <remarks>The network's mean μ of every frame, the differentiable half of its output.</remarks>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        if (_usingCustomLayers) return base.ForwardForTraining(input);
        var output = Network(Windows(PreprocessFrames(input)));
        return AsFrames(Engine.TensorNarrow(output, 1, 0, Channels), input);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The negative log-likelihood of each target frame under the network's Gaussian for its window. With the noisy
    /// frames as the target this is the paper's unsupervised objective.
    /// </remarks>
    protected override IReadOnlyList<TrainingObjective<T>>? DefineTrainingObjectives(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (_usingCustomLayers || !_useNativeMode) return null;
        double sigma = WindowNoiseSigma(PreprocessFrames(input));
        return new[]
        {
            new TrainingObjective<T>(new[] { input, expectedOutput }, data =>
            {
                var noisy = PreprocessFrames(data[0]);
                var target = PreprocessTarget(data[1]);
                if (target.Rank == 5)
                    target = Engine.Reshape(target, new[] { target.Shape[0] * target.Shape[1], target.Shape[2], target.Shape[3], target.Shape[4] });
                return NegativeLogLikelihood(Network(Windows(noisy)), target, sigma);
            })
            {
                Optimizer = _optimizer,
            }
        };
    }

    #endregion

    /// <inheritdoc/>
    protected override Tensor<T> PreprocessFrames(Tensor<T> rawFrames) => NormalizeFrames(rawFrames);

    /// <inheritdoc/>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput) => DenormalizeFrames(modelOutput);

    /// <inheritdoc/>
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
        => _optimizer ?? base.GetOrCreateBaseOptimizer();

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelName", "UDVD" },
                { "EncoderFeatures", _options.EncoderFeatures },
                { "DecoderFeatures", _options.DecoderFeatures },
                { "FirstStageFeatures", _options.FirstStageFeatures }
            },
            ModelDataProvider = () => SerializeForMetadata()
        };
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(UDVD<T>));
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing) OnnxModel?.Dispose();
        base.Dispose(disposing);
    }
}

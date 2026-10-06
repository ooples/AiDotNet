using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Finance.Probabilistic;

/// <summary>
/// Diffusion-TS's denoiser (Yuan &amp; Qiao, ICLR 2024, arXiv:2403.01742): an encoder-decoder transformer whose decoder
/// splits its representation into an interpretable trend (a low-order polynomial basis) and seasonality (the dominant
/// Fourier components), and that predicts the clean window x_0 from x_t and the diffusion step.
/// </summary>
/// <remarks>
/// <para>
/// Follows the authors' implementation (<c>interpretable_diffusion/transformer.py</c>): a 3-tap convolutional embedding
/// with separate learnable positional embeddings for encoder and decoder; encoder blocks of AdaLN (a parameter-free
/// LayerNorm scaled and shifted from the sinusoidal step embedding) -> self-attention, then LayerNorm -> GELU MLP;
/// decoder blocks adding AdaLN cross-attention to the encoder, a 1x1 projection over the time axis split into a
/// TrendBlock input and a FourierLayer input, the MLP, and removal of the per-block mean, which a linear head turns into
/// a level term. The output is trend (summed block trends + combined means + residual mean) plus seasonality (combined
/// block seasonals + residual).
/// </para>
/// <para>
/// The FourierLayer keeps, per channel, the int(log K) frequencies of largest amplitude (DC and Nyquist excluded) and
/// extrapolates them back to the time axis. The transform is a matmul against fixed DFT bases and the top-k selection a
/// rank computed from element-wise comparisons, so every step is a recorded engine operation.
/// </para>
/// </remarks>
internal sealed class DiffusionTSNetwork<T>
{
    private const int TrendPolynomialOrder = 3;

    private readonly int _features;
    private readonly int _window;
    private readonly int _modelDim;
    private readonly int _heads;
    private readonly int _encoderLayers;
    private readonly int _decoderLayers;
    private readonly int _mlpRatio;
    private readonly double _dropout;
    private readonly int _combineKernel;
    private readonly int _fourierTopK;
    private readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    // Fixed bases: DFT over the retained frequencies (FourierLayer) and the trend polynomials.
    private readonly Tensor<T> _fourierCos;
    private readonly Tensor<T> _fourierSin;
    private readonly Tensor<T> _fourierCosInverse;
    private readonly Tensor<T> _fourierSinInverse;
    private readonly Tensor<T> _trendBasis;

    private readonly List<ILayer<T>> _layers = new();
    private IReadOnlyList<ILayer<T>>? _bindSource;

    private Conv1DLayer<T> _embedding;
    private DropoutLayer<T>? _embeddingDropout;
    private LearnedPositionalEmbeddingLayer<T> _encoderPosition;
    private LearnedPositionalEmbeddingLayer<T> _decoderPosition;
    private readonly List<EncoderBlock> _encoder = new();
    private readonly List<DecoderBlock> _decoder = new();
    private Conv1DLayer<T> _inverse;
    private DropoutLayer<T>? _inverseDropout;
    private Conv1DLayer<T> _combineSeasonal;
    private Conv1DLayer<T> _combineMean;

    /// <summary>Builds the denoiser for windows of <paramref name="window"/> steps of <paramref name="features"/> series.</summary>
    public DiffusionTSNetwork(int features, int window, int modelDim, int heads, int encoderLayers, int decoderLayers,
        int mlpRatio, double dropout, int fourierTopKFactor)
    {
        if (features <= 0) throw new ArgumentOutOfRangeException(nameof(features));
        if (window < 2) throw new ArgumentOutOfRangeException(nameof(window), "A window needs at least two steps.");
        if (modelDim <= 0 || modelDim % 2 != 0) throw new ArgumentOutOfRangeException(nameof(modelDim), "The model width must be positive and even.");
        if (heads <= 0 || modelDim % heads != 0) throw new ArgumentOutOfRangeException(nameof(heads), "The model width must divide into the heads.");
        if (encoderLayers <= 0) throw new ArgumentOutOfRangeException(nameof(encoderLayers));
        if (decoderLayers <= 0) throw new ArgumentOutOfRangeException(nameof(decoderLayers));
        if (mlpRatio <= 0) throw new ArgumentOutOfRangeException(nameof(mlpRatio));
        if (double.IsNaN(dropout) || dropout < 0 || dropout >= 1) throw new ArgumentOutOfRangeException(nameof(dropout));
        if (fourierTopKFactor <= 0) throw new ArgumentOutOfRangeException(nameof(fourierTopKFactor));

        _features = features;
        _window = window;
        _modelDim = modelDim;
        _heads = heads;
        _encoderLayers = encoderLayers;
        _decoderLayers = decoderLayers;
        _mlpRatio = mlpRatio;
        _dropout = dropout;
        // The reference combines seasonality with a 1x1 convolution for small problems and a circular 5-tap one otherwise.
        _combineKernel = features < 32 && window < 64 ? 1 : 5;

        // Retained frequencies 1..K: DC dropped, and Nyquist too for an even window, as the reference does.
        int retained = window % 2 == 0 ? window / 2 - 1 : (window - 1) / 2;
        _fourierTopK = retained > 0 ? (int)(fourierTopKFactor * Math.Log(retained)) : 0;
        int columns = Math.Max(retained, 1);
        _fourierCos = new Tensor<T>(new[] { window, columns });
        _fourierSin = new Tensor<T>(new[] { window, columns });
        _fourierCosInverse = new Tensor<T>(new[] { columns, window });
        _fourierSinInverse = new Tensor<T>(new[] { columns, window });
        for (int t = 0; t < window; t++)
            for (int j = 0; j < retained; j++)
            {
                double angle = 2.0 * Math.PI * (j + 1) * t / window;
                _fourierCos[t * columns + j] = _numOps.FromDouble(Math.Cos(angle));
                _fourierSin[t * columns + j] = _numOps.FromDouble(-Math.Sin(angle));
                // A real signal's components at f and -f add up to (2 / L) Re(X e^{i angle}) (the inverse DFT's 1 / L).
                _fourierCosInverse[j * window + t] = _numOps.FromDouble(2.0 * Math.Cos(angle) / window);
                _fourierSinInverse[j * window + t] = _numOps.FromDouble(-2.0 * Math.Sin(angle) / window);
            }

        _trendBasis = new Tensor<T>(new[] { TrendPolynomialOrder, window });
        for (int p = 0; p < TrendPolynomialOrder; p++)
            for (int t = 0; t < window; t++)
                _trendBasis[p * window + t] = _numOps.FromDouble(Math.Pow((t + 1.0) / (window + 1.0), p + 1));

        Build();
    }

    /// <summary>The denoiser's layers in their fixed order, for the owning model to publish.</summary>
    public IReadOnlyList<ILayer<T>> Layers => _layers;

    /// <summary>
    /// Points every role at the layer at the same position of <paramref name="layers"/>, refusing a list whose types or
    /// convolution settings differ from this layout. Does nothing when the roles already point there.
    /// </summary>
    public void BindTo(IReadOnlyList<ILayer<T>> layers)
    {
        if (layers is null) throw new ArgumentNullException(nameof(layers));
        bool same = layers.Count == _layers.Count;
        for (int i = 0; same && i < _layers.Count; i++) same = ReferenceEquals(layers[i], _layers[i]);
        if (same) return;
        if (layers.Count != _layers.Count)
            throw new InvalidOperationException(
                $"Diffusion-TS's graph has {_layers.Count} layers; the supplied list has {layers.Count}.");

        var previous = _layers.ToList();
        _bindSource = layers;
        try
        {
            Build();
        }
        catch
        {
            _bindSource = previous;
            Build();
            throw;
        }
        finally
        {
            _bindSource = null;
        }
    }

    [System.Diagnostics.CodeAnalysis.MemberNotNull(
        nameof(_embedding), nameof(_encoderPosition), nameof(_decoderPosition), nameof(_inverse),
        nameof(_combineSeasonal), nameof(_combineMean))]
    private void Build()
    {
        _layers.Clear();
        _encoder.Clear();
        _decoder.Clear();
        var gelu = (IActivationFunction<T>)new GELUActivation<T>();
        var identity = (IActivationFunction<T>)new IdentityActivation<T>();
        int headDim = _modelDim / _heads;

        _embedding = Add(new Conv1DLayer<T>(_modelDim, 3, padding: 1));
        _embeddingDropout = _dropout > 0 ? Add(new DropoutLayer<T>(_dropout)) : null;
        _encoderPosition = Add(new LearnedPositionalEmbeddingLayer<T>(_window, _modelDim));
        _decoderPosition = Add(new LearnedPositionalEmbeddingLayer<T>(_window, _modelDim));

        for (int i = 0; i < _encoderLayers; i++)
        {
            _encoder.Add(new EncoderBlock(
                Add(new DenseLayer<T>(2 * _modelDim, identity)),
                Add(new MultiHeadAttentionLayer<T>(_heads, headDim, activationFunction: identity)),
                Add(new LayerNormalizationLayer<T>(_modelDim)),
                Add(new DenseLayer<T>(_mlpRatio * _modelDim, gelu)),
                Add(new DenseLayer<T>(_modelDim, identity)),
                _dropout > 0 ? Add(new DropoutLayer<T>(_dropout)) : null));
        }

        for (int i = 0; i < _decoderLayers; i++)
        {
            _decoder.Add(new DecoderBlock(
                Add(new DenseLayer<T>(2 * _modelDim, identity)),
                Add(new MultiHeadAttentionLayer<T>(_heads, headDim, activationFunction: identity)),
                Add(new DenseLayer<T>(2 * _modelDim, identity)),
                Add(new CrossAttentionLayer<T>(_modelDim, _modelDim, _heads, _window, zeroOutputProjection: false)),
                Add(new Conv1DLayer<T>(2 * _window, 1, padding: 0)),
                Add(new Conv1DLayer<T>(TrendPolynomialOrder, 3, padding: 1, activation: gelu)),
                Add(new Conv1DLayer<T>(_features, 3, padding: 1)),
                Add(new LayerNormalizationLayer<T>(_modelDim)),
                Add(new DenseLayer<T>(_mlpRatio * _modelDim, gelu)),
                Add(new DenseLayer<T>(_modelDim, identity)),
                _dropout > 0 ? Add(new DropoutLayer<T>(_dropout)) : null,
                Add(new DenseLayer<T>(_features, identity))));
        }

        _inverse = Add(new Conv1DLayer<T>(_features, 3, padding: 1));
        _inverseDropout = _dropout > 0 ? Add(new DropoutLayer<T>(_dropout)) : null;
        _combineSeasonal = Add(new Conv1DLayer<T>(_features, _combineKernel, padding: 0));
        _combineMean = Add(new Conv1DLayer<T>(1, 1, padding: 0));
    }

    private TLayer Add<TLayer>(TLayer layer) where TLayer : class, ILayer<T>
    {
        if (_bindSource is not null)
        {
            int index = _layers.Count;
            if (index >= _bindSource.Count || _bindSource[index] is not TLayer existing)
                throw new InvalidOperationException(
                    $"The layer list does not match Diffusion-TS's layout at position {index}: expected {typeof(TLayer).Name}, " +
                    $"found {(index < _bindSource.Count ? _bindSource[index].GetType().Name : "the end")}.");
            if (existing is Conv1DLayer<T> boundConvolution && layer is Conv1DLayer<T> layoutConvolution)
            {
                var expected = layoutConvolution.GetMetadata();
                var found = boundConvolution.GetMetadata();
                foreach (var key in new[] { "OutputChannels", "KernelSize", "Dilation", "Stride", "Padding", "Groups" })
                {
                    expected.TryGetValue(key, out var want);
                    found.TryGetValue(key, out var got);
                    if (!string.Equals(want, got, StringComparison.Ordinal))
                        throw new InvalidOperationException(
                            $"The layer list does not match Diffusion-TS's layout at position {index}: the Conv1DLayer's {key} " +
                            $"is {got ?? "unset"}, but the layout needs {want ?? "unset"}.");
                }
            }

            layer = existing;
        }

        _layers.Add(layer);
        return layer;
    }

    /// <summary>
    /// Predicts the clean window x_0 <c>[B, L, F]</c> from the noisy window <paramref name="noisy"/> <c>[B, L, F]</c>
    /// and the diffusion step of each row, <paramref name="steps"/> <c>[B, 1]</c>.
    /// </summary>
    public Tensor<T> PredictCleanWindow(IEngine engine, Tensor<T> noisy, Tensor<T> steps)
    {
        if (noisy is null) throw new ArgumentNullException(nameof(noisy));
        if (steps is null) throw new ArgumentNullException(nameof(steps));
        if (noisy.Rank != 3 || noisy.Shape[1] != _window || noisy.Shape[2] != _features)
            throw new ArgumentException(
                $"Diffusion-TS's denoiser takes [B, {_window}, {_features}]; got [{string.Join(", ", noisy.Shape.ToArray())}].",
                nameof(noisy));
        int batch = noisy.Shape[0];
        var stepEmbedding = StepEmbedding(engine, steps, batch);

        // Conv_MLP embedding over time, then the two learnable positional embeddings.
        var embedded = engine.TensorPermute(_embedding.Forward(engine.TensorPermute(noisy, new[] { 0, 2, 1 })), new[] { 0, 2, 1 });
        if (_embeddingDropout is not null) embedded = _embeddingDropout.Forward(embedded);

        var encoded = _encoderPosition.Forward(embedded);
        foreach (var block in _encoder)
        {
            encoded = engine.TensorAdd(encoded, block.Attention.Forward(AdaptiveNorm(engine, block.AdaNorm, encoded, stepEmbedding)));
            encoded = engine.TensorAdd(encoded, Mlp(engine, block.Norm, block.Up, block.Down, block.Dropout, encoded));
        }

        var x = _decoderPosition.Forward(embedded);
        Tensor<T>? trend = null;
        Tensor<T>? season = null;
        var means = new Tensor<T>[_decoder.Count];
        for (int i = 0; i < _decoder.Count; i++)
        {
            var block = _decoder[i];
            x = engine.TensorAdd(x, block.SelfAttention.Forward(AdaptiveNorm(engine, block.SelfAdaNorm, x, stepEmbedding)));
            x = engine.TensorAdd(x, block.CrossAttention.Forward(AdaptiveNorm(engine, block.CrossAdaNorm, x, stepEmbedding), encoded));
            // The 1x1 projection mixes time steps (the sequence axis is its channel axis); its halves feed the two components.
            var projected = block.Projection.Forward(x);
            var blockTrend = Trend(engine, block, engine.TensorNarrow(projected, 1, 0, _window));
            var blockSeason = Seasonal(engine, engine.TensorNarrow(projected, 1, _window, _window));
            trend = trend is null ? blockTrend : engine.TensorAdd(trend, blockTrend);
            season = season is null ? blockSeason : engine.TensorAdd(season, blockSeason);

            x = engine.TensorAdd(x, Mlp(engine, block.Norm, block.Up, block.Down, block.Dropout, x));
            var mean = engine.ReduceMean(x, new[] { 1 }, keepDims: true);
            x = engine.TensorSubtract(x, engine.TensorBroadcastTo(mean, x.Shape.ToArray()));
            means[i] = engine.Reshape(block.Level.Forward(engine.Reshape(mean, new[] { batch, _modelDim })), new[] { batch, 1, _features });
        }

        if (trend is null || season is null)
            throw new InvalidOperationException("Diffusion-TS's decoder has no blocks.");

        var residual = engine.TensorPermute(_inverse.Forward(engine.TensorPermute(x, new[] { 0, 2, 1 })), new[] { 0, 2, 1 });
        if (_inverseDropout is not null) residual = _inverseDropout.Forward(residual);
        var residualMean = engine.ReduceMean(residual, new[] { 1 }, keepDims: true);
        var full = new[] { batch, _window, _features };

        var seasonFeatures = engine.TensorPermute(
            _combineSeasonal.Forward(CircularPad(engine, engine.TensorPermute(season, new[] { 0, 2, 1 }), _combineKernel / 2)),
            new[] { 0, 2, 1 });
        var seasonError = engine.TensorSubtract(
            engine.TensorAdd(seasonFeatures, residual), engine.TensorBroadcastTo(residualMean, full));
        var level = engine.TensorAdd(
            _combineMean.Forward(engine.TensorConcatenate(means, axis: 1)), residualMean);
        var trendTotal = engine.TensorAdd(trend, engine.TensorBroadcastTo(level, full));
        return engine.TensorAdd(trendTotal, seasonError);
    }

    // SinusoidalPosEmb of the reference: [sin(k w_i), cos(k w_i)], w_i = 10000^(-i/(half-1)), from a [B, 1] step column.
    private Tensor<T> StepEmbedding(IEngine engine, Tensor<T> steps, int batch)
    {
        int half = _modelDim / 2;
        var frequencies = new Tensor<T>(new[] { 1, half });
        double scale = half > 1 ? Math.Log(10000.0) / (half - 1) : 0.0;
        for (int i = 0; i < half; i++) frequencies[i] = _numOps.FromDouble(Math.Exp(-scale * i));
        var shape = new[] { batch, half };
        var phase = engine.TensorMultiply(
            engine.TensorBroadcastTo(engine.Reshape(steps, new[] { batch, 1 }), shape),
            engine.TensorBroadcastTo(frequencies, shape));
        return engine.TensorConcatenate(new[] { engine.TensorSin(phase), engine.TensorCos(phase) }, axis: 1);
    }

    // AdaLayerNorm: a LayerNorm without its own affine, scaled by (1 + scale) and shifted, both from the step embedding.
    private Tensor<T> AdaptiveNorm(IEngine engine, DenseLayer<T> modulation, Tensor<T> x, Tensor<T> stepEmbedding)
    {
        int batch = x.Shape[0];
        var full = x.Shape.ToArray();
        var parameters = modulation.Forward(engine.Swish(stepEmbedding));
        var scale = engine.TensorBroadcastTo(
            engine.Reshape(engine.TensorNarrow(parameters, 1, 0, _modelDim), new[] { batch, 1, _modelDim }), full);
        var shift = engine.TensorBroadcastTo(
            engine.Reshape(engine.TensorNarrow(parameters, 1, _modelDim, _modelDim), new[] { batch, 1, _modelDim }), full);
        var mean = engine.TensorBroadcastTo(engine.ReduceMean(x, new[] { 2 }, keepDims: true), full);
        var variance = engine.TensorBroadcastTo(engine.ReduceVariance(x, new[] { 2 }, keepDims: true), full);
        var normalized = engine.TensorDivide(
            engine.TensorSubtract(x, mean), engine.TensorSqrt(engine.TensorAddScalar(variance, _numOps.FromDouble(1e-5))));
        return engine.TensorAdd(engine.TensorMultiply(normalized, engine.TensorAddScalar(scale, _numOps.One)), shift);
    }

    // LayerNorm -> Linear(d, ratio d) + GELU -> Linear(ratio d, d) [-> dropout], per time step.
    private Tensor<T> Mlp(IEngine engine, LayerNormalizationLayer<T> norm, DenseLayer<T> up, DenseLayer<T> down,
        DropoutLayer<T>? dropout, Tensor<T> x)
    {
        int batch = x.Shape[0], steps = x.Shape[1];
        var rows = engine.Reshape(norm.Forward(x), new[] { batch * steps, _modelDim });
        var output = engine.Reshape(down.Forward(up.Forward(rows)), new[] { batch, steps, _modelDim });
        return dropout is null ? output : dropout.Forward(output);
    }

    // TrendBlock: a 3-tap convolution over the time-mixed input to the polynomial order, a convolution to the features,
    // then the coefficients times the basis (t / (L + 1))^p, p = 1..3.
    private Tensor<T> Trend(IEngine engine, DecoderBlock block, Tensor<T> x)
    {
        int batch = x.Shape[0];
        var coefficients = block.TrendFeatures.Forward(engine.TensorPermute(block.TrendOrder.Forward(x), new[] { 0, 2, 1 }));
        var values = engine.TensorMatMul(
            engine.Reshape(coefficients, new[] { batch * _features, TrendPolynomialOrder }), _trendBasis);
        return engine.TensorPermute(engine.Reshape(values, new[] { batch, _features, _window }), new[] { 0, 2, 1 });
    }

    /// <summary>The number of frequencies the seasonal block keeps per channel.</summary>
    internal int FourierTopK => _fourierTopK;

    /// <summary>The seasonal block on its own, <c>[B, L, d_model]</c> to <c>[B, L, d_model]</c>.</summary>
    internal Tensor<T> SeasonalComponent(IEngine engine, Tensor<T> x) => Seasonal(engine, x);

    // FourierLayer: per channel, keep the top-k amplitude frequencies and extrapolate them back over the window.
    private Tensor<T> Seasonal(IEngine engine, Tensor<T> x)
    {
        int batch = x.Shape[0];
        int rows = batch * _modelDim;
        if (_fourierTopK <= 0)
            return engine.TensorMultiplyScalar(x, _numOps.Zero);

        var signal = engine.Reshape(engine.TensorPermute(x, new[] { 0, 2, 1 }), new[] { rows, _window });
        var real = engine.TensorMatMul(signal, _fourierCos);
        var imaginary = engine.TensorMatMul(signal, _fourierSin);
        int retained = real.Shape[1];
        var power = engine.TensorAdd(engine.TensorMultiply(real, real), engine.TensorMultiply(imaginary, imaginary));

        // Rank of each frequency within its row: how many retained frequencies are strictly stronger.
        var cube = new[] { rows, retained, retained };
        var own = engine.TensorBroadcastTo(engine.Reshape(power, new[] { rows, retained, 1 }), cube);
        var others = engine.TensorBroadcastTo(engine.Reshape(power, new[] { rows, 1, retained }), cube);
        var stronger = engine.ReduceSum(engine.TensorGreaterThan(others, own), new[] { 2 }, keepDims: false);
        var keep = engine.TensorGreaterThan(
            engine.TensorAddScalar(engine.TensorMultiplyScalar(stronger, _numOps.FromDouble(-1)), _numOps.FromDouble(_fourierTopK)),
            _numOps.FromDouble(0.5));

        var time = engine.TensorAdd(
            engine.TensorMatMul(engine.TensorMultiply(real, keep), _fourierCosInverse),
            engine.TensorMatMul(engine.TensorMultiply(imaginary, keep), _fourierSinInverse));
        return engine.TensorPermute(engine.Reshape(time, new[] { batch, _modelDim, _window }), new[] { 0, 2, 1 });
    }

    private static Tensor<T> CircularPad(IEngine engine, Tensor<T> x, int padding)
    {
        if (padding == 0) return x;
        int length = x.Shape[2];
        int wraps = (padding + length - 1) / length;
        var copies = new Tensor<T>[2 * wraps + 1];
        for (int i = 0; i < copies.Length; i++) copies[i] = x;
        return engine.TensorNarrow(engine.TensorConcatenate(copies, axis: 2), 2, wraps * length - padding, length + 2 * padding);
    }

    private sealed class EncoderBlock
    {
        public EncoderBlock(DenseLayer<T> adaNorm, MultiHeadAttentionLayer<T> attention, LayerNormalizationLayer<T> norm,
            DenseLayer<T> up, DenseLayer<T> down, DropoutLayer<T>? dropout)
        {
            AdaNorm = adaNorm; Attention = attention; Norm = norm; Up = up; Down = down; Dropout = dropout;
        }

        public DenseLayer<T> AdaNorm { get; }
        public MultiHeadAttentionLayer<T> Attention { get; }
        public LayerNormalizationLayer<T> Norm { get; }
        public DenseLayer<T> Up { get; }
        public DenseLayer<T> Down { get; }
        public DropoutLayer<T>? Dropout { get; }
    }

    private sealed class DecoderBlock
    {
        public DecoderBlock(DenseLayer<T> selfAdaNorm, MultiHeadAttentionLayer<T> selfAttention, DenseLayer<T> crossAdaNorm,
            CrossAttentionLayer<T> crossAttention, Conv1DLayer<T> projection, Conv1DLayer<T> trendOrder,
            Conv1DLayer<T> trendFeatures, LayerNormalizationLayer<T> norm, DenseLayer<T> up, DenseLayer<T> down,
            DropoutLayer<T>? dropout, DenseLayer<T> level)
        {
            SelfAdaNorm = selfAdaNorm; SelfAttention = selfAttention; CrossAdaNorm = crossAdaNorm; CrossAttention = crossAttention;
            Projection = projection; TrendOrder = trendOrder; TrendFeatures = trendFeatures; Norm = norm; Up = up; Down = down;
            Dropout = dropout; Level = level;
        }

        public DenseLayer<T> SelfAdaNorm { get; }
        public MultiHeadAttentionLayer<T> SelfAttention { get; }
        public DenseLayer<T> CrossAdaNorm { get; }
        public CrossAttentionLayer<T> CrossAttention { get; }
        public Conv1DLayer<T> Projection { get; }
        public Conv1DLayer<T> TrendOrder { get; }
        public Conv1DLayer<T> TrendFeatures { get; }
        public LayerNormalizationLayer<T> Norm { get; }
        public DenseLayer<T> Up { get; }
        public DenseLayer<T> Down { get; }
        public DropoutLayer<T>? Dropout { get; }
        public DenseLayer<T> Level { get; }
    }
}

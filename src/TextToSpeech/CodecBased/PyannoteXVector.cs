using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// SincNet's parametrized band-pass filterbank (Ravanelli and Bengio 2018), as asteroid-filterbanks' <c>ParamSincFB</c>
/// builds it with its odd extension (Pariente et al. 2020): half the filters are cosine (even) band-pass sincs, half
/// their sine (odd) counterparts, from learned low frequencies and bandwidths, windowed by a Hamming window; applied as a
/// strided convolution of a <c>[batch, 1, samples]</c> waveform.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>Band edges are <c>low = min_low + |low_hz|</c> and <c>high = clamp(low + min_band + |band_hz|, min_low,
/// sample_rate / 2)</c>; the parameters start at mel-spaced bands between 30 Hz and
/// <c>sample_rate / 2 − (min_low + min_band)</c>, computed in single precision as the reference does.</remarks>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, TestInputShape = "1, 1, 40", TestConstructorArgs = "4, 11, 2")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Time, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Time, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
internal sealed partial class ParamSincFilterbankLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _filters;
    private readonly int _kernel;
    private readonly int _stride;
    private readonly double _sampleRate;
    private readonly double _minLowHz;
    private readonly double _minBandHz;
    private readonly double[] _window;      // the first half of a Hamming window
    private readonly double[] _time;        // 2π · n / sample_rate for n = −half … −1

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _lowHz;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _bandHz;

    public override bool SupportsTraining => true;

    public ParamSincFilterbankLayer([LayerState] int filters, [LayerState] int kernelSize, [LayerState] int stride,
        [LayerState] double sampleRate = 16000.0, [LayerState] double minLowHz = 50.0, [LayerState] double minBandHz = 50.0)
        : base(new[] { 1 }, new[] { filters })
    {
        if (filters <= 0 || filters % 2 != 0) throw new ArgumentOutOfRangeException(nameof(filters), "The filter count must be even.");
        if (kernelSize <= 0 || kernelSize % 2 == 0) throw new ArgumentOutOfRangeException(nameof(kernelSize), "The kernel must be odd.");
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride));
        _filters = filters;
        _kernel = kernelSize;
        _stride = stride;
        _sampleRate = sampleRate;
        _minLowHz = minLowHz;
        _minBandHz = minBandHz;
        int half = kernelSize / 2, bands = filters / 2;
        _window = new double[half];
        _time = new double[half];
        for (int n = 0; n < half; n++)
        {
            _window[n] = (float)(0.54 - 0.46 * Math.Cos(2 * Math.PI * n / (kernelSize - 1)));      // np.hamming, stored as float32
            _time[n] = (float)(2 * Math.PI) * ((float)(n - half) / (float)sampleRate);              // float32 arithmetic
        }
        // Mel-spaced initial bands: the endpoints and np.linspace in double precision, cast to float32, then the
        // inverse mel scale in float32 (asteroid's _initialize_filters).
        double lowMel = ToMel(30.0), highMel = ToMel(sampleRate / 2 - (minLowHz + minBandHz));
        var hz = new float[bands + 1];
        for (int i = 0; i <= bands; i++)
        {
            float mel = i == bands ? (float)highMel : (float)(lowMel + i * ((highMel - lowMel) / bands));
            hz[i] = 700f * ((float)Math.Pow(10f, mel / 2595f) - 1f);
        }
        _lowHz = new Tensor<T>(new[] { bands, 1 });
        _bandHz = new Tensor<T>(new[] { bands, 1 });
        for (int i = 0; i < bands; i++)
        {
            _lowHz[i, 0] = NumOps.FromDouble(hz[i]);
            _bandHz[i, 0] = NumOps.FromDouble(hz[i + 1] - hz[i]);
        }
        RegisterTrainableParameter(_lowHz, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_bandHz, PersistentTensorRole.Weights);
    }

    private static double ToMel(double hz) => 2595 * Math.Log10(1 + hz / 700);

    internal Tensor<T> LowHz => _lowHz;
    internal Tensor<T> BandHz => _bandHz;

    /// <summary>Throws unless a checkpoint's half window and half time vector equal the ones this layer builds.</summary>
    internal void CheckBuffers(double[] window, double[] time)
    {
        for (int n = 0; n < _window.Length; n++)
        {
            if (Math.Abs(window[n] - _window[n]) > 1e-7 || Math.Abs(time[n] - _time[n]) > 1e-7 * Math.Max(1, Math.Abs(_time[n])))
                throw new System.IO.InvalidDataException(
                    "The checkpoint's sinc filterbank buffers differ from this layer's (a different kernel size or sample rate).");
        }
    }

    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => null;

    /// <summary>The filters <c>[filters, 1, kernel]</c>: cosine band-passes, then sine.</summary>
    internal Tensor<T> Filters()
    {
        int bands = _filters / 2, half = _kernel / 2;
        var low = Engine.TensorAddScalar(Engine.TensorAbs(_lowHz), NumOps.FromDouble(_minLowHz));                       // [B, 1]
        var high = Engine.TensorClamp(
            Engine.TensorAdd(Engine.TensorAddScalar(low, NumOps.FromDouble(_minBandHz)), Engine.TensorAbs(_bandHz)),
            NumOps.FromDouble(_minLowHz), NumOps.FromDouble(_sampleRate / 2));
        var band = Engine.TensorSubtract(high, low);                                                                     // [B, 1]
        var time = new Tensor<T>(new[] { 1, half });
        var window = new Tensor<T>(new[] { 1, half });
        var halfTime = new Tensor<T>(new[] { 1, half });
        for (int n = 0; n < half; n++)
        {
            time[0, n] = NumOps.FromDouble(_time[n]);
            window[0, n] = NumOps.FromDouble(_window[n]);
            halfTime[0, n] = NumOps.FromDouble(_time[n] / 2);
        }
        var ftLow = Engine.TensorMatMul(low, time);                                                                      // [B, half]
        var ftHigh = Engine.TensorMatMul(high, time);
        var windowTiled = Engine.TensorTile(window, new[] { bands, 1 });
        var halfTimeTiled = Engine.TensorTile(halfTime, new[] { bands, 1 });
        var twiceBand = Engine.TensorMultiplyScalar(band, NumOps.FromDouble(2));                                        // [B, 1]

        Tensor<T> Flip(Tensor<T> x)
        {
            var columns = new Tensor<T>[half];
            for (int n = 0; n < half; n++) columns[n] = Engine.TensorSlice(x, new[] { 0, half - 1 - n }, new[] { bands, 1 });
            return Engine.TensorConcatenate(columns, 1);
        }

        Tensor<T> Assemble(Tensor<T> left, Tensor<T> centre, bool negateRight)
        {
            var right = Flip(left);
            if (negateRight) right = Engine.TensorNegate(right);
            var filter = Engine.TensorConcatenate(new[] { left, centre, right }, 1);                                     // [B, kernel]
            return Engine.TensorDivide(filter, Engine.TensorTile(twiceBand, new[] { 1, _kernel }));
        }

        var cosLeft = Engine.TensorMultiply(
            Engine.TensorDivide(Engine.TensorSubtract(Engine.TensorSin(ftHigh), Engine.TensorSin(ftLow)), halfTimeTiled), windowTiled);
        var sinLeft = Engine.TensorMultiply(
            Engine.TensorDivide(Engine.TensorSubtract(Engine.TensorCos(ftLow), Engine.TensorCos(ftHigh)), halfTimeTiled), windowTiled);
        var cosFilters = Assemble(cosLeft, twiceBand, negateRight: false);
        var sinFilters = Assemble(sinLeft, new Tensor<T>(new[] { bands, 1 }), negateRight: true);
        return Engine.Reshape(Engine.TensorConcatenate(new[] { cosFilters, sinFilters }, 0), new[] { _filters, 1, _kernel });
    }

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank != 3 || input.Shape[1] != 1)
            throw new ArgumentException("Expected a waveform [batch, 1, samples].", nameof(input));
        int batch = input.Shape[0], samples = input.Shape[2];
        var filters = Engine.Reshape(Filters(), new[] { _filters, 1, 1, _kernel });
        var x4 = Engine.Reshape(input, new[] { batch, 1, 1, samples });
        var y = Engine.Conv2D(x4, filters, new[] { 1, _stride }, new[] { 0, 0 }, new[] { 1, 1 });                        // [B, F, 1, frames]
        return Engine.Reshape(y, new[] { batch, _filters, y.Shape[3] });
    }

    public override void ResetState()
    {
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var invariant = System.Globalization.CultureInfo.InvariantCulture;
        metadata["Filters"] = _filters.ToString(invariant);
        metadata["KernelSize"] = _kernel.ToString(invariant);
        metadata["Stride"] = _stride.ToString(invariant);
        metadata["SampleRate"] = _sampleRate.ToString("R", invariant);
        metadata["MinLowHz"] = _minLowHz.ToString("R", invariant);
        metadata["MinBandHz"] = _minBandHz.ToString("R", invariant);
        return metadata;
    }
}

/// <summary>
/// The x-vector speaker encoder with a SincNet front end that pyannote.audio ships as <c>XVectorSincNet</c> and Pheme
/// uses through <c>pyannote/embedding</c> (Snyder et al. 2018 with Ravanelli and Bengio 2018's SincNet; Bredin 2023).
/// </summary>
/// <remarks>
/// SincNet: instance-normalized waveform → 80 parametrized sinc filters of 251 taps at stride 10 → |·| → max-pool 3 →
/// instance norm → leaky ReLU; then twice a 5-wide convolution to 60 channels → max-pool 3 → instance norm → leaky ReLU.
/// Five TDNN layers (dilated 1-D convolutions to 512, 512, 512, 512 and 1500 channels with kernels 5, 3, 3, 1, 1 and
/// dilations 1, 2, 3, 1, 1), each followed by leaky ReLU and batch norm; statistics pooling (the mean and unbiased
/// standard deviation over time); and a linear projection to a 512-dimensional embedding. 16 kHz input.
/// </remarks>
internal sealed class PyannoteXVector<T>
{
    private static readonly int[] TdnnChannels = { 512, 512, 512, 512, 1500 };
    private static readonly int[] TdnnKernels = { 5, 3, 3, 1, 1 };
    private static readonly int[] TdnnDilations = { 1, 2, 3, 1, 1 };
    private readonly IEngine _engine;

    public PyannoteXVector(IEngine engine, List<LayerBase<T>> layers, int dimension = 512)
    {
        _engine = engine;
        Dimension = dimension;
        WaveNorm = Own(layers, new InstanceNormalizationLayer<T>(1));
        Sinc = Own(layers, new ParamSincFilterbankLayer<T>(80, 251, 10));
        SincNorms.Add(Own(layers, new InstanceNormalizationLayer<T>(80)));
        SincConvs.Add(Own(layers, Conv(80, 60, 5, 1)));
        SincNorms.Add(Own(layers, new InstanceNormalizationLayer<T>(60)));
        SincConvs.Add(Own(layers, Conv(60, 60, 5, 1)));
        SincNorms.Add(Own(layers, new InstanceNormalizationLayer<T>(60)));
        int input = 60;
        for (int i = 0; i < TdnnChannels.Length; i++)
        {
            Tdnns.Add(Own(layers, Conv(input, TdnnChannels[i], TdnnKernels[i], TdnnDilations[i])));
            TdnnNorms.Add(Own(layers, new BatchNormalizationLayer<T>(TdnnChannels[i], epsilon: 1e-5)));
            input = TdnnChannels[i];
        }
        Embedding = Own(layers, new DenseLayer<T>(dimension, new AiDotNet.ActivationFunctions.IdentityActivation<T>() as IActivationFunction<T>));
    }

    public int Dimension { get; }
    public InstanceNormalizationLayer<T> WaveNorm { get; }
    public ParamSincFilterbankLayer<T> Sinc { get; }
    public List<InstanceNormalizationLayer<T>> SincNorms { get; } = new();
    public List<NormedConv1DLayer<T>> SincConvs { get; } = new();
    public List<NormedConv1DLayer<T>> Tdnns { get; } = new();
    public List<BatchNormalizationLayer<T>> TdnnNorms { get; } = new();
    public DenseLayer<T> Embedding { get; }

    private static NormedConv1DLayer<T> Conv(int input, int output, int kernel, int dilation) =>
        new(input, output, kernel, 1, dilation, 1, 0, false, ConvolutionNormalization.None);

    private static TLayer Own<TLayer>(List<LayerBase<T>> layers, TLayer layer) where TLayer : LayerBase<T>
    {
        layers.Add(layer);
        return layer;
    }

    private Tensor<T> LeakyRelu(Tensor<T> x) => _engine.LeakyReLU(x, MathHelper.GetNumericOperations<T>().FromDouble(0.01));

    // MaxPool1d(3, stride 3) over the last axis of [batch, channels, time], dropping a partial window.
    private Tensor<T> MaxPool3(Tensor<T> x)
    {
        int batch = x.Shape[0], channels = x.Shape[1], frames = x.Shape[2] / 3;
        var kept = _engine.TensorSlice(x, new[] { 0, 0, 0 }, new[] { batch, channels, frames * 3 });
        return _engine.ReduceMax(_engine.Reshape(kept, new[] { batch, channels, frames, 3 }), new[] { 3 }, keepDims: false, out _);
    }

    /// <summary>The speaker embedding <c>[batch, dimension]</c> of 16 kHz waveforms <c>[batch, 1, samples]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> waveform)
    {
        var x = Sinc.Forward(WaveNorm.Forward(waveform));
        x = LeakyRelu(SincNorms[0].Forward(MaxPool3(_engine.TensorAbs(x))));
        for (int i = 0; i < SincConvs.Count; i++)
            x = LeakyRelu(SincNorms[i + 1].Forward(MaxPool3(SincConvs[i].Forward(x))));
        for (int i = 0; i < Tdnns.Count; i++)
        {
            var activated = LeakyRelu(Tdnns[i].Forward(x));                                                            // [B, C, T]
            int batch = activated.Shape[0], channels = activated.Shape[1], frames = activated.Shape[2];
            var normed = TdnnNorms[i].Forward(_engine.Reshape(activated, new[] { batch, channels, frames, 1 }));        // NCHW: channels on axis 1
            x = _engine.Reshape(normed, new[] { batch, channels, frames });
        }
        return Embedding.Forward(StatisticsPool(x));
    }

    // The mean and the unbiased standard deviation over time, concatenated: [batch, 2 · channels].
    private Tensor<T> StatisticsPool(Tensor<T> x)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int channels = x.Shape[1], frames = x.Shape[2];
        var mean = _engine.ReduceMean(x, new[] { 2 }, keepDims: true);                                                 // [B, C, 1]
        var centred = _engine.TensorSubtract(x, _engine.TensorTile(mean, new[] { 1, 1, frames }));
        var variance = _engine.TensorMultiplyScalar(
            _engine.ReduceSum(_engine.TensorMultiply(centred, centred), new[] { 2 }, keepDims: false), ops.FromDouble(1.0 / (frames - 1)));
        var std = _engine.TensorSqrt(variance);
        return _engine.TensorConcatenate(new[] { _engine.Reshape(mean, new[] { x.Shape[0], channels }), std }, 1);
    }

    /// <summary>Loads a pyannote.audio <c>XVectorSincNet</c> state dict (the <c>pyannote/embedding</c> checkpoint's
    /// layout); <paramref name="read"/> checks each tensor's shape.</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        void Fill(Tensor<T> target, double[] values)
        {
            for (int i = 0; i < values.Length; i++) target[i] = ops.FromDouble(values[i]);
            _engine.InvalidatePersistentTensor(target);
        }
        void Instance(string name, InstanceNormalizationLayer<T> norm, int channels)
        {
            Fill(norm.GetGammaTensor(), read(name + ".weight", new[] { channels }));
            Fill(norm.GetBetaTensor(), read(name + ".bias", new[] { channels }));
        }
        void Convolution(string name, NormedConv1DLayer<T> conv, int input, int output, int kernel) =>
            conv.LoadTorchWeights(read(name + ".weight", new[] { output, input, kernel }), null, read(name + ".bias", new[] { output }));

        Instance("sincnet.wav_norm1d", WaveNorm, 1);
        // The filterbank's fixed buffers are rebuilt here; they must agree with the checkpoint's.
        Sinc.CheckBuffers(read("sincnet.conv1d.0.filterbank.window_", new[] { 125 }),
            read("sincnet.conv1d.0.filterbank.n_", new[] { 1, 125 }));
        Fill(Sinc.LowHz, read("sincnet.conv1d.0.filterbank.low_hz_", new[] { 40, 1 }));
        Fill(Sinc.BandHz, read("sincnet.conv1d.0.filterbank.band_hz_", new[] { 40, 1 }));
        Instance("sincnet.norm1d.0", SincNorms[0], 80);
        Convolution("sincnet.conv1d.1", SincConvs[0], 80, 60, 5);
        Instance("sincnet.norm1d.1", SincNorms[1], 60);
        Convolution("sincnet.conv1d.2", SincConvs[1], 60, 60, 5);
        Instance("sincnet.norm1d.2", SincNorms[2], 60);
        int input = 60;
        for (int i = 0; i < Tdnns.Count; i++)
        {
            int channels = TdnnChannels[i];
            Convolution($"tdnns.{3 * i}", Tdnns[i], input, channels, TdnnKernels[i]);
            var norm = TdnnNorms[i];
            using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
                norm.Forward(new Tensor<T>(new[] { 1, channels, 2, 1 }));                                               // allocate the statistics
            Fill(norm.GetGamma(), read($"tdnns.{3 * i + 2}.weight", new[] { channels }));
            Fill(norm.GetBeta(), read($"tdnns.{3 * i + 2}.bias", new[] { channels }));
            Fill(norm.GetRunningMean(), read($"tdnns.{3 * i + 2}.running_mean", new[] { channels }));
            Fill(norm.GetRunningVariance(), read($"tdnns.{3 * i + 2}.running_var", new[] { channels }));
            input = channels;
        }
        TorchParameters.Linear(_engine, Embedding, 2 * input, Dimension,
            read("embedding.weight", new[] { Dimension, 2 * input }), read("embedding.bias", new[] { Dimension }));
    }
}

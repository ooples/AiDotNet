using AiDotNet.ActivationFunctions;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.SpeechRecognition.Foundation;

/// <summary>
/// The BEST-RQ network (Chiu et al. 2022): Conformer XL (Zhang et al. 2020, Table 1) over normalized log-mel features,
/// the fixed random-projection quantizer that gives the masked-prediction targets (section 3.1), the pre-training
/// softmax layer (section 3.2) and the fine-tuning projection with a CTC layer on top (section 3.3).
/// </summary>
/// <remarks>
/// The encoder is the 0.6B non-streaming model of section 4.1.1: two stride-2 2-D convolutions giving 4× temporal
/// reduction, then a linear layer and a stack of Conformer blocks without relative positional embedding (Zhang et al.
/// 2020, section 2.1). The paper fine-tunes with an RNN-T decoder; this class puts a CTC layer on the projection instead.
/// Its layers are added to the owner's list; the quantizer's matrix and codebook are not layers, since they are fixed.
/// </remarks>
internal sealed class BestRqNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly BESTRQOptions _o;
    private readonly MelSpectrogram<T> _mel;
    private readonly int _subsampledBins;
    // The quantizer (section 3.1): A is CodeDim × (NumMels · StackedFrames), Xavier-initialized; the codebook's entries
    // are standard normal and stored l2-normalized, since Equation 1 only compares normalized vectors.
    private readonly double[,] _projection;
    private readonly double[,] _codebook;

    public BestRqNetwork(IEngine engine, List<LayerBase<T>> layers, BESTRQOptions options, Random quantizerRandom)
    {
        _engine = engine ?? throw new ArgumentNullException(nameof(engine));
        _o = options ?? throw new ArgumentNullException(nameof(options));
        if (quantizerRandom is null) throw new ArgumentNullException(nameof(quantizerRandom));
        TLayer Own<TLayer>(TLayer layer) where TLayer : LayerBase<T>
        {
            layers.Add(layer);
            return layer;
        }
        var identity = (IActivationFunction<T>)new IdentityActivation<T>();
        int d = _o.EncoderDim;
        if (d % _o.NumAttentionHeads != 0)
            throw new ArgumentException($"EncoderDim {d} must divide into {_o.NumAttentionHeads} heads.", nameof(options));
        _mel = new MelSpectrogram<T>(sampleRate: _o.SampleRate, nMels: _o.NumMels, nFft: _o.WindowLength,
            hopLength: _o.HopLength, logMel: true, logOffset: _o.LogMelFloor);
        // The paper gives the strides; the 3×3 kernels with ReLU are ESPnet's Conv2dSubsampling, as W2vBertNetwork uses.
        FirstConv = Own(new Conv2DLayer<T>(1, d, 3, 3, 2, 2, 0, 0));
        SecondConv = Own(new Conv2DLayer<T>(d, d, 3, 3, 2, 2, 0, 0));
        _subsampledBins = Subsampled(Subsampled(_o.NumMels));
        if (_subsampledBins <= 0) throw new ArgumentException("Too few mel bins for two stride-2 3×3 convolutions.", nameof(options));
        InputProjection = Own(new DenseLayer<T>(d, identity));
        for (int i = 0; i < _o.NumEncoderLayers; i++)
            Blocks.Add(Own(new ConformerBlockLayer<T>(d, _o.NumAttentionHeads, _o.FeedForwardMultiplier, _o.ConvKernel, ropeTheta: 0)));
        PretrainingHead = Own(new DenseLayer<T>(_o.CodebookSize, identity));
        FineTuningProjection = Own(new DenseLayer<T>(d, identity));
        CtcHead = Own(new DenseLayer<T>(_o.VocabSize, identity));

        int inputs = _o.NumMels * _o.StackedFrames, h = _o.CodeDim;
        double bound = Math.Sqrt(6.0 / (inputs + h));
        _projection = new double[h, inputs];
        for (int r = 0; r < h; r++)
            for (int c = 0; c < inputs; c++)
                _projection[r, c] = (2 * quantizerRandom.NextDouble() - 1) * bound;
        _codebook = new double[_o.CodebookSize, h];
        for (int k = 0; k < _o.CodebookSize; k++)
        {
            double norm = 0;
            for (int c = 0; c < h; c++)
            {
                double u1 = 1.0 - quantizerRandom.NextDouble(), u2 = quantizerRandom.NextDouble();
                _codebook[k, c] = Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
                norm += _codebook[k, c] * _codebook[k, c];
            }
            norm = Math.Sqrt(norm);
            for (int c = 0; c < h; c++) _codebook[k, c] /= norm;
        }
    }

    private static int Subsampled(int length) => (length - 3) / 2 + 1;

    public Conv2DLayer<T> FirstConv { get; }
    public Conv2DLayer<T> SecondConv { get; }
    public DenseLayer<T> InputProjection { get; }
    public List<ConformerBlockLayer<T>> Blocks { get; } = new();
    public DenseLayer<T> PretrainingHead { get; }
    public DenseLayer<T> FineTuningProjection { get; }
    public DenseLayer<T> CtcHead { get; }

    /// <summary>Encoder frames of <paramref name="melFrames"/> log-mel frames (two stride-2 3×3 convolutions).</summary>
    public int EncodedLength(int melFrames) => Subsampled(Subsampled(melFrames));

    /// <summary>Log-mel features <c>[frames, bins]</c> of a waveform <c>[S]</c>, each bin normalized to zero mean and unit
    /// standard deviation over the utterance (section 3: "The input data is normalized to have 0 mean and standard
    /// deviation of 1").</summary>
    public double[,] Features(IReadOnlyList<double> waveform)
    {
        var samples = new Tensor<T>(new[] { waveform.Count });
        for (int i = 0; i < waveform.Count; i++) samples[i] = NumOps.FromDouble(waveform[i]);
        var mel = _mel.Forward(samples);
        int frames = mel.Shape[0], bins = mel.Shape[1];
        var features = new double[frames, bins];
        for (int b = 0; b < bins; b++)
        {
            double mean = 0, square = 0;
            for (int t = 0; t < frames; t++) mean += NumOps.ToDouble(mel[t, b]);
            mean /= frames;
            for (int t = 0; t < frames; t++)
            {
                double v = NumOps.ToDouble(mel[t, b]) - mean;
                square += v * v;
            }
            double std = Math.Sqrt(square / frames);
            for (int t = 0; t < frames; t++)
                features[t, b] = std > 1e-8 ? (NumOps.ToDouble(mel[t, b]) - mean) / std : 0.0;
        }
        return features;
    }

    /// <summary>
    /// The quantized label of every encoder frame (Equation 1): each group of <see cref="BESTRQOptions.StackedFrames"/>
    /// input frames, stacked, is projected by A, and the label is the index of the nearest codebook entry after both are
    /// l2-normalized. Frames past the last full group are not labelled.
    /// </summary>
    public int[] Quantize(double[,] features)
    {
        int frames = features.GetLength(0), bins = features.GetLength(1), stack = _o.StackedFrames, h = _o.CodeDim;
        var labels = new int[frames / stack];
        var projected = new double[h];
        for (int g = 0; g < labels.Length; g++)
        {
            double norm = 0;
            for (int r = 0; r < h; r++)
            {
                double sum = 0;
                for (int s = 0; s < stack; s++)
                    for (int b = 0; b < bins; b++)
                        sum += _projection[r, s * bins + b] * features[g * stack + s, b];
                projected[r] = sum;
                norm += sum * sum;
            }
            norm = Math.Sqrt(norm);
            int best = 0;
            double bestDistance = double.PositiveInfinity;
            for (int k = 0; k < _o.CodebookSize; k++)
            {
                double distance = 0;
                for (int r = 0; r < h; r++)
                {
                    double diff = _codebook[k, r] - (norm > 0 ? projected[r] / norm : 0.0);
                    distance += diff * diff;
                }
                if (distance < bestDistance)
                {
                    bestDistance = distance;
                    best = k;
                }
            }
            labels[g] = best;
        }
        return labels;
    }

    /// <summary>
    /// Masks the input frames (section 3): every frame starts a span of <see cref="BESTRQOptions.MaskLength"/> frames with
    /// probability <see cref="BESTRQOptions.MaskProbability"/>, spans may overlap, and masked frames are replaced by
    /// N(0, <see cref="BESTRQOptions.MaskNoiseStd"/>²) noise. Returns the masked features and the mask.
    /// </summary>
    public (double[,] Features, bool[] Mask) MaskInput(double[,] features, Random random)
    {
        int frames = features.GetLength(0), bins = features.GetLength(1);
        var mask = new bool[frames];
        for (int t = 0; t < frames; t++)
            if (random.NextDouble() < _o.MaskProbability)
                for (int s = t; s < Math.Min(frames, t + _o.MaskLength); s++) mask[s] = true;
        var masked = (double[,])features.Clone();
        for (int t = 0; t < frames; t++)
        {
            if (!mask[t]) continue;
            for (int b = 0; b < bins; b++)
            {
                double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
                masked[t, b] = _o.MaskNoiseStd * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
            }
        }
        return (masked, mask);
    }

    /// <summary>The encoder output <c>[L, EncoderDim]</c> of features <c>[frames, bins]</c>.</summary>
    public Tensor<T> Encode(double[,] features, bool training)
    {
        int frames = features.GetLength(0), bins = features.GetLength(1), d = _o.EncoderDim;
        var mel = new Tensor<T>(new[] { 1, 1, frames, bins });
        for (int t = 0; t < frames; t++)
            for (int b = 0; b < bins; b++)
                mel[0, 0, t, b] = NumOps.FromDouble(features[t, b]);
        var x = _engine.ReLU(FirstConv.Forward(mel));
        x = _engine.ReLU(SecondConv.Forward(x));                                                               // [1, D, L, F']
        int length = x.Shape[2];
        if (length <= 0) throw new ArgumentException("The audio is too short for the feature encoder.");
        var steps = _engine.Reshape(_engine.TensorPermute(x, new[] { 0, 2, 1, 3 }), new[] { length, d * _subsampledBins });
        var h = InputProjection.Forward(steps);
        foreach (var block in Blocks)
        {
            block.SetTrainingMode(training);
            h = block.Forward(h);
        }
        return h;
    }

    /// <summary>The CTC logits <c>[L, VocabSize]</c>: the fine-tuning projection and the CTC layer on the encoder.</summary>
    public Tensor<T> CtcLogits(Tensor<T> encoded) => CtcHead.Forward(FineTuningProjection.Forward(encoded));

    /// <summary>
    /// The pre-training loss of one waveform (section 3.2): the cross-entropy of the softmax layer's prediction of the
    /// quantized label at every encoder frame whose input frames were masked, averaged over those frames. The labels come
    /// from the unmasked features; the encoder sees the masked ones. Zero when nothing is masked.
    /// </summary>
    public Tensor<T> PretrainingLoss(IReadOnlyList<double> waveform, bool training, Random random)
    {
        var features = Features(waveform);
        var labels = Quantize(features);
        var (masked, mask) = MaskInput(features, random);
        var logits = PretrainingHead.Forward(Encode(masked, training));                                       // [L, codes]
        int length = Math.Min(logits.Shape[0], labels.Length), stack = _o.StackedFrames;
        var weights = new Tensor<T>(logits._shape);
        int count = 0;
        for (int g = 0; g < length; g++)
        {
            bool any = false;
            for (int s = 0; s < stack; s++) any |= mask[g * stack + s];
            if (!any) continue;
            weights[g, labels[g]] = NumOps.FromDouble(-1.0);
            count++;
        }
        var total = _engine.ReduceSum(_engine.TensorMultiply(_engine.TensorLogSoftmax(logits, axis: -1), weights), new[] { 0, 1 }, keepDims: false);
        return count == 0 ? total : _engine.TensorMultiplyScalar(total, NumOps.FromDouble(1.0 / count));
    }
}

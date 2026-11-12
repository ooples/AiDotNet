using AiDotNet.ActivationFunctions;
using AiDotNet.Audio.Foundations;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.SpeechRecognition.Foundation;

/// <summary>A SPIRAL network's shape (Huang et al. 2022, Table 1; defaults are the BASE model).</summary>
internal sealed record SpiralConfiguration
{
    public int SampleRate { get; init; } = 16000;
    public int NumMels { get; init; } = 128;
    public int WindowLength { get; init; } = 320;
    public int HopLength { get; init; } = 160;
    public double LogMelFloor { get; init; } = 1e-6;

    /// <summary>The first block's convolutions: kernel, channels and stride per layer; the last has no ReLU.</summary>
    public (int Kernel, int Channels, int Stride)[] FirstConvolutions { get; init; } = { (5, 384, 2), (5, 512, 2), (1, 512, 1) };
    public int FirstLayers { get; init; } = 2;
    public int FirstHeads { get; init; } = 8;
    public int FirstFeedForward { get; init; } = 2048;
    public double FirstLayerDrop { get; init; }

    /// <summary>The second block's convolutions; the last sets the encoder width and has no ReLU.</summary>
    public (int Kernel, int Channels, int Stride)[] SecondConvolutions { get; init; } = { (5, 1536, 2), (1, 768, 1) };
    public int SecondLayers { get; init; } = 10;
    public int SecondHeads { get; init; } = 12;
    public int SecondFeedForward { get; init; } = 3072;
    public double SecondLayerDrop { get; init; } = 0.05;

    public int PositionalConvKernel { get; init; } = 128;
    public int PositionalConvGroups { get; init; } = 16;
    public double Dropout { get; init; } = 0.1;
    public int ProjectionDim { get; init; } = 256;
    public int PredictorChannels { get; init; } = 256;
    public int PredictorKernel { get; init; } = 5;
    public int ClassifierChannels { get; init; } = 512;
    public int ClassifierKernel { get; init; } = 5;
    public int VocabSize { get; init; } = 1024;
}

/// <summary>
/// The SPIRAL network (Huang et al. 2022, section 3.2, Figure 2): an encoder of two blocks, each temporal convolutions
/// that down-sample (layer norm and ReLU after each) followed by a pre-norm Transformer with wav2vec 2.0's convolutional
/// relative position encoding, then a linear projection head and the student's predictor (two temporal convolutions with
/// batch norm and ReLU, and a linear layer). Fine-tuning puts the convolutional classifier of section 3.5 (two
/// convolutions with layer norm and ReLU, and a linear output) on the encoder.
/// </summary>
/// <remarks>
/// The input is 128-bin log-mel at a 10 ms stride (section 4.2); the blocks reduce it to 40 ms and then 80 ms frames.
/// Layer details the paper leaves open follow the authors' released configuration (huawei-noah/Speech-Backbones):
/// "same" padding for the convolutions, no ReLU after the kernel-1 convolution closing each block, and fairseq's pre-norm
/// wav2vec 2.0 Transformer (<see cref="PreNormTransformerEncoder{T}"/>) with 128-tap positional convolutions in 16 groups.
/// Its layers are added to the given list.
/// </remarks>
internal sealed class SpiralNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly SpiralConfiguration _c;
    private readonly MelSpectrogram<T> _mel;

    internal sealed record Convolution(DenseLayer<T> Map, LayerNormalizationLayer<T> Norm, int Kernel, int Stride, bool Relu);

    internal sealed record Block(List<Convolution> Convolutions, PreNormTransformerEncoder<T> Transformer, double LayerDrop);

    internal sealed record PredictorConvolution(DenseLayer<T> Map, BatchNormalizationLayer<T> Norm);

    public SpiralNetwork(IEngine engine, List<LayerBase<T>> layers, SpiralConfiguration configuration)
    {
        _engine = engine ?? throw new ArgumentNullException(nameof(engine));
        _c = configuration ?? throw new ArgumentNullException(nameof(configuration));
        TLayer Own<TLayer>(TLayer layer) where TLayer : LayerBase<T>
        {
            layers.Add(layer);
            return layer;
        }
        var identity = (IActivationFunction<T>)new IdentityActivation<T>();
        DenseLayer<T> Linear(int outputs) => Own(new DenseLayer<T>(outputs, identity));
        _mel = new MelSpectrogram<T>(sampleRate: _c.SampleRate, nMels: _c.NumMels, nFft: _c.WindowLength,
            hopLength: _c.HopLength, logMel: true, logOffset: _c.LogMelFloor);

        Block Build((int Kernel, int Channels, int Stride)[] convolutions, int count, int heads, int feedForward, double layerDrop)
        {
            var convs = new List<Convolution>();
            for (int i = 0; i < convolutions.Length; i++)
            {
                var (kernel, channels, stride) = convolutions[i];
                convs.Add(new Convolution(Linear(channels), Own(new LayerNormalizationLayer<T>(channels)), kernel, stride,
                    Relu: i < convolutions.Length - 1));
            }
            var transformer = new PreNormTransformerEncoder<T>(_engine, layers, convolutions[^1].Channels, count, heads, feedForward,
                _c.PositionalConvKernel, _c.PositionalConvGroups);
            return new Block(convs, transformer, layerDrop);
        }

        First = Build(_c.FirstConvolutions, _c.FirstLayers, _c.FirstHeads, _c.FirstFeedForward, _c.FirstLayerDrop);
        Second = Build(_c.SecondConvolutions, _c.SecondLayers, _c.SecondHeads, _c.SecondFeedForward, _c.SecondLayerDrop);
        ProjectionHead = Linear(_c.ProjectionDim);
        for (int i = 0; i < 2; i++)
            PredictorConvolutions.Add(new PredictorConvolution(Linear(_c.PredictorChannels), Own(new BatchNormalizationLayer<T>(_c.PredictorChannels))));
        PredictorOutput = Linear(_c.ProjectionDim);
        for (int i = 0; i < 2; i++)
            ClassifierConvolutions.Add(new Convolution(Linear(_c.ClassifierChannels), Own(new LayerNormalizationLayer<T>(_c.ClassifierChannels)),
                _c.ClassifierKernel, 1, Relu: true));
        ClassifierOutput = Linear(_c.VocabSize);
    }

    /// <summary>Initializes both blocks' weight-normalized position convolutions as fairseq does.</summary>
    public void InitializePositionalConvolutions(Random random)
    {
        First.Transformer.InitializePositionalConvolution(random);
        Second.Transformer.InitializePositionalConvolution(random);
    }

    public Block First { get; }
    public Block Second { get; }
    public DenseLayer<T> ProjectionHead { get; }
    public List<PredictorConvolution> PredictorConvolutions { get; } = new();
    public DenseLayer<T> PredictorOutput { get; }
    public List<Convolution> ClassifierConvolutions { get; } = new();
    public DenseLayer<T> ClassifierOutput { get; }

    /// <summary>The input frames per encoder frame: the product of every convolution stride (8 for BASE and LARGE).</summary>
    public int TotalStride => _c.FirstConvolutions.Concat(_c.SecondConvolutions).Aggregate(1, (s, c) => s * c.Stride);

    /// <summary>Encoder frames of <paramref name="frames"/> input frames: each strided "same" convolution rounds up.</summary>
    public int EncodedLength(int frames)
    {
        foreach (var c in _c.FirstConvolutions.Concat(_c.SecondConvolutions)) frames = (frames + c.Stride - 1) / c.Stride;
        return frames;
    }

    /// <summary>Log-mel features <c>[frames, bins]</c> of a waveform: the waveform scaled to zero mean and unit variance,
    /// then each bin normalized over the utterance, as the released preprocessing does (normalize_time_domain,
    /// per_feature).</summary>
    public double[,] Features(IReadOnlyList<double> waveform)
    {
        int n = waveform.Count;
        double mean = waveform.Average(), std = Math.Sqrt(waveform.Sum(v => (v - mean) * (v - mean)) / Math.Max(1, n));
        var samples = new Tensor<T>(new[] { n });
        for (int i = 0; i < n; i++) samples[i] = NumOps.FromDouble(std > 1e-8 ? (waveform[i] - mean) / std : 0.0);
        var mel = _mel.Forward(samples);
        int frames = mel.Shape[0], bins = mel.Shape[1];
        var features = new double[frames, bins];
        for (int b = 0; b < bins; b++)
        {
            double m = 0, square = 0;
            for (int t = 0; t < frames; t++) m += NumOps.ToDouble(mel[t, b]);
            m /= frames;
            for (int t = 0; t < frames; t++)
            {
                double v = NumOps.ToDouble(mel[t, b]) - m;
                square += v * v;
            }
            double s = Math.Sqrt(square / frames);
            for (int t = 0; t < frames; t++) features[t, b] = s > 1e-8 ? (NumOps.ToDouble(mel[t, b]) - m) / s : 0.0;
        }
        return features;
    }

    public Tensor<T> ToTensor(double[,] features)
    {
        int frames = features.GetLength(0), bins = features.GetLength(1);
        var x = new Tensor<T>(new[] { frames, bins });
        for (int t = 0; t < frames; t++)
            for (int b = 0; b < bins; b++) x[t, b] = NumOps.FromDouble(features[t, b]);
        return x;
    }

    /// <summary>The encoder output <c>[L, width]</c> of features <c>[frames, bins]</c>.</summary>
    public Tensor<T> Encode(Tensor<T> features, bool training, Random random)
    {
        var x = RunBlock(First, features, training, random);
        return RunBlock(Second, x, training, random);
    }

    /// <summary>The projection head's output <c>[L, ProjectionDim]</c> (the teacher's representation).</summary>
    public Tensor<T> Project(Tensor<T> encoded) => ProjectionHead.Forward(encoded);

    /// <summary>The predictor on the projected representation (the student's final output).</summary>
    public Tensor<T> Predictor(Tensor<T> projected, bool training)
    {
        var x = projected;
        foreach (var p in PredictorConvolutions)
        {
            p.Norm.SetTrainingMode(training);
            x = _engine.ReLU(p.Norm.Forward(Convolve(p.Map, x, _c.PredictorKernel, 1)));
        }
        return PredictorOutput.Forward(x);
    }

    /// <summary>The CTC logits <c>[L, VocabSize]</c>: the convolutional classifier on the encoder output.</summary>
    public Tensor<T> CtcLogits(Tensor<T> encoded, bool training, Random random)
    {
        var x = encoded;
        foreach (var conv in ClassifierConvolutions) x = Drop(ConvBlock(conv, x), _c.Dropout, training, random);
        return ClassifierOutput.Forward(x);
    }

    private Tensor<T> RunBlock(Block block, Tensor<T> x, bool training, Random random)
    {
        foreach (var conv in block.Convolutions)
        {
            x = ConvBlock(conv, x);
            if (conv.Relu) x = Drop(x, _c.Dropout, training, random);
        }
        return block.Transformer.Forward(x, training, random, _c.Dropout, _c.Dropout, block.LayerDrop);
    }

    // A strided "same" convolution on [L, Cin] (TensorFlow padding: ⌈L/stride⌉ outputs, the padding split with the extra
    // frame at the end) as a linear map of the unfolded windows.
    private Tensor<T> Convolve(DenseLayer<T> map, Tensor<T> x, int kernel, int stride)
    {
        int length = x.Shape[0], channels = x.Shape[1];
        int frames = (length + stride - 1) / stride;
        int pad = Math.Max((frames - 1) * stride + kernel - length, 0), left = pad / 2, right = pad - left;
        var padded = pad == 0 ? x : _engine.TensorConcatenate(new[]
        {
            new Tensor<T>(new[] { left, channels }), x, new Tensor<T>(new[] { right, channels }),
        }.Where(t => t.Shape[0] > 0).ToArray(), 0);
        var indices = new int[frames * kernel];
        for (int t = 0; t < frames; t++)
            for (int k = 0; k < kernel; k++) indices[t * kernel + k] = t * stride + k;
        var windows = _engine.Reshape(_engine.TensorGather(padded, new Tensor<int>(new[] { indices.Length }, new Vector<int>(indices)), axis: 0),
            new[] { frames, kernel * channels });
        return map.Forward(windows);
    }

    private Tensor<T> ConvBlock(Convolution conv, Tensor<T> x)
    {
        var y = conv.Norm.Forward(Convolve(conv.Map, x, conv.Kernel, conv.Stride));
        return conv.Relu ? _engine.ReLU(y) : y;
    }

    private Tensor<T> Drop(Tensor<T> x, double rate, bool training, Random random)
    {
        if (!training || rate <= 0) return x;
        var mask = new Tensor<T>(x._shape);
        var keep = NumOps.FromDouble(1.0 / (1.0 - rate));
        for (int i = 0; i < mask.Length; i++) mask[i] = random.NextDouble() < rate ? NumOps.Zero : keep;
        return _engine.TensorMultiply(x, mask);
    }

    /// <summary>
    /// The in-utterance contrastive loss of Equation 2: for every student output z_i, the cross-entropy of picking the
    /// teacher's z'_i among it and <paramref name="negatives"/> distractors z'_j drawn from other positions of the same
    /// utterance, with cosine similarity over the temperature κ. Summed over positions, as Equation 2 is.
    /// </summary>
    /// <param name="student">The student's outputs <c>[L, D]</c>.</param>
    /// <param name="teacher">The teacher's outputs at the same positions <c>[L, D]</c> (no gradient flows into them).</param>
    public Tensor<T> ContrastiveLoss(Tensor<T> student, Tensor<T> teacher, int negatives, double temperature, Random random)
    {
        int length = student.Shape[0];
        if (length < 2) throw new ArgumentException("The contrastive loss needs at least two positions.", nameof(student));
        int k = Math.Min(negatives, length - 1);
        // Candidate c of position i: c = 0 is the positive, c > 0 a distractor index drawn from positions other than i.
        var index = new int[length * (k + 1)];
        for (int i = 0; i < length; i++)
        {
            index[i * (k + 1)] = i;
            for (int c = 1; c <= k; c++)
            {
                int j = random.Next(length - 1);
                index[i * (k + 1) + c] = j >= i ? j + 1 : j;
            }
        }
        var normalizedStudent = Normalize(student);
        var normalizedTeacher = Normalize(teacher);
        var candidates = _engine.Reshape(_engine.TensorGather(normalizedTeacher, new Tensor<int>(new[] { index.Length }, new Vector<int>(index)), axis: 0),
            new[] { length, k + 1, teacher.Shape[1] });
        var similarity = _engine.ReduceSum(_engine.TensorMultiply(candidates,
            _engine.Reshape(normalizedStudent, new[] { length, 1, student.Shape[1] })), new[] { 2 }, keepDims: false);   // [L, k+1]
        var logits = _engine.TensorMultiplyScalar(similarity, NumOps.FromDouble(1.0 / temperature));
        var pick = new Tensor<T>(new[] { length, k + 1 });
        for (int i = 0; i < length; i++) pick[i, 0] = NumOps.FromDouble(-1.0);
        return _engine.ReduceSum(_engine.TensorMultiply(_engine.TensorLogSoftmax(logits, axis: -1), pick), new[] { 0, 1 }, keepDims: false);
    }

    private Tensor<T> Normalize(Tensor<T> x)
    {
        var norm = _engine.TensorSqrt(_engine.TensorAdd(_engine.ReduceSum(_engine.TensorMultiply(x, x), new[] { 1 }, keepDims: true),
            new Tensor<T>(new[] { 1 }, new Vector<T>(new[] { NumOps.FromDouble(1e-12) }))));
        return _engine.TensorDivide(x, norm);
    }
}

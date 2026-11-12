using AiDotNet.ActivationFunctions;
using AiDotNet.Audio.Foundations;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.SpeechRecognition.Robust;

/// <summary>An AV-HuBERT network's shape (Shi et al. 2022, section 4 and appendix B; defaults are the BASE model).</summary>
internal sealed record AvHubertConfiguration
{
    public int SampleRate { get; init; } = 16000;
    /// <summary>Log filterbank bins per 10 ms frame (26, stacked four at a time into 104).</summary>
    public int FilterbankBins { get; init; } = 26;
    public int WindowLength { get; init; } = 400;
    public int HopLength { get; init; } = 160;
    public int StackedFrames { get; init; } = 4;
    public double LogFloor { get; init; } = 1e-6;
    public int LipSize { get; init; } = 88;
    public int EncoderDim { get; init; } = 768;
    public int Layers { get; init; } = 12;
    public int Heads { get; init; } = 12;
    public int FeedForwardDim { get; init; } = 3072;
    public double Dropout { get; init; } = 0.1;
    public double LayerDrop { get; init; } = 0.05;
    public double ModalityDropout { get; init; } = 0.5;
    public double AudioDropout { get; init; } = 0.5;
    public double AudioMaskProbability { get; init; } = 0.8;
    public int AudioMaskLength { get; init; } = 10;
    public double VideoMaskProbability { get; init; } = 0.3;
    public int VideoMaskLength { get; init; } = 5;
    public int FinalDim { get; init; } = 256;
    public int Clusters { get; init; } = 2000;
    public double LogitTemperature { get; init; } = 0.1;
    public double FeaturePenaltyWeight { get; init; } = 10.0;
    public int VocabSize { get; init; } = 46;
}

/// <summary>
/// The AV-HuBERT network (Shi et al. 2022, section 3 and appendix B.3) as the authors' fairseq code builds it: a linear
/// audio encoder over 104-dimensional stacked log filterbanks, a modified ResNet-18 lip encoder (a 5×7×7 3-D convolution
/// front, BatchNorm, PReLU and max-pooling, then four stages of PReLU basic blocks and average pooling), channel-wise
/// concatenation with modality dropout (Equation 2), a layer norm and projection, the pre-norm wav2vec 2.0 Transformer,
/// and on top either the cosine cluster-prediction head of pre-training or the CTC projection of fine-tuning.
/// </summary>
/// <remarks>
/// Audio and video share one 25 Hz frame rate: four 10 ms filterbank frames per 40 ms video frame. The 3-D front's kernel
/// spans five frames at stride 1, so it is computed as a 2-D convolution over each frame's five-frame neighbourhood (zero
/// padded) taken as input channels, which is the same operation. Its layers are added to the given list.
/// </remarks>
internal sealed class AvHubertNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly AvHubertConfiguration _c;
    private readonly MelSpectrogram<T> _filterbank;

    internal sealed record ResBlock(ConvolutionalLayer<T> Conv1, BatchNormalizationLayer<T> Norm1, PReLULayer<T> Act1,
        ConvolutionalLayer<T> Conv2, BatchNormalizationLayer<T> Norm2, PReLULayer<T> Act2,
        ConvolutionalLayer<T>? DownsampleConv, BatchNormalizationLayer<T>? DownsampleNorm);

    public AvHubertNetwork(IEngine engine, List<LayerBase<T>> layers, AvHubertConfiguration configuration)
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
        ConvolutionalLayer<T> Conv(int inputs, int outputs, int kernel, int stride, int padding) => Own(ConvolutionalLayer<T>.WithInputDepth(
            inputs, outputs, kernel, stride, padding, identity, biasMode: Enums.BiasMode.Never));
        BatchNormalizationLayer<T> Norm(int channels) => Own(new BatchNormalizationLayer<T>(channels, 1e-5, 0.9));
        PReLULayer<T> PRelu(int channels) => Own(new PReLULayer<T>(channels, channelAxis: 1));
        _filterbank = new MelSpectrogram<T>(sampleRate: _c.SampleRate, nMels: _c.FilterbankBins, nFft: _c.WindowLength,
            hopLength: _c.HopLength, logMel: true, logOffset: _c.LogFloor);

        AudioEncoder = Linear(_c.EncoderDim);
        FrontConv = Conv(5, 64, 7, 2, 3);
        FrontNorm = Norm(64);
        FrontAct = PRelu(64);
        int inputs = 64;
        foreach (var (planes, stride) in new[] { (64, 1), (128, 2), (256, 2), (512, 2) })
            for (int b = 0; b < 2; b++)
            {
                int s = b == 0 ? stride : 1;
                bool project = s != 1 || inputs != planes;
                Blocks.Add(new ResBlock(Conv(inputs, planes, 3, s, 1), Norm(planes), PRelu(planes), Conv(planes, planes, 3, 1, 1),
                    Norm(planes), PRelu(planes), project ? Conv(inputs, planes, 1, s, 0) : null, project ? Norm(planes) : null));
                inputs = planes;
            }
        VideoEncoder = Linear(_c.EncoderDim);
        AudioMaskEmbedding = Own(new BiasFreeLinearLayer<T>(1, _c.FilterbankBins * _c.StackedFrames));
        FusionNorm = Own(new LayerNormalizationLayer<T>(2 * _c.EncoderDim));
        FusionProjection = Linear(_c.EncoderDim);
        Encoder = new PreNormTransformerEncoder<T>(_engine, layers, _c.EncoderDim, _c.Layers, _c.Heads, _c.FeedForwardDim);
        FinalProjection = Linear(_c.FinalDim);
        ClusterEmbeddings = Own(new BiasFreeLinearLayer<T>(_c.FinalDim, _c.Clusters));
        CtcHead = Linear(_c.VocabSize);
    }

    public DenseLayer<T> AudioEncoder { get; }
    public ConvolutionalLayer<T> FrontConv { get; }
    public BatchNormalizationLayer<T> FrontNorm { get; }
    public PReLULayer<T> FrontAct { get; }
    public List<ResBlock> Blocks { get; } = new();
    public DenseLayer<T> VideoEncoder { get; }
    /// <summary>The learned vector that replaces masked audio frames (a [1 → 104] map of a constant one).</summary>
    public BiasFreeLinearLayer<T> AudioMaskEmbedding { get; }
    public LayerNormalizationLayer<T> FusionNorm { get; }
    public DenseLayer<T> FusionProjection { get; }
    public PreNormTransformerEncoder<T> Encoder { get; }
    public DenseLayer<T> FinalProjection { get; }
    /// <summary>The cluster embeddings e_c as a [FinalDim → Clusters] map: its weights are the embedding matrix.</summary>
    public BiasFreeLinearLayer<T> ClusterEmbeddings { get; }
    public DenseLayer<T> CtcHead { get; }

    public int AudioWidth => _c.FilterbankBins * _c.StackedFrames;

    /// <summary>
    /// The audio stream <c>[frames, 104]</c> of a 16 kHz waveform: 26-bin log filterbanks every 10 ms, four consecutive
    /// frames stacked into one 25 Hz frame, each frame normalized by its own mean and standard deviation (appendix B.3).
    /// </summary>
    public double[,] AudioFeatures(IReadOnlyList<double> waveform)
    {
        var samples = new Tensor<T>(new[] { waveform.Count });
        for (int i = 0; i < waveform.Count; i++) samples[i] = NumOps.FromDouble(waveform[i]);
        var bank = _filterbank.Forward(samples);
        int frames = bank.Shape[0] / _c.StackedFrames, bins = _c.FilterbankBins, width = AudioWidth;
        var features = new double[frames, width];
        for (int t = 0; t < frames; t++)
        {
            double mean = 0, square = 0;
            for (int s = 0; s < _c.StackedFrames; s++)
                for (int b = 0; b < bins; b++)
                {
                    double v = NumOps.ToDouble(bank[t * _c.StackedFrames + s, b]);
                    features[t, s * bins + b] = v;
                    mean += v;
                }
            mean /= width;
            for (int i = 0; i < width; i++) square += (features[t, i] - mean) * (features[t, i] - mean);
            double std = Math.Sqrt(square / width + 1e-5);
            for (int i = 0; i < width; i++) features[t, i] = (features[t, i] - mean) / std;
        }
        return features;
    }

    /// <summary>
    /// Input masking for pre-training (appendix B.4, "masking_type: input"): spans drawn as wav2vec 2.0 draws them — a
    /// proportion p of the frames start spans of l frames (at least two spans) — with audio frames replaced by the learned
    /// mask embedding and video frames by a same-length segment from elsewhere in the utterance.
    /// </summary>
    public bool[] DrawMask(int frames, double probability, int length, Random random)
    {
        var mask = new bool[frames];
        if (probability <= 0 || frames == 0) return mask;
        int spans = Math.Max(2, (int)(probability * frames / length + random.NextDouble()));
        int room = Math.Max(1, frames - length + 1);
        var starts = Enumerable.Range(0, room).OrderBy(_ => random.Next()).Take(Math.Min(spans, room));
        foreach (int start in starts)
            for (int t = start; t < Math.Min(frames, start + length); t++) mask[t] = true;
        return mask;
    }

    /// <summary>The audio encoder's output <c>[T, D]</c>; masked frames take the learned mask embedding.</summary>
    public Tensor<T> EncodeAudio(double[,] features, bool[]? mask)
    {
        int frames = features.GetLength(0), width = AudioWidth;
        var x = new Tensor<T>(new[] { frames, width });
        var keep = new Tensor<T>(new[] { frames, width });
        for (int t = 0; t < frames; t++)
            for (int i = 0; i < width; i++)
            {
                bool masked = mask is not null && mask[t];
                x[t, i] = masked ? NumOps.Zero : NumOps.FromDouble(features[t, i]);
                keep[t, i] = masked ? NumOps.Zero : NumOps.One;
            }
        if (mask is not null && mask.Any(m => m))
        {
            var embedding = AudioMaskEmbedding.Forward(new Tensor<T>(new[] { 1, 1 }, new Vector<T>(new[] { NumOps.One })));   // [1, 104]
            var drop = _engine.TensorSubtract(new Tensor<T>(keep._shape, new Vector<T>(Enumerable.Repeat(NumOps.One, keep.Length).ToArray())), keep);
            x = _engine.TensorAdd(x, _engine.TensorMultiply(drop, _engine.TensorBroadcastTo(embedding, new[] { frames, width })));
        }
        return AudioEncoder.Forward(x);
    }

    /// <summary>The lip encoder's output <c>[T, D]</c> of grayscale mouth crops <c>[T, H, W]</c>; masked frames are
    /// replaced by a same-length segment drawn from elsewhere in the utterance.</summary>
    public Tensor<T> EncodeVideo(Tensor<T> lips, bool[]? mask, Random random, bool training)
    {
        int frames = lips.Shape[0], h = lips.Shape[1], w = lips.Shape[2];
        var source = lips;
        if (mask is not null && mask.Any(m => m))
        {
            // Each masked span copies the frames of a span starting elsewhere (outside the span and the span before it).
            var copy = Enumerable.Range(0, frames).ToArray();
            int t = 0;
            while (t < frames)
            {
                if (!mask[t]) { t++; continue; }
                int start = t;
                while (t < frames && mask[t]) t++;
                int length = t - start;
                var others = Enumerable.Range(0, frames).Where(o => o < Math.Max(0, start - length) || o >= t).ToArray();
                int other = others.Length > 0 ? others[random.Next(others.Length)] : 0;
                for (int k = 0; k < length; k++) copy[start + k] = Math.Min(frames - 1, other + k);
            }
            source = _engine.TensorGather(lips, new Tensor<int>(new[] { frames }, new Vector<int>(copy)), axis: 0);
        }
        // The 5-frame neighbourhood of every frame as channels: [T, 5, H, W], zero beyond the ends.
        var padded = _engine.TensorConcatenate(new[] { new Tensor<T>(new[] { 2, h, w }), source, new Tensor<T>(new[] { 2, h, w }) }, 0);
        var index = new int[frames * 5];
        for (int f = 0; f < frames; f++)
            for (int k = 0; k < 5; k++) index[f * 5 + k] = f + k;
        var neighbourhoods = _engine.Reshape(_engine.TensorGather(padded, new Tensor<int>(new[] { index.Length }, new Vector<int>(index)), axis: 0),
            new[] { frames, 5, h, w });
        FrontNorm.SetTrainingMode(training);
        var x = FrontAct.Forward(FrontNorm.Forward(FrontConv.Forward(neighbourhoods)));
        x = AiDotNet.ComputerVision.CvTensorOps<T>.MaxPoolPadded(x, kernelSize: 3, stride: 2, padding: 1);
        foreach (var block in Blocks)
        {
            block.Norm1.SetTrainingMode(training);
            block.Norm2.SetTrainingMode(training);
            block.DownsampleNorm?.SetTrainingMode(training);
            var y = block.Act1.Forward(block.Norm1.Forward(block.Conv1.Forward(x)));
            y = block.Norm2.Forward(block.Conv2.Forward(y));
            var residual = block.DownsampleConv is not null && block.DownsampleNorm is not null
                ? block.DownsampleNorm.Forward(block.DownsampleConv.Forward(x))
                : x;
            x = block.Act2.Forward(_engine.TensorAdd(y, residual));
        }
        var pooled = _engine.ReduceMean(x, new[] { 2, 3 }, keepDims: false);                                   // [T, 512]
        return VideoEncoder.Forward(pooled);
    }

    /// <summary>
    /// The fused features <c>[T, D]</c> (Equation 2): channel-wise concatenation of the two streams, a missing stream as
    /// zeros, then the layer norm, the projection and input dropout.
    /// </summary>
    public Tensor<T> Fuse(Tensor<T>? audio, Tensor<T>? video, bool training, Random random)
    {
        var reference = audio ?? video ?? throw new ArgumentException("At least one modality is needed.");
        int frames = reference.Shape[0], d = _c.EncoderDim;
        var a = audio ?? new Tensor<T>(new[] { frames, d });
        var v = video ?? new Tensor<T>(new[] { frames, d });
        if (a.Shape[0] != v.Shape[0])
        {
            // The streams can differ by a frame at the end; both are cut to the shorter.
            frames = Math.Min(a.Shape[0], v.Shape[0]);
            a = _engine.TensorSlice(a, new[] { 0, 0 }, new[] { frames, d });
            v = _engine.TensorSlice(v, new[] { 0, 0 }, new[] { frames, d });
        }
        var fused = FusionProjection.Forward(FusionNorm.Forward(_engine.TensorConcatenate(new[] { a, v }, 1)));
        return Drop(fused, _c.Dropout, training, random);
    }

    /// <summary>The Transformer's output <c>[T, D]</c>.</summary>
    public Tensor<T> Encode(Tensor<T> fused, bool training, Random random, List<Tensor<T>>? states = null)
        => Encoder.Forward(fused, training, random, _c.Dropout, 0.0, _c.LayerDrop, states);

    /// <summary>The cluster logits <c>[T, Clusters]</c>: cosine similarity of the projected output with each cluster
    /// embedding over the temperature (fairseq's <c>compute_logits</c> with <c>sim_type="cosine"</c>, the denominator
    /// floored at 1e-6).</summary>
    public Tensor<T> ClusterLogits(Tensor<T> encoded)
    {
        var projected = FinalProjection.Forward(encoded);                                                       // [T, F]
        var dots = ClusterEmbeddings.Forward(projected);                                                        // [T, C]
        var weights = ClusterEmbeddings.Weights;                                                                // [F, C]
        var rowNorm = _engine.TensorSqrt(_engine.ReduceSum(_engine.TensorMultiply(projected, projected), new[] { 1 }, keepDims: true));   // [T, 1]
        var embeddingNorm = _engine.TensorSqrt(_engine.ReduceSum(_engine.TensorMultiply(weights, weights), new[] { 0 }, keepDims: true)); // [1, C]
        var denominator = _engine.TensorMultiply(_engine.TensorBroadcastTo(rowNorm, dots._shape), _engine.TensorBroadcastTo(embeddingNorm, dots._shape));
        var floor = new Tensor<T>(dots._shape);
        var dv = denominator.ToArray();
        for (int i = 0; i < dv.Length; i++) floor[i] = NumOps.FromDouble(Math.Max(0.0, 1e-6 - NumOps.ToDouble(dv[i])));
        return _engine.TensorMultiplyScalar(_engine.TensorDivide(dots, _engine.TensorAdd(denominator, floor)), NumOps.FromDouble(1.0 / _c.LogitTemperature));
    }

    /// <summary>The CTC logits <c>[T, VocabSize]</c>: a projection of the Transformer output.</summary>
    public Tensor<T> CtcLogits(Tensor<T> encoded) => CtcHead.Forward(encoded);

    private Tensor<T> Drop(Tensor<T> x, double rate, bool training, Random random)
    {
        if (!training || rate <= 0) return x;
        var mask = new Tensor<T>(x._shape);
        var keep = NumOps.FromDouble(1.0 / (1.0 - rate));
        for (int i = 0; i < mask.Length; i++) mask[i] = random.NextDouble() < rate ? NumOps.Zero : keep;
        return _engine.TensorMultiply(x, mask);
    }
}

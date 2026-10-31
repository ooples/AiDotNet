using AiDotNet.ActivationFunctions;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Audio.Speaker;

/// <summary>
/// The x-vector network (Snyder et al. 2018, "X-Vectors: Robust DNN Embeddings for Speaker Recognition", Table 1): five
/// frame-level TDNN layers with temporal contexts [t-2, t+2], {t-2, t, t+2}, {t-3, t, t+3}, {t} and {t} (512, 512, 512,
/// 512 and 1500 units), statistics pooling of the mean and standard deviation over the segment, two segment-level layers
/// of 512 units and a softmax over the training speakers. The x-vector is the first segment layer's affine output, before
/// its nonlinearity (section 2.2). Every hidden layer applies ReLU then batch normalization, as Kaldi's x-vector recipe
/// does. Input features are 24-dimensional log-mel filterbanks over 25 ms frames, mean-normalized over a sliding window
/// of up to 3 seconds (section 3.2).
/// </summary>
internal sealed class XVectorNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly MelSpectrogram<T> _mel;
    private readonly int _normalizationWindow;

    public XVectorNetwork(IEngine engine, List<LayerBase<T>> layers, int sampleRate, int numFilterbanks, int fftSize, int hopLength,
        int frameDim, int poolingDim, int embeddingDim, int numSpeakers, int normalizationWindow)
    {
        _engine = engine ?? throw new ArgumentNullException(nameof(engine));
        _normalizationWindow = normalizationWindow;
        _mel = new MelSpectrogram<T>(sampleRate: sampleRate, nMels: numFilterbanks, nFft: fftSize, hopLength: hopLength, logMel: true, logOffset: 1e-6);
        TLayer Own<TLayer>(TLayer layer) where TLayer : LayerBase<T> { layers.Add(layer); return layer; }
        var identity = (IActivationFunction<T>)new IdentityActivation<T>();
        // (input channels, output channels, kernel, dilation) of frame1-frame5.
        var frames = new (int In, int Out, int Kernel, int Dilation)[]
        {
            (numFilterbanks, frameDim, 5, 1), (frameDim, frameDim, 3, 2), (frameDim, frameDim, 3, 3), (frameDim, frameDim, 1, 1), (frameDim, poolingDim, 1, 1),
        };
        foreach (var f in frames)
        {
            FrameLayers.Add(Own(new Conv1DLayer<T>(inputChannels: f.In, outputChannels: f.Out, kernelSize: f.Kernel, dilation: f.Dilation, stride: 1, padding: 0, activation: identity)));
            FrameNorms.Add(Own(new BatchNormalizationLayer<T>(f.Out)));
        }
        Segment6 = Own(new FullyConnectedLayer<T>(2 * poolingDim, embeddingDim, identity));
        Segment6Norm = Own(new BatchNormalizationLayer<T>(embeddingDim));
        Segment7 = Own(new FullyConnectedLayer<T>(embeddingDim, embeddingDim, identity));
        Segment7Norm = Own(new BatchNormalizationLayer<T>(embeddingDim));
        if (numSpeakers > 0) Classifier = Own(new FullyConnectedLayer<T>(embeddingDim, numSpeakers, identity));
    }

    public List<Conv1DLayer<T>> FrameLayers { get; } = new();
    public List<BatchNormalizationLayer<T>> FrameNorms { get; } = new();
    public FullyConnectedLayer<T> Segment6 { get; }
    public BatchNormalizationLayer<T> Segment6Norm { get; }
    public FullyConnectedLayer<T> Segment7 { get; }
    public BatchNormalizationLayer<T> Segment7Norm { get; }
    public FullyConnectedLayer<T>? Classifier { get; }

    /// <summary>Frames the TDNN consumes as context: 14, seven on either side (2 + 2 + 3).</summary>
    public const int Context = 14;

    /// <summary>Log-mel filterbanks <c>[frames, bins]</c> of a waveform, each frame's mean over the surrounding window
    /// (up to the normalization window, centred) subtracted.</summary>
    public Tensor<T> Features(Tensor<T> waveform)
    {
        var mel = _mel.Forward(waveform);
        int frames = mel.Shape[0], bins = mel.Shape[1], half = _normalizationWindow / 2;
        var normalized = new Tensor<T>(new[] { frames, bins });
        var prefix = new double[frames + 1, bins];
        for (int t = 0; t < frames; t++)
            for (int b = 0; b < bins; b++) prefix[t + 1, b] = prefix[t, b] + NumOps.ToDouble(mel[t, b]);
        for (int t = 0; t < frames; t++)
        {
            int start = Math.Max(0, t - half), end = Math.Min(frames, t + half + 1);
            for (int b = 0; b < bins; b++)
                normalized[t, b] = NumOps.FromDouble(NumOps.ToDouble(mel[t, b]) - (prefix[end, b] - prefix[start, b]) / (end - start));
        }
        return normalized;
    }

    // ReLU then batch normalization over channels of [B, C, T] or [B, C].
    private Tensor<T> ReluNorm(BatchNormalizationLayer<T> norm, Tensor<T> x)
    {
        var relu = _engine.ReLU(x);
        if (relu.Rank == 2) return norm.Forward(relu);
        int b = relu.Shape[0], c = relu.Shape[1], t = relu.Shape[2];
        var flat = _engine.Reshape(_engine.TensorPermute(relu, new[] { 0, 2, 1 }), new[] { b * t, c });
        return _engine.TensorPermute(_engine.Reshape(norm.Forward(flat), new[] { b, t, c }), new[] { 0, 2, 1 });
    }

    /// <summary>The x-vectors <c>[B, embedding]</c> of features <c>[B, frames, bins]</c> (segment6's affine output).</summary>
    public Tensor<T> Embed(Tensor<T> features, bool training)
    {
        if (features.Shape[1] <= Context) throw new ArgumentException($"An x-vector needs more than {Context} frames.", nameof(features));
        foreach (var norm in FrameNorms) norm.SetTrainingMode(training);
        Segment6Norm.SetTrainingMode(training);
        Segment7Norm.SetTrainingMode(training);
        var x = _engine.TensorPermute(features, new[] { 0, 2, 1 });                 // [B, bins, T]
        for (int i = 0; i < FrameLayers.Count; i++) x = ReluNorm(FrameNorms[i], FrameLayers[i].Forward(x));
        // Statistics pooling (section 2.2): the mean and standard deviation over time.
        int frames = x.Shape[2];
        var mean = _engine.ReduceMean(x, new[] { 2 }, keepDims: true);
        var centred = _engine.TensorSubtract(x, _engine.TensorBroadcastTo(mean, x.Shape.ToArray()));
        var variance = _engine.ReduceMean(_engine.TensorMultiply(centred, centred), new[] { 2 }, keepDims: false);
        var std = _engine.TensorPower(_engine.TensorAddScalar(variance, NumOps.FromDouble(1e-10)), NumOps.FromDouble(0.5));
        var pooled = _engine.TensorConcatenate(new[] { _engine.Reshape(mean, new[] { x.Shape[0], x.Shape[1] }), std }, 1);
        return Segment6.Forward(pooled);
    }

    /// <summary>The speaker logits <c>[B, speakers]</c> of x-vectors: segment6's nonlinearity, segment7 and the softmax layer.</summary>
    public Tensor<T> SpeakerLogits(Tensor<T> xvectors)
    {
        var classifier = Classifier ?? throw new InvalidOperationException("Set NumTrainingSpeakers to train the speaker classifier.");
        var h = ReluNorm(Segment6Norm, xvectors);
        h = ReluNorm(Segment7Norm, Segment7.Forward(h));
        return classifier.Forward(h);
    }
}

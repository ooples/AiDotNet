using AiDotNet.ActivationFunctions;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.Audio.Foundations;

/// <summary>
/// The w2v-BERT network (Chung et al. 2021, §3.1, Figure 1): log-mel features, a feature encoder of two stride-2 2-D
/// convolutions and a projection, a contrastive module (a linear projection and Conformer blocks) and a masked-prediction
/// module (Conformer blocks and a softmax over the quantizer's codebook).
/// </summary>
internal sealed class W2vBertNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly W2vBertOptions _o;
    private readonly MelSpectrogram<T> _mel;
    private readonly int _subsampledBins;

    public W2vBertNetwork(IEngine engine, List<LayerBase<T>> layers, W2vBertOptions options)
    {
        _engine = engine ?? throw new ArgumentNullException(nameof(engine));
        _o = options ?? throw new ArgumentNullException(nameof(options));
        TLayer Own<TLayer>(TLayer layer) where TLayer : LayerBase<T>
        {
            layers.Add(layer);
            return layer;
        }
        var identity = (IActivationFunction<T>)new IdentityActivation<T>();
        int d = _o.ModelDim;
        _mel = new MelSpectrogram<T>(sampleRate: _o.SampleRate, nMels: _o.MelBins, nFft: _o.WindowLength,
            hopLength: _o.HopLength, logMel: true, logOffset: _o.LogMelFloor);
        // Two 3×3 convolutions with stride 2 and no padding, each followed by ReLU (ESPnet Conv2dSubsampling), then a
        // projection of the channels × remaining bins to the model width.
        FirstConv = Own(new Conv2DLayer<T>(1, d, 3, 3, 2, 2, 0, 0));
        SecondConv = Own(new Conv2DLayer<T>(d, d, 3, 3, 2, 2, 0, 0));
        _subsampledBins = Subsampled(Subsampled(_o.MelBins));
        if (_subsampledBins <= 0) throw new ArgumentException("Too few mel bins for two stride-2 3×3 convolutions.", nameof(options));
        FeatureProjection = Own(new DenseLayer<T>(d, identity));
        ContrastiveProjection = Own(new DenseLayer<T>(d, identity));
        for (int i = 0; i < _o.ContrastiveLayers; i++)
            ContrastiveBlocks.Add(Own(new ConformerBlockLayer<T>(d, _o.NumHeads, _o.FeedForwardMultiplier, _o.ConvKernel, ropeTheta: 0)));
        for (int i = 0; i < _o.MaskedLayers; i++)
            MaskedBlocks.Add(Own(new ConformerBlockLayer<T>(d, _o.NumHeads, _o.FeedForwardMultiplier, _o.ConvKernel, ropeTheta: 0)));
        PredictionHead = Own(new DenseLayer<T>(_o.CodebookSize, identity));
    }

    private static int Subsampled(int length) => (length - 3) / 2 + 1;

    public Conv2DLayer<T> FirstConv { get; }
    public Conv2DLayer<T> SecondConv { get; }
    public DenseLayer<T> FeatureProjection { get; }
    public DenseLayer<T> ContrastiveProjection { get; }
    public List<ConformerBlockLayer<T>> ContrastiveBlocks { get; } = new();
    public List<ConformerBlockLayer<T>> MaskedBlocks { get; } = new();
    public DenseLayer<T> PredictionHead { get; }

    /// <summary>Encoder frames of <paramref name="melFrames"/> log-mel frames (two stride-2 3×3 convolutions).</summary>
    public int EncodedLength(int melFrames) => Subsampled(Subsampled(melFrames));

    /// <summary>One forward pass: the unmasked encoded features <c>[B, L, D]</c>, the contrastive module's context
    /// vectors, the masked-prediction module's output and logits, the output of every block (contrastive first), and
    /// the mask applied.</summary>
    internal sealed record Output(Tensor<T> Features, Tensor<T> Context, Tensor<T> Final, Tensor<T> Logits,
        IReadOnlyList<Tensor<T>> BlockStates, bool[,]? Mask);

    /// <summary>Log-mel features <c>[B, frames, bins]</c> of waveforms <c>[B, S]</c> or <c>[S]</c>.</summary>
    public Tensor<T> Features(Tensor<T> audio)
    {
        var waveforms = audio.Rank == 1 ? _engine.Reshape(audio, new[] { 1, audio.Length }) : audio;
        int batch = waveforms.Shape[0], samples = waveforms.Shape[1];
        var mels = new Tensor<T>[batch];
        for (int b = 0; b < batch; b++)
        {
            var one = _engine.Reshape(_engine.TensorSlice(waveforms, new[] { b, 0 }, new[] { 1, samples }), new[] { samples });
            var mel = _mel.Forward(one);                                                                        // [frames, bins]
            mels[b] = _engine.Reshape(mel, new[] { 1, mel.Shape[0], mel.Shape[1] });
        }
        return batch == 1 ? mels[0] : _engine.TensorConcatenate(mels, 0);
    }

    /// <summary>The feature encoder: <c>[B, frames, bins]</c> to <c>[B, L, D]</c>.</summary>
    public Tensor<T> Encode(Tensor<T> mel)
    {
        int batch = mel.Shape[0], frames = mel.Shape[1], bins = mel.Shape[2], d = _o.ModelDim;
        var x = _engine.ReLU(FirstConv.Forward(_engine.Reshape(mel, new[] { batch, 1, frames, bins })));
        x = _engine.ReLU(SecondConv.Forward(x));                                                               // [B, D, L, F']
        int length = x.Shape[2];
        if (length <= 0) throw new ArgumentException("The audio is too short for the feature encoder.");
        var steps = _engine.Reshape(_engine.TensorPermute(x, new[] { 0, 2, 1, 3 }), new[] { batch * length, d * _subsampledBins });
        return _engine.Reshape(FeatureProjection.Forward(steps), new[] { batch, length, d });
    }

    /// <summary>
    /// Masks spans of <c>[B, L]</c>: each step starts a span of <see cref="W2vBertOptions.MaskLength"/> steps with
    /// probability <paramref name="probability"/>, and spans may overlap (§4.2).
    /// </summary>
    public bool[,] DrawMask(int batch, int length, double probability, Random random)
    {
        var mask = new bool[batch, length];
        for (int b = 0; b < batch; b++)
            for (int t = 0; t < length; t++)
                if (random.NextDouble() < probability)
                    for (int s = t; s < Math.Min(length, t + _o.MaskLength); s++) mask[b, s] = true;
        return mask;
    }

    /// <summary>The network on log-mel features <c>[B, frames, bins]</c>; masked steps of the encoded features are
    /// replaced by N(0, 1) vectors before the contrastive module (§3.2).</summary>
    public Output Forward(
        Tensor<T> mel,
        bool[,]? mask,
        bool training,
        Random random,
        Tensor<T>? dynamicMaskFlat = null,
        Tensor<T>? fixedMaskNoise = null)
    {
        var features = Encode(mel);
        int batch = features.Shape[0], length = features.Shape[1], d = _o.ModelDim;
        var input = features;
        if (dynamicMaskFlat is not null)
        {
            var drop = _engine.TensorTile(_engine.Reshape(dynamicMaskFlat, new[] { batch, length, 1 }), new[] { 1, 1, d });
            var keep = _engine.ScalarMinusTensor(NumOps.One, drop);
            var stdNormal = fixedMaskNoise ?? AiDotNet.Training.CompiledStepRandom<T>.StandardNormal(new[] { batch, length, d }, random);
            input = _engine.TensorAdd(_engine.TensorMultiply(features, keep), _engine.TensorMultiply(stdNormal, drop));
        }
        else if (mask is not null)
        {
            if (mask.GetLength(0) != batch || mask.GetLength(1) != length)
                throw new ArgumentException($"The mask must be [{batch}, {length}].", nameof(mask));
            int total = batch * length * d;
            if (fixedMaskNoise is not null)
            {
                Tensor<T> keep;
                using (AiDotNet.Tensors.Engines.Compilation.GraphMode.SuspendRecording())
                {
                    var keepArr = new double[total];
                    for (int b = 0; b < batch; b++)
                        for (int t = 0; t < length; t++)
                            if (!mask[b, t])
                            {
                                int rowOffset = (b * length + t) * d;
                                for (int i = 0; i < d; i++) keepArr[rowOffset + i] = 1.0;
                            }
                    keep = new Tensor<double>(keepArr, new[] { batch, length, d }).Cast<T>();
                }
                input = _engine.TensorAdd(_engine.TensorMultiply(features, keep), fixedMaskNoise);
            }
            else if (AiDotNet.Tensors.Engines.Compilation.GraphMode.IsActive)
            {
                Tensor<T> keep;
                Tensor<T> drop;
                using (AiDotNet.Tensors.Engines.Compilation.GraphMode.SuspendRecording())
                {
                    var keepArr = new double[total];
                    var dropArr = new double[total];
                    for (int b = 0; b < batch; b++)
                        for (int t = 0; t < length; t++)
                        {
                            int rowOffset = (b * length + t) * d;
                            if (!mask[b, t])
                            {
                                for (int i = 0; i < d; i++) keepArr[rowOffset + i] = 1.0;
                            }
                            else
                            {
                                for (int i = 0; i < d; i++) dropArr[rowOffset + i] = 1.0;
                            }
                        }
                    keep = new Tensor<double>(keepArr, new[] { batch, length, d }).Cast<T>();
                    drop = new Tensor<double>(dropArr, new[] { batch, length, d }).Cast<T>();
                }
                var stdNormal = AiDotNet.Training.CompiledStepRandom<T>.StandardNormal(new[] { batch, length, d }, random);
                input = _engine.TensorAdd(_engine.TensorMultiply(features, keep), _engine.TensorMultiply(stdNormal, drop));
            }
            else
            {
                var keepArr = new double[total];
                var noiseArr = new double[total];
                for (int b = 0; b < batch; b++)
                    for (int t = 0; t < length; t++)
                    {
                        int rowOffset = (b * length + t) * d;
                        for (int i = 0; i < d; i++)
                        {
                            if (!mask[b, t])
                            {
                                keepArr[rowOffset + i] = 1.0;
                                continue;
                            }
                            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
                            noiseArr[rowOffset + i] = Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
                        }
                    }
                var keep = new Tensor<double>(keepArr, new[] { batch, length, d }).Cast<T>();
                var noise = new Tensor<double>(noiseArr, new[] { batch, length, d }).Cast<T>();
                input = _engine.TensorAdd(_engine.TensorMultiply(features, keep), noise);
            }
        }
        var states = new List<Tensor<T>>();
        var h = _engine.Reshape(ContrastiveProjection.Forward(_engine.Reshape(input, new[] { batch * length, d })), new[] { batch, length, d });
        foreach (var block in ContrastiveBlocks)
        {
            block.SetTrainingMode(training);
            h = block.Forward(h);
            states.Add(h);
        }
        var context = h;
        foreach (var block in MaskedBlocks)
        {
            block.SetTrainingMode(training);
            h = block.Forward(h);
            states.Add(h);
        }
        var logits = _engine.Reshape(PredictionHead.Forward(_engine.Reshape(h, new[] { batch * length, d })),
            new[] { batch, length, _o.CodebookSize });
        return new Output(features, context, h, logits, states, mask);
    }

    /// <summary>The masked-prediction loss (§3.2): the cross-entropy of each masked step's assigned codebook entry,
    /// summed over the masked steps.</summary>
    public Tensor<T> MaskedPredictionLoss(
        Output output,
        int[,] targets,
        Tensor<T>? codeOneHot = null,
        Tensor<T>? dynamicMaskFlat = null)
    {
        var logProbabilities = _engine.TensorLogSoftmax(output.Logits, axis: -1);
        Tensor<T> weights;
        if (dynamicMaskFlat is not null && codeOneHot is not null)
        {
            int batch = output.Logits.Shape[0], length = output.Logits.Shape[1], vocab = output.Logits.Shape[2];
            var maskScale = _engine.TensorMultiplyScalar(_engine.Reshape(dynamicMaskFlat, new[] { batch, length, 1 }), NumOps.FromDouble(-1.0));
            weights = _engine.TensorMultiply(codeOneHot, _engine.TensorTile(maskScale, new[] { 1, 1, vocab }));
        }
        else if (codeOneHot is not null)
        {
            var mask = output.Mask ?? throw new ArgumentException("Masked prediction needs the forward's mask.", nameof(output));
            int batch = mask.GetLength(0), length = mask.GetLength(1), vocab = output.Logits.Shape[2];
            Tensor<T> maskScale;
            using (AiDotNet.Tensors.Engines.Compilation.GraphMode.SuspendRecording())
            {
                var maskScaleArr = new double[batch * length];
                for (int b = 0; b < batch; b++)
                    for (int t = 0; t < length; t++)
                        if (mask[b, t]) maskScaleArr[b * length + t] = -1.0;
                maskScale = new Tensor<double>(maskScaleArr, new[] { batch, length, 1 }).Cast<T>();
            }
            weights = _engine.TensorMultiply(codeOneHot, _engine.TensorTile(maskScale, new[] { 1, 1, vocab }));
        }
        else
        {
            var mask = output.Mask ?? throw new ArgumentException("Masked prediction needs the forward's mask.", nameof(output));
            int batch = mask.GetLength(0), length = mask.GetLength(1), vocab = output.Logits.Shape[2];
            var weightsArr = new double[batch * length * vocab];
            for (int b = 0; b < batch; b++)
                for (int t = 0; t < length; t++)
                    if (mask[b, t]) weightsArr[(b * length + t) * vocab + targets[b, t]] = -1.0;
            weights = new Tensor<double>(weightsArr, output.Logits._shape).Cast<T>();
        }
        return _engine.ReduceSum(_engine.TensorMultiply(logProbabilities, weights), new[] { 0, 1, 2 }, keepDims: false);
    }
}

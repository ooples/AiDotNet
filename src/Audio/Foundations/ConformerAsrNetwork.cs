using AiDotNet.ActivationFunctions;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Audio.Foundations;

/// <summary>The decoder a <see cref="ConformerAsrNetwork{T}"/> trains and decodes with.</summary>
internal enum ConformerAsrDecoder
{
    /// <summary>A linear CTC layer on the encoder (Graves et al. 2006).</summary>
    Ctc,

    /// <summary>An RNN transducer: an LSTM prediction network and a joint network (Graves 2012).</summary>
    Transducer,
}

/// <summary>The encoder block a <see cref="ConformerAsrNetwork{T}"/> stacks.</summary>
internal enum ConformerAsrBlock
{
    /// <summary>Conformer (Gulati et al. 2020): macaron feed-forward modules around self-attention and a convolution
    /// module.</summary>
    Conformer,

    /// <summary>Branchformer (Peng et al. 2022): self-attention and a convolutional gating MLP in parallel branches,
    /// concatenated and projected.</summary>
    Branchformer,

    /// <summary>E-Branchformer (Kim et al. 2022): Branchformer's branches merged through a depthwise convolution, between
    /// macaron feed-forward modules.</summary>
    EBranchformer,

    /// <summary>A pre-norm Transformer encoder layer (ESPnet's): self-attention and a ReLU feed-forward network, each with
    /// a residual, after absolute sinusoidal positions are added to the scaled input.</summary>
    Transformer,
}

/// <summary>The convolutional front end that shortens the feature sequence.</summary>
internal enum ConformerAsrSubsampling
{
    /// <summary>Two 3x3 stride-2 convolutions with ReLU: 4x (Gulati et al. 2020, Figure 1).</summary>
    Conv4x,

    /// <summary>Fast Conformer's 8x depthwise-separable striding (Rekesh et al. 2023, section 3.1): a 3x3 stride-2
    /// convolution, then two depthwise 3x3 stride-2 convolutions each followed by a pointwise convolution, all with ReLU
    /// and padding 1.</summary>
    DepthwiseStriding8x,
}

/// <summary>A Conformer recognizer's shape: front end, encoder and decoder.</summary>
internal sealed record ConformerAsrConfiguration
{
    public int SampleRate { get; init; } = 16000;
    public int NumMels { get; init; } = 80;
    public int WindowLength { get; init; } = 400;
    public int HopLength { get; init; } = 160;
    public double LogMelFloor { get; init; } = 1e-6;
    public ConformerAsrSubsampling Subsampling { get; init; } = ConformerAsrSubsampling.Conv4x;
    /// <summary>Channels of the 8x depthwise-striding front end (Fast Conformer: 256).</summary>
    public int SubsamplingChannels { get; init; } = 256;
    public int EncoderDim { get; init; } = 256;
    public int NumEncoderLayers { get; init; } = 16;
    public int NumAttentionHeads { get; init; } = 4;
    public int FeedForwardMultiplier { get; init; } = 4;
    public int ConvKernel { get; init; } = 32;
    public ConformerAsrBlock Block { get; init; } = ConformerAsrBlock.Conformer;
    /// <summary>Hidden units of the convolutional gating MLP (Branchformer and E-Branchformer).</summary>
    public int CgmlpDim { get; init; } = 2048;
    /// <summary>Depthwise kernel of E-Branchformer's merge convolution.</summary>
    public int MergeKernel { get; init; } = 31;
    /// <summary>Encoder layers (1-based) whose outputs get an intermediate CTC loss through the shared CTC layer.</summary>
    public int[] IntermediateCtcLayers { get; init; } = Array.Empty<int>();
    /// <summary>Weight of the mean intermediate CTC loss: L = (1 - w) L_CTC + w L_inter.</summary>
    public double IntermediateCtcWeight { get; init; }
    /// <summary>Whether each intermediate prediction is fed back into the next layer's input (self-conditioned CTC,
    /// Nozaki and Komatsu 2021, Equation 9).</summary>
    public bool SelfConditioning { get; init; }
    /// <summary>Rotary base standing in for the relative positional encoding; zero for none.</summary>
    public double RopeTheta { get; init; } = 10000.0;
    /// <summary>Encoder frames per attention chunk (streaming); zero attends over the whole input.</summary>
    public int AttentionChunkSize { get; init; }
    /// <summary>Whether the blocks see only past frames (causal attention and convolution), for streaming.</summary>
    public bool Causal { get; init; }
    public ConformerAsrDecoder Decoder { get; init; } = ConformerAsrDecoder.Ctc;
    public int PredictionDim { get; init; } = 640;
    public int PredictionLayers { get; init; } = 1;
    public int JointDim { get; init; } = 640;
    /// <summary>Output vocabulary, blank (0) included.</summary>
    public int VocabSize { get; init; } = 1024;
    public int FrequencyMasks { get; init; } = 2;
    public int FrequencyMaskWidth { get; init; } = 27;
    public int TimeMasks { get; init; } = 10;
    public double TimeMaskRatio { get; init; } = 0.05;
    /// <summary>FastEmit's λ on the transducer loss (Yu et al. 2021); zero for the plain loss.</summary>
    public double FastEmitLambda { get; init; }
}

/// <summary>
/// A Conformer speech recognizer (Gulati et al. 2020): log-mel features, a convolutional subsampling front end that
/// reduces the frame rate 4x, a linear layer and a stack of Conformer blocks (section 2, Figure 1), with either a CTC
/// layer or an RNN-T decoder (a single-layer LSTM prediction network and a joint network, section 3.1) on top.
/// SpecAugment (Park et al. 2019) masks the features in training (section 3.1).
/// </summary>
/// <remarks>
/// The subsampling is two 3x3 stride-2 2-D convolutions with ReLU (the convolution subsampling of Figure 1, as ESPnet and
/// NeMo build it). The paper's blocks use Transformer-XL relative positional encoding; these use rotary position encoding,
/// the relative scheme <see cref="ConformerBlockLayer{T}"/> provides. The layers are added to the owner's list.
/// </remarks>
internal sealed class ConformerAsrNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly ConformerAsrConfiguration _o;
    private readonly MelSpectrogram<T> _mel;
    private readonly int _subsampledBins;

    public ConformerAsrNetwork(IEngine engine, List<LayerBase<T>> layers, ConformerAsrConfiguration options)
    {
        _engine = engine ?? throw new ArgumentNullException(nameof(engine));
        _o = options ?? throw new ArgumentNullException(nameof(options));
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
        if (_o.Subsampling == ConformerAsrSubsampling.Conv4x)
        {
            FirstConv = Own(new Conv2DLayer<T>(1, d, 3, 3, 2, 2, 0, 0));
            SecondConv = Own(new Conv2DLayer<T>(d, d, 3, 3, 2, 2, 0, 0));
            _subsampledBins = Subsampled(Subsampled(_o.NumMels));
            _channels = d;
        }
        else
        {
            int c = _o.SubsamplingChannels;
            FirstConv = Own(new Conv2DLayer<T>(1, c, 3, 3, 2, 2, 1, 1));
            DepthwiseStages.Add(Own(new DepthwiseSeparableConvolutionalLayer<T>(c, 3, stride: 2, padding: 1, activation: (IActivationFunction<T>)new ReLUActivation<T>())));
            DepthwiseStages.Add(Own(new DepthwiseSeparableConvolutionalLayer<T>(c, 3, stride: 2, padding: 1, activation: (IActivationFunction<T>)new ReLUActivation<T>())));
            _subsampledBins = Padded(Padded(Padded(_o.NumMels)));
            _channels = c;
        }
        if (_subsampledBins <= 0) throw new ArgumentException("Too few mel bins for two stride-2 3x3 convolutions.", nameof(options));
        InputProjection = Own(new DenseLayer<T>(d, identity));
        for (int i = 0; i < _o.NumEncoderLayers; i++)
        {
            if (_o.Block == ConformerAsrBlock.Conformer)
                Blocks.Add(Own(new ConformerBlockLayer<T>(d, _o.NumAttentionHeads, _o.FeedForwardMultiplier, _o.ConvKernel,
                    ropeTheta: _o.RopeTheta, attentionChunkSize: _o.AttentionChunkSize, causal: _o.Causal)));
            else if (_o.Block == ConformerAsrBlock.Transformer) TransformerBlocks.Add(new TransformerBlock(_o, layer => Own(layer)));
            else BranchBlocks.Add(new BranchBlock(_o, layer => Own(layer)));
        }
        foreach (int layer in _o.IntermediateCtcLayers)
            if (layer < 1 || layer >= _o.NumEncoderLayers)
                throw new ArgumentException($"Intermediate CTC layer {layer} is not below the last of {_o.NumEncoderLayers}.", nameof(options));
        if (_o.SelfConditioning || _o.Block == ConformerAsrBlock.Transformer)
            CtcNorm = Own(new LayerNormalizationLayer<T>(d));
        if (_o.Decoder == ConformerAsrDecoder.Ctc)
        {
            CtcHead = Own(new DenseLayer<T>(_o.VocabSize, identity));
            if (_o.SelfConditioning) Conditioning = Own(new DenseLayer<T>(d, identity));
            return;
        }
        // The prediction network reads the previous label, the blank (0) standing for "no label yet".
        Embedding = Own(new EmbeddingLayer<T>(_o.VocabSize, _o.PredictionDim));
        for (int i = 0; i < _o.PredictionLayers; i++) PredictionLstms.Add(Own(new LSTMLayer<T>(_o.PredictionDim)));
        JointEncoder = Own(new DenseLayer<T>(_o.JointDim, identity));
        JointPrediction = Own(new BiasFreeLinearLayer<T>(_o.PredictionDim, _o.JointDim));
        JointOutput = Own(new DenseLayer<T>(_o.VocabSize, identity));
    }

    private readonly int _channels;

    private static int Subsampled(int length) => (length - 3) / 2 + 1;

    // A 3x3 stride-2 convolution with padding 1.
    private static int Padded(int length) => (length - 1) / 2 + 1;

    public Conv2DLayer<T> FirstConv { get; }
    public Conv2DLayer<T>? SecondConv { get; }
    public List<DepthwiseSeparableConvolutionalLayer<T>> DepthwiseStages { get; } = new();
    public DenseLayer<T> InputProjection { get; }
    public List<ConformerBlockLayer<T>> Blocks { get; } = new();
    public List<BranchBlock> BranchBlocks { get; } = new();

    /// <summary>
    /// A Branchformer or E-Branchformer block (ESPnet's <c>BranchformerEncoderLayer</c> and <c>EBranchformerEncoderLayer</c>).
    /// The global branch is pre-norm self-attention; the local branch is the convolutional gating MLP: a linear layer
    /// with GELU, the convolutional spatial gating unit (half the channels layer-normalized, convolved depthwise over time
    /// and multiplied into the other half) and a linear layer back (Peng et al. 2022, section 3.2). Branchformer merges
    /// by concatenation and projection; E-Branchformer adds a depthwise convolution over the concatenation with a
    /// residual before projecting, and wraps the branches in half-step macaron feed-forward modules (Kim et al. 2022,
    /// section 3). Both close with a layer norm.
    /// </summary>
    internal sealed class BranchBlock
    {
        public BranchBlock(ConformerAsrConfiguration o, Func<LayerBase<T>, LayerBase<T>> own)
        {
            int d = o.EncoderDim, c = o.CgmlpDim;
            var identity = (IActivationFunction<T>)new IdentityActivation<T>();
            var swish = (IActivationFunction<T>)new SwishActivation<T>();
            var gelu = (IActivationFunction<T>)new GELUActivation<T>();
            Enhanced = o.Block == ConformerAsrBlock.EBranchformer;
            TLayer Own<TLayer>(TLayer layer) where TLayer : LayerBase<T> { own(layer); return layer; }
            if (Enhanced)
            {
                FeedForwardNorm = Own(new LayerNormalizationLayer<T>(d));
                FeedForwardExpand = Own(new FullyConnectedLayer<T>(d, d * o.FeedForwardMultiplier, swish));
                FeedForwardProject = Own(new FullyConnectedLayer<T>(d * o.FeedForwardMultiplier, d, identity));
            }
            AttentionNorm = Own(new LayerNormalizationLayer<T>(d));
            var attention = new MultiHeadAttentionLayer<T>(o.NumAttentionHeads, d / o.NumAttentionHeads);
            if (o.RopeTheta > 0) attention.ConfigurePositionalEncoding(PositionalEncodingType.Rotary, o.RopeTheta, 2048);
            Attention = Own(attention);
            MlpNorm = Own(new LayerNormalizationLayer<T>(d));
            ChannelExpand = Own(new FullyConnectedLayer<T>(d, c, gelu));
            GateNorm = Own(new LayerNormalizationLayer<T>(c / 2));
            GateConv = Own(new DepthwiseConv1DLayer<T>(c / 2, o.ConvKernel, multiplier: 1, stride: 1, padding: o.ConvKernel / 2, activation: identity));
            ChannelProject = Own(new FullyConnectedLayer<T>(c / 2, d, identity));
            if (Enhanced)
                MergeConv = Own(new DepthwiseConv1DLayer<T>(2 * d, o.MergeKernel, multiplier: 1, stride: 1, padding: o.MergeKernel / 2, activation: identity));
            MergeProject = Own(new FullyConnectedLayer<T>(2 * d, d, identity));
            if (Enhanced)
            {
                MacaronNorm = Own(new LayerNormalizationLayer<T>(d));
                MacaronExpand = Own(new FullyConnectedLayer<T>(d, d * o.FeedForwardMultiplier, swish));
                MacaronProject = Own(new FullyConnectedLayer<T>(d * o.FeedForwardMultiplier, d, identity));
            }
            FinalNorm = Own(new LayerNormalizationLayer<T>(d));
        }

        public bool Enhanced { get; }
        public LayerNormalizationLayer<T>? FeedForwardNorm { get; }
        public FullyConnectedLayer<T>? FeedForwardExpand { get; }
        public FullyConnectedLayer<T>? FeedForwardProject { get; }
        public LayerNormalizationLayer<T> AttentionNorm { get; }
        public MultiHeadAttentionLayer<T> Attention { get; }
        public LayerNormalizationLayer<T> MlpNorm { get; }
        public FullyConnectedLayer<T> ChannelExpand { get; }
        public LayerNormalizationLayer<T> GateNorm { get; }
        public DepthwiseConv1DLayer<T> GateConv { get; }
        public FullyConnectedLayer<T> ChannelProject { get; }
        public DepthwiseConv1DLayer<T>? MergeConv { get; }
        public FullyConnectedLayer<T> MergeProject { get; }
        public LayerNormalizationLayer<T>? MacaronNorm { get; }
        public FullyConnectedLayer<T>? MacaronExpand { get; }
        public FullyConnectedLayer<T>? MacaronProject { get; }
        public LayerNormalizationLayer<T> FinalNorm { get; }

        // A depthwise convolution over time of [L, C] features.
        private static Tensor<T> ConvolveTime(IEngine engine, DepthwiseConv1DLayer<T> conv, Tensor<T> x)
        {
            int length = x.Shape[0], channels = x.Shape[1];
            var y = conv.Forward(engine.TensorPermute(engine.Reshape(x, new[] { 1, length, channels }), new[] { 0, 2, 1 }));
            return engine.Reshape(engine.TensorPermute(y, new[] { 0, 2, 1 }), new[] { length, channels });
        }

        /// <summary>The block's output <c>[L, D]</c> of <c>[L, D]</c>.</summary>
        public Tensor<T> Forward(IEngine engine, Tensor<T> x)
        {
            var half = NumOps.FromDouble(0.5);
            int length = x.Shape[0], d = x.Shape[1];
            if (FeedForwardNorm is not null && FeedForwardExpand is not null && FeedForwardProject is not null)
                x = engine.TensorAdd(x, engine.TensorMultiplyScalar(FeedForwardProject.Forward(FeedForwardExpand.Forward(FeedForwardNorm.Forward(x))), half));
            var global = engine.Reshape(Attention.Forward(engine.Reshape(AttentionNorm.Forward(x), new[] { 1, length, d })), new[] { length, d });
            var hidden = ChannelExpand.Forward(MlpNorm.Forward(x));
            int c = hidden.Shape[1] / 2;
            var gate = ConvolveTime(engine, GateConv, GateNorm.Forward(engine.TensorSlice(hidden, new[] { 0, c }, new[] { length, c })));
            var local = ChannelProject.Forward(engine.TensorMultiply(engine.TensorSlice(hidden, new[] { 0, 0 }, new[] { length, c }), gate));
            var merged = engine.TensorConcatenate(new[] { global, local }, 1);
            if (MergeConv is not null) merged = engine.TensorAdd(merged, ConvolveTime(engine, MergeConv, merged));
            x = engine.TensorAdd(x, MergeProject.Forward(merged));
            if (MacaronNorm is not null && MacaronExpand is not null && MacaronProject is not null)
                x = engine.TensorAdd(x, engine.TensorMultiplyScalar(MacaronProject.Forward(MacaronExpand.Forward(MacaronNorm.Forward(x))), half));
            return FinalNorm.Forward(x);
        }
    }
    public DenseLayer<T>? CtcHead { get; }
    /// <summary>The layer norm before the CTC layer (self-conditioned CTC's Equation 6; the Transformer encoder's final norm).</summary>
    public LayerNormalizationLayer<T>? CtcNorm { get; }
    /// <summary>Self-conditioned CTC's shared Linear_{|V|->D} that feeds each intermediate prediction back.</summary>
    public DenseLayer<T>? Conditioning { get; }
    public List<TransformerBlock> TransformerBlocks { get; } = new();

    /// <summary>A pre-norm Transformer encoder layer: x + MHSA(LN(x)), then + FFN(LN(x)) with ReLU.</summary>
    internal sealed class TransformerBlock
    {
        public TransformerBlock(ConformerAsrConfiguration o, Func<LayerBase<T>, LayerBase<T>> own)
        {
            int d = o.EncoderDim;
            TLayer Own<TLayer>(TLayer layer) where TLayer : LayerBase<T> { own(layer); return layer; }
            AttentionNorm = Own(new LayerNormalizationLayer<T>(d));
            Attention = Own(new MultiHeadAttentionLayer<T>(o.NumAttentionHeads, d / o.NumAttentionHeads));
            FeedForwardNorm = Own(new LayerNormalizationLayer<T>(d));
            Expand = Own(new FullyConnectedLayer<T>(d, d * o.FeedForwardMultiplier, (IActivationFunction<T>)new ReLUActivation<T>()));
            Project = Own(new FullyConnectedLayer<T>(d * o.FeedForwardMultiplier, d, (IActivationFunction<T>)new IdentityActivation<T>()));
        }

        public LayerNormalizationLayer<T> AttentionNorm { get; }
        public MultiHeadAttentionLayer<T> Attention { get; }
        public LayerNormalizationLayer<T> FeedForwardNorm { get; }
        public FullyConnectedLayer<T> Expand { get; }
        public FullyConnectedLayer<T> Project { get; }

        public Tensor<T> Forward(IEngine engine, Tensor<T> x)
        {
            int length = x.Shape[0], d = x.Shape[1];
            x = engine.TensorAdd(x, engine.Reshape(Attention.Forward(engine.Reshape(AttentionNorm.Forward(x), new[] { 1, length, d })), new[] { length, d }));
            return engine.TensorAdd(x, Project.Forward(Expand.Forward(FeedForwardNorm.Forward(x))));
        }
    }
    public EmbeddingLayer<T>? Embedding { get; }
    public List<LSTMLayer<T>> PredictionLstms { get; } = new();
    public DenseLayer<T>? JointEncoder { get; }
    public BiasFreeLinearLayer<T>? JointPrediction { get; }
    public DenseLayer<T>? JointOutput { get; }

    public ConformerAsrConfiguration Configuration => _o;

    /// <summary>Encoder frames of <paramref name="melFrames"/> log-mel frames.</summary>
    public int EncodedLength(int melFrames) => _o.Subsampling == ConformerAsrSubsampling.Conv4x
        ? Subsampled(Subsampled(melFrames)) : Padded(Padded(Padded(melFrames)));

    /// <summary>Log-mel features <c>[frames, bins]</c> of a waveform, each bin normalized over the utterance.</summary>
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
            for (int t = 0; t < frames; t++) { double v = NumOps.ToDouble(mel[t, b]) - mean; square += v * v; }
            double std = Math.Sqrt(square / frames);
            for (int t = 0; t < frames; t++) features[t, b] = std > 1e-8 ? (NumOps.ToDouble(mel[t, b]) - mean) / std : 0.0;
        }
        return features;
    }

    /// <summary>SpecAugment (Park et al. 2019, section 2) as the Conformer paper sets it (section 3.1): frequency masks of
    /// width up to F and time masks of length up to a ratio of the utterance, set to zero (the normalized mean).</summary>
    public double[,] SpecAugment(double[,] features, Random random)
    {
        int frames = features.GetLength(0), bins = features.GetLength(1);
        var masked = (double[,])features.Clone();
        for (int m = 0; m < _o.FrequencyMasks; m++)
        {
            int width = random.Next(Math.Min(_o.FrequencyMaskWidth, bins) + 1), start = random.Next(Math.Max(1, bins - width + 1));
            for (int t = 0; t < frames; t++)
                for (int b = start; b < Math.Min(bins, start + width); b++) masked[t, b] = 0.0;
        }
        int maxLength = (int)(_o.TimeMaskRatio * frames);
        for (int m = 0; m < _o.TimeMasks && maxLength > 0; m++)
        {
            int length = random.Next(maxLength + 1), start = random.Next(Math.Max(1, frames - length + 1));
            for (int t = start; t < Math.Min(frames, start + length); t++)
                for (int b = 0; b < bins; b++) masked[t, b] = 0.0;
        }
        return masked;
    }

    /// <summary>The encoder output <c>[L, EncoderDim]</c> of features <c>[frames, bins]</c>.</summary>
    public Tensor<T> Encode(double[,] features, bool training) => EncodeWithIntermediates(features, training).Output;

    /// <summary>The encoder output and the CTC logits of each intermediate CTC layer, in order.</summary>
    public (Tensor<T> Output, List<Tensor<T>> IntermediateLogits) EncodeWithIntermediates(double[,] features, bool training)
    {
        var intermediates = new List<Tensor<T>>();
        int frames = features.GetLength(0), bins = features.GetLength(1), d = _o.EncoderDim;
        var mel = new Tensor<T>(new[] { 1, 1, frames, bins });
        for (int t = 0; t < frames; t++)
            for (int b = 0; b < bins; b++) mel[0, 0, t, b] = NumOps.FromDouble(features[t, b]);
        var x = _engine.ReLU(FirstConv.Forward(mel));
        if (SecondConv is not null) x = _engine.ReLU(SecondConv.Forward(x));
        foreach (var stage in DepthwiseStages) x = stage.Forward(x);
        int length = x.Shape[2];
        if (length <= 0) throw new ArgumentException("The audio is too short for the convolution subsampling.");
        var steps = _engine.Reshape(_engine.TensorPermute(x, new[] { 0, 2, 1, 3 }), new[] { length, _channels * _subsampledBins });
        var h = InputProjection.Forward(steps);
        if (_o.Block == ConformerAsrBlock.Transformer) h = _engine.TensorAdd(_engine.TensorMultiplyScalar(h, NumOps.FromDouble(Math.Sqrt(d))), Positions(length, d));
        int count = Math.Max(Blocks.Count, Math.Max(BranchBlocks.Count, TransformerBlocks.Count));
        for (int i = 0; i < count; i++)
        {
            if (Blocks.Count > 0)
            {
                Blocks[i].SetTrainingMode(training);
                h = Blocks[i].Forward(h);
            }
            else if (TransformerBlocks.Count > 0) h = TransformerBlocks[i].Forward(_engine, h);
            else h = BranchBlocks[i].Forward(_engine, h);
            if (Array.IndexOf(_o.IntermediateCtcLayers, i + 1) < 0) continue;
            // Self-conditioned CTC (Equations 6, 7 and 9): the shared norm and CTC layer give Z_l, fed back through a shared
            // linear layer; InterCTC (Lee and Watanabe 2021, Equation 13) only reads the prediction.
            var normed = CtcNorm is not null ? CtcNorm.Forward(h) : h;
            var logits = CtcLogitsOfNormed(normed);
            intermediates.Add(logits);
            if (_o.SelfConditioning && Conditioning is not null)
                h = _engine.TensorAdd(normed, Conditioning.Forward(_engine.Softmax(logits, axis: -1)));
        }
        return (h, intermediates);
    }

    // Absolute sinusoidal positions [L, D] (Vaswani et al. 2017).
    private Tensor<T> Positions(int length, int d)
    {
        var positions = new Tensor<T>(new[] { length, d });
        for (int t = 0; t < length; t++)
            for (int i = 0; i < d; i += 2)
            {
                double angle = t / Math.Pow(10000.0, (double)i / d);
                positions[t, i] = NumOps.FromDouble(Math.Sin(angle));
                if (i + 1 < d) positions[t, i + 1] = NumOps.FromDouble(Math.Cos(angle));
            }
        return positions;
    }

    private Tensor<T> CtcLogitsOfNormed(Tensor<T> normed) =>
        (CtcHead ?? throw new InvalidOperationException("This network decodes with a transducer.")).Forward(normed);

    /// <summary>The CTC logits <c>[L, VocabSize]</c>.</summary>
    public Tensor<T> CtcLogits(Tensor<T> encoded) => CtcLogitsOfNormed(CtcNorm is not null ? CtcNorm.Forward(encoded) : encoded);

    /// <summary>The CTC loss of labels for CTC logits <c>[L, VocabSize]</c>.</summary>
    public Tensor<T> CtcLoss(Tensor<T> logits, int[] labels) =>
        _engine.TensorCTCLoss(_engine.Reshape(_engine.TensorLogSoftmax(logits, axis: -1), new[] { logits.Shape[0], 1, _o.VocabSize }),
            new Tensor<int>(labels, new[] { labels.Length }), new[] { logits.Shape[0] }, new[] { labels.Length }, 0);

    /// <summary>
    /// The CTC training loss of an utterance: the final layer's CTC loss, mixed with the mean intermediate CTC loss as
    /// L = (1 - w) L_CTC + w L_inter when intermediate layers are configured (InterCTC Equation 14; self-conditioned CTC
    /// Equations 11 and 12).
    /// </summary>
    public Tensor<T> CtcTrainingLoss(double[,] features, int[] labels, bool training)
    {
        var (output, intermediates) = EncodeWithIntermediates(features, training);
        var loss = CtcLoss(CtcLogits(output), labels);
        if (intermediates.Count == 0) return loss;
        Tensor<T>? inter = null;
        foreach (var logits in intermediates)
        {
            var term = CtcLoss(logits, labels);
            inter = inter is null ? term : _engine.TensorAdd(inter, term);
        }
        double w = _o.IntermediateCtcWeight;
        return _engine.TensorAdd(_engine.TensorMultiplyScalar(loss, NumOps.FromDouble(1 - w)),
            _engine.TensorMultiplyScalar(inter ?? loss, NumOps.FromDouble(w / intermediates.Count)));
    }

    /// <summary>The prediction network's output <c>[U + 1, PredictionDim]</c> for the label history: the blank, then each
    /// label.</summary>
    public Tensor<T> Predict(IReadOnlyList<int> labels)
    {
        var embedding = Embedding ?? throw new InvalidOperationException("This network decodes with CTC.");
        var history = new Tensor<T>(new[] { 1, labels.Count + 1 });
        for (int u = 0; u < labels.Count; u++) history[0, u + 1] = NumOps.FromDouble(labels[u]);
        var g = embedding.Forward(history);                                                                // [1, U+1, P]
        foreach (var lstm in PredictionLstms) g = lstm.Forward(g);
        return _engine.Reshape(g, new[] { labels.Count + 1, _o.PredictionDim });
    }

    /// <summary>The joint network's log-probabilities <c>[T, U + 1, VocabSize]</c>:
    /// log softmax(W tanh(W_enc f_t + W_pred g_u)) (Graves 2012, section 2, with the additive joint of He et al. 2019).</summary>
    public Tensor<T> Joint(Tensor<T> encoded, Tensor<T> predicted)
    {
        if (JointEncoder is null || JointPrediction is null || JointOutput is null) throw new InvalidOperationException("This network decodes with CTC.");
        int frames = encoded.Shape[0], width = predicted.Shape[0], j = _o.JointDim;
        var f = _engine.Reshape(JointEncoder.Forward(encoded), new[] { frames, 1, j });
        var g = _engine.Reshape(JointPrediction.Forward(predicted), new[] { 1, width, j });
        var hidden = _engine.Tanh(_engine.TensorAdd(_engine.TensorBroadcastTo(f, new[] { frames, width, j }), _engine.TensorBroadcastTo(g, new[] { frames, width, j })));
        var logits = JointOutput.Forward(_engine.Reshape(hidden, new[] { frames * width, j }));
        return _engine.TensorLogSoftmax(_engine.Reshape(logits, new[] { frames, width, _o.VocabSize }), axis: -1);
    }

    /// <summary>The transducer loss of a label sequence for an encoded utterance.</summary>
    public Tensor<T> TransducerLoss(Tensor<T> encoded, IReadOnlyList<int> labels) =>
        Foundations.TransducerLoss.Compute(_engine, Joint(encoded, Predict(labels)), labels, blank: 0, _o.FastEmitLambda);

    /// <summary>Greedy transducer decoding: at each frame emit the most likely symbol until it is the blank (at most
    /// <paramref name="maxSymbolsPerFrame"/> labels per frame).</summary>
    public List<int> GreedyTransducer(Tensor<T> encoded, int maxSymbolsPerFrame = 10)
    {
        var labels = new List<int>();
        int frames = encoded.Shape[0], d = encoded.Shape[1];
        for (int t = 0; t < frames; t++)
        {
            var frame = _engine.TensorSlice(encoded, new[] { t, 0 }, new[] { 1, d });
            for (int s = 0; s < maxSymbolsPerFrame; s++)
            {
                var g = Predict(labels);
                var last = _engine.TensorSlice(g, new[] { labels.Count, 0 }, new[] { 1, _o.PredictionDim });
                var logProbs = Joint(frame, last);
                int best = 0;
                for (int k = 1; k < _o.VocabSize; k++)
                    if (NumOps.GreaterThan(logProbs[0, 0, k], logProbs[0, 0, best])) best = k;
                if (best == 0) break;
                labels.Add(best);
            }
        }
        return labels;
    }

    /// <summary>Greedy CTC decoding: the most likely symbol per frame, repeats merged and blanks removed.</summary>
    public static List<int> GreedyCtc(Tensor<T> logits)
    {
        var labels = new List<int>();
        int frames = logits.Shape[0], vocabulary = logits.Shape[1], previous = -1;
        for (int t = 0; t < frames; t++)
        {
            int best = 0;
            for (int k = 1; k < vocabulary; k++)
                if (NumOps.GreaterThan(logits[t, k], logits[t, best])) best = k;
            if (best != previous && best != 0) labels.Add(best);
            previous = best;
        }
        return labels;
    }
}

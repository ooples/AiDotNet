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

    /// <summary>A token-and-duration transducer (Xu et al. 2023): the joint network also predicts how many frames each
    /// emission advances.</summary>
    TokenAndDuration,

    /// <summary>An attention decoder alone, without a CTC layer.</summary>
    Attention,
}

/// <summary>The activation of a feed-forward network.</summary>
internal enum ConformerAsrFeedForward
{
    /// <summary>Linear, ReLU, linear.</summary>
    Relu,

    /// <summary>Linear, GELU, linear.</summary>
    Gelu,

    /// <summary>A SwiGLU gate (Shazeer 2020): one linear layer to twice the units, Swish of one half times the other,
    /// then a linear layer back.</summary>
    SwiGlu,
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

    /// <summary>Moonshine's stem over the raw waveform (Jeffries et al. 2024, Figure 3): a width-127 stride-64 convolution
    /// with tanh and group normalization, a width-7 stride-3 convolution with GELU and a width-3 stride-2 convolution with
    /// GELU, 384x in all.</summary>
    RawWaveformStem,

    /// <summary>Mohamed et al. 2019's convolutional context (section 2.2): two 2-D convolution blocks of 64 and 128 channels,
    /// each two 3x3 convolutions with layer norm and ReLU followed by 2x2 max pooling, 4x in time.</summary>
    VggBlocks,
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
    /// <summary>Whether sinusoidal absolute positions are added to the scaled input (always for the Transformer block).</summary>
    public bool AbsolutePositions { get; init; }
    /// <summary>Hidden units of the convolutional gating MLP (Branchformer and E-Branchformer).</summary>
    public int CgmlpDim { get; init; } = 2048;
    /// <summary>Depthwise kernel of E-Branchformer's merge convolution.</summary>
    public int MergeKernel { get; init; } = 31;
    /// <summary>Encoder layers (1-based) whose outputs get an intermediate CTC loss through the shared CTC layer.</summary>
    public int[] IntermediateCtcLayers { get; init; } = Array.Empty<int>();
    /// <summary>Weight of the mean intermediate CTC loss: L = (1 - w) L_CTC + w L_inter.</summary>
    public double IntermediateCtcWeight { get; init; }
    /// <summary>Activation of the Transformer encoder block's feed-forward network.</summary>
    public ConformerAsrFeedForward EncoderFeedForward { get; init; } = ConformerAsrFeedForward.Relu;
    /// <summary>Activation of the attention decoder's feed-forward networks.</summary>
    public ConformerAsrFeedForward DecoderFeedForward { get; init; } = ConformerAsrFeedForward.Relu;
    /// <summary>Causal 1-D convolutions (width 3, layer norm, ReLU) over the decoder's token embeddings, which give it
    /// positions in place of encodings (Mohamed et al. 2019, section 2.2); zero for none.</summary>
    public int DecoderContextConvolutions { get; init; }
    /// <summary>Width of the decoder's token embedding when context convolutions project it to the model width.</summary>
    public int DecoderEmbeddingDim { get; init; }
    /// <summary>Whether the encoder adds no positional encoding at all (the convolutional context supplies order).</summary>
    public bool NoPositions { get; init; }
    /// <summary>Whether the attention decoder's self-attention uses rotary positions instead of added sinusoids.</summary>
    public bool DecoderRotary { get; init; }
    /// <summary>Whether the decoder's output layer shares the token embedding's table (logits h E^T).</summary>
    public bool DecoderTiedOutput { get; init; }
    /// <summary>Channels of the 4x convolution subsampling; the encoder width when zero.</summary>
    public int Conv4xChannels { get; init; }
    /// <summary>The token that starts the decoder's input; the vocabulary's last when null (ESPnet's convention).</summary>
    public int? DecoderStartToken { get; init; }
    /// <summary>The token that ends a transcript; the vocabulary's last when null.</summary>
    public int? DecoderEndToken { get; init; }
    /// <summary>Transformer decoder layers of a hybrid CTC/attention model (Watanabe et al. 2017); zero for none.</summary>
    public int AttentionDecoderLayers { get; init; }
    /// <summary>Feed-forward units of each attention-decoder layer.</summary>
    public int AttentionDecoderFeedForward { get; init; } = 2048;
    /// <summary>Weight of the CTC loss in the hybrid objective: L = w L_CTC + (1 - w) L_attention.</summary>
    public double CtcWeight { get; init; } = 0.3;
    /// <summary>Label smoothing of the attention decoder's cross-entropy.</summary>
    public double LabelSmoothing { get; init; } = 0.1;
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
    /// <summary>The durations a token-and-duration joint network chooses among (Parakeet-TDT: 0 to 4).</summary>
    public int[] Durations { get; init; } = { 0, 1, 2, 3, 4 };
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
        if (_o.Subsampling == ConformerAsrSubsampling.RawWaveformStem)
        {
            var tanh = (IActivationFunction<T>)new TanhActivation<T>();
            var gelu = (IActivationFunction<T>)new GELUActivation<T>();
            StemConvs.Add(Own(new Conv1DLayer<T>(inputChannels: 1, outputChannels: d, kernelSize: 127, dilation: 1, stride: 64, padding: 0, activation: tanh)));
            StemNorm = Own(new GroupNormalizationLayer<T>(1, d));
            StemConvs.Add(Own(new Conv1DLayer<T>(inputChannels: d, outputChannels: 2 * d, kernelSize: 7, dilation: 1, stride: 3, padding: 0, activation: gelu)));
            StemConvs.Add(Own(new Conv1DLayer<T>(inputChannels: 2 * d, outputChannels: d, kernelSize: 3, dilation: 1, stride: 2, padding: 0, activation: gelu)));
            _subsampledBins = 1;
            _channels = d;
        }
        else if (_o.Subsampling == ConformerAsrSubsampling.VggBlocks)
        {
            int[] widths = { 64, 128 };
            int inChannels = 1, bins = _o.NumMels;
            foreach (int c in widths)
            {
                for (int k = 0; k < 2; k++)
                {
                    VggConvs.Add(Own(new Conv2DLayer<T>(inChannels, c, 3, 3, 1, 1, 1, 1)));
                    VggNorms.Add(Own(new LayerNormalizationLayer<T>(c)));
                    inChannels = c;
                }
                bins /= 2;
            }
            _subsampledBins = bins;
            _channels = widths[widths.Length - 1];
        }
        else if (_o.Subsampling == ConformerAsrSubsampling.Conv4x)
        {
            int c = _o.Conv4xChannels > 0 ? _o.Conv4xChannels : d;
            FirstConv = Own(new Conv2DLayer<T>(1, c, 3, 3, 2, 2, 0, 0));
            SecondConv = Own(new Conv2DLayer<T>(c, c, 3, 3, 2, 2, 0, 0));
            _subsampledBins = Subsampled(Subsampled(_o.NumMels));
            _channels = c;
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
        if (_o.Subsampling != ConformerAsrSubsampling.RawWaveformStem) InputProjection = Own(new DenseLayer<T>(d, identity));
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
        if (_o.Decoder is ConformerAsrDecoder.Ctc or ConformerAsrDecoder.Attention)
        {
            if (_o.Decoder == ConformerAsrDecoder.Ctc) CtcHead = Own(new DenseLayer<T>(_o.VocabSize, identity));
            else if (_o.AttentionDecoderLayers <= 0) throw new ArgumentException("An attention-only network needs decoder layers.", nameof(options));
            if (_o.SelfConditioning) Conditioning = Own(new DenseLayer<T>(d, identity));
            if (_o.AttentionDecoderLayers > 0)
            {
                // The last token is the start and end of a sentence, as ESPnet sets them.
                int embedding = _o.DecoderEmbeddingDim > 0 ? _o.DecoderEmbeddingDim : d;
                TokenEmbedding = Own(new EmbeddingLayer<T>(_o.VocabSize, embedding));
                for (int i = 0; i < _o.DecoderContextConvolutions; i++)
                {
                    DecoderContextConvs.Add(Own(new Conv1DLayer<T>(inputChannels: embedding, outputChannels: embedding, kernelSize: 3, dilation: 1, stride: 1, padding: 0, activation: identity)));
                    DecoderContextNorms.Add(Own(new LayerNormalizationLayer<T>(embedding)));
                }
                if (embedding != d) DecoderEmbeddingProjection = Own(new DenseLayer<T>(d, identity));
                for (int i = 0; i < _o.AttentionDecoderLayers; i++) DecoderBlocks.Add(new DecoderBlock(_o, layer => Own(layer)));
                DecoderNorm = Own(new LayerNormalizationLayer<T>(d));
                DecoderOutput = _o.DecoderTiedOutput ? Own(new TiedEmbeddingHeadLayer<T>(TokenEmbedding)) : Own(new DenseLayer<T>(_o.VocabSize, identity));
            }
            return;
        }
        // The prediction network reads the previous label, the blank (0) standing for "no label yet".
        Embedding = Own(new EmbeddingLayer<T>(_o.VocabSize, _o.PredictionDim));
        for (int i = 0; i < _o.PredictionLayers; i++) PredictionLstms.Add(Own(new LSTMLayer<T>(_o.PredictionDim)));
        JointEncoder = Own(new DenseLayer<T>(_o.JointDim, identity));
        JointPrediction = Own(new BiasFreeLinearLayer<T>(_o.PredictionDim, _o.JointDim));
        JointOutput = Own(new DenseLayer<T>(_o.VocabSize + (_o.Decoder == ConformerAsrDecoder.TokenAndDuration ? _o.Durations.Length : 0), identity));
    }

    private readonly int _channels;

    private static int Subsampled(int length) => (length - 3) / 2 + 1;

    // A 3x3 stride-2 convolution with padding 1.
    private static int Padded(int length) => (length - 1) / 2 + 1;

    public Conv2DLayer<T>? FirstConv { get; }
    public List<Conv1DLayer<T>> StemConvs { get; } = new();
    public List<Conv2DLayer<T>> VggConvs { get; } = new();
    public List<LayerNormalizationLayer<T>> VggNorms { get; } = new();
    public List<Conv1DLayer<T>> DecoderContextConvs { get; } = new();
    public List<LayerNormalizationLayer<T>> DecoderContextNorms { get; } = new();
    public DenseLayer<T>? DecoderEmbeddingProjection { get; }
    public GroupNormalizationLayer<T>? StemNorm { get; }
    public Conv2DLayer<T>? SecondConv { get; }
    public List<DepthwiseSeparableConvolutionalLayer<T>> DepthwiseStages { get; } = new();
    public DenseLayer<T>? InputProjection { get; }
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
    public EmbeddingLayer<T>? TokenEmbedding { get; }
    public List<DecoderBlock> DecoderBlocks { get; } = new();
    public LayerNormalizationLayer<T>? DecoderNorm { get; }
    public LayerBase<T>? DecoderOutput { get; }

    /// <summary>The start-and-end-of-sentence token of the attention decoder: the vocabulary's last.</summary>
    public int SentenceBoundary => _o.VocabSize - 1;

    /// <summary>The token that starts the decoder's input.</summary>
    public int StartToken => _o.DecoderStartToken ?? SentenceBoundary;

    /// <summary>The token that ends a transcript.</summary>
    public int EndToken => _o.DecoderEndToken ?? SentenceBoundary;

    /// <summary>A feed-forward network of the given activation: ReLU or GELU between two linear layers, or a SwiGLU gate.</summary>
    internal sealed class FeedForward
    {
        private readonly bool _gated;

        public FeedForward(int d, int units, ConformerAsrFeedForward kind, Func<LayerBase<T>, LayerBase<T>> own)
        {
            _gated = kind == ConformerAsrFeedForward.SwiGlu;
            var activation = kind == ConformerAsrFeedForward.Gelu ? (IActivationFunction<T>)new GELUActivation<T>()
                : kind == ConformerAsrFeedForward.Relu ? new ReLUActivation<T>() : new IdentityActivation<T>();
            Expand = new FullyConnectedLayer<T>(d, _gated ? 2 * units : units, activation);
            Project = new FullyConnectedLayer<T>(units, d, (IActivationFunction<T>)new IdentityActivation<T>());
            own(Expand);
            own(Project);
        }

        public FullyConnectedLayer<T> Expand { get; }
        public FullyConnectedLayer<T> Project { get; }

        public Tensor<T> Forward(IEngine engine, Tensor<T> x)
        {
            var h = Expand.Forward(x);
            if (_gated)
            {
                var shape = h.Shape.ToArray();
                int rank = shape.Length, units = shape[rank - 1] / 2;
                var start = new int[rank];
                var length = (int[])shape.Clone();
                length[rank - 1] = units;
                var gate = engine.TensorSlice(h, start, length);
                start[rank - 1] = units;
                h = engine.TensorMultiply(engine.Swish(gate), engine.TensorSlice(h, start, length));
            }
            return Project.Forward(h);
        }
    }

    /// <summary>A pre-norm Transformer decoder layer (ESPnet's <c>DecoderLayer</c>): causal self-attention, attention over
    /// the encoder output and a feed-forward network, each with a residual.</summary>
    internal sealed class DecoderBlock
    {
        public DecoderBlock(ConformerAsrConfiguration o, Func<LayerBase<T>, LayerBase<T>> own)
        {
            int d = o.EncoderDim;
            TLayer Own<TLayer>(TLayer layer) where TLayer : LayerBase<T> { own(layer); return layer; }
            SelfNorm = Own(new LayerNormalizationLayer<T>(d));
            var self = new MultiHeadAttentionLayer<T>(o.NumAttentionHeads, d / o.NumAttentionHeads) { UseCausalMask = true };
            if (o.DecoderRotary) self.ConfigurePositionalEncoding(PositionalEncodingType.Rotary, o.RopeTheta > 0 ? o.RopeTheta : 10000.0, 2048);
            SelfAttention = Own(self);
            SourceNorm = Own(new LayerNormalizationLayer<T>(d));
            SourceAttention = Own(new MultiHeadAttentionLayer<T>(o.NumAttentionHeads, d / o.NumAttentionHeads));
            FeedForwardNorm = Own(new LayerNormalizationLayer<T>(d));
            Network = new FeedForward(d, o.AttentionDecoderFeedForward, o.DecoderFeedForward, own);
        }

        public LayerNormalizationLayer<T> SelfNorm { get; }
        public MultiHeadAttentionLayer<T> SelfAttention { get; }
        public LayerNormalizationLayer<T> SourceNorm { get; }
        public MultiHeadAttentionLayer<T> SourceAttention { get; }
        public LayerNormalizationLayer<T> FeedForwardNorm { get; }
        public FeedForward Network { get; }

        /// <summary>Tokens' states <c>[1, n, D]</c> given the encoder output <c>[1, L, D]</c>.</summary>
        public Tensor<T> Forward(IEngine engine, Tensor<T> x, Tensor<T> memory)
        {
            x = engine.TensorAdd(x, SelfAttention.Forward(SelfNorm.Forward(x)));
            x = engine.TensorAdd(x, SourceAttention.Forward(SourceNorm.Forward(x), memory));
            return engine.TensorAdd(x, Network.Forward(engine, FeedForwardNorm.Forward(x)));
        }
    }

    /// <summary>The attention decoder's logits <c>[n, VocabSize]</c> for a token history, reading the encoder output
    /// <c>[L, D]</c>: embedding scaled by sqrt(D) plus sinusoidal positions, the decoder layers and a final norm.</summary>
    public Tensor<T> DecoderLogits(IReadOnlyList<int> tokens, Tensor<T> encoded)
    {
        if (TokenEmbedding is null || DecoderNorm is null || DecoderOutput is null) throw new InvalidOperationException("This network has no attention decoder.");
        int n = tokens.Count, d = _o.EncoderDim;
        var ids = new Tensor<T>(new[] { 1, n });
        for (int i = 0; i < n; i++) ids[0, i] = NumOps.FromDouble(tokens[i]);
        int embedding = _o.DecoderEmbeddingDim > 0 ? _o.DecoderEmbeddingDim : d;
        var x = _engine.Reshape(TokenEmbedding.Forward(ids), new[] { n, embedding });
        for (int i = 0; i < DecoderContextConvs.Count; i++)
        {
            // Causal: two frames of left padding, so position i sees tokens i - 2 to i.
            var padded = _engine.TensorConstantPad(_engine.TensorPermute(_engine.Reshape(x, new[] { 1, n, embedding }), new[] { 0, 2, 1 }), new[] { 2, 0 }, NumOps.Zero);
            var y = _engine.Reshape(_engine.TensorPermute(DecoderContextConvs[i].Forward(padded), new[] { 0, 2, 1 }), new[] { n, embedding });
            x = _engine.ReLU(DecoderContextNorms[i].Forward(y));
        }
        if (DecoderEmbeddingProjection is not null) x = DecoderEmbeddingProjection.Forward(x);
        x = _o.DecoderRotary || DecoderContextConvs.Count > 0 ? _engine.Reshape(x, new[] { 1, n, d })
            : _engine.Reshape(_engine.TensorAdd(_engine.TensorMultiplyScalar(x, NumOps.FromDouble(Math.Sqrt(d))), Positions(n, d)), new[] { 1, n, d });
        // The decoder reads the encoder's normalized output (the Transformer encoder's final norm).
        var memory = _engine.Reshape(CtcNorm is not null ? CtcNorm.Forward(encoded) : encoded, new[] { 1, encoded.Shape[0], d });
        foreach (var block in DecoderBlocks) x = block.Forward(_engine, x, memory);
        return DecoderOutput.Forward(_engine.Reshape(DecoderNorm.Forward(x), new[] { n, d }));
    }

    /// <summary>The attention decoder's label-smoothed cross-entropy of a transcript: it reads the boundary token then
    /// the labels, and predicts the labels then the boundary token (Watanabe et al. 2017, section 3.2).</summary>
    public Tensor<T> AttentionLoss(Tensor<T> encoded, IReadOnlyList<int> labels)
    {
        var input = new List<int> { StartToken };
        input.AddRange(labels);
        var logits = DecoderLogits(input, encoded);
        int n = input.Count, v = _o.VocabSize;
        double smooth = _o.LabelSmoothing;
        var weights = new Tensor<T>(new[] { n, v });
        for (int i = 0; i < n; i++)
        {
            int target = i < labels.Count ? labels[i] : EndToken;
            for (int k = 0; k < v; k++)
                weights[i, k] = NumOps.FromDouble(-((k == target ? 1 - smooth : 0) + smooth / v) / n);
        }
        return _engine.ReduceSum(_engine.TensorMultiply(_engine.TensorLogSoftmax(logits, axis: -1), weights), new[] { 0, 1 }, keepDims: false);
    }

    /// <summary>Greedy attention decoding: the most likely next token until the boundary token or
    /// <paramref name="maxLength"/> tokens.</summary>
    public List<int> GreedyAttention(Tensor<T> encoded, int maxLength)
    {
        var tokens = new List<int> { StartToken };
        for (int step = 0; step < maxLength; step++)
        {
            var logits = DecoderLogits(tokens, encoded);
            int last = tokens.Count - 1, best = 0;
            for (int k = 1; k < _o.VocabSize; k++)
                if (NumOps.GreaterThan(logits[last, k], logits[last, best])) best = k;
            if (best == EndToken) break;
            tokens.Add(best);
        }
        tokens.RemoveAt(0);
        return tokens;
    }

    /// <summary>A pre-norm Transformer encoder layer: x + MHSA(LN(x)), then + FFN(LN(x)) with ReLU.</summary>
    internal sealed class TransformerBlock
    {
        public TransformerBlock(ConformerAsrConfiguration o, Func<LayerBase<T>, LayerBase<T>> own)
        {
            int d = o.EncoderDim;
            TLayer Own<TLayer>(TLayer layer) where TLayer : LayerBase<T> { own(layer); return layer; }
            AttentionNorm = Own(new LayerNormalizationLayer<T>(d));
            var attention = new MultiHeadAttentionLayer<T>(o.NumAttentionHeads, d / o.NumAttentionHeads);
            // A positive rotary base gives rotary attention (Moonshine); otherwise positions are added sinusoids.
            if (o.RopeTheta > 0) attention.ConfigurePositionalEncoding(PositionalEncodingType.Rotary, o.RopeTheta, 2048);
            Attention = Own(attention);
            FeedForwardNorm = Own(new LayerNormalizationLayer<T>(d));
            Network = new FeedForward(d, d * o.FeedForwardMultiplier, o.EncoderFeedForward, own);
        }

        public LayerNormalizationLayer<T> AttentionNorm { get; }
        public MultiHeadAttentionLayer<T> Attention { get; }
        public LayerNormalizationLayer<T> FeedForwardNorm { get; }
        public FeedForward Network { get; }

        public Tensor<T> Forward(IEngine engine, Tensor<T> x)
        {
            int length = x.Shape[0], d = x.Shape[1];
            x = engine.TensorAdd(x, engine.Reshape(Attention.Forward(engine.Reshape(AttentionNorm.Forward(x), new[] { 1, length, d })), new[] { length, d }));
            return engine.TensorAdd(x, Network.Forward(engine, FeedForwardNorm.Forward(x)));
        }
    }
    public EmbeddingLayer<T>? Embedding { get; }
    public List<LSTMLayer<T>> PredictionLstms { get; } = new();
    public DenseLayer<T>? JointEncoder { get; }
    public BiasFreeLinearLayer<T>? JointPrediction { get; }
    public DenseLayer<T>? JointOutput { get; }

    public ConformerAsrConfiguration Configuration => _o;

    /// <summary>Encoder frames of <paramref name="melFrames"/> log-mel frames.</summary>
    public int EncodedLength(int melFrames) => _o.Subsampling switch
    {
        ConformerAsrSubsampling.Conv4x => Subsampled(Subsampled(melFrames)),
        ConformerAsrSubsampling.RawWaveformStem => ((melFrames - 127) / 64 + 1 - 7) / 3 + 1 is int y ? (y - 3) / 2 + 1 : 0,
        _ => Padded(Padded(Padded(melFrames))),
    };

    /// <summary>Log-mel features <c>[frames, bins]</c> of a waveform, each bin normalized over the utterance.</summary>
    public double[,] Features(IReadOnlyList<double> waveform)
    {
        if (_o.Subsampling == ConformerAsrSubsampling.RawWaveformStem)
        {
            // Moonshine reads the waveform itself: one channel of samples.
            var raw = new double[waveform.Count, 1];
            for (int i = 0; i < waveform.Count; i++) raw[i, 0] = waveform[i];
            return raw;
        }
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
        Tensor<T> h;
        int length;
        if (InputProjection is null)
        {
            var wave = new Tensor<T>(new[] { 1, 1, frames });
            for (int t = 0; t < frames; t++) wave[0, 0, t] = NumOps.FromDouble(features[t, 0]);
            var y = StemConvs[0].Forward(wave);
            // Group normalization reads rank 4 as [batch, channels, height, width].
            if (StemNorm is not null) y = _engine.Reshape(StemNorm.Forward(_engine.Reshape(y, new[] { 1, y.Shape[1], y.Shape[2], 1 })), new[] { 1, y.Shape[1], y.Shape[2] });
            for (int i = 1; i < StemConvs.Count; i++) y = StemConvs[i].Forward(y);
            length = y.Shape[2];
            if (length <= 0) throw new ArgumentException("The audio is too short for the convolution stem.");
            h = _engine.Reshape(_engine.TensorPermute(y, new[] { 0, 2, 1 }), new[] { length, d });
        }
        else
        {
            var mel = new Tensor<T>(new[] { 1, 1, frames, bins });
            for (int t = 0; t < frames; t++)
                for (int b = 0; b < bins; b++) mel[0, 0, t, b] = NumOps.FromDouble(features[t, b]);
            Tensor<T> x;
            if (VggConvs.Count > 0)
            {
                x = mel;
                for (int i = 0; i < VggConvs.Count; i++)
                {
                    x = VggConvs[i].Forward(x);
                    x = _engine.ReLU(_engine.TensorPermute(VggNorms[i].Forward(_engine.TensorPermute(x, new[] { 0, 2, 3, 1 })), new[] { 0, 3, 1, 2 }));
                    if (i % 2 == 1) x = _engine.MaxPool2D(x, 2, 2);
                }
            }
            else x = _engine.ReLU((FirstConv ?? throw new InvalidOperationException("No convolution front end.")).Forward(mel));
            if (SecondConv is not null) x = _engine.ReLU(SecondConv.Forward(x));
            foreach (var stage in DepthwiseStages) x = stage.Forward(x);
            length = x.Shape[2];
            if (length <= 0) throw new ArgumentException("The audio is too short for the convolution subsampling.");
            var steps = _engine.Reshape(_engine.TensorPermute(x, new[] { 0, 2, 1, 3 }), new[] { length, _channels * _subsampledBins });
            h = InputProjection.Forward(steps);
        }
        if (!_o.NoPositions && ((_o.Block == ConformerAsrBlock.Transformer && _o.RopeTheta <= 0) || _o.AbsolutePositions)) h = _engine.TensorAdd(_engine.TensorMultiplyScalar(h, NumOps.FromDouble(Math.Sqrt(d))), Positions(length, d));
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
        if (_o.Decoder == ConformerAsrDecoder.Attention) return AttentionLoss(output, labels);
        var loss = CtcLoss(CtcLogits(output), labels);
        // Hybrid CTC/attention (Watanabe et al. 2017, Equation 13): L = w L_CTC + (1 - w) L_attention.
        if (DecoderBlocks.Count > 0)
            loss = _engine.TensorAdd(_engine.TensorMultiplyScalar(loss, NumOps.FromDouble(_o.CtcWeight)),
                _engine.TensorMultiplyScalar(AttentionLoss(output, labels), NumOps.FromDouble(1 - _o.CtcWeight)));
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
        var logits = _engine.Reshape(JointOutput.Forward(_engine.Reshape(hidden, new[] { frames * width, j })), new[] { frames, width, _o.VocabSize + (_o.Decoder == ConformerAsrDecoder.TokenAndDuration ? _o.Durations.Length : 0) });
        if (_o.Decoder != ConformerAsrDecoder.TokenAndDuration) return _engine.TensorLogSoftmax(logits, axis: -1);
        // The token and duration distributions are normalized separately (Xu et al. 2023, section 3).
        int v = _o.VocabSize, n = _o.Durations.Length;
        var tokens = _engine.TensorLogSoftmax(_engine.TensorSlice(logits, new[] { 0, 0, 0 }, new[] { frames, width, v }), axis: -1);
        var lengths = _engine.TensorLogSoftmax(_engine.TensorSlice(logits, new[] { 0, 0, v }, new[] { frames, width, n }), axis: -1);
        return _engine.TensorConcatenate(new[] { tokens, lengths }, 2);
    }

    /// <summary>The transducer loss of a label sequence for an encoded utterance.</summary>
    public Tensor<T> TransducerLoss(Tensor<T> encoded, IReadOnlyList<int> labels) => _o.Decoder == ConformerAsrDecoder.TokenAndDuration
        ? Foundations.TransducerLoss.ComputeTokenAndDuration(_engine, Joint(encoded, Predict(labels)), _o.VocabSize, _o.Durations, labels, blank: 0)
        : Foundations.TransducerLoss.Compute(_engine, Joint(encoded, Predict(labels)), labels, blank: 0, _o.FastEmitLambda);

    /// <summary>Greedy transducer decoding: at each frame emit the most likely symbol until it is the blank (at most
    /// <paramref name="maxSymbolsPerFrame"/> labels per frame).</summary>
    public List<int> GreedyTransducer(Tensor<T> encoded, int maxSymbolsPerFrame = 10)
    {
        if (_o.Decoder == ConformerAsrDecoder.TokenAndDuration) return GreedyTokenAndDuration(encoded, maxSymbolsPerFrame);
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

    // TDT greedy decoding (Xu et al. 2023, Algorithm 1): the most likely token and duration at the current frame; a token
    // is emitted and the frame advances by the duration (a blank advances at least one frame).
    private List<int> GreedyTokenAndDuration(Tensor<T> encoded, int maxSymbolsPerFrame)
    {
        var labels = new List<int>();
        int frames = encoded.Shape[0], d = encoded.Shape[1], v = _o.VocabSize, symbols = 0;
        for (int t = 0; t < frames;)
        {
            var g = Predict(labels);
            var last = _engine.TensorSlice(g, new[] { labels.Count, 0 }, new[] { 1, _o.PredictionDim });
            var logProbs = Joint(_engine.TensorSlice(encoded, new[] { t, 0 }, new[] { 1, d }), last);
            int token = 0, duration = 0;
            for (int k = 1; k < v; k++)
                if (NumOps.GreaterThan(logProbs[0, 0, k], logProbs[0, 0, token])) token = k;
            for (int i = 1; i < _o.Durations.Length; i++)
                if (NumOps.GreaterThan(logProbs[0, 0, v + i], logProbs[0, 0, v + duration])) duration = i;
            int step = _o.Durations[duration];
            if (token != 0) { labels.Add(token); symbols++; }
            if (token == 0 || step > 0 || symbols >= maxSymbolsPerFrame) { t += Math.Max(step, 1); symbols = 0; }
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

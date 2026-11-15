using AiDotNet.ActivationFunctions;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.ThreeD;

/// <summary>
/// The asymmetric U-Net of LGM: a convolutional U-Net over the stacked input views whose attention blocks attend
/// across all views at once, and whose up path is one level shorter than its down path.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This is <c>core/unet.py</c> of the official LGM code (Tang et al., ECCV 2024, arXiv 2402.05054), block for block:
/// </para>
/// <list type="bullet">
/// <item><description><b>conv_in</b>: 3x3 convolution from the 9 input channels (RGB plus the 6-channel Plucker ray
/// embedding) to the first down width.</description></item>
/// <item><description><b>Down block</b> i: <c>LayersPerBlock</c> residual blocks (GroupNorm, SiLU, 3x3 conv, GroupNorm,
/// SiLU, 3x3 conv, plus a 1x1 shortcut when the width changes, the sum scaled by <c>SkipScale</c>), each followed by
/// multi-view attention where <c>DownAttention[i]</c> is set, then a stride-2 3x3 convolution on every block but the
/// last. Every residual output and every downsample output is kept as a skip.</description></item>
/// <item><description><b>Middle block</b>: residual, then (attention, residual).</description></item>
/// <item><description><b>Up block</b> i: <c>LayersPerBlock + 1</c> residual blocks, each fed the running tensor
/// concatenated with the next skip popped off the stack, each followed by attention where <c>UpAttention[i]</c> is set,
/// then nearest x2 upsampling and a 3x3 convolution on every block but the last.</description></item>
/// <item><description><b>Head</b>: GroupNorm, SiLU, 3x3 convolution to the output channels.</description></item>
/// </list>
/// <para>
/// <b>Multi-view attention</b> (<c>MVAttention</c>) normalizes the features, flattens the views of one object into a
/// single sequence of <c>views x height x width</c> tokens, runs multi-head self-attention over that sequence (query,
/// key and value projections without bias, output projection with bias, as <c>MemEffAttention</c>), and adds the
/// residual, scaled by <c>SkipScale</c>. Because every token sees every view, this is where the four views exchange
/// information.
/// </para>
/// <para>
/// The blocks are the library's existing layers (<see cref="DiffusionResBlock{T}"/> with no time embedding,
/// <see cref="DiffusionAttentionLayer{T}"/>, <see cref="GroupNormalizationLayer{T}"/>,
/// <see cref="ConvolutionalLayer{T}"/>); this type composes them in the LGM order and owns their parameter surface.
/// The weights start from the library's initializers; load the released LGM checkpoint through
/// <see cref="SetParameters"/> (in <see cref="Layers"/> order) for meaningful reconstructions.
/// </para>
/// <para><b>For Beginners:</b> the network looks at four photos of the same object at once and, for every pixel of a
/// half-resolution grid in each photo, predicts one small 3D blob (a Gaussian).</para>
/// </remarks>
public class LGMAsymmetricUNet<T> : IParameterSource<T>
{
    private sealed class DownBlock
    {
        public DownBlock(List<DiffusionResBlock<T>> resBlocks, List<AttentionBlock?> attentions,
            ConvolutionalLayer<T>? downsample)
        {
            ResBlocks = resBlocks;
            Attentions = attentions;
            Downsample = downsample;
        }

        public List<DiffusionResBlock<T>> ResBlocks { get; }
        public List<AttentionBlock?> Attentions { get; }
        public ConvolutionalLayer<T>? Downsample { get; }
    }

    private sealed class UpBlock
    {
        public UpBlock(List<DiffusionResBlock<T>> resBlocks, List<AttentionBlock?> attentions,
            ConvolutionalLayer<T>? upsample)
        {
            ResBlocks = resBlocks;
            Attentions = attentions;
            Upsample = upsample;
        }

        public List<DiffusionResBlock<T>> ResBlocks { get; }
        public List<AttentionBlock?> Attentions { get; }
        public ConvolutionalLayer<T>? Upsample { get; }
    }

    private sealed class AttentionBlock
    {
        public AttentionBlock(GroupNormalizationLayer<T> norm, DiffusionAttentionLayer<T> attention)
        {
            Norm = norm;
            Attention = attention;
        }

        public GroupNormalizationLayer<T> Norm { get; }
        public DiffusionAttentionLayer<T> Attention { get; }
    }

    private readonly ConvolutionalLayer<T> _convIn;
    private readonly List<DownBlock> _downBlocks = new List<DownBlock>();
    private readonly DiffusionResBlock<T> _midFirst;
    private readonly List<AttentionBlock?> _midAttentions = new List<AttentionBlock?>();
    private readonly List<DiffusionResBlock<T>> _midResBlocks = new List<DiffusionResBlock<T>>();
    private readonly List<UpBlock> _upBlocks = new List<UpBlock>();
    private readonly GroupNormalizationLayer<T> _normOut;
    private readonly ConvolutionalLayer<T> _convOut;
    private readonly List<ILayer<T>> _layers = new List<ILayer<T>>();
    private readonly double _skipScale;
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly int _numViews;

    /// <summary>
    /// Builds the U-Net from an LGM configuration.
    /// </summary>
    /// <param name="config">The configuration; its channel and attention lists set the architecture.</param>
    /// <param name="inputChannels">Input channels per view (9: RGB plus the Plucker embedding).</param>
    /// <param name="outputChannels">Output channels per pixel (14 Gaussian parameters).</param>
    /// <param name="seed">Seed for the initial weights, or null for a non-reproducible draw.</param>
    public LGMAsymmetricUNet(LGMConfig config, int inputChannels = 9, int outputChannels = 14, int? seed = null)
    {
        if (config is null) throw new ArgumentNullException(nameof(config));
        Validate(config, inputChannels, outputChannels);

        Config = config;
        InputChannels = inputChannels;
        OutputChannels = outputChannels;
        _numViews = config.NumInputViews;
        _skipScale = config.SkipScale;

        // One deterministic per-layer seed sequence for everything built below, as NoisePredictorBase does for the
        // diffusion U-Nets: a seed reproduces the weights, no seed keeps the non-reproducible default.
        LayerInitializationSeedScope.ResetForModelConstruction(seed);

        var down = config.DownChannels;
        var up = config.UpChannels;
        int layersPerBlock = config.LayersPerBlock;

        _convIn = Register(Conv(inputChannels, down[0], kernel: 3, stride: 1, padding: 1));

        // The skip stack holds channel counts in push order, so each up block's concatenation width is read off the
        // stack exactly as the forward pass pops it.
        var skipChannels = new List<int> { down[0] };
        int resolution = config.InputSize;
        int channels = down[0];
        for (int i = 0; i < down.Length; i++)
        {
            int cin = channels;
            int cout = down[i];
            var resBlocks = new List<DiffusionResBlock<T>>();
            var attentions = new List<AttentionBlock?>();
            for (int l = 0; l < layersPerBlock; l++)
            {
                resBlocks.Add(Register(Res(l == 0 ? cin : cout, cout, resolution, config)));
                attentions.Add(config.DownAttention[i] ? Attention(cout, config) : null);
                skipChannels.Add(cout);
            }

            ConvolutionalLayer<T>? downsample = null;
            if (i != down.Length - 1)
            {
                downsample = Register(Conv(cout, cout, kernel: 3, stride: 2, padding: 1));
                skipChannels.Add(cout);
                resolution = (resolution + 1) / 2;
            }

            _downBlocks.Add(new DownBlock(resBlocks, attentions, downsample));
            channels = cout;
        }

        _midFirst = Register(Res(channels, channels, resolution, config));
        _midAttentions.Add(config.MidAttention ? Attention(channels, config) : null);
        _midResBlocks.Add(Register(Res(channels, channels, resolution, config)));

        channels = up[0];
        if (channels != down[down.Length - 1])
            throw new ArgumentException(
                $"UpChannels[0] ({up[0]}) must equal the last down width ({down[down.Length - 1]}).", nameof(config));
        for (int i = 0; i < up.Length; i++)
        {
            int cin = channels;
            int cout = up[i];
            var resBlocks = new List<DiffusionResBlock<T>>();
            var attentions = new List<AttentionBlock?>();
            for (int l = 0; l < layersPerBlock + 1; l++)
            {
                int skip = skipChannels[skipChannels.Count - 1];
                skipChannels.RemoveAt(skipChannels.Count - 1);
                resBlocks.Add(Register(Res((l == 0 ? cin : cout) + skip, cout, resolution, config)));
                attentions.Add(config.UpAttention[i] ? Attention(cout, config) : null);
            }

            ConvolutionalLayer<T>? upsample = null;
            if (i != up.Length - 1)
            {
                upsample = Register(Conv(cout, cout, kernel: 3, stride: 1, padding: 1));
                resolution *= 2;
            }

            _upBlocks.Add(new UpBlock(resBlocks, attentions, upsample));
            channels = cout;
        }

        _normOut = Register(new GroupNormalizationLayer<T>(Groups(channels, config.NormGroups), channels, config.NormEpsilon));
        _convOut = Register(Conv(channels, outputChannels, kernel: 3, stride: 1, padding: 1));
        OutputSize = resolution;

        // Inference is what this network is used for here; the training branch of every layer records a tape.
        SetTrainingMode(false);
    }

    /// <summary>The configuration the U-Net was built from.</summary>
    public LGMConfig Config { get; }

    /// <summary>Input channels per view.</summary>
    public int InputChannels { get; }

    /// <summary>Output channels per pixel.</summary>
    public int OutputChannels { get; }

    /// <summary>Side length of the output maps for an input of <see cref="LGMConfig.InputSize"/> (128 by default).</summary>
    public int OutputSize { get; }

    /// <summary>Every layer, in parameter (checkpoint) order: construction order, which is forward order.</summary>
    public IReadOnlyList<ILayer<T>> Layers => _layers;

    /// <inheritdoc />
    public long ParameterCount
    {
        get
        {
            long total = 0;
            foreach (var layer in _layers) total += layer.ParameterCount;
            return total;
        }
    }

    /// <inheritdoc />
    public Vector<T> GetParameters()
    {
        var parts = new List<Vector<T>>(_layers.Count);
        foreach (var layer in _layers) parts.Add(layer.GetParameters());
        return Vector<T>.Concatenate(parts.ToArray());
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        long expected = ParameterCount;
        if (parameters.Length != expected)
            throw new ArgumentException($"Expected {expected} parameters, got {parameters.Length}.", nameof(parameters));
        int offset = 0;
        foreach (var layer in _layers)
        {
            int count = checked((int)layer.ParameterCount);
            layer.SetParameters(parameters.GetSubVector(offset, count));
            offset += count;
        }
    }

    /// <summary>Switches every layer between training and inference execution.</summary>
    public void SetTrainingMode(bool isTraining)
    {
        foreach (var layer in _layers) layer.SetTrainingMode(isTraining);
    }

    /// <summary>
    /// Runs the U-Net.
    /// </summary>
    /// <param name="input">Views of one or more objects, <c>[objects x views, InputChannels, H, W]</c>, the views of
    /// each object consecutive.</param>
    /// <returns>The output maps, <c>[objects x views, OutputChannels, H', W']</c> with <c>H' = OutputSize</c> for an
    /// input of <see cref="LGMConfig.InputSize"/>.</returns>
    public Tensor<T> Forward(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 4)
            throw new ArgumentException("The LGM U-Net expects [objects x views, channels, height, width].", nameof(input));
        if (input.Shape[1] != InputChannels)
            throw new ArgumentException($"Expected {InputChannels} input channels, got {input.Shape[1]}.", nameof(input));
        if (input.Shape[0] % _numViews != 0)
            throw new ArgumentException(
                $"The batch ({input.Shape[0]}) is not a whole number of {_numViews}-view objects.", nameof(input));

        var engine = AiDotNetEngine.Current;
        var x = _convIn.Forward(input);
        var skips = new List<Tensor<T>> { x };

        foreach (var block in _downBlocks)
        {
            for (int l = 0; l < block.ResBlocks.Count; l++)
            {
                x = Residual(engine, block.ResBlocks[l], x);
                var attention = block.Attentions[l];
                if (attention is not null) x = Attend(engine, attention, x);
                skips.Add(x);
            }

            if (block.Downsample is not null)
            {
                x = block.Downsample.Forward(x);
                skips.Add(x);
            }
        }

        x = Residual(engine, _midFirst, x);
        for (int l = 0; l < _midResBlocks.Count; l++)
        {
            var attention = _midAttentions[l];
            if (attention is not null) x = Attend(engine, attention, x);
            x = Residual(engine, _midResBlocks[l], x);
        }

        foreach (var block in _upBlocks)
        {
            for (int l = 0; l < block.ResBlocks.Count; l++)
            {
                var skip = skips[skips.Count - 1];
                skips.RemoveAt(skips.Count - 1);
                x = engine.TensorConcatenate(new[] { x, skip }, 1);
                x = Residual(engine, block.ResBlocks[l], x);
                var attention = block.Attentions[l];
                if (attention is not null) x = Attend(engine, attention, x);
            }

            if (block.Upsample is not null)
            {
                x = engine.Upsample(x, 2, 2);
                x = block.Upsample.Forward(x);
            }
        }

        x = _normOut.Forward(x);
        x = engine.TensorSiLU(x);
        return _convOut.Forward(x);
    }

    /// <summary>A residual block's output scaled by <c>SkipScale</c>: <c>(h + shortcut(x)) * skip_scale</c>.</summary>
    private Tensor<T> Residual(IEngine engine, DiffusionResBlock<T> block, Tensor<T> x)
        => engine.TensorMultiplyScalar(block.Forward(x), NumOps.FromDouble(_skipScale));

    /// <summary>
    /// Multi-view self-attention: every pixel of every view of one object attends to every pixel of every view.
    /// </summary>
    private Tensor<T> Attend(IEngine engine, AttentionBlock block, Tensor<T> x)
    {
        int batchViews = x.Shape[0];
        int channels = x.Shape[1];
        int height = x.Shape[2];
        int width = x.Shape[3];
        int objects = batchViews / _numViews;

        var h = block.Norm.Forward(x);
        // [B*V, C, H, W] -> [B, V, H, W, C] -> [B, V*H*W, C]
        h = engine.Reshape(h, new[] { objects, _numViews, channels, height, width });
        h = engine.TensorPermute(h, new[] { 0, 1, 3, 4, 2 }).Contiguous();
        h = engine.Reshape(h, new[] { objects, _numViews * height * width, channels });
        h = block.Attention.Forward(h, h);
        // [B, V*H*W, C] -> [B, V, C, H, W] -> [B*V, C, H, W]
        h = engine.Reshape(h, new[] { objects, _numViews, height, width, channels });
        h = engine.TensorPermute(h, new[] { 0, 1, 4, 2, 3 }).Contiguous();
        h = engine.Reshape(h, new[] { batchViews, channels, height, width });
        return engine.TensorMultiplyScalar(engine.TensorAdd(h, x), NumOps.FromDouble(_skipScale));
    }

    private TLayer Register<TLayer>(TLayer layer) where TLayer : ILayer<T>
    {
        _layers.Add(layer);
        return layer;
    }

    private AttentionBlock Attention(int channels, LGMConfig config)
    {
        if (channels % config.AttentionHeads != 0)
            throw new ArgumentException(
                $"AttentionHeads ({config.AttentionHeads}) must divide every attended width ({channels}).",
                nameof(config));
        var norm = Register(new GroupNormalizationLayer<T>(Groups(channels, config.NormGroups), channels, config.NormEpsilon));
        var attention = Register(new DiffusionAttentionLayer<T>(channels, channels, config.AttentionHeads));
        return new AttentionBlock(norm, attention);
    }

    private static DiffusionResBlock<T> Res(int inChannels, int outChannels, int resolution, LGMConfig config)
        => new DiffusionResBlock<T>(inChannels, outChannels, resolution, timeEmbedDim: 0,
            numGroups: config.NormGroups, epsilon: config.NormEpsilon);

    private static ConvolutionalLayer<T> Conv(int inChannels, int outChannels, int kernel, int stride, int padding)
        => ConvolutionalLayer<T>.WithInputDepth(
            inputDepth: inChannels,
            outputDepth: outChannels,
            kernelSize: kernel,
            stride: stride,
            padding: padding,
            activationFunction: new IdentityActivation<T>(),
            initializationStrategy: InitializationStrategies<T>.Lazy,
            biasMode: AiDotNet.Enums.BiasMode.Always);

    /// <summary>The largest group count no greater than <paramref name="target"/> that divides the channels, the same
    /// rule <see cref="DiffusionResBlock{T}"/> applies; with the paper's widths it is always 32.</summary>
    internal static int Groups(int channels, int target)
    {
        for (int groups = System.Math.Min(channels, target); groups >= 1; groups--)
            if (channels % groups == 0) return groups;
        return 1;
    }

    private static void Validate(LGMConfig config, int inputChannels, int outputChannels)
    {
        if (inputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inputChannels));
        if (outputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(outputChannels));
        if (config.NumInputViews <= 0)
            throw new ArgumentException("NumInputViews must be positive.", nameof(config));
        if (config.InputSize <= 0)
            throw new ArgumentException("InputSize must be positive.", nameof(config));
        if (config.DownChannels is null || config.DownChannels.Length == 0)
            throw new ArgumentException("DownChannels must list at least one width.", nameof(config));
        if (config.UpChannels is null || config.UpChannels.Length == 0)
            throw new ArgumentException("UpChannels must list at least one width.", nameof(config));
        if (config.DownAttention is null || config.DownAttention.Length != config.DownChannels.Length)
            throw new ArgumentException("DownAttention needs one entry per down block.", nameof(config));
        if (config.UpAttention is null || config.UpAttention.Length != config.UpChannels.Length)
            throw new ArgumentException("UpAttention needs one entry per up block.", nameof(config));
        if (config.UpChannels.Length > config.DownChannels.Length)
            throw new ArgumentException(
                "The up path cannot be longer than the down path: it would run out of skip connections.",
                nameof(config));
        if (config.LayersPerBlock <= 0)
            throw new ArgumentException("LayersPerBlock must be positive.", nameof(config));
        if (config.AttentionHeads <= 0)
            throw new ArgumentException("AttentionHeads must be positive.", nameof(config));
        if (config.NormGroups <= 0)
            throw new ArgumentException("NormGroups must be positive.", nameof(config));
        foreach (int width in config.DownChannels)
            if (width <= 0) throw new ArgumentException("Every down width must be positive.", nameof(config));
        foreach (int width in config.UpChannels)
            if (width <= 0) throw new ArgumentException("Every up width must be positive.", nameof(config));
    }
}

using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// The MIA-VSR network (Zhou et al., CVPR 2024, arXiv:2401.06312): shallow features, SPyNet patch
/// alignment, bidirectional second-order propagation through inter-and-intra-frame attention blocks
/// with adaptive masked processing, and a pixel-shuffle reconstruction head.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>Propagation.</b> Branch m (backward, forward, backward, forward) refines every frame from the
/// previous branch's output: X_{m+1}^t = FPM(X_m^t, X_{m+1}^{t-1}, X_{m+1}^{t-2}), where t-1 and t-2
/// are the two frames this branch already processed in its own direction. A branch is a residual group
/// of attention blocks closed by a 3x3 convolution.
/// </para>
/// <para>
/// <b>Inter-and-intra-frame attention (IIAB).</b> Within each window, queries come from the current
/// frame only, while keys and values come from [X^{t-1}; X^{t-2}; X^t] through projections shared between
/// inter and intra tokens, plus a learnable relative position bias. LayerNorm and an FFN follow,
/// Swin-style.
/// </para>
/// <para>
/// <b>Patch alignment</b> follows PSRT, which MIA-VSR builds on. SPyNet estimates flow between
/// low-resolution frames, and each window of a neighbouring frame is moved by its mean motion vector,
/// rounded to whole pixels, with no interpolation. The second-order neighbour composes two such moves:
/// the second is measured where the first one landed. The moves are discrete, so no gradient reaches
/// the flow estimator, which is a fixed, pretrained component as in PSRT; the flow is therefore computed
/// outside the gradient tape.
/// </para>
/// <para>
/// <b>Adaptive masked processing.</b> A 1x1 projection of |LN(X^t) − LN(X^{t-1})|, the same block's
/// input on consecutive frames, scores each position. In training a two-way Gumbel-softmax at
/// temperature τ samples a binary keep-mask, with straight-through gradients to the scorer. The
/// difference of the two Gumbel draws is a logistic variable, so the keep probability is
/// σ((score + logistic) / τ); its soft values feed the mask-sparsity loss. At inference a position is
/// kept when its score is positive. Both after the attention and after the FFN, a dropped position
/// copies that block's previous-frame result. At inference this is real sparse computation: only kept
/// positions run the query projection, attention, output projection and FFN, while keys and values are
/// always computed, as in the paper's cost analysis (Fig. 3).
/// </para>
/// </remarks>
internal sealed class MiaVsrNetwork<T>
{
    private readonly int _inputChannels;
    private readonly int _channels;
    private readonly int _window;
    private readonly int _heads;
    private readonly int _ffnRatio;
    private readonly int _branches;
    private readonly int _blocksPerBranch;
    private readonly int _scale;
    private readonly int _reconChannels;
    private readonly double _temperature;
    private readonly Tensor<T> _relativeIndex;

    private readonly List<ILayer<T>> _layers = new();
    private IReadOnlyList<ILayer<T>>? _bindSource;

    // Roles over _layers, assigned by Build.
    private ConvolutionalLayer<T> _shallow;
    // The pretrained flow estimator, outside the layer graph: the optimizer never sees it, so it stays
    // frozen, and it runs only under NoGradScope. Null means no motion estimate (windows align in place).
    private readonly SpyNetLayer<T>? _flow;
    private readonly List<AttentionBlock[]> _branchBlocks = new();
    private readonly List<ConvolutionalLayer<T>> _branchConvs = new();
    private readonly List<(ConvolutionalLayer<T> Conv, PixelShuffleLayer<T> Shuffle)> _upsample = new();
    private ConvolutionalLayer<T> _hrConv;
    private ConvolutionalLayer<T> _lastConv;

    public MiaVsrNetwork(MIAVSROptions options, int inputChannels, SpyNetLayer<T>? flowEstimator)
    {
        if (options is null) throw new ArgumentNullException(nameof(options));
        if (inputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inputChannels));
        if (options.NumFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(options), "NumFeatures must be positive.");
        if (options.NumHeads <= 0 || options.NumFeatures % options.NumHeads != 0)
            throw new ArgumentException(
                $"NumFeatures ({options.NumFeatures}) must be divisible by NumHeads ({options.NumHeads}).", nameof(options));
        if (options.WindowSize <= 0) throw new ArgumentOutOfRangeException(nameof(options), "WindowSize must be positive.");
        if (options.FeedForwardRatio <= 0) throw new ArgumentOutOfRangeException(nameof(options), "FeedForwardRatio must be positive.");
        if (options.NumPropagationBranches <= 0) throw new ArgumentOutOfRangeException(nameof(options), "NumPropagationBranches must be positive.");
        if (options.BlocksPerBranch <= 0) throw new ArgumentOutOfRangeException(nameof(options), "BlocksPerBranch must be positive.");
        if (options.ScaleFactor <= 0 || (options.ScaleFactor & (options.ScaleFactor - 1)) != 0)
            throw new ArgumentOutOfRangeException(nameof(options), $"ScaleFactor must be a positive power of two; got {options.ScaleFactor}.");
        if (options.ReconstructionChannels <= 0) throw new ArgumentOutOfRangeException(nameof(options), "ReconstructionChannels must be positive.");
        if (double.IsNaN(options.GumbelTemperature) || options.GumbelTemperature <= 0) throw new ArgumentOutOfRangeException(nameof(options), "GumbelTemperature must be positive.");
        // Zero disables the sparsity term; anything negative or non-finite would silently drop or invert it.
        if (double.IsNaN(options.MaskLossWeight) || double.IsInfinity(options.MaskLossWeight) || options.MaskLossWeight < 0)
            throw new ArgumentOutOfRangeException(nameof(options), "MaskLossWeight must be finite and non-negative.");

        _inputChannels = inputChannels;
        _channels = options.NumFeatures;
        _window = options.WindowSize;
        _heads = options.NumHeads;
        _ffnRatio = options.FeedForwardRatio;
        _branches = options.NumPropagationBranches;
        _blocksPerBranch = options.BlocksPerBranch;
        _scale = options.ScaleFactor;
        _flow = flowEstimator;
        _reconChannels = options.ReconstructionChannels;
        _temperature = options.GumbelTemperature;
        _relativeIndex = BuildRelativeIndex(_window);

        Build();
    }

    /// <summary>Every layer of the network, in a fixed order, for the owning model to publish.</summary>
    public IReadOnlyList<ILayer<T>> Layers => _layers;

    /// <summary>
    /// When true, inference computes every position and blends by the mask instead of computing only the
    /// kept ones. The two are mathematically identical; this exists so tests can prove the sparse path is.
    /// </summary>
    internal bool DenseInference { get; set; }

    /// <summary>The smallest frame side the flow estimator accepts; smaller frames are padded for it.</summary>
    public int MinimumFlowSize => _flow is null ? 1 : 1 << (_flow.NumLevels - 1);

    /// <summary>
    /// Points every role at the model's current layers, position for position, after a deserialize or
    /// eager clone replaced the instances; does nothing while the graph is unchanged.
    /// </summary>
    public void BindTo(IReadOnlyList<ILayer<T>> layers)
    {
        if (layers is null) throw new ArgumentNullException(nameof(layers));
        bool same = layers.Count == _layers.Count;
        for (int i = 0; same && i < layers.Count; i++)
        {
            same = ReferenceEquals(layers[i], _layers[i]);
        }

        if (same) return;
        if (layers.Count != _layers.Count)
            throw new InvalidOperationException(
                $"The layer graph has {layers.Count} layers but the MIA-VSR layout has {_layers.Count}.");

        // Rebinding replaces every role at once; a layer of the wrong type part-way through must leave the
        // network exactly as it was, not half-bound.
        var previous = (Layers: _layers.ToList(), Blocks: _branchBlocks.ToList(), Convs: _branchConvs.ToList(),
            Upsample: _upsample.ToList(), Shallow: _shallow, Hr: _hrConv, Last: _lastConv);
        _bindSource = layers;
        try
        {
            Build();
        }
        catch
        {
            _layers.Clear(); _layers.AddRange(previous.Layers);
            _branchBlocks.Clear(); _branchBlocks.AddRange(previous.Blocks);
            _branchConvs.Clear(); _branchConvs.AddRange(previous.Convs);
            _upsample.Clear(); _upsample.AddRange(previous.Upsample);
            _shallow = previous.Shallow;
            _hrConv = previous.Hr;
            _lastConv = previous.Last;
            throw;
        }
        finally
        {
            _bindSource = null;
        }
    }

    /// <summary>
    /// Super-resolves a clip <c>[B, T, C, H, W]</c> (or one frame <c>[B, C, H, W]</c>) by the scale factor.
    /// </summary>
    /// <param name="input">The low-resolution clip.</param>
    /// <param name="training">True for the training forward: Gumbel-sampled masks, dense computation.</param>
    /// <param name="random">The noise source for the training-time Gumbel samples.</param>
    /// <param name="maskLoss">In training, the mean of every soft keep-mask as a tape scalar; otherwise null.</param>
    public Tensor<T> Forward(Tensor<T> input, bool training, Random random, out Tensor<T>? maskLoss)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (random is null) throw new ArgumentNullException(nameof(random));
        bool singleFrame = input.Shape.Length == 4;
        if (!singleFrame && input.Shape.Length != 5)
            throw new ArgumentException(
                $"MIA-VSR expects a clip [B, T, C, H, W] or a frame [B, C, H, W]; got rank {input.Shape.Length}.", nameof(input));

        var engine = AiDotNetEngine.Current;
        var clip = singleFrame
            ? engine.Reshape(input, new[] { input.Shape[0], 1, input.Shape[1], input.Shape[2], input.Shape[3] })
            : input;
        int batch = clip.Shape[0], frames = clip.Shape[1], channels = clip.Shape[2];
        int height = clip.Shape[3], width = clip.Shape[4];
        if (channels != _inputChannels)
            throw new ArgumentException($"MIA-VSR was built for {_inputChannels} input channels; got {channels}.", nameof(input));

        // Frames are zero-padded to whole windows; the output is cropped back.
        int paddedH = (height + _window - 1) / _window * _window;
        int paddedW = (width + _window - 1) / _window * _window;
        var lowRes = PadFrames(engine, clip, paddedH, paddedW);
        var geometry = new WindowGeometry(batch, paddedH, paddedW, _window);
        var shallow = new Tensor<T>[frames];
        for (int t = 0; t < frames; t++)
        {
            shallow[t] = ToTokens(engine, _shallow.Forward(lowRes[t]));
        }

        var alignment = new PatchAlignment(engine, _flow, lowRes, geometry, MinimumFlowSize, _channels);
        var context = new ForwardContext(training, random, _temperature, DenseInference);
        var current = shallow;
        for (int m = 0; m < _branches; m++)
        {
            current = Propagate(engine, m, current, alignment, geometry, context);
        }

        var hr = new Tensor<T>[frames];
        for (int t = 0; t < frames; t++)
        {
            var frame = Reconstruct(engine, engine.TensorAdd(current[t], shallow[t]), lowRes[t], geometry);
            if (paddedH != height || paddedW != width)
            {
                frame = engine.TensorNarrow(engine.TensorNarrow(frame, dim: 2, start: 0, length: height * _scale),
                    dim: 3, start: 0, length: width * _scale);
            }

            hr[t] = engine.Reshape(frame, new[] { batch, 1, channels, height * _scale, width * _scale });
        }

        maskLoss = context.MaskLoss(engine);
        var result = frames == 1 ? hr[0] : engine.TensorConcatenate(hr, axis: 1);
        return singleFrame
            ? engine.Reshape(result, new[] { batch, channels, height * _scale, width * _scale })
            : result;
    }

    /// <summary>Splits a clip into frames, zero-padded to whole windows: <c>[B, C, H', W']</c> each.</summary>
    private static Tensor<T>[] PadFrames(IEngine engine, Tensor<T> clip, int paddedH, int paddedW)
    {
        int batch = clip.Shape[0], frames = clip.Shape[1], channels = clip.Shape[2];
        int height = clip.Shape[3], width = clip.Shape[4];
        var zero = MathHelper.GetNumericOperations<T>().Zero;
        var lowRes = new Tensor<T>[frames];
        for (int t = 0; t < frames; t++)
        {
            var frame = engine.Reshape(engine.TensorNarrow(clip, dim: 1, start: t, length: 1),
                new[] { batch, channels, height, width });
            lowRes[t] = paddedH == height && paddedW == width
                ? frame
                : engine.Pad(frame, 0, paddedH - height, 0, paddedW - width, zero);
        }

        return lowRes;
    }

    /// <summary>
    /// One feature propagation module: branch <paramref name="branch"/> refines every frame in its
    /// direction (even branches backward, odd forward) from its own outputs at the two frames it has
    /// already processed, as a residual group of attention blocks closed by a 3x3 convolution.
    /// </summary>
    private Tensor<T>[] Propagate(IEngine engine, int branch, Tensor<T>[] inputs, PatchAlignment alignment,
        WindowGeometry geometry, ForwardContext context)
    {
        int frames = inputs.Length;
        bool backward = branch % 2 == 0;
        int step = backward ? 1 : -1;
        var states = new BlockState[_blocksPerBranch];
        for (int n = 0; n < states.Length; n++) states[n] = new BlockState();

        var outputs = new Tensor<T>?[frames];
        var completed = new Tensor<T>[frames];
        for (int k = 0; k < frames; k++)
        {
            int t = backward ? frames - 1 - k : k;
            var first = alignment.Neighbour(outputs, t, step, order: 1);
            var second = alignment.Neighbour(outputs, t, step, order: 2);

            var x = inputs[t];
            for (int n = 0; n < _blocksPerBranch; n++)
            {
                x = _branchBlocks[branch][n].Forward(engine, x, first, second, _relativeIndex, geometry, states[n], context);
            }

            completed[t] = engine.TensorAdd(inputs[t], ToTokens(engine, _branchConvs[branch].Forward(ToImage(engine, x, geometry))));
            outputs[t] = completed[t];
        }

        return completed;
    }

    /// <summary>
    /// The reconstruction head on one frame's propagated features: pixel-shuffle upsampling with
    /// LeakyReLU(0.1), a 3x3 convolution to the output channels, plus the bilinearly upsampled input.
    /// </summary>
    private Tensor<T> Reconstruct(IEngine engine, Tensor<T> tokens, Tensor<T> lowRes, WindowGeometry geometry)
    {
        var leaky = MathHelper.GetNumericOperations<T>().FromDouble(0.1);
        var feature = ToImage(engine, tokens, geometry);
        foreach (var (conv, shuffle) in _upsample)
        {
            feature = engine.LeakyReLU(shuffle.Forward(conv.Forward(feature)), leaky);
        }

        feature = engine.LeakyReLU(_hrConv.Forward(feature), leaky);
        var residual = engine.Interpolate(lowRes, new[] { geometry.Height * _scale, geometry.Width * _scale }, InterpolateMode.Bilinear);
        return engine.TensorAdd(_lastConv.Forward(feature), residual);
    }

    private static Tensor<T> BuildRelativeIndex(int window)
    {
        int w2 = window * window, span = 2 * window - 1;
        var numOps = MathHelper.GetNumericOperations<T>();
        var index = new Tensor<T>(new[] { w2 * w2 });
        for (int q = 0; q < w2; q++)
        {
            for (int k = 0; k < w2; k++)
            {
                int dy = q / window - k / window + window - 1;
                int dx = q % window - k % window + window - 1;
                index[q * w2 + k] = numOps.FromDouble(dy * span + dx);
            }
        }

        return index;
    }

    private static Tensor<T> ToTokens(IEngine engine, Tensor<T> image)
    {
        int b = image.Shape[0], c = image.Shape[1], h = image.Shape[2], w = image.Shape[3];
        return engine.Reshape(engine.TensorPermute(image, new[] { 0, 2, 3, 1 }), new[] { b * h * w, c });
    }

    private Tensor<T> ToImage(IEngine engine, Tensor<T> tokens, WindowGeometry g)
    {
        var grid = engine.Reshape(tokens, new[] { g.Batch, g.Height, g.Width, _channels });
        return engine.Reshape(engine.TensorPermute(grid, new[] { 0, 3, 1, 2 }), new[] { g.Batch, _channels, g.Height, g.Width });
    }

    [System.Diagnostics.CodeAnalysis.MemberNotNull(
        nameof(_shallow), nameof(_hrConv), nameof(_lastConv))]
    private void Build()
    {
        _layers.Clear();
        _branchBlocks.Clear();
        _branchConvs.Clear();
        _upsample.Clear();

        IActivationFunction<T> identity = new IdentityActivation<T>();
        _shallow = Add(new ConvolutionalLayer<T>(_channels, 3, 1, 1));

        int span = 2 * _window - 1;
        for (int m = 0; m < _branches; m++)
        {
            var blocks = new AttentionBlock[_blocksPerBranch];
            for (int n = 0; n < blocks.Length; n++)
            {
                blocks[n] = new AttentionBlock(
                    _channels, _heads, _window,
                    norm: Add(new LayerNormalizationLayer<T>(_channels)),
                    query: Add(new DenseLayer<T>(_channels, identity)),
                    key: Add(new DenseLayer<T>(_channels, identity)),
                    value: Add(new DenseLayer<T>(_channels, identity)),
                    bias: Add(new EmbeddingLayer<T>(span * span, _heads)),
                    projection: Add(new DenseLayer<T>(_channels, identity)),
                    maskNorm: Add(new LayerNormalizationLayer<T>(_channels)),
                    maskScore: Add(new DenseLayer<T>(1, identity)),
                    ffnNorm: Add(new LayerNormalizationLayer<T>(_channels)),
                    ffnUp: Add(new DenseLayer<T>(_channels * _ffnRatio, (IActivationFunction<T>)new GELUActivation<T>())),
                    ffnDown: Add(new DenseLayer<T>(_channels, identity)));
            }

            _branchBlocks.Add(blocks);
            _branchConvs.Add(Add(new ConvolutionalLayer<T>(_channels, 3, 1, 1)));
        }

        for (int s = 1; s < _scale; s *= 2)
        {
            _upsample.Add((Add(new ConvolutionalLayer<T>(_reconChannels * 4, 3, 1, 1)), Add(new PixelShuffleLayer<T>(2))));
        }

        _hrConv = Add(new ConvolutionalLayer<T>(_reconChannels, 3, 1, 1));
        _lastConv = Add(new ConvolutionalLayer<T>(_inputChannels, 3, 1, 1));
    }

    private TLayer Add<TLayer>(TLayer layer) where TLayer : class, ILayer<T>
    {
        if (_bindSource is not null)
        {
            int index = _layers.Count;
            if (index >= _bindSource.Count || _bindSource[index] is not TLayer existing)
            {
                throw new InvalidOperationException(
                    $"The layer graph does not match the MIA-VSR layout at position {index}: expected " +
                    $"{typeof(TLayer).Name}, found {(index < _bindSource.Count ? _bindSource[index].GetType().Name : "the end")}.");
            }

            layer = existing;
        }

        _layers.Add(layer);
        return layer;
    }

    /// <summary>Row bookkeeping for window partitioning of a padded <c>[B, H, W]</c> token grid.</summary>
    private sealed class WindowGeometry
    {
        public WindowGeometry(int batch, int height, int width, int window)
        {
            Batch = batch;
            Height = height;
            Width = width;
            Window = window;
            WindowsAcross = width / window;
            WindowsPerImage = height / window * WindowsAcross;
            Tokens = batch * height * width;

            WindowOfRow = new int[Tokens];
            PositionOfRow = new int[Tokens];
            var partition = new int[Tokens];
            var merge = new int[Tokens];
            int w2 = window * window;
            for (int b = 0; b < batch; b++)
            {
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int row = (b * height + y) * width + x;
                        int windowIndex = b * WindowsPerImage + y / window * WindowsAcross + x / window;
                        int position = y % window * window + x % window;
                        int ordered = windowIndex * w2 + position;
                        WindowOfRow[row] = windowIndex;
                        PositionOfRow[row] = position;
                        partition[ordered] = row;
                        merge[row] = ordered;
                    }
                }
            }

            Partition = new Tensor<int>(partition, new[] { Tokens });
            Merge = new Tensor<int>(merge, new[] { Tokens });
        }

        public int Batch { get; }
        public int Height { get; }
        public int Width { get; }
        public int Window { get; }
        public int WindowsAcross { get; }
        public int WindowsPerImage { get; }
        public int Windows => Batch * WindowsPerImage;
        public int Tokens { get; }
        public int[] WindowOfRow { get; }
        public int[] PositionOfRow { get; }

        /// <summary>Window-ordered position to row-major row.</summary>
        public Tensor<int> Partition { get; }

        /// <summary>Row-major row to window-ordered position (the inverse of <see cref="Partition"/>).</summary>
        public Tensor<int> Merge { get; }

        /// <summary>
        /// Window-ordered rows of a neighbouring frame, each window moved by its own whole-pixel offset
        /// and clamped to the frame.
        /// </summary>
        public Tensor<int> ShiftedPartition(int[] offsetY, int[] offsetX)
        {
            var rows = new int[Tokens];
            int w2 = Window * Window;
            for (int windowIndex = 0; windowIndex < Windows; windowIndex++)
            {
                int b = windowIndex / WindowsPerImage, local = windowIndex % WindowsPerImage;
                int top = local / WindowsAcross * Window, left = local % WindowsAcross * Window;
                for (int p = 0; p < w2; p++)
                {
                    int y = Clamp(top + p / Window + offsetY[windowIndex], Height);
                    int x = Clamp(left + p % Window + offsetX[windowIndex], Width);
                    rows[windowIndex * w2 + p] = (b * Height + y) * Width + x;
                }
            }

            return new Tensor<int>(rows, new[] { Tokens });
        }

        private static int Clamp(int value, int size) => value < 0 ? 0 : value >= size ? size - 1 : value;
    }

    /// <summary>
    /// PSRT patch alignment: per-window whole-pixel motion between frames, from SPyNet flow computed
    /// once per frame pair outside the gradient tape.
    /// </summary>
    private sealed class PatchAlignment
    {
        private readonly IEngine _engine;
        private readonly SpyNetLayer<T>? _flow;
        private readonly Tensor<T>[] _frames;
        private readonly WindowGeometry _geometry;
        private readonly int _minimumFlowSize;
        private readonly int _channels;
        private readonly Dictionary<(int From, int To), (double[] Dx, double[] Dy)> _fields = new();
        private Tensor<T>? _zeros;

        public PatchAlignment(IEngine engine, SpyNetLayer<T>? flow, Tensor<T>[] frames, WindowGeometry geometry,
            int minimumFlowSize, int channels)
        {
            _engine = engine;
            _flow = flow;
            _frames = frames;
            _geometry = geometry;
            _minimumFlowSize = minimumFlowSize;
            _channels = channels;
        }

        /// <summary>
        /// The aligned window-ordered tokens of the frame <paramref name="order"/> steps behind
        /// <paramref name="t"/> in this branch's direction, or zeros when that frame is outside the clip.
        /// </summary>
        public Tensor<T> Neighbour(Tensor<T>?[] outputs, int t, int step, int order)
        {
            int source = t + order * step;
            var neighbour = source >= 0 && source < outputs.Length ? outputs[source] : null;
            if (neighbour is null)
            {
                return _zeros ??= new Tensor<T>(new[] { _geometry.Tokens, _channels });
            }

            int windows = _geometry.Windows;
            var offsetY = new int[windows];
            var offsetX = new int[windows];
            int via = t;
            for (int hop = 1; _flow is not null && hop <= order; hop++)
            {
                int next = t + hop * step;
                var (dx, dy) = Field(via, next);
                for (int w = 0; w < windows; w++)
                {
                    // Mean motion over the window region where the previous hops left it.
                    var (mx, my) = WindowMean(dx, dy, w, offsetY[w], offsetX[w]);
                    offsetX[w] += (int)Math.Round(mx);
                    offsetY[w] += (int)Math.Round(my);
                }

                via = next;
            }

            return _engine.TensorIndexSelect(neighbour, _geometry.ShiftedPartition(offsetY, offsetX), 0);
        }

        private (double Dx, double Dy) WindowMean(double[] dx, double[] dy, int windowIndex, int shiftY, int shiftX)
        {
            var g = _geometry;
            int b = windowIndex / g.WindowsPerImage, local = windowIndex % g.WindowsPerImage;
            int top = local / g.WindowsAcross * g.Window + shiftY, left = local % g.WindowsAcross * g.Window + shiftX;
            double sumX = 0, sumY = 0;
            int count = 0;
            for (int i = 0; i < g.Window; i++)
            {
                int y = top + i;
                if (y < 0 || y >= g.Height) continue;
                for (int j = 0; j < g.Window; j++)
                {
                    int x = left + j;
                    if (x < 0 || x >= g.Width) continue;
                    int at = (b * g.Height + y) * g.Width + x;
                    sumX += dx[at];
                    sumY += dy[at];
                    count++;
                }
            }

            return count == 0 ? (0, 0) : (sumX / count, sumY / count);
        }

        private (double[] Dx, double[] Dy) Field(int from, int to)
        {
            if (_fields.TryGetValue((from, to), out var cached)) return cached;

            var g = _geometry;
            int flowH = Math.Max(g.Height, _minimumFlowSize), flowW = Math.Max(g.Width, _minimumFlowSize);
            var numOps = MathHelper.GetNumericOperations<T>();
            var estimator = _flow ?? throw new InvalidOperationException("No flow estimator is configured.");
            Tensor<T> flow;
            using (new NoGradScope<T>())
            {
                // Flow(from, to) gives, for each pixel of `from`, its displacement into `to`.
                flow = estimator.EstimateFlow(Fit(_frames[from], flowH, flowW, numOps.Zero), Fit(_frames[to], flowH, flowW, numOps.Zero));
            }

            var dx = new double[g.Tokens];
            var dy = new double[g.Tokens];
            int plane = flowH * flowW;
            for (int b = 0; b < g.Batch; b++)
            {
                for (int y = 0; y < g.Height; y++)
                {
                    for (int x = 0; x < g.Width; x++)
                    {
                        int at = (b * g.Height + y) * g.Width + x;
                        int source = b * 2 * plane + y * flowW + x;
                        double fx = numOps.ToDouble(flow[source]);
                        double fy = numOps.ToDouble(flow[source + plane]);
                        dx[at] = double.IsNaN(fx) || double.IsInfinity(fx) ? 0 : fx;
                        dy[at] = double.IsNaN(fy) || double.IsInfinity(fy) ? 0 : fy;
                    }
                }
            }

            var field = (dx, dy);
            _fields[(from, to)] = field;
            return field;
        }

        private Tensor<T> Fit(Tensor<T> frame, int height, int width, T zero)
        {
            int padH = height - frame.Shape[2], padW = width - frame.Shape[3];
            return padH == 0 && padW == 0 ? frame : _engine.Pad(frame, 0, padH, 0, padW, zero);
        }
    }

    /// <summary>Per-forward training state: the Gumbel noise source and every soft mask's mean.</summary>
    private sealed class ForwardContext
    {
        private readonly List<Tensor<T>> _maskMeans = new();

        public ForwardContext(bool training, Random random, double temperature, bool denseInference)
        {
            Training = training;
            DenseInference = denseInference;
            Random = random;
            Temperature = temperature;
        }

        public bool Training { get; }
        public bool DenseInference { get; }
        public Random Random { get; }
        public double Temperature { get; }

        public void AddMaskMean(Tensor<T> mean) => _maskMeans.Add(mean);

        /// <summary>L_mask: the mean over blocks and frames of each soft mask's mean, or null.</summary>
        public Tensor<T>? MaskLoss(IEngine engine)
        {
            if (!Training || _maskMeans.Count == 0) return null;
            var total = _maskMeans[0];
            for (int i = 1; i < _maskMeans.Count; i++) total = engine.TensorAdd(total, _maskMeans[i]);
            return engine.TensorMultiplyScalar(total, MathHelper.GetNumericOperations<T>().FromDouble(1.0 / _maskMeans.Count));
        }
    }

    /// <summary>One block's previous-frame tensors within the current branch.</summary>
    private sealed class BlockState
    {
        public Tensor<T>? Input { get; set; }
        public Tensor<T>? AfterAttention { get; set; }
        public Tensor<T>? Output { get; set; }
    }

    /// <summary>The inter-and-intra-frame attention block with adaptive masked processing.</summary>
    private sealed class AttentionBlock
    {
        private readonly int _channels;
        private readonly int _heads;
        private readonly int _headDim;
        private readonly int _window;
        private readonly LayerNormalizationLayer<T> _norm;
        private readonly DenseLayer<T> _query;
        private readonly DenseLayer<T> _key;
        private readonly DenseLayer<T> _value;
        private readonly EmbeddingLayer<T> _bias;
        private readonly DenseLayer<T> _projection;
        private readonly LayerNormalizationLayer<T> _maskNorm;
        private readonly DenseLayer<T> _maskScore;
        private readonly LayerNormalizationLayer<T> _ffnNorm;
        private readonly DenseLayer<T> _ffnUp;
        private readonly DenseLayer<T> _ffnDown;

        public AttentionBlock(int channels, int heads, int window,
            LayerNormalizationLayer<T> norm, DenseLayer<T> query, DenseLayer<T> key, DenseLayer<T> value,
            EmbeddingLayer<T> bias, DenseLayer<T> projection, LayerNormalizationLayer<T> maskNorm,
            DenseLayer<T> maskScore, LayerNormalizationLayer<T> ffnNorm, DenseLayer<T> ffnUp, DenseLayer<T> ffnDown)
        {
            _channels = channels;
            _heads = heads;
            _headDim = channels / heads;
            _window = window;
            _norm = norm;
            _query = query;
            _key = key;
            _value = value;
            _bias = bias;
            _projection = projection;
            _maskNorm = maskNorm;
            _maskScore = maskScore;
            _ffnNorm = ffnNorm;
            _ffnUp = ffnUp;
            _ffnDown = ffnDown;
        }

        /// <summary>
        /// Runs the block on one frame: inter-and-intra-frame attention and the FFN, each followed by the keep-mask
        /// blend with this block's previous-frame result once a previous frame exists.
        /// </summary>
        /// <param name="engine">The engine.</param>
        /// <param name="x">The block input, row-major tokens <c>[B·H·W, C]</c>.</param>
        /// <param name="first">The aligned t-1 neighbour, window-ordered <c>[B·H·W, C]</c>.</param>
        /// <param name="second">The aligned t-2 neighbour, window-ordered.</param>
        /// <param name="relativeIndex">The relative-position index of every (query, key) pair in a window.</param>
        /// <param name="g">The window geometry.</param>
        /// <param name="state">This block's previous-frame tensors in the current branch.</param>
        /// <param name="context">Training mode, the noise source and the mask-loss accumulator.</param>
        public Tensor<T> Forward(IEngine engine, Tensor<T> x, Tensor<T> first, Tensor<T> second,
            Tensor<T> relativeIndex, WindowGeometry g, BlockState state, ForwardContext context)
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            int w2 = _window * _window, windows = g.Windows;

            var normed = _norm.Forward(x);
            var normedWindows = engine.TensorIndexSelect(normed, g.Partition, 0);

            // Keys and values over [t-1; t-2; t] per window, with projections shared by all three.
            var keySource = engine.TensorConcatenate(new[]
            {
                engine.Reshape(_norm.Forward(first), new[] { windows, w2, _channels }),
                engine.Reshape(_norm.Forward(second), new[] { windows, w2, _channels }),
                engine.Reshape(normedWindows, new[] { windows, w2, _channels })
            }, axis: 1);
            var keySourceRows = engine.Reshape(keySource, new[] { windows * 3 * w2, _channels });
            var keys = SplitHeads(engine, _key.Forward(keySourceRows), windows, 3 * w2);
            var values = SplitHeads(engine, _value.Forward(keySourceRows), windows, 3 * w2);

            // Relative position bias [heads, W², 3W²]: the same spatial offsets for each key frame.
            var table = engine.Reshape(_bias.Forward(relativeIndex), new[] { w2, w2, _heads });
            var spatialBias = engine.TensorPermute(table, new[] { 2, 0, 1 });
            var bias = engine.TensorConcatenate(new[] { spatialBias, spatialBias, spatialBias }, axis: 2);

            // The keep-mask exists from the second frame of the branch on.
            Tensor<T>? keep = null;
            var previousInput = state.Input;
            if (previousInput is not null)
            {
                var change = engine.TensorAbs(engine.TensorSubtract(_maskNorm.Forward(x), _maskNorm.Forward(previousInput)));
                var score = _maskScore.Forward(change);
                keep = context.Training ? SampleTrainingMask(engine, score, context) : engine.TensorGreaterThan(score, numOps.Zero);
            }

            var previousAttention = state.AfterAttention;
            var previousOutput = state.Output;
            Tensor<T> afterAttention, output;
            if (keep is not null && !context.Training && !context.DenseInference && previousAttention is not null && previousOutput is not null)
            {
                (afterAttention, output) = SparseInference(engine, x, normed, keys, values, bias, keep, g,
                    previousAttention, previousOutput);
            }
            else
            {
                afterAttention = engine.TensorAdd(x, DenseAttention(engine, normedWindows, keys, values, bias, g));
                if (keep is not null && previousAttention is not null) afterAttention = Blend(engine, afterAttention, previousAttention, keep);
                output = engine.TensorAdd(afterAttention, FeedForward(afterAttention));
                if (keep is not null && previousOutput is not null) output = Blend(engine, output, previousOutput, keep);
            }

            state.Input = x;
            state.AfterAttention = afterAttention;
            state.Output = output;
            return output;
        }

        private Tensor<T> FeedForward(Tensor<T> x) => _ffnDown.Forward(_ffnUp.Forward(_ffnNorm.Forward(x)));

        private static Tensor<T> SampleTrainingMask(IEngine engine, Tensor<T> score, ForwardContext context)
        {
            // Two-way Gumbel-softmax: g1 − g2 is logistic, so the soft keep value is σ((s + L)/τ).
            var numOps = MathHelper.GetNumericOperations<T>();
            var noise = new Tensor<T>(score._shape);
            for (int i = 0; i < noise.Length; i++)
            {
                double u = context.Random.NextDouble();
                u = Math.Min(Math.Max(u, 1e-10), 1 - 1e-10);
                noise[i] = numOps.FromDouble(Math.Log(u) - Math.Log(1 - u));
            }

            var soft = engine.Sigmoid(engine.TensorMultiplyScalar(engine.TensorAdd(score, noise), numOps.FromDouble(1.0 / context.Temperature)));
            context.AddMaskMean(engine.Reshape(engine.ReduceMean(soft, new[] { 0, 1 }, keepDims: false), new[] { 1 }));

            // Straight-through: the forward uses the binary mask, the backward the soft one.
            var hard = engine.TensorGreaterThan(engine.StopGradient(soft), numOps.FromDouble(0.5));
            return engine.TensorAdd(hard, engine.TensorSubtract(soft, engine.StopGradient(soft)));
        }

        private static Tensor<T> Blend(IEngine engine, Tensor<T> fresh, Tensor<T> previous, Tensor<T> keep)
        {
            // keep ⊙ fresh + (1 − keep) ⊙ previous.
            var gate = engine.TensorBroadcastTo(keep, fresh._shape);
            return engine.TensorAdd(previous, engine.TensorMultiply(gate, engine.TensorSubtract(fresh, previous)));
        }

        private Tensor<T> SplitHeads(IEngine engine, Tensor<T> rows, int groups, int tokens)
        {
            var split = engine.Reshape(rows, new[] { groups, tokens, _heads, _headDim });
            return engine.Reshape(engine.TensorPermute(split, new[] { 0, 2, 1, 3 }), new[] { groups * _heads, tokens, _headDim });
        }

        private Tensor<T> DenseAttention(IEngine engine, Tensor<T> normedWindows, Tensor<T> keys, Tensor<T> values,
            Tensor<T> bias, WindowGeometry g)
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            int w2 = _window * _window, windows = g.Windows;
            var queries = SplitHeads(engine, _query.Forward(normedWindows), windows, w2);
            var scores = engine.TensorMultiplyScalar(
                engine.BatchMatMul(queries, engine.TensorPermute(keys, new[] { 0, 2, 1 })),
                numOps.FromDouble(1.0 / Math.Sqrt(_headDim)));
            scores = engine.Reshape(scores, new[] { windows, _heads, w2, 3 * w2 });
            scores = engine.TensorAdd(scores, engine.TensorBroadcastTo(
                engine.Reshape(bias, new[] { 1, _heads, w2, 3 * w2 }), new[] { windows, _heads, w2, 3 * w2 }));
            var probabilities = engine.Reshape(engine.TensorSoftmax(scores, axis: 3), new[] { windows * _heads, w2, 3 * w2 });

            var attended = engine.Reshape(engine.BatchMatMul(probabilities, values), new[] { windows, _heads, w2, _headDim });
            var rows = engine.Reshape(engine.TensorPermute(attended, new[] { 0, 2, 1, 3 }), new[] { windows * w2, _channels });
            return engine.TensorIndexSelect(_projection.Forward(rows), g.Merge, 0);
        }

        private (Tensor<T> AfterAttention, Tensor<T> Output) SparseInference(IEngine engine, Tensor<T> x, Tensor<T> normed,
            Tensor<T> keys, Tensor<T> values, Tensor<T> bias, Tensor<T> keep, WindowGeometry g,
            Tensor<T> previousAttention, Tensor<T> previousOutput)
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            var kept = new List<int>();
            for (int row = 0; row < g.Tokens; row++)
            {
                if (numOps.ToDouble(keep[row]) > 0.5) kept.Add(row);
            }

            if (kept.Count == 0) return (previousAttention, previousOutput);

            int count = kept.Count, w2 = _window * _window;
            var keptRows = new Tensor<int>(kept.ToArray(), new[] { count });
            var keptWindows = new int[count];
            var keptPositions = new int[count];
            var map = new int[g.Tokens];
            for (int row = 0; row < g.Tokens; row++) map[row] = row;
            for (int i = 0; i < count; i++)
            {
                keptWindows[i] = g.WindowOfRow[kept[i]];
                keptPositions[i] = g.PositionOfRow[kept[i]];
                map[kept[i]] = g.Tokens + i;
            }

            // Only kept positions run the query, attention, output projection and FFN.
            var queries = SplitHeads(engine, _query.Forward(engine.TensorIndexSelect(normed, keptRows, 0)), count, 1);
            var windowIndex = new Tensor<int>(keptWindows, new[] { count });
            // A kept position attends to its own window's keys and values. Selecting windows along axis 0
            // is a row gather on the [windows, heads·3W²·d] view, which is the rank the engine selects on.
            var keptKeys = engine.Reshape(engine.TensorIndexSelect(
                engine.Reshape(keys, new[] { g.Windows, _heads * 3 * w2 * _headDim }), windowIndex, 0),
                new[] { count * _heads, 3 * w2, _headDim });
            var keptValues = engine.Reshape(engine.TensorIndexSelect(
                engine.Reshape(values, new[] { g.Windows, _heads * 3 * w2 * _headDim }), windowIndex, 0),
                new[] { count * _heads, 3 * w2, _headDim });

            var scores = engine.TensorMultiplyScalar(
                engine.BatchMatMul(queries, engine.TensorPermute(keptKeys, new[] { 0, 2, 1 })),
                numOps.FromDouble(1.0 / Math.Sqrt(_headDim)));
            var biasByPosition = engine.Reshape(engine.TensorPermute(bias, new[] { 1, 0, 2 }), new[] { w2, _heads * 3 * w2 });
            var biasRows = engine.Reshape(
                engine.TensorIndexSelect(biasByPosition, new Tensor<int>(keptPositions, new[] { count }), 0),
                new[] { count, _heads, 3 * w2 });
            scores = engine.TensorAdd(engine.Reshape(scores, new[] { count, _heads, 3 * w2 }), biasRows);
            var probabilities = engine.Reshape(engine.TensorSoftmax(scores, axis: 2), new[] { count * _heads, 1, 3 * w2 });
            var attended = engine.Reshape(engine.BatchMatMul(probabilities, keptValues), new[] { count, _channels });

            var keptAttention = engine.TensorAdd(engine.TensorIndexSelect(x, keptRows, 0), _projection.Forward(attended));
            var keptOutput = engine.TensorAdd(keptAttention, FeedForward(keptAttention));

            var mapTensor = new Tensor<int>(map, new[] { g.Tokens });
            var afterAttention = engine.TensorIndexSelect(engine.TensorConcatenate(new[] { previousAttention, keptAttention }, axis: 0), mapTensor, 0);
            var output = engine.TensorIndexSelect(engine.TensorConcatenate(new[] { previousOutput, keptOutput }, axis: 0), mapTensor, 0);
            return (afterAttention, output);
        }
    }
}

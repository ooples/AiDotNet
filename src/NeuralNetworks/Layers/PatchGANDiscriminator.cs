using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

// File-level, deliberately: two Tensors namespaces in the project's global usings also define a
// TensorLayout, so [TensorLayout(...)] only binds when this import shadows them from a nearer scope.
using AiDotNet.Attributes;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// PatchGAN: the convolutional "Markovian" discriminator introduced by pix2pix (Isola et al., 2017).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> when a generator is trained with an adversarial loss, a second network — the
/// discriminator — has to say whether an image looks real. A PatchGAN does not output one verdict for
/// the whole image. It slides over the image and outputs a GRID of verdicts, one per square patch, then
/// the loss averages them. Two things follow: the discriminator is much smaller than a whole-image
/// classifier (it only ever looks at one patch at a time), and it works on any input size, because a
/// bigger image simply produces a bigger grid.
/// </para>
/// <para>
/// <b>Paper (section 3.2.2, "Markovian discriminator (PatchGAN)").</b> The discriminator "tries to
/// classify if each N x N patch in an image is real or fake", is "run convolutionally across the image,
/// averaging all responses", and models the image as a Markov random field — pixels further apart than
/// one patch diameter are treated as independent. The L1/L2 reconstruction term is what enforces
/// low-frequency correctness; the PatchGAN's job is only the high frequencies.
/// </para>
/// <para>
/// <b>Architecture (section 6.1.2, verbatim).</b> The 70x70 discriminator is <c>C64-C128-C256-C512</c>;
/// "after the last layer, a convolution is applied to map to a 1-dimensional output, followed by a
/// Sigmoid function"; "as an exception to the above notation, BatchNorm is not applied to the first C64
/// layer"; "all ReLUs are leaky, with slope 0.2". <c>Ck</c> is a 4x4 Convolution-BatchNorm-LeakyReLU
/// block with k filters. Other receptive fields "follow the same basic architecture, with depth varied
/// to modify the receptive field size" — hence <see cref="PatchGANReceptiveField"/> selects a depth
/// rather than a different design.
/// </para>
/// <para>
/// <b>Why depth alone sets the patch size.</b> Every stride-2 block doubles the distance an output unit
/// can see. Applying the receptive-field recurrence r_in = r_out * stride + (kernel - stride) backwards
/// through <c>C64(s2) C128(s2) C256(s2) C512(s1)</c> plus the 1-channel output convolution (s1) gives
/// 1 -&gt; 4 -&gt; 7 -&gt; 16 -&gt; 34 -&gt; 70, which is exactly the paper's 70x70. Adding two more stride-2
/// C512 blocks continues 70 -&gt; 142 -&gt; 286, its ImageGAN. This class therefore makes the LAST block
/// stride 1 and every earlier block stride 2, and sizes channels as
/// <c>numFilters * min(2^i, 8)</c> — which yields 64, 128, 256, 512, 512, 512.
/// </para>
/// <para>
/// <b>Usage:</b>
/// <code>
/// // Paper default: 70x70 patches.
/// var d = new PatchGANDiscriminator&lt;float&gt;();
/// var verdictGrid = d.Forward(image);            // [1, Hp, Wp] of per-patch scores
///
/// // Logits instead of probabilities, for a numerically stable with-logits loss.
/// var dLogits = new PatchGANDiscriminator&lt;float&gt;(applySigmoid: false);
/// </code>
/// </para>
/// <para>
/// Every sub-layer is a normal library layer registered via <c>RegisterSubLayer</c>, and the forward
/// pass is composed purely from those layers' own <c>Forward</c> calls. Gradients therefore flow
/// through the autodiff tape with no manual backward pass and no raw buffer writes, which is what lets
/// this be dropped into any model's adversarial objective.
/// </para>
/// </remarks>
// Rank 3 [Channels, Height, Width] -> [1, Hp, Wp], matching the ResolveShapes call in OnFirstForward.
// OutputAxesFor is hand-written because the spatial relation depends on the kernel/stride/padding
// schedule, which is built from constructor arguments.
[TensorLayout(TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class PatchGANDiscriminator<T> : LayerBase<T>, IShapeContract
{
    #region Constants

    /// <summary>
    /// Paper default filter count for the first block (the "64" in C64), section 6.1.2.
    /// </summary>
    public const int DefaultNumFilters = 64;

    /// <summary>
    /// Paper default convolution kernel size. Section 6.1.1 defines Ck as a 4x4 convolution.
    /// </summary>
    public const int DefaultKernelSize = 4;

    /// <summary>
    /// Paper default LeakyReLU slope: "all ReLUs are leaky, with slope 0.2" (section 6.1.2).
    /// </summary>
    public const double DefaultLeakySlope = 0.2;

    /// <summary>
    /// Largest channel multiplier. The paper's deeper variants repeat C512 rather than growing to
    /// C1024/C2048, so the multiplier saturates at 8x <see cref="DefaultNumFilters"/>.
    /// </summary>
    private const int MaxChannelMultiplier = 8;

    #endregion

    #region Fields

    private readonly int _numFilters;
    private readonly int _kernelSize;
    private readonly int _numLayers;
    private readonly bool _applySigmoid;
    private readonly bool _allStrideOne;

    /// <summary>The convolution of each Ck block. The activation is a separate layer so the paper's
    /// Convolution -&gt; BatchNorm -&gt; LeakyReLU order is preserved exactly.</summary>
    private readonly ConvolutionalLayer<T>[] _convBlocks;

    /// <summary>
    /// BatchNorm for each Ck block, or null at index 0 — "BatchNorm is not applied to the first C64
    /// layer" (section 6.1.2).
    /// </summary>
    private readonly BatchNormalizationLayer<T>?[] _norms;

    /// <summary>The LeakyReLU of each Ck block, applied after the block's BatchNorm.</summary>
    private readonly ActivationLayer<T>[] _activations;

    /// <summary>The final convolution mapping to a 1-dimensional (single-channel) output.</summary>
    private readonly ConvolutionalLayer<T> _convOut;

    #endregion

    #region Properties

    /// <summary>
    /// Gets the receptive field, in pixels, of one output unit — the "N" in "N x N patch".
    /// </summary>
    /// <remarks>
    /// Computed from the actual kernel/stride schedule via the recurrence
    /// r_in = r_out * stride + (kernel - stride), so it stays correct for any custom depth or kernel
    /// size rather than reporting the paper's number regardless of configuration.
    /// </remarks>
    public int ReceptiveField
    {
        get
        {
            int r = 1;
            // Walk output -> input: the 1-channel output convolution, then the Ck blocks in reverse.
            r = r * 1 + (_kernelSize - 1);
            for (int i = _numLayers - 1; i >= 0; i--)
            {
                int stride = StrideAt(i);
                r = r * stride + (_kernelSize - stride);
            }

            return r;
        }
    }

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// The channel axis collapses to the single real/fake score per patch, so it is
    /// <see cref="AxisRelation.Fixed"/> at 1. Height and Width both follow the whole convolution
    /// stack.
    /// </para>
    /// <para>
    /// <b>A stack of convolutions IS a single convolution.</b> This looked at first like it needed a
    /// composite relation the vocabulary does not have — one window per block. It does not. Composing
    /// <c>(k1,s1,p1)</c> then <c>(k2,s2,p2)</c> gives exactly <c>k = k1 + (k2-1)*s1</c>,
    /// <c>s = s1*s2</c>, <c>p = p1 + p2*s1</c>, because
    /// <c>floor((floor(a/s1) + b) / s2) == floor((a + b*s1) / (s1*s2))</c> for integer <c>b</c>. The
    /// identity is exact, not an approximation, so folding the schedule left to right yields ONE
    /// <see cref="AxisRelation.Window"/> that reproduces the resolved shape for every input size.
    /// </para>
    /// <para>
    /// The folded kernel is the receptive field: this method's <c>kernel</c> always equals
    /// <see cref="ReceptiveField"/>, which the class computes independently by walking the schedule
    /// backwards. Two derivations, one number - that is the cross-check that this is right.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank != 3) return null;

        int padding = PaddingFor(_kernelSize);

        // Fold the Ck blocks, then the 1-channel output convolution (kernel, stride 1, same padding).
        int kernel = _kernelSize;
        int stride = StrideAt(0);
        int pad = padding;
        for (int i = 1; i <= _numLayers; i++)
        {
            int s = i < _numLayers ? StrideAt(i) : 1;
            kernel += (_kernelSize - 1) * stride;
            pad += padding * stride;
            stride *= s;
        }

        return new[]
        {
            new OutputAxisContract(TensorAxis.Channels, AxisRelation.Fixed(1)),
            new OutputAxisContract(
                TensorAxis.Height, AxisRelation.Window(TensorAxis.Height, kernel, stride, pad)),
            new OutputAxisContract(
                TensorAxis.Width, AxisRelation.Window(TensorAxis.Width, kernel, stride, pad)),
        };
    }

    /// <summary>
    /// Gets the number of Ck blocks preceding the output convolution.
    /// </summary>
    public int NumLayers => _numLayers;

    /// <inheritdoc/>
    /// <remarks>
    /// A discriminator only exists to be trained, and every sub-layer holds trainable weights.
    /// </remarks>
    public override bool SupportsTraining => true;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a PatchGAN discriminator at one of the receptive-field sizes reported in the paper.
    /// </summary>
    /// <param name="receptiveField">Which patch size to build. Defaults to
    /// <see cref="PatchGANReceptiveField.Patch70x70"/>, the paper's default and best-scoring variant.</param>
    /// <param name="numFilters">Filters in the first block. Default 64, per section 6.1.2.</param>
    /// <param name="leakySlope">LeakyReLU slope. Default 0.2, per section 6.1.2.</param>
    /// <param name="applySigmoid">Whether to apply the paper's final Sigmoid. Default true (paper).
    /// Pass false to emit raw logits for a numerically stable with-logits adversarial loss.</param>
    public PatchGANDiscriminator(
        PatchGANReceptiveField receptiveField = PatchGANReceptiveField.Patch70x70,
        int numFilters = DefaultNumFilters,
        double leakySlope = DefaultLeakySlope,
        bool applySigmoid = true)
        : this(
            numLayers: LayerCountFor(receptiveField),
            numFilters: numFilters,
            // "1 x 1 discriminator: C64-C128 (note, in this special case, all convolutions are
            // 1 x 1 spatial filters)" — section 6.1.2.
            kernelSize: receptiveField == PatchGANReceptiveField.Pixel1x1 ? 1 : DefaultKernelSize,
            leakySlope: leakySlope,
            applySigmoid: applySigmoid)
    {
    }

    /// <summary>
    /// Creates a PatchGAN discriminator with an explicit depth, for receptive fields other than the
    /// four the paper tabulates.
    /// </summary>
    /// <param name="numLayers">Number of Ck blocks. The paper's variants correspond to 2 (16x16),
    /// 4 (70x70) and 6 (286x286).</param>
    /// <param name="numFilters">Filters in the first block. Default 64, per section 6.1.2.</param>
    /// <param name="kernelSize">Convolution kernel size. Default 4, per section 6.1.1's Ck definition.
    /// A value of 1 selects the paper's all-1x1 PixelGAN behaviour.</param>
    /// <param name="leakySlope">LeakyReLU slope. Default 0.2, per section 6.1.2.</param>
    /// <param name="applySigmoid">Whether to apply the paper's final Sigmoid. Default true.</param>
    public PatchGANDiscriminator(
        int numLayers,
        int numFilters = DefaultNumFilters,
        int kernelSize = DefaultKernelSize,
        double leakySlope = DefaultLeakySlope,
        bool applySigmoid = true)
        : base([-1, -1, -1], [1, -1, -1])
    {
        if (numLayers <= 0)
            throw new ArgumentOutOfRangeException(
                nameof(numLayers), numLayers, "PatchGAN needs at least one Ck block.");
        if (numFilters <= 0)
            throw new ArgumentOutOfRangeException(
                nameof(numFilters), numFilters, "Filter count must be positive.");
        if (kernelSize <= 0)
            throw new ArgumentOutOfRangeException(
                nameof(kernelSize), kernelSize, "Kernel size must be positive.");

        _numLayers = numLayers;
        _numFilters = numFilters;
        _kernelSize = kernelSize;
        _applySigmoid = applySigmoid;

        // The PixelGAN variant uses 1x1 filters throughout; downsampling there would discard pixels
        // the discriminator is supposed to judge individually, so it also keeps stride 1 everywhere.
        _allStrideOne = kernelSize == 1;

        int padding = PaddingFor(kernelSize);

        _convBlocks = new ConvolutionalLayer<T>[numLayers];
        _norms = new BatchNormalizationLayer<T>?[numLayers];
        _activations = new ActivationLayer<T>[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            // Channels: numFilters * min(2^i, 8) -> 64, 128, 256, 512, 512, 512 ...
            // The convolution itself is linear: Ck is Convolution -> BatchNorm -> LeakyReLU, so the
            // nonlinearity must come AFTER the normalization, not inside the convolution.
            _convBlocks[i] = new ConvolutionalLayer<T>(
                outputDepth: ChannelsAt(i),
                kernelSize: kernelSize,
                stride: StrideAt(i),
                padding: padding,
                activationFunction: null);

            // "BatchNorm is not applied to the first C64 layer."
            _norms[i] = i == 0 ? null : new BatchNormalizationLayer<T>();

            // Each block gets its own activation layer instance rather than sharing one, so training
            // mode and any per-layer state stay independent.
            // LeakyReLU implements both the scalar and vector activation interfaces; it is elementwise,
            // so the scalar overload is the correct one and the cast resolves the ambiguity.
            _activations[i] = new ActivationLayer<T>(
                (IActivationFunction<T>)new LeakyReLUActivation<T>(leakySlope));
        }

        // "After the last layer, a convolution is applied to map to a 1-dimensional output, followed
        // by a Sigmoid function." Stride 1 so the patch grid keeps the resolution the last Ck block
        // produced.
        _convOut = new ConvolutionalLayer<T>(
            outputDepth: 1,
            kernelSize: kernelSize,
            stride: 1,
            padding: padding,
            activationFunction: applySigmoid ? new SigmoidActivation<T>() : null);

        for (int i = 0; i < numLayers; i++)
        {
            RegisterSubLayer(_convBlocks[i]);
            if (_norms[i] is { } norm) RegisterSubLayer(norm);
            RegisterSubLayer(_activations[i]);
        }

        RegisterSubLayer(_convOut);
    }

    #endregion

    #region Shape Resolution

    /// <inheritdoc/>
    /// <remarks>
    /// Drives each sub-layer's lazy resolution down the spatial pyramid so all weights are allocated
    /// before the first forward pass, and records the resulting patch-grid shape.
    /// </remarks>
    protected override void OnFirstForward(Tensor<T> input)
    {
        var s = input._shape;
        int inC, inH, inW;
        if (s.Length == 3) { inC = s[0]; inH = s[1]; inW = s[2]; }
        else if (s.Length == 4) { inC = s[1]; inH = s[2]; inW = s[3]; }
        else
        {
            throw new ArgumentException(
                $"PatchGANDiscriminator requires rank-3 [C,H,W] or rank-4 [B,C,H,W] input; " +
                $"got rank {s.Length}.",
                nameof(input));
        }

        int padding = PaddingFor(_kernelSize);
        int c = inC, h = inH, w = inW;
        for (int i = 0; i < _numLayers; i++)
        {
            _convBlocks[i].ResolveFromShape(new[] { c, h, w });
            _convBlocks[i].SetTrainingMode(IsTrainingMode);

            int stride = StrideAt(i);
            c = ChannelsAt(i);
            h = ConvOut(h, _kernelSize, stride, padding);
            w = ConvOut(w, _kernelSize, stride, padding);

            if (h <= 0 || w <= 0)
            {
                throw new ArgumentException(
                    $"PatchGANDiscriminator with numLayers={_numLayers} and kernelSize={_kernelSize} " +
                    $"collapses the input to a non-positive size at block {i} (from [{inH}, {inW}]). " +
                    "Use a smaller receptive field or a larger input.",
                    nameof(input));
            }

            if (_norms[i] is { } norm)
            {
                norm.ResolveFromShape(new[] { c, h, w });
                norm.SetTrainingMode(IsTrainingMode);
            }

            _activations[i].ResolveFromShape(new[] { c, h, w });
            _activations[i].SetTrainingMode(IsTrainingMode);
        }

        _convOut.ResolveFromShape(new[] { c, h, w });
        _convOut.SetTrainingMode(IsTrainingMode);

        int outH = ConvOut(h, _kernelSize, 1, padding);
        int outW = ConvOut(w, _kernelSize, 1, padding);

        ResolveShapes(new[] { inC, inH, inW }, new[] { 1, outH, outW });
    }

    #endregion

    #region Forward Pass

    /// <inheritdoc/>
    /// <returns>
    /// A single-channel grid of per-patch scores — probabilities when <c>applySigmoid</c> is true,
    /// logits otherwise. Callers average the grid, which is the paper's "averaging all responses".
    /// </returns>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (!IsShapeResolved) OnFirstForward(input);

        var x = input;
        for (int i = 0; i < _numLayers; i++)
        {
            // Ck = Convolution -> BatchNorm -> LeakyReLU, with the BatchNorm absent for block 0.
            x = _convBlocks[i].Forward(x);
            if (_norms[i] is { } norm) x = norm.Forward(x);
            x = _activations[i].Forward(x);
        }

        return _convOut.Forward(x);
    }

    #endregion

    #region Parameter Management

    /// <summary>
    /// Enumerates every sub-layer in a single fixed order: per block the convolution, then its
    /// BatchNorm (when present), then its activation, and finally the output convolution.
    /// </summary>
    /// <remarks>
    /// Every parameter-facing override below is driven from this one enumeration, so the flat
    /// parameter vector, the flat gradient vector and the per-layer walks can never disagree about
    /// ordering — a mismatch would silently apply one layer's weights to another.
    /// </remarks>
    private IEnumerable<ILayer<T>> AllSubLayers()
    {
        for (int i = 0; i < _numLayers; i++)
        {
            yield return _convBlocks[i];
            if (_norms[i] is { } norm) yield return norm;
            yield return _activations[i];
        }

        yield return _convOut;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Concatenates the sub-layers' gradients in the same order as <see cref="GetParameters"/>. The
    /// base implementation returns only this layer's own gradient buffer, which for a composite is an
    /// all-zero vector of the aggregate length — an optimizer consuming that would take no step while
    /// appearing to have gradients.
    /// </remarks>
    public override Vector<T> GetParameterGradients()
    {
        var all = new List<T>();
        foreach (var layer in AllSubLayers())
        {
            var g = layer.GetParameterGradients();
            for (int i = 0; i < g.Length; i++) all.Add(g[i]);
        }

        return new Vector<T>([.. all]);
    }

    /// <inheritdoc/>
    public override void ClearGradients()
    {
        foreach (var layer in AllSubLayers()) layer.ClearGradients();
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        foreach (var layer in AllSubLayers()) layer.UpdateParameters(learningRate);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        foreach (var layer in AllSubLayers()) layer.ResetState();
    }

    #endregion

    #region Helpers

    /// <summary>
    /// Maps a paper receptive field to the number of Ck blocks that produces it.
    /// </summary>
    private static int LayerCountFor(PatchGANReceptiveField field) => field switch
    {
        // "1 x 1 discriminator: C64-C128"
        PatchGANReceptiveField.Pixel1x1 => 2,
        // "16 x 16 discriminator: C64-C128"
        PatchGANReceptiveField.Patch16x16 => 2,
        // "The 70 x 70 discriminator architecture is: C64-C128-C256-C512"
        PatchGANReceptiveField.Patch70x70 => 4,
        // "C64-C128-C256-C512-C512-C512"
        PatchGANReceptiveField.Image286x286 => 6,
        _ => throw new ArgumentOutOfRangeException(
            nameof(field), field, "Unknown PatchGAN receptive field.")
    };

    /// <summary>
    /// Channels for block <paramref name="i"/>: <c>numFilters * min(2^i, 8)</c>.
    /// </summary>
    private int ChannelsAt(int i) => _numFilters * Math.Min(1 << Math.Min(i, 30), MaxChannelMultiplier);

    /// <summary>
    /// Stride for block <paramref name="i"/>: 2 for every block except the last, which uses 1 so the
    /// final blocks refine the patch verdict instead of shrinking the grid further. This schedule is
    /// what makes the depths in <see cref="LayerCountFor"/> land on 16, 70 and 286 exactly.
    /// </summary>
    private int StrideAt(int i) => _allStrideOne || i == _numLayers - 1 ? 1 : 2;

    /// <summary>
    /// "Same"-style padding for an even kernel: 1 for the paper's 4x4 filters, 0 for 1x1.
    /// </summary>
    private static int PaddingFor(int kernelSize) => kernelSize <= 1 ? 0 : (kernelSize - 2) / 2 + 1;

    /// <summary>Standard convolution output extent.</summary>
    private static int ConvOut(int size, int kernel, int stride, int padding)
        => (size + 2 * padding - kernel) / stride + 1;

    #endregion
}

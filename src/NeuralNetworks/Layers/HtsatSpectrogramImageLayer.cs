using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// HTS-AT's <c>reshape_wav2img</c> (Chen et al. 2022): turns a log-mel spectrogram into the square image its Swin
/// Transformer reads, by stretching it to the image's time budget and folding time into stacked frequency bands.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// With <c>specSize = S</c> and <c>freqRatio = r</c>, the target spectrogram is <c>S·r</c> frames by <c>S/r</c> mel
/// bins (HTS-AT: 1024 × 64 for S = 256, r = 4). A shorter input is stretched with bicubic interpolation
/// (<c>align_corners = True</c>, PyTorch's cubic coefficient <c>A = −0.75</c>), first along time and then along
/// frequency, exactly as the reference calls <c>F.interpolate</c>. The <c>[S·r, S/r]</c> map is then cut into
/// <c>r</c> consecutive time chunks of <c>S</c> frames that are stacked along the frequency axis, giving an
/// <c>S × S</c> image whose rows are the <c>r</c> chunks' mel bins and whose columns are time within a chunk.
/// </para>
/// <para>Input: <c>[batch, channels, frames, mels]</c>. Output: <c>[batch, channels, S, S]</c>. Channels are
/// carried through independently, so the four mel views of feature fusion stay four images.</para>
/// <para><b>For Beginners:</b> A 10-second spectrogram is long and thin. Vision transformers want square pictures,
/// so this slices the long strip into four pieces and stacks them on top of each other.</para>
/// </remarks>
[LayerCategory(LayerCategory.Structural)]
[LayerTask(LayerTask.SpatialProcessing)]
[LayerProperty(IsTrainable = false, ChangesShape = true, TestInputShape = "1, 1, 12, 4", TestConstructorArgs = "8, 2")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class HtsatSpectrogramImageLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _specSize;
    private readonly int _freqRatio;

    /// <inheritdoc/>
    public override bool SupportsTraining => false;

    /// <summary>Creates the fold. HTS-AT: <c>specSize = 256</c>, <c>freqRatio = specSize / mel bins = 4</c>.</summary>
    /// <param name="specSize">Side of the square image.</param>
    /// <param name="freqRatio">Number of time chunks stacked along frequency; must divide <paramref name="specSize"/>.</param>
    public HtsatSpectrogramImageLayer([LayerState] int specSize, [LayerState] int freqRatio)
        : base(new[] { -1, -1, -1 }, new[] { -1, specSize, specSize })
    {
        if (specSize <= 0) throw new ArgumentOutOfRangeException(nameof(specSize));
        if (freqRatio <= 0 || specSize % freqRatio != 0)
            throw new ArgumentException($"freqRatio ({freqRatio}) must be a positive divisor of specSize ({specSize}).", nameof(freqRatio));
        _specSize = specSize;
        _freqRatio = freqRatio;
    }

    /// <summary>Frames the spectrogram is stretched to: <c>specSize · freqRatio</c>.</summary>
    public int TargetFrames => _specSize * _freqRatio;

    /// <summary>Mel bins the spectrogram is stretched to: <c>specSize / freqRatio</c>.</summary>
    public int TargetMelBins => _specSize / _freqRatio;

    /// <inheritdoc/>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank == 4
        ? new[]
        {
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Channels, AxisRelation.Same(TensorAxis.Channels)),
            new OutputAxisContract(TensorAxis.Height, AxisRelation.Fixed(_specSize)),
            new OutputAxisContract(TensorAxis.Width, AxisRelation.Fixed(_specSize)),
        }
        : null;

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank != 4)
            throw new ArgumentException(
                $"HtsatSpectrogramImageLayer expects [batch, channels, frames, mels]; got rank {input.Rank}.", nameof(input));
        int batch = input.Shape[0], channels = input.Shape[1], frames = input.Shape[2], mels = input.Shape[3];
        int targetFrames = TargetFrames, targetMels = TargetMelBins;
        if (frames > targetFrames || mels > targetMels)
            throw new ArgumentException(
                $"The spectrogram dimensions ({frames} frames x {mels} mel bins) exceed the {targetFrames} x {targetMels} "
                + "HTS-AT input; shorten the clip or raise specSize.", nameof(input));

        var x = input;
        if (frames < targetFrames)
        {
            // [B, C, T, F] -> [T, B*C*F], interpolate T, and back.
            var timeFirst = Engine.Reshape(Engine.TensorPermute(x, new[] { 2, 0, 1, 3 }), new[] { frames, batch * channels * mels });
            var stretched = Engine.TensorMatMul(BicubicMatrix(targetFrames, frames), timeFirst);
            x = Engine.TensorPermute(Engine.Reshape(stretched, new[] { targetFrames, batch, channels, mels }), new[] { 1, 2, 0, 3 });
            frames = targetFrames;
        }
        if (mels < targetMels)
        {
            var rows = Engine.Reshape(x, new[] { batch * channels * frames, mels });
            var stretched = Engine.TensorMatMul(rows, Engine.TensorTranspose(BicubicMatrix(targetMels, mels)));
            x = Engine.Reshape(stretched, new[] { batch, channels, frames, targetMels });
            mels = targetMels;
        }

        // x.permute(0,1,3,2).reshape(B, C, F, r, T/r).permute(0,1,3,2,4).reshape(B, C, r*F, T/r)
        int chunk = frames / _freqRatio;
        var freqFirst = Engine.TensorPermute(x, new[] { 0, 1, 3, 2 });
        var split = Engine.Reshape(freqFirst, new[] { batch, channels, mels, _freqRatio, chunk });
        var stacked = Engine.TensorPermute(split, new[] { 0, 1, 3, 2, 4 });
        return Engine.Reshape(stacked, new[] { batch, channels, _freqRatio * mels, chunk });
    }

    /// <summary>
    /// The <c>[outputs, inputs]</c> matrix of PyTorch's 1-D bicubic interpolation with <c>align_corners = True</c>:
    /// source coordinate <c>i·(in−1)/(out−1)</c>, Keys' cubic convolution with <c>A = −0.75</c>, indices clamped to
    /// the border.
    /// </summary>
    internal static Tensor<T> BicubicMatrix(int outputs, int inputs)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        double scale = outputs > 1 ? (inputs - 1) / (double)(outputs - 1) : 0.0;
        const double a = -0.75;
        T aT = ops.FromDouble(a);
        T nearLinear = ops.FromDouble(a + 2), nearConstant = ops.FromDouble(-(a + 3));
        T farFive = ops.FromDouble(-5 * a), farEight = ops.FromDouble(8 * a), farFour = ops.FromDouble(-4 * a);
        T lastInput = ops.FromDouble(inputs - 1);

        // Keys' cubic convolution: Near(x) = ((a+2)x - (a+3))x^2 + 1 for |x| <= 1,
        // Far(x) = ((ax - 5a)x + 8a)x - 4a for 1 < |x| < 2.
        Tensor<T> Near(Tensor<T> x) => engine.TensorAddScalar(
            engine.TensorMultiply(engine.TensorMultiply(
                engine.TensorAddScalar(engine.TensorMultiplyScalar(x, nearLinear), nearConstant), x), x),
            ops.One);
        Tensor<T> Far(Tensor<T> x) => engine.TensorAddScalar(engine.TensorMultiply(
            engine.TensorAddScalar(engine.TensorMultiply(
                engine.TensorAddScalar(engine.TensorMultiplyScalar(x, aT), farFive), x), farEight), x), farFour);

        // Source coordinate i * scale per output row, its integer part and fractional offset.
        var real = engine.TensorMultiplyScalar(engine.TensorArange<T>(0, outputs, 1), ops.FromDouble(scale));
        var index = engine.TensorFloor(real);
        var t = engine.TensorSubtract(real, index);
        var oneMinusT = engine.ScalarMinusTensor(ops.One, t);

        // Tap k sits at source clamp(index - 1 + k, 0, inputs - 1); taps that clamp onto the same
        // border column accumulate, as the reference's "weights[i, source] += w[k]" does.
        Tensor<T> Tap(T offset, Tensor<T> weight)
        {
            var source = engine.TensorClamp(engine.TensorAddScalar(index, offset), ops.Zero, lastInput).Cast<int>();
            return engine.TensorMultiply(engine.TensorOneHot<T>(source, inputs), weight.Reshape(new[] { outputs, 1 }));
        }

        var matrix = Tap(ops.FromDouble(-1), Far(engine.TensorAddScalar(t, ops.One)));
        matrix = engine.TensorAdd(matrix, Tap(ops.Zero, Near(t)));
        matrix = engine.TensorAdd(matrix, Tap(ops.One, Near(oneMinusT)));
        matrix = engine.TensorAdd(matrix, Tap(ops.FromDouble(2), Far(engine.TensorAddScalar(oneMinusT, ops.One))));
        return matrix;
    }

    /// <inheritdoc/>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["SpecSize"] = _specSize.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["FreqRatio"] = _freqRatio.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    /// <inheritdoc/>
    public override void ResetState() { }
}

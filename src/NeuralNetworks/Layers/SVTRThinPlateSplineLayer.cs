using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// SVTR's trainable STN_ON rectifier: a six-block localization CNN predicts control
/// points and a thin-plate-spline grid resamples the source image to 32x100.
/// </summary>
/// <typeparam name="T">The numeric type used for image values and trainable control points.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> The localization network predicts how curved or tilted text should
/// be straightened, then a smooth thin-plate-spline warp resamples it into the fixed canvas that
/// the recognizer expects.</para>
/// <para><b>Reference:</b> Du et al., "SVTR: Scene Text Recognition with a Single Visual Model",
/// IJCAI 2022 (STN_ON), and Jaderberg et al., "Spatial Transformer Networks", NeurIPS 2015.</para>
/// </remarks>
[LayerCategory(LayerCategory.Other)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, Cost = ComputeCost.High,
    TestInputShape = "1, 3, 32, 100", TestConstructorArgs = "3, 32, 64, 32, 100, 20")]
[GenerateDoubleTestScaffold]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Output)]
public sealed partial class SVTRThinPlateSplineLayer<T> : LayerBase<T>, IShapeContract
{
    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank != 4) return null;
        var channels = new OutputAxisContract(
            TensorAxis.Channels, AxisRelation.Same(TensorAxis.Channels));
        var height = new OutputAxisContract(
            TensorAxis.Height, AxisRelation.Fixed(_outputHeight));
        var width = new OutputAxisContract(
            TensorAxis.Width, AxisRelation.Fixed(_outputWidth));
        return
        [
            new(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            channels, height, width
        ];
    }

    private readonly int _inputChannels;
    private readonly int _localizationHeight;
    private readonly int _localizationWidth;
    private readonly int _outputHeight;
    private readonly int _outputWidth;
    private readonly int _controlPointCount;
    private readonly double _marginX;
    private readonly double _marginY;
    private readonly double _localizationMargin;
    private readonly int _localizationFeatureSize;

    // The released localization head damps features before predicting control-point offsets,
    // keeping the early warp close to its identity initialization.
    private const double LocalizationScale = 0.1;

    private readonly List<ILayer<T>> _localizationLayers = [];
    private readonly DenseLayer<T> _featureProjection;

    // NOT readonly, and the class is partial: both are required for [TrainableParameter] to do
    // anything at all. The generator emits its restore surface INTO the declaring class, so a
    // non-partial class gets no generated file and the attributes are inert -- which is what had
    // happened here. The generated SetTrainableParameters then has to rebind these two fields, which
    // readonly forbids outside the constructor. They are still only assigned in the constructor and
    // by that generated restore.
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _controlWeights;

    [TrainableParameter(Role = PersistentTensorRole.Biases)]
    private Tensor<T> _controlBias;

    private readonly Tensor<T> _inverseKernel;
    private readonly Tensor<T> _targetCoordinateRepresentation;

    private IEnumerable<ILayer<T>> ParameterLayers =>
        _localizationLayers.Append(_featureProjection);

    public int ControlPointCount => _controlPointCount;
    public int LocalizationHeight => _localizationHeight;
    public int LocalizationWidth => _localizationWidth;
    public int OutputHeight => _outputHeight;
    public int OutputWidth => _outputWidth;
    public double MarginX => _marginX;
    public double MarginY => _marginY;

    /// <summary>
    /// Inset used for the localization network's identity bias, separately from the output grid's
    /// <see cref="MarginX"/> / <see cref="MarginY"/>. Internal: this layer is an implementation
    /// detail of SVTR rather than a supported public surface, and callers configure the model
    /// through its options rather than by reaching into the layer.
    /// </summary>
    /// <remarks>
    /// The released SVTR rectifier does NOT reuse one margin for both point sets. Its localization
    /// network seeds the SOURCE control points it regresses at a 0.01 inset, while the TARGET grid
    /// the sampler warps onto is built at the configured 0.05 margins. Collapsing the two makes the
    /// initial transform the identity by construction and removes the small source/target offset the
    /// published initialization starts from.
    /// </remarks>
    internal double LocalizationMargin => _localizationMargin;
    /// <inheritdoc />
    public override bool SupportsTraining => true;
    public SVTRThinPlateSplineLayer(
        int inputChannels = 3,
        int localizationHeight = 32,
        int localizationWidth = 64,
        int outputHeight = 32,
        int outputWidth = 100,
        int controlPointCount = 20,
        double marginX = 0.05,
        double marginY = 0.05,
        double localizationMargin = 0.01)
        : base([inputChannels, -1, -1], [inputChannels, outputHeight, outputWidth])
    {
        if (inputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inputChannels));
        if (localizationHeight <= 0 || localizationWidth <= 0 || outputHeight <= 1 || outputWidth <= 1)
            throw new ArgumentOutOfRangeException(nameof(outputHeight));
        if (controlPointCount < 4 || controlPointCount % 2 != 0)
            throw new ArgumentException("TPS requires an even control-point count of at least four.", nameof(controlPointCount));
        if (double.IsNaN(marginX) || double.IsInfinity(marginX)
            || double.IsNaN(marginY) || double.IsInfinity(marginY)
            || marginX <= 0 || marginX >= 0.5
            || marginY <= 0 || marginY >= 0.5)
        {
            // Same reason the localization margin below is checked this way: NaN fails every
            // ordered comparison, so a bare range test lets it reach BuildControlPoints and the TPS
            // matrices fill with NaN instead of failing here.
            throw new ArgumentOutOfRangeException(nameof(marginX));
        }
        if (double.IsNaN(localizationMargin)
            || double.IsInfinity(localizationMargin)
            || localizationMargin <= 0
            || localizationMargin >= 0.5)
        {
            // NaN fails every ordered comparison, so a bare range check lets it through and the
            // control-point grid, the TPS kernel and its inverse all fill with NaN instead of
            // failing here.
            throw new ArgumentOutOfRangeException(nameof(localizationMargin));
        }

        _inputChannels = inputChannels;
        _localizationHeight = localizationHeight;
        _localizationWidth = localizationWidth;
        _outputHeight = outputHeight;
        _outputWidth = outputWidth;
        _controlPointCount = controlPointCount;
        _marginX = marginX;
        _marginY = marginY;
        _localizationMargin = localizationMargin;

        AddConvBlock(32, pool: true);
        AddConvBlock(64, pool: true);
        AddConvBlock(128, pool: true);
        AddConvBlock(256, pool: true);
        AddConvBlock(256, pool: true);
        AddConvBlock(256, pool: false);
        int pooledHeight = localizationHeight / 32;
        int pooledWidth = localizationWidth / 32;
        if (pooledHeight < 1 || pooledWidth < 1)
            throw new ArgumentOutOfRangeException(
                nameof(localizationHeight),
                "The localization input must be at least 32x32 so five pooling stages leave a non-empty grid.");
        _localizationFeatureSize = checked(256 * pooledHeight * pooledWidth);
        _featureProjection = new DenseLayer<T>(512, new ReLUActivation<T>() as IActivationFunction<T>);
        RegisterSubLayer(_featureProjection);

        _controlWeights = new Tensor<T>([512, controlPointCount * 2]);
        _controlBias = new Tensor<T>([controlPointCount * 2]);
        // Zero weights are intentional: the bias alone reproduces the SOURCE control points, so the
        // localization network starts by predicting exactly the grid it was seeded with. That grid
        // sits at localizationMargin while the sampler's target grid sits at marginX / marginY, so
        // the initial transform is NOT the identity -- it preserves the small source-to-target
        // offset the released initialization starts from.
        InitializeIdentityControlBias(_controlBias, controlPointCount, localizationMargin, localizationMargin);
        RegisterTrainableParameter(_controlWeights, PersistentTensorRole.Weights);
        AppendTrainableParameter(_controlBias, PersistentTensorRole.Biases);

        var targetControlPoints = BuildControlPoints(controlPointCount, marginX, marginY);
        _inverseKernel = Invert(BuildTpsKernel(targetControlPoints)).Cast<T>();
        _targetCoordinateRepresentation =
            BuildTargetCoordinateRepresentation(outputHeight, outputWidth, targetControlPoints).Cast<T>();
        RegisterBuffer(_inverseKernel, "inverse_kernel", PersistentTensorRole.Constant);
        RegisterBuffer(_targetCoordinateRepresentation, "target_coordinate_representation", PersistentTensorRole.Constant);
    }

    private void AddConvBlock(int outputChannels, bool pool)
    {
        var convolution = new ConvolutionalLayer<T>(outputChannels, 3, 1, 1,
            new IdentityActivation<T>() as IActivationFunction<T>);
        // The convolution's output width is known here. Eagerly size BatchNorm so a fresh
        // checkpoint recipient exposes the same complete parameter manifest as a materialized
        // donor; leaving it lazy deferred all 1,984 gamma/beta values until after their first use.
        var normalization = new BatchNormalizationLayer<T>(outputChannels);
        var activation = new ActivationLayer<T>(new ReLUActivation<T>() as IActivationFunction<T>);
        _localizationLayers.Add(convolution);
        _localizationLayers.Add(normalization);
        _localizationLayers.Add(activation);
        RegisterSubLayer(convolution);
        RegisterSubLayer(normalization);
        RegisterSubLayer(activation);
        if (pool)
        {
            var pooling = new MaxPoolingLayer<T>(2, 2);
            _localizationLayers.Add(pooling);
            RegisterSubLayer(pooling);
        }
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank != 4 || input.Shape[1] != _inputChannels)
            throw new ArgumentException($"Expected [B,{_inputChannels},H,W].", nameof(input));
        int batch = input.Shape[0];
        var localization = Engine.Interpolate(
            input, [_localizationHeight, _localizationWidth],
            InterpolateMode.Bilinear, alignCorners: true);
        foreach (var layer in _localizationLayers) localization = layer.Forward(localization);
        localization = Engine.Reshape(localization, [batch, _localizationFeatureSize]);
        var features = _featureProjection.Forward(localization);
        features = Engine.TensorMultiplyScalar(features, NumOps.FromDouble(LocalizationScale));
        var flatControl = Engine.TensorAdd(
            Engine.TensorMatMul(features, _controlWeights), _controlBias);
        var sourceControl = Engine.Reshape(flatControl, [batch, _controlPointCount, 2]);
        var padding = new Tensor<T>([batch, 3, 2]);
        var rightHandSide = Engine.TensorConcatenate([sourceControl, padding], axis: 1);

        var inverse = Engine.TensorBroadcastTo(
            Engine.Reshape(_inverseKernel, [1, _controlPointCount + 3, _controlPointCount + 3]),
            [batch, _controlPointCount + 3, _controlPointCount + 3]);
        var target = Engine.TensorBroadcastTo(
            Engine.Reshape(_targetCoordinateRepresentation,
                [1, _outputHeight * _outputWidth, _controlPointCount + 3]),
            [batch, _outputHeight * _outputWidth, _controlPointCount + 3]);
        var mapping = Engine.TensorBatchMatMul(inverse, rightHandSide);
        var coordinates = Engine.TensorBatchMatMul(target, mapping);
        // Do not clamp the normalized source coordinates: GridSamplePadding.Zeros is the
        // deliberate out-of-bounds policy and preserves useful coordinate gradients at edges.
        coordinates = Engine.TensorSubtract(
            Engine.TensorMultiplyScalar(coordinates, NumOps.FromDouble(2.0)),
            Tensor<T>.CreateDefault(coordinates.Shape.ToArray(), NumOps.One));
        var grid = Engine.Reshape(coordinates, [batch, _outputHeight, _outputWidth, 2]);
        return Engine.GridSample(
            input, grid, GridSampleMode.Bilinear, GridSamplePadding.Zeros, alignCorners: true);
    }

    // GetParameterGradients and UpdateParameters are deliberately NOT overridden.
    //
    // Both used to be hand-rolled walks, and they disagreed with the canonical one and with each
    // other. LayerBase.FillParameterGradients exists precisely to prevent this and says so: it
    // "MIRRORS FillParameters DELIBERATELY, rather than building its own ordering ... A separate
    // walk that merely intends to agree will drift the first time either side gains a member, and
    // misaligned gradients are far worse than the missing ones they replace: every parameter would
    // be updated by some other parameter's derivative."
    //
    // It had drifted, and measurably so. GetParameters() emits 94,649 scalars for this layer, but
    // the old GetParameterGradients() override returned 3,765,049 -- roughly 40x too long, matching
    // neither the parameter vector nor the 20,520 own scalars. It concatenated each child's own
    // already-recursive gradient vector in front of a base value that is itself already recursive,
    // duplicating the whole subtree. A gradient vector that cannot be index-aligned with the
    // parameter vector silently updates every parameter by some other parameter's derivative.
    //
    // The two overrides also asserted incompatible contracts on the SAME base call: the gradient
    // override treated base.GetParameterGradients() as own-only and prepended children, while
    // UpdateParameters asserted its length equalled _controlWeights.Length + _controlBias.Length.
    // Both cannot hold once the base walk recurses, which it does.
    //
    // Composite layers that carry BOTH registered trainable tensors and RegisterSubLayer children
    // -- ConvNeXtV2Block is the reference shape -- override neither method and let the base walk
    // handle the whole subtree. _controlWeights and _controlBias are registered trainable tensors,
    // so they are covered by that walk with no bespoke code and, more importantly, no second
    // ordering that can drift again.

    /// <inheritdoc />
    public override void ResetState()
    {
        foreach (var layer in ParameterLayers) layer.ResetState();
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["InputChannels"] = _inputChannels.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["LocalizationHeight"] = _localizationHeight.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["LocalizationWidth"] = _localizationWidth.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["OutputHeight"] = _outputHeight.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["OutputWidth"] = _outputWidth.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["ControlPointCount"] = _controlPointCount.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["MarginX"] = _marginX.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["MarginY"] = _marginY.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["LocalizationMargin"] = _localizationMargin.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    private void InitializeIdentityControlBias(
        Tensor<T> bias, int count, double marginX, double marginY)
    {
        // Row-major (x0, y0, x1, y1, ...) is the [count, 2] point table flattened.
        var points = BuildControlPoints(count, marginX, marginY);
        Engine.TensorCopy(Engine.Reshape(points, [count * 2]).Cast<T>(), bias);
    }

    /// <summary>
    /// The <c>[count, 2]</c> fiducial grid: <c>count / 2</c> evenly spaced x positions in
    /// <c>[marginX, 1 - marginX]</c>, first along the top edge (<c>y = marginY</c>) and then the bottom
    /// (<c>y = 1 - marginY</c>).
    /// </summary>
    private Tensor<double> BuildControlPoints(int count, double marginX, double marginY)
    {
        int side = count / 2;
        var x = Engine.TensorLinspace(marginX, 1.0 - marginX, side);
        var top = Tensor<double>.CreateDefault([side], marginY);
        var bottom = Tensor<double>.CreateDefault([side], 1.0 - marginY);
        var xs = Engine.TensorConcatenate([x, x], 0);
        var ys = Engine.TensorConcatenate([top, bottom], 0);
        return Engine.TensorConcatenate([xs.Reshape([count, 1]), ys.Reshape([count, 1])], 1);
    }

    /// <summary>TPS geometry and its inverse retain double precision before casting constant buffers.</summary>
    private Tensor<double> BuildTpsKernel(Tensor<double> control)
    {
        int count = control.Shape[0];
        var controlX = Engine.TensorSlice(control, [0, 0], [count, 1]);
        var controlY = Engine.TensorSlice(control, [0, 1], [count, 1]);
        int[] pairShape = [count, count];
        var dx = Engine.TensorSubtract(
            Engine.TensorBroadcastTo(controlX, pairShape),
            Engine.TensorBroadcastTo(controlX.Reshape([1, count]), pairShape));
        var dy = Engine.TensorSubtract(
            Engine.TensorBroadcastTo(controlY, pairShape),
            Engine.TensorBroadcastTo(controlY.Reshape([1, count]), pairShape));
        var affine = Engine.TensorConcatenate([Tensor<double>.CreateDefault([count, 1], 1.0), control], 1);
        var top = Engine.TensorConcatenate([PartialRepresentation(dx, dy), affine], 1);
        var bottom = Engine.TensorConcatenate([Engine.TensorTranspose(affine), new Tensor<double>([3, 3])], 1);
        return Engine.TensorConcatenate([top, bottom], 0);
    }

    private Tensor<double> BuildTargetCoordinateRepresentation(int height, int width, Tensor<double> control)
    {
        int count = control.Shape[0];
        int pixels = height * width;
        int[] gridShape = [height, width];
        var columnX = Engine.TensorLinspace(0.0, 1.0, width);
        var rowY = Engine.TensorLinspace(0.0, 1.0, height);
        var nx = Engine.TensorBroadcastTo(columnX.Reshape([1, width]), gridShape).Reshape([pixels, 1]);
        var ny = Engine.TensorBroadcastTo(rowY.Reshape([height, 1]), gridShape).Reshape([pixels, 1]);
        int[] pairShape = [pixels, count];
        var controlX = Engine.TensorSlice(control, [0, 0], [count, 1]).Reshape([1, count]);
        var controlY = Engine.TensorSlice(control, [0, 1], [count, 1]).Reshape([1, count]);
        var dx = Engine.TensorSubtract(
            Engine.TensorBroadcastTo(nx, pairShape), Engine.TensorBroadcastTo(controlX, pairShape));
        var dy = Engine.TensorSubtract(
            Engine.TensorBroadcastTo(ny, pairShape), Engine.TensorBroadcastTo(controlY, pairShape));
        return Engine.TensorConcatenate([
            PartialRepresentation(dx, dy), Tensor<double>.CreateDefault([pixels, 1], 1.0), nx, ny], 1);
    }

    /// <summary>The radial basis 0.5 r^2 log(r^2), defined to be zero at the origin.</summary>
    private Tensor<double> PartialRepresentation(Tensor<double> dx, Tensor<double> dy)
    {
        var squared = Engine.TensorAdd(Engine.TensorMultiply(dx, dx), Engine.TensorMultiply(dy, dy));
        var positive = Engine.TensorGreaterThan(squared, 0.0);
        var safe = Engine.TensorWhere(positive, squared, Tensor<double>.CreateDefault(squared.Shape.ToArray(), 1.0));
        return Engine.TensorMultiplyScalar(Engine.TensorMultiply(squared, Engine.TensorLog(safe)), 0.5);
    }

    private Tensor<double> Invert(Tensor<double> source)
    {
        int size = source.Shape[0];
        var (inverse, info) = LinalgDispatch<double>.SolveEx(source, Engine.TensorEye<double>(size));
        if (Engine.TensorCountNonzero(info) > 0)
            throw new InvalidOperationException("TPS kernel is singular.");
        return inverse;
    }
}

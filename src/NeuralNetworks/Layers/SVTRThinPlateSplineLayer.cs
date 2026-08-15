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
[LayerCategory(LayerCategory.Other)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, Cost = ComputeCost.High,
    TestInputShape = "1, 3, 32, 100", TestConstructorArgs = "3, 32, 64, 32, 100, 20")]
public sealed class SVTRThinPlateSplineLayer<T> : LayerBase<T>
{
    private readonly int _inputChannels;
    private readonly int _localizationHeight;
    private readonly int _localizationWidth;
    private readonly int _outputHeight;
    private readonly int _outputWidth;
    private readonly int _controlPointCount;
    private readonly double _marginX;
    private readonly double _marginY;

    private readonly List<ILayer<T>> _localizationLayers = [];
    private readonly DenseLayer<T> _featureProjection;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private readonly Tensor<T> _controlWeights;

    [TrainableParameter(Role = PersistentTensorRole.Biases)]
    private readonly Tensor<T> _controlBias;

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
    public override bool SupportsTraining => true;
    public override long ParameterCount =>
        ParameterLayers.Sum(layer => layer.ParameterCount) + _controlWeights.Length + _controlBias.Length;

    public SVTRThinPlateSplineLayer(
        int inputChannels = 3,
        int localizationHeight = 32,
        int localizationWidth = 64,
        int outputHeight = 32,
        int outputWidth = 100,
        int controlPointCount = 20,
        double marginX = 0.05,
        double marginY = 0.05)
        : base([inputChannels, -1, -1], [inputChannels, outputHeight, outputWidth])
    {
        if (inputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inputChannels));
        if (localizationHeight <= 0 || localizationWidth <= 0 || outputHeight <= 1 || outputWidth <= 1)
            throw new ArgumentOutOfRangeException(nameof(outputHeight));
        if (controlPointCount < 4 || controlPointCount % 2 != 0)
            throw new ArgumentException("TPS requires an even control-point count of at least four.", nameof(controlPointCount));
        if (marginX <= 0 || marginX >= 0.5 || marginY <= 0 || marginY >= 0.5)
            throw new ArgumentOutOfRangeException(nameof(marginX));

        _inputChannels = inputChannels;
        _localizationHeight = localizationHeight;
        _localizationWidth = localizationWidth;
        _outputHeight = outputHeight;
        _outputWidth = outputWidth;
        _controlPointCount = controlPointCount;
        _marginX = marginX;
        _marginY = marginY;

        AddConvBlock(32, pool: true);
        AddConvBlock(64, pool: true);
        AddConvBlock(128, pool: true);
        AddConvBlock(256, pool: true);
        AddConvBlock(256, pool: true);
        AddConvBlock(256, pool: false);
        _featureProjection = new DenseLayer<T>(512, new ReLUActivation<T>() as IActivationFunction<T>);
        RegisterSubLayer(_featureProjection);

        _controlWeights = new Tensor<T>([512, controlPointCount * 2]);
        _controlBias = new Tensor<T>([controlPointCount * 2]);
        InitializeIdentityControlBias(_controlBias, controlPointCount);
        RegisterTrainableParameter(_controlWeights, PersistentTensorRole.Weights);
        AppendTrainableParameter(_controlBias, PersistentTensorRole.Biases);

        var targetControlPoints = BuildControlPoints(controlPointCount, marginX, marginY);
        _inverseKernel = ToTensor(Invert(BuildTpsKernel(targetControlPoints)));
        _targetCoordinateRepresentation = ToTensor(
            BuildTargetCoordinateRepresentation(outputHeight, outputWidth, targetControlPoints));
        RegisterBuffer(_inverseKernel, "inverse_kernel", PersistentTensorRole.Constant);
        RegisterBuffer(_targetCoordinateRepresentation, "target_coordinate_representation", PersistentTensorRole.Constant);
    }

    private void AddConvBlock(int outputChannels, bool pool)
    {
        var convolution = new ConvolutionalLayer<T>(outputChannels, 3, 1, 1,
            new IdentityActivation<T>() as IActivationFunction<T>);
        var normalization = new BatchNormalizationLayer<T>();
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

    public override Tensor<T> Forward(Tensor<T> input)
    {
        if (input.Rank != 4 || input.Shape[1] != _inputChannels)
            throw new ArgumentException($"Expected [B,{_inputChannels},H,W].", nameof(input));
        int batch = input.Shape[0];
        var localization = Engine.Interpolate(
            input, [_localizationHeight, _localizationWidth],
            InterpolateMode.Bilinear, alignCorners: true);
        foreach (var layer in _localizationLayers) localization = layer.Forward(localization);
        localization = Engine.Reshape(localization, [batch, 512]);
        var features = _featureProjection.Forward(localization);
        features = Engine.TensorMultiplyScalar(features, NumOps.FromDouble(0.1));
        var flatControl = Engine.TensorBroadcastAdd(
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
        coordinates = Engine.TensorClamp(coordinates, NumOps.Zero, NumOps.One);
        coordinates = Engine.TensorSubtract(
            Engine.TensorMultiplyScalar(coordinates, NumOps.FromDouble(2.0)),
            Tensor<T>.CreateDefault(coordinates.Shape.ToArray(), NumOps.One));
        var grid = Engine.Reshape(coordinates, [batch, _outputHeight, _outputWidth, 2]);
        return Engine.GridSample(
            input, grid, GridSampleMode.Bilinear, GridSamplePadding.Zeros, alignCorners: true);
    }

    public override Vector<T> GetParameters()
    {
        var values = new List<T>();
        foreach (var layer in ParameterLayers) values.AddRange(layer.GetParameters());
        values.AddRange(_controlWeights.Data.ToArray());
        values.AddRange(_controlBias.Data.ToArray());
        return new Vector<T>([.. values]);
    }

    public override Vector<T> GetParameterGradients()
    {
        var values = new List<T>();
        foreach (var layer in ParameterLayers) values.AddRange(layer.GetParameterGradients());
        var own = base.GetParameterGradients();
        values.AddRange(own);
        return new Vector<T>([.. values]);
    }

    public override void SetParameters(Vector<T> parameters)
    {
        if (ParameterLayers.Any(layer => layer.ParameterCount == 0))
            _ = Forward(new Tensor<T>([1, _inputChannels, _localizationHeight, _localizationWidth]));
        if (parameters.Length != ParameterCount)
            throw new ArgumentException($"Expected {ParameterCount} parameters, got {parameters.Length}.");
        int offset = 0;
        foreach (var layer in ParameterLayers)
        {
            int count = checked((int)layer.ParameterCount);
            layer.SetParameters(parameters.Slice(offset, count));
            offset += count;
        }
        parameters.Slice(offset, _controlWeights.Length).AsSpan().CopyTo(_controlWeights.Data.Span);
        offset += _controlWeights.Length;
        parameters.Slice(offset, _controlBias.Length).AsSpan().CopyTo(_controlBias.Data.Span);
        Engine.InvalidatePersistentTensor(_controlWeights);
        Engine.InvalidatePersistentTensor(_controlBias);
    }

    public override void UpdateParameters(T learningRate)
    {
        foreach (var layer in ParameterLayers) layer.UpdateParameters(learningRate);
        var own = base.GetParameterGradients();
        if (own.Length == _controlWeights.Length + _controlBias.Length)
        {
            for (int i = 0; i < _controlWeights.Length; i++)
                _controlWeights[i] = NumOps.Subtract(_controlWeights[i], NumOps.Multiply(learningRate, own[i]));
            for (int i = 0; i < _controlBias.Length; i++)
                _controlBias[i] = NumOps.Subtract(_controlBias[i],
                    NumOps.Multiply(learningRate, own[_controlWeights.Length + i]));
        }
    }

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
        return metadata;
    }

    private void InitializeIdentityControlBias(Tensor<T> bias, int count)
    {
        var points = BuildControlPoints(count, 0.01, 0.01);
        for (int i = 0; i < count; i++)
        {
            bias[i * 2] = NumOps.FromDouble(points[i, 0]);
            bias[i * 2 + 1] = NumOps.FromDouble(points[i, 1]);
        }
    }

    private Tensor<T> ToTensor(double[,] source)
    {
        int rows = source.GetLength(0), columns = source.GetLength(1);
        var data = new T[rows * columns];
        for (int row = 0; row < rows; row++)
        for (int column = 0; column < columns; column++)
            data[row * columns + column] = NumOps.FromDouble(source[row, column]);
        return new Tensor<T>(data, [rows, columns]);
    }

    private static double[,] BuildControlPoints(int count, double marginX, double marginY)
    {
        int side = count / 2;
        var points = new double[count, 2];
        for (int i = 0; i < side; i++)
        {
            double x = marginX + (1.0 - 2.0 * marginX) * i / (side - 1);
            points[i, 0] = x; points[i, 1] = marginY;
            points[side + i, 0] = x; points[side + i, 1] = 1.0 - marginY;
        }
        return points;
    }

    private static double[,] BuildTpsKernel(double[,] control)
    {
        int count = control.GetLength(0);
        var kernel = new double[count + 3, count + 3];
        for (int row = 0; row < count; row++)
        for (int column = 0; column < count; column++)
            kernel[row, column] = PartialRepresentation(
                control[row, 0] - control[column, 0], control[row, 1] - control[column, 1]);
        for (int i = 0; i < count; i++)
        {
            kernel[i, count] = 1.0;
            kernel[count, i] = 1.0;
            kernel[i, count + 1] = control[i, 0];
            kernel[i, count + 2] = control[i, 1];
            kernel[count + 1, i] = control[i, 0];
            kernel[count + 2, i] = control[i, 1];
        }
        return kernel;
    }

    private static double[,] BuildTargetCoordinateRepresentation(int height, int width, double[,] control)
    {
        int count = control.GetLength(0);
        var result = new double[height * width, count + 3];
        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            int row = y * width + x;
            double nx = (double)x / (width - 1);
            double ny = (double)y / (height - 1);
            for (int i = 0; i < count; i++)
                result[row, i] = PartialRepresentation(nx - control[i, 0], ny - control[i, 1]);
            result[row, count] = 1.0;
            result[row, count + 1] = nx;
            result[row, count + 2] = ny;
        }
        return result;
    }

    private static double PartialRepresentation(double dx, double dy)
    {
        double squaredDistance = dx * dx + dy * dy;
        return squaredDistance <= 0.0 ? 0.0 : 0.5 * squaredDistance * Math.Log(squaredDistance);
    }

    private static double[,] Invert(double[,] source)
    {
        int size = source.GetLength(0);
        var augmented = new double[size, size * 2];
        for (int row = 0; row < size; row++)
        for (int column = 0; column < size; column++)
        {
            augmented[row, column] = source[row, column];
            augmented[row, size + column] = row == column ? 1.0 : 0.0;
        }
        for (int pivot = 0; pivot < size; pivot++)
        {
            int best = pivot;
            for (int row = pivot + 1; row < size; row++)
                if (Math.Abs(augmented[row, pivot]) > Math.Abs(augmented[best, pivot])) best = row;
            if (Math.Abs(augmented[best, pivot]) < 1e-12)
                throw new InvalidOperationException("TPS kernel is singular.");
            if (best != pivot)
                for (int column = 0; column < size * 2; column++)
                    (augmented[pivot, column], augmented[best, column]) =
                        (augmented[best, column], augmented[pivot, column]);
            double divisor = augmented[pivot, pivot];
            for (int column = 0; column < size * 2; column++) augmented[pivot, column] /= divisor;
            for (int row = 0; row < size; row++)
            {
                if (row == pivot) continue;
                double factor = augmented[row, pivot];
                for (int column = 0; column < size * 2; column++)
                    augmented[row, column] -= factor * augmented[pivot, column];
            }
        }
        var inverse = new double[size, size];
        for (int row = 0; row < size; row++)
        for (int column = 0; column < size; column++)
            inverse[row, column] = augmented[row, size + column];
        return inverse;
    }
}

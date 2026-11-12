using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.DomainSpecific;

/// <summary>
/// Feature selection for image data.
/// </summary>
/// <remarks>
/// <para>
/// ImageFeatureSelector is designed for selecting features from image data,
/// considering spatial locality and the correlation patterns typical of image features.
/// It can handle flattened image data and respects the original image structure.
/// </para>
/// <para><b>For Beginners:</b> Images have special structure - nearby pixels are usually
/// similar. This selector understands that and picks features while considering their
/// spatial relationships. It's like choosing representative spots on a photo rather
/// than random pixels.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ImageFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _imageWidth;
    private readonly int _imageHeight;
    private readonly int _numChannels;
    private readonly bool _useSpatialCorrelation;

    private Vector<T>? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int ImageWidth => _imageWidth;
    public int ImageHeight => _imageHeight;
    public int NumChannels => _numChannels;
    public bool UseSpatialCorrelation => _useSpatialCorrelation;
    public Vector<T>? FeatureScores => _featureScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public ImageFeatureSelector(
        int nFeaturesToSelect = 10,
        int imageWidth = 28,
        int imageHeight = 28,
        int numChannels = 1,
        bool useSpatialCorrelation = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (imageWidth < 1 || imageHeight < 1)
            throw new ArgumentException("Image dimensions must be positive.");
        if (numChannels < 1)
            throw new ArgumentException("Number of channels must be at least 1.", nameof(numChannels));

        _nFeaturesToSelect = nFeaturesToSelect;
        _imageWidth = imageWidth;
        _imageHeight = imageHeight;
        _numChannels = numChannels;
        _useSpatialCorrelation = useSpatialCorrelation;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ImageFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");
        if (data.Rows == 0 || data.Columns == 0)
            throw new ArgumentException("Data must contain at least one row and one column.", nameof(data));

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Compute base scores (correlation with target)
        var scores = ComputeBaseScores(data, target);

        if (_useSpatialCorrelation)
        {
            // Adjust scores based on spatial diversity
            scores = AdjustForSpatialDiversity(scores, p);
        }

        _featureScores = scores.ToVector();

        // Select top features
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, numToSelect);

        IsFitted = true;
    }

    private Tensor<T> ComputeBaseScores(Matrix<T> data, Vector<T> target)
    {
        var engine = Engine;
        int n = data.Rows;
        var X = Tensor<T>.FromMatrix(data);

        // |correlation| with target, plus the standard deviation (pixels with high variance are more informative)
        var corr = engine.TensorAbs(PreprocessingTensorOps.TargetCorrelations(X, Tensor<T>.FromVector(target)));
        var variance = engine.TensorDivideScalar(FeatureSelectionTensorOps.ColumnSumSquaredDeviations(X), NumOps.FromDouble(n));

        return engine.TensorAdd(
            engine.TensorMultiplyScalar(corr, NumOps.FromDouble(0.7)),
            engine.TensorMultiplyScalar(engine.TensorSqrt(variance), NumOps.FromDouble(0.3)));
    }

    private Tensor<T> AdjustForSpatialDiversity(Tensor<T> scores, int p)
    {
        // Greedy selection ensuring spatial diversity
        var engine = Engine;
        var adjustedScores = scores;
        var columns = new int[p];
        var rows = new int[p];
        for (int j = 0; j < p; j++)
        {
            var (x, y, _) = GetSpatialPosition(j);
            columns[j] = x;
            rows[j] = y;
        }

        var xs = new Tensor<int>(columns, new[] { p }).Cast<T>();
        var ys = new Tensor<int>(rows, new[] { p }).Cast<T>();
        var positions = PreprocessingTensorOps.Range<T>(p);
        var selected = new Tensor<T>(new[] { p });
        var lowest = PreprocessingTensorOps.Filled(new[] { p }, NumOps.MinValue);
        var unpenalized = PreprocessingTensorOps.Filled(new[] { p }, NumOps.One);
        T minValue = NumOps.MinValue;
        T radius = NumOps.FromDouble(3);
        T negativeHalf = NumOps.FromDouble(-0.5);
        T half = NumOps.FromDouble(0.5);

        for (int k = 0; k < Math.Min(_nFeaturesToSelect * 2, p); k++)
        {
            // Find best unselected feature (first among ties)
            var candidates = engine.TensorWhere(selected, lowest, adjustedScores);
            if (!PreprocessingTensorOps.IsAbove(engine.TensorMaxValue(candidates), minValue))
                break;

            int bestIdx = PreprocessingTensorOps.FirstArgMax(candidates);
            var isBest = engine.TensorEquals(positions, PreprocessingTensorOps.CountAsScalar<T>(bestIdx));
            selected = engine.TensorMax(selected, isBest);

            // Reduce scores of spatially nearby features (within 3 pixels) by 1 - exp(-distance / 2) * 0.5
            var (x1, y1, _) = GetSpatialPosition(bestIdx);
            var distance = engine.TensorSqrt(engine.TensorAdd(
                engine.TensorSquare(engine.TensorSubtractScalar(xs, PreprocessingTensorOps.CountAsScalar<T>(x1))),
                engine.TensorSquare(engine.TensorSubtractScalar(ys, PreprocessingTensorOps.CountAsScalar<T>(y1)))));
            var penalty = engine.TensorExp(engine.TensorMultiplyScalar(distance, negativeHalf));
            var factor = PreprocessingTensorOps.OneMinus(engine.TensorMultiplyScalar(penalty, half));
            var nearby = engine.TensorMultiply(PreprocessingTensorOps.Not(selected), engine.TensorLessThan(distance, radius));
            adjustedScores = engine.TensorMultiply(adjustedScores, engine.TensorWhere(nearby, factor, unpenalized));
        }

        return adjustedScores;
    }
    private (int X, int Y, int Channel) GetSpatialPosition(int featureIdx)
    {
        int pixelIdx = featureIdx / _numChannels;
        int channel = featureIdx % _numChannels;
        int x = pixelIdx % _imageWidth;
        int y = pixelIdx / _imageWidth;
        return (x, y, channel);
    }

    /// <inheritdoc />
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ImageFeatureSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ImageFeatureSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ImageFeatureSelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return [];

        if (inputFeatureNames is null)
        {
            return _selectedIndices.Select(i =>
            {
                var (x, y, c) = GetSpatialPosition(i);
                return $"Pixel_{x}_{y}_Ch{c}";
            }).ToArray();
        }

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}

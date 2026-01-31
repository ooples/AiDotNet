using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

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
public class ImageFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _imageWidth;
    private readonly int _imageHeight;
    private readonly int _numChannels;
    private readonly bool _useSpatialCorrelation;

    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int ImageWidth => _imageWidth;
    public int ImageHeight => _imageHeight;
    public int NumChannels => _numChannels;
    public bool UseSpatialCorrelation => _useSpatialCorrelation;
    public double[]? FeatureScores => _featureScores;
    public int[]? SelectedIndices => _selectedIndices;
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

        _nFeaturesToSelect = nFeaturesToSelect;
        _imageWidth = imageWidth;
        _imageHeight = imageHeight;
        _numChannels = numChannels;
        _useSpatialCorrelation = useSpatialCorrelation;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ImageFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Compute base scores (correlation with target)
        _featureScores = ComputeBaseScores(data, target);

        if (_useSpatialCorrelation)
        {
            // Adjust scores based on spatial diversity
            AdjustForSpatialDiversity(p);
        }

        // Select top features
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _featureScores
            .Select((s, idx) => (Score: s, Index: idx))
            .OrderByDescending(x => x.Score)
            .Take(numToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[] ComputeBaseScores(Matrix<T> data, Vector<T> target)
    {
        int n = data.Rows;
        int p = data.Columns;
        var scores = new double[p];

        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        for (int j = 0; j < p; j++)
        {
            double xMean = 0;
            for (int i = 0; i < n; i++)
                xMean += NumOps.ToDouble(data[i, j]);
            xMean /= n;

            double sxy = 0, sxx = 0, syy = 0;
            for (int i = 0; i < n; i++)
            {
                double xDiff = NumOps.ToDouble(data[i, j]) - xMean;
                double yDiff = NumOps.ToDouble(target[i]) - yMean;
                sxy += xDiff * yDiff;
                sxx += xDiff * xDiff;
                syy += yDiff * yDiff;
            }

            // Also consider variance (pixels with high variance are more informative)
            double corr = (sxx > 1e-10 && syy > 1e-10) ? Math.Abs(sxy / Math.Sqrt(sxx * syy)) : 0;
            double variance = sxx / n;

            scores[j] = corr * 0.7 + Math.Sqrt(variance) * 0.3;
        }

        return scores;
    }

    private void AdjustForSpatialDiversity(int p)
    {
        // Greedy selection ensuring spatial diversity
        var adjustedScores = (double[])_featureScores!.Clone();
        var selected = new HashSet<int>();

        for (int k = 0; k < Math.Min(_nFeaturesToSelect * 2, p); k++)
        {
            // Find best unselected feature
            int bestIdx = -1;
            double bestScore = double.MinValue;

            for (int j = 0; j < p; j++)
            {
                if (!selected.Contains(j) && adjustedScores[j] > bestScore)
                {
                    bestScore = adjustedScores[j];
                    bestIdx = j;
                }
            }

            if (bestIdx < 0) break;

            selected.Add(bestIdx);

            // Reduce scores of spatially nearby features
            var (x1, y1, _) = GetSpatialPosition(bestIdx);
            for (int j = 0; j < p; j++)
            {
                if (!selected.Contains(j))
                {
                    var (x2, y2, _) = GetSpatialPosition(j);
                    double distance = Math.Sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
                    if (distance < 3) // Within 3 pixels
                    {
                        double penalty = Math.Exp(-distance / 2);
                        adjustedScores[j] *= (1 - penalty * 0.5);
                    }
                }
            }
        }

        _featureScores = adjustedScores;
    }

    private (int X, int Y, int Channel) GetSpatialPosition(int featureIdx)
    {
        int pixelIdx = featureIdx / _numChannels;
        int channel = featureIdx % _numChannels;
        int x = pixelIdx % _imageWidth;
        int y = pixelIdx / _imageWidth;
        return (x, y, channel);
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ImageFeatureSelector has not been fitted.");

        int numRows = data.Rows;
        int numCols = _selectedIndices.Length;
        var result = new T[numRows, numCols];

        for (int i = 0; i < numRows; i++)
            for (int j = 0; j < numCols; j++)
                result[i, j] = data[i, _selectedIndices[j]];

        return new Matrix<T>(result);
    }

    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ImageFeatureSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ImageFeatureSelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

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

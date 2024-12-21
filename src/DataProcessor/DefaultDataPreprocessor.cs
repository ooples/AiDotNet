namespace AiDotNet.DataProcessor;

public class DefaultDataPreprocessor<T> : IDataPreprocessor<T>
{
    private readonly INormalizer<T> _normalizer;
    private readonly IFeatureSelector<T> _featureSelector;
    private readonly IOutlierRemoval<T> _outlierRemoval;
    private readonly DataProcessorOptions _options;

    public DefaultDataPreprocessor(INormalizer<T> normalizer, IFeatureSelector<T> featureSelector, IOutlierRemoval<T> outlierRemoval, DataProcessorOptions? options = null)
    {
        _normalizer = normalizer;
        _featureSelector = featureSelector;
        _outlierRemoval = outlierRemoval;
        _options = options ?? new();
    }

    public (Matrix<T> X, Vector<T> y, NormalizationInfo<T> normInfo) PreprocessData(Matrix<T> X, Vector<T> y)
    {
        NormalizationInfo<T> normInfo = new();

        (X, y) = _outlierRemoval.RemoveOutliers(X, y);

        if (_options.NormalizeBeforeFeatureSelection)
        {
            (X, normInfo.XParams) = _normalizer.NormalizeMatrix(X);
            (y, normInfo.YParams) = _normalizer.NormalizeVector(y);
            X = _featureSelector.SelectFeatures(X);
        }
        else
        {
            X = _featureSelector.SelectFeatures(X);
            (X, normInfo.XParams) = _normalizer.NormalizeMatrix(X);
            (y, normInfo.YParams) = _normalizer.NormalizeVector(y);
        }

        return (X, y, normInfo);
    }

    public (Matrix<T> XTrain, Vector<T> yTrain, Matrix<T> XValidation, Vector<T> yValidation, Matrix<T> XTest, Vector<T> yTest) SplitData(Matrix<T> X, Vector<T> y)
    {
        int totalSamples = X.Rows;
        int trainSize = (int)(totalSamples * _options.TrainingSplitPercentage);
        int validationSize = (int)(totalSamples * _options.ValidationSplitPercentage);
        int testSize = totalSamples - trainSize - validationSize;

        // Shuffle the data
        var random = new Random(_options.RandomSeed);
        var indices = Enumerable.Range(0, totalSamples).ToList();
        indices = [.. indices.OrderBy(x => random.Next())];

        // Split the data
        var XTrain = new Matrix<T>(trainSize, X.Columns);
        var yTrain = new Vector<T>(trainSize);
        var XValidation = new Matrix<T>(validationSize, X.Columns);
        var yValidation = new Vector<T>(validationSize);
        var XTest = new Matrix<T>(testSize, X.Columns);
        var yTest = new Vector<T>(testSize);

        for (int i = 0; i < trainSize; i++)
        {
            XTrain.SetRow(i, X.GetRow(indices[i]));
            yTrain[i] = y[indices[i]];
        }

        for (int i = 0; i < validationSize; i++)
        {
            XValidation.SetRow(i, X.GetRow(indices[i + trainSize]));
            yValidation[i] = y[indices[i + trainSize]];
        }

        for (int i = 0; i < testSize; i++)
        {
            XTest.SetRow(i, X.GetRow(indices[i + trainSize + validationSize]));
            yTest[i] = y[indices[i + trainSize + validationSize]];
        }

        return (XTrain, yTrain, XValidation, yValidation, XTest, yTest);
    }
}
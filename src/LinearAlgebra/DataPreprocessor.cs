namespace AiDotNet.LinearAlgebra;

public class DataPreprocessor : IDataPreprocessor
{
    private readonly INormalizer _normalizer;
    private readonly IFeatureSelector _featureSelector;
    private readonly bool _normalizeBeforeFeatureSelection;
    private readonly PredictionModelOptions _options;

    public DataPreprocessor(INormalizer normalizer, IFeatureSelector featureSelector, PredictionModelOptions options)
    {
        _normalizer = normalizer;
        _featureSelector = featureSelector;
        _options = options;
        _normalizeBeforeFeatureSelection = options.NormalizeBeforeFeatureSelection;
    }

    public (Matrix<double> X, Vector<double> y, NormalizationInfo normInfo) PreprocessData(Matrix<double> X, Vector<double> y)
    {
        NormalizationInfo normInfo = new();

        if (_normalizeBeforeFeatureSelection)
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

    public (Matrix<double> XTrain, Vector<double> yTrain, Matrix<double> XValidation, Vector<double> yValidation, Matrix<double> XTest, Vector<double> yTest) 
        SplitData(Matrix<double> X, Vector<double> y)
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
        var XTrain = new Matrix<double>(trainSize, X.Columns);
        var yTrain = new Vector<double>(trainSize);
        var XValidation = new Matrix<double>(validationSize, X.Columns);
        var yValidation = new Vector<double>(validationSize);
        var XTest = new Matrix<double>(testSize, X.Columns);
        var yTest = new Vector<double>(testSize);

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
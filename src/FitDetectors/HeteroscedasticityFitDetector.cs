using AiDotNet.Helpers;
using AiDotNet.Models.Results;

namespace AiDotNet.FitDetectors;

public class HeteroscedasticityFitDetector<T> : FitDetectorBase<T>
{
    private readonly HeteroscedasticityFitDetectorOptions _options;

    public HeteroscedasticityFitDetector(HeteroscedasticityFitDetectorOptions? options = null)
    {
        _options = options ?? new HeteroscedasticityFitDetectorOptions();
    }

    public override FitDetectorResult<T> DetectFit(ModelEvaluationData<T> evaluationData)
    {
        var fitType = DetermineFitType(evaluationData);
        var confidenceLevel = CalculateConfidenceLevel(evaluationData);
        var recommendations = GenerateRecommendations(fitType, evaluationData);

        return new FitDetectorResult<T>
        {
            FitType = fitType,
            ConfidenceLevel = confidenceLevel,
            Recommendations = recommendations,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "BreuschPaganTestStatistic", Convert.ToDouble(CalculateBreuschPaganTestStatistic(evaluationData)) },
                { "WhiteTestStatistic", Convert.ToDouble(CalculateWhiteTestStatistic(evaluationData)) }
            }
        };
    }

    protected override FitType DetermineFitType(ModelEvaluationData<T> evaluationData)
    {
        var breuschPaganTestStatistic = CalculateBreuschPaganTestStatistic(evaluationData);
        var whiteTestStatistic = CalculateWhiteTestStatistic(evaluationData);

        if (_numOps.GreaterThan(breuschPaganTestStatistic, _numOps.FromDouble(_options.HeteroscedasticityThreshold)) ||
            _numOps.GreaterThan(whiteTestStatistic, _numOps.FromDouble(_options.HeteroscedasticityThreshold)))
        {
            return FitType.Unstable;
        }
        else if (_numOps.LessThan(breuschPaganTestStatistic, _numOps.FromDouble(_options.HomoscedasticityThreshold)) &&
                 _numOps.LessThan(whiteTestStatistic, _numOps.FromDouble(_options.HomoscedasticityThreshold)))
        {
            return FitType.GoodFit;
        }
        else
        {
            return FitType.Moderate;
        }
    }

    protected override T CalculateConfidenceLevel(ModelEvaluationData<T> evaluationData)
    {
        var breuschPaganTestStatistic = CalculateBreuschPaganTestStatistic(evaluationData);
        var whiteTestStatistic = CalculateWhiteTestStatistic(evaluationData);

        var maxTestStatistic = _numOps.GreaterThan(breuschPaganTestStatistic, whiteTestStatistic) ? breuschPaganTestStatistic : whiteTestStatistic;
        var normalizedTestStatistic = _numOps.Divide(maxTestStatistic, _numOps.FromDouble(_options.HeteroscedasticityThreshold));

        // Invert the normalized test statistic to get a confidence level (higher test statistic = lower confidence)
        return _numOps.Subtract(_numOps.One, _numOps.LessThan(_numOps.One, normalizedTestStatistic) ? _numOps.One : normalizedTestStatistic);
    }

    private T CalculateBreuschPaganTestStatistic(ModelEvaluationData<T> evaluationData)
    {
        var X = evaluationData.ModelStats.FeatureMatrix;
        var y = evaluationData.ModelStats.Actual;
        var yPredicted = evaluationData.ModelStats.Model?.Predict(X) ?? Vector<T>.Empty();
        var residuals = y.Subtract(yPredicted);

        var squaredResiduals = residuals.Select(r => _numOps.Multiply(r, r));
        var meanSquaredResidual = _numOps.Divide(squaredResiduals.Sum(), _numOps.FromDouble(squaredResiduals.Length));

        var scaledResiduals = squaredResiduals.Divide(meanSquaredResidual);
        var auxiliaryRegression = new SimpleRegression<T>();
        auxiliaryRegression.Train(X, scaledResiduals);

        // Calculate R-squared using PredictionStats
        var predictionStatsInputs = new PredictionStatsInputs<T>
        {
            Actual = new Vector<T>(scaledResiduals),
            Predicted = auxiliaryRegression.Predict(X),
            NumberOfParameters = X.Columns,
        };
        var predictionStats = new PredictionStats<T>(predictionStatsInputs);

        return _numOps.Multiply(_numOps.FromDouble(X.Rows), predictionStats.R2);
    }

    private T CalculateWhiteTestStatistic(ModelEvaluationData<T> evaluationData)
    {
        var X = evaluationData.ModelStats.FeatureMatrix;
        var y = evaluationData.ModelStats.Actual;
        var yPredicted = evaluationData.ModelStats.Model?.Predict(X) ?? new Vector<T>(y.Length);
        var residuals = y.Subtract(yPredicted);

        var squaredResiduals = residuals.Select(r => _numOps.Multiply(r, r));

        // Create augmented X matrix with squared terms and cross products
        var augmentedX = new Matrix<T>(X.Rows, X.Columns * (X.Columns + 3) / 2 + 1);
        int column = 0;
        for (int i = 0; i < X.Columns; i++)
        {
            augmentedX.SetColumn(column++, X.GetColumn(i));
            augmentedX.SetColumn(column++, X.GetColumn(i).Select(x => _numOps.Multiply(x, x)));
            for (int j = i + 1; j < X.Columns; j++)
                {
                augmentedX.SetColumn(column++, new Vector<T>(X.GetColumn(i).Zip(X.GetColumn(j), (a, b) => _numOps.Multiply(a, b))));
            }
        }
        augmentedX.SetColumn(column, Vector<T>.CreateDefault(X.Rows, _numOps.One));

        var auxiliaryRegression = new SimpleRegression<T>();
        auxiliaryRegression.Train(augmentedX, new Vector<T>(squaredResiduals));
        var predictionStatsInputs = new PredictionStatsInputs<T>
        {
            Actual = new Vector<T>(squaredResiduals),
            Predicted = auxiliaryRegression.Predict(augmentedX),
            NumberOfParameters = augmentedX.Columns,
        };

        var predictionStats = new PredictionStats<T>(predictionStatsInputs);

        return _numOps.Multiply(_numOps.FromDouble(X.Rows), predictionStats.R2);
    }

    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T> evaluationData)
    {
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.Unstable:
                recommendations.Add("The model shows signs of heteroscedasticity. Consider the following:");
                recommendations.Add("1. Transform the dependent variable (e.g., log transformation).");
                recommendations.Add("2. Use weighted least squares regression.");
                recommendations.Add("3. Consider using robust standard errors.");
                recommendations.Add("4. Investigate if important predictors are missing from the model.");
                break;

            case FitType.Moderate:
                recommendations.Add("The model shows some signs of heteroscedasticity. Consider the following:");
                recommendations.Add("1. Investigate potential causes of heteroscedasticity in your data.");
                recommendations.Add("2. Consider mild transformations of variables.");
                recommendations.Add("3. Use diagnostic plots to visualize the residuals.");
                break;

            case FitType.GoodFit:
                recommendations.Add("The model appears to have homoscedastic residuals. Consider the following:");
                recommendations.Add("1. Validate the model on new, unseen data.");
                recommendations.Add("2. Monitor model performance over time for potential changes in residual patterns.");
                recommendations.Add("3. Consider other aspects of model fit, such as linearity and normality of residuals.");
                break;
        }

        var breuschPaganTestStatistic = CalculateBreuschPaganTestStatistic(evaluationData);
        var whiteTestStatistic = CalculateWhiteTestStatistic(evaluationData);

        recommendations.Add($"Breusch-Pagan test statistic: {breuschPaganTestStatistic}");
        recommendations.Add($"White test statistic: {whiteTestStatistic}");

        return recommendations;
    }
}

public class HeteroscedasticityFitDetectorOptions
{
    public double HeteroscedasticityThreshold { get; set; } = 0.05; // p-value threshold for heteroscedasticity
    public double HomoscedasticityThreshold { get; set; } = 0.1; // p-value threshold for homoscedasticity
}
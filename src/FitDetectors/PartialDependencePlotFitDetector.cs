using AiDotNet.Models.Options;

namespace AiDotNet.FitDetectors;

public class PartialDependencePlotFitDetector<T> : FitDetectorBase<T>
{
    private readonly PartialDependencePlotFitDetectorOptions _options;

    public PartialDependencePlotFitDetector(PartialDependencePlotFitDetectorOptions? options = null)
    {
        _options = options ?? new PartialDependencePlotFitDetectorOptions();
    }

    public override FitDetectorResult<T> DetectFit(ModelEvaluationData<T> evaluationData)
    {
        var pdpResults = CalculatePartialDependencePlots(evaluationData);
        var fitType = DetermineFitType(pdpResults);
        var confidenceLevel = CalculateConfidenceLevel(evaluationData);
        var recommendations = GenerateRecommendations(fitType, pdpResults);

        return new FitDetectorResult<T>
        {
            FitType = fitType,
            ConfidenceLevel = confidenceLevel,
            Recommendations = recommendations,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "PartialDependencePlots", pdpResults }
            }
        };
    }

    protected override FitType DetermineFitType(ModelEvaluationData<T> evaluationData)
    {
        var pdpResults = CalculatePartialDependencePlots(evaluationData);
        return DetermineFitType(pdpResults);
    }

    private FitType DetermineFitType(Dictionary<string, Vector<T>> pdpResults)
    {
        var nonlinearityScores = CalculateNonlinearityScores(pdpResults);
        var sumNonlinearity = nonlinearityScores.Aggregate(_numOps.Zero, (acc, score) => _numOps.Add(acc, score));
        var averageNonlinearity = _numOps.Divide(sumNonlinearity, _numOps.FromDouble(nonlinearityScores.Count));

        if (_numOps.GreaterThan(averageNonlinearity, _numOps.FromDouble(_options.OverfitThreshold)))
        {
            return FitType.Overfit;
        }
        else if (_numOps.LessThan(averageNonlinearity, _numOps.FromDouble(_options.UnderfitThreshold)))
        {
            return FitType.Underfit;
        }
        else
        {
            return FitType.GoodFit;
        }
    }

    protected override T CalculateConfidenceLevel(ModelEvaluationData<T> evaluationData)
    {
        var pdpResults = CalculatePartialDependencePlots(evaluationData);
        var nonlinearityScores = CalculateNonlinearityScores(pdpResults);
        var sumNonlinearity = nonlinearityScores.Aggregate(_numOps.Zero, (acc, score) => _numOps.Add(acc, score));
        var averageNonlinearity = _numOps.Divide(sumNonlinearity, _numOps.FromDouble(nonlinearityScores.Count));

        // Normalize confidence level to be between 0 and 1
        return _numOps.Subtract(_numOps.One, 
            _numOps.Divide(
                averageNonlinearity, 
                _numOps.FromDouble(_options.OverfitThreshold)
            )
        );
    }

    private Dictionary<string, Vector<T>> CalculatePartialDependencePlots(ModelEvaluationData<T> evaluationData)
    {
        var pdpResults = new Dictionary<string, Vector<T>>();
        var features = evaluationData.ModelStats.FeatureNames;

        foreach (var feature in features)
        {
            var featureValues = evaluationData.ModelStats.FeatureValues[feature];
            var pdp = CalculatePartialDependencePlot(evaluationData, feature, featureValues);
            pdpResults[feature] = pdp;
        }

        return pdpResults;
    }

    private Vector<T> CalculatePartialDependencePlot(ModelEvaluationData<T> evaluationData, string feature, Vector<T> featureValues)
    {
        var uniqueValues = featureValues.Distinct().OrderBy(v => v).ToList();
        var pdp = new Vector<T>(uniqueValues.Count);

        for (int i = 0; i < uniqueValues.Count; i++)
        {
            var value = uniqueValues[i];
            var modifiedData = CreateModifiedDataset(evaluationData, feature, value);
            var predictions = evaluationData.ModelStats.Model?.Predict(modifiedData) ?? Vector<T>.Empty();
            pdp[i] = predictions.Average();
        }

        return pdp;
    }

    private Matrix<T> CreateModifiedDataset(ModelEvaluationData<T> evaluationData, string feature, T value)
    {
        var modifiedData = evaluationData.ModelStats.FeatureMatrix.Copy();
        var featureIndex = evaluationData.ModelStats.FeatureNames.IndexOf(feature);

        for (int i = 0; i < modifiedData.Rows; i++)
        {
            modifiedData[i, featureIndex] = value;
        }

        return modifiedData;
    }

    private List<T> CalculateNonlinearityScores(Dictionary<string, Vector<T>> pdpResults)
    {
        var nonlinearityScores = new List<T>();

        foreach (var pdp in pdpResults.Values)
        {
            var nonlinearity = CalculateNonlinearity(pdp);
            nonlinearityScores.Add(nonlinearity);
        }

        return nonlinearityScores;
    }

    private T CalculateNonlinearity(Vector<T> pdp)
    {
        var differences = new Vector<T>(pdp.Length - 1);
        for (int i = 0; i < pdp.Length - 1; i++)
        {
            differences[i] = _numOps.Abs(_numOps.Subtract(pdp[i + 1], pdp[i]));
        }

        return StatisticsHelper<T>.CalculateStandardDeviation(differences);
    }

    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T> evaluationData)
    {
        var pdpResults = CalculatePartialDependencePlots(evaluationData);
        return GenerateRecommendations(fitType, pdpResults);
    }

    private List<string> GenerateRecommendations(FitType fitType, Dictionary<string, Vector<T>> pdpResults)
    {
        var recommendations = new List<string>();
        var nonlinearityScores = CalculateNonlinearityScores(pdpResults);
        var sortedFeatures = pdpResults.Keys.OrderByDescending(f => nonlinearityScores[pdpResults.Keys.ToList().IndexOf(f)]).ToList();

        switch (fitType)
        {
            case FitType.Overfit:
                recommendations.Add("The model may be overfitting. Consider the following:");
                recommendations.Add("1. Increase regularization strength.");
                recommendations.Add("2. Reduce model complexity.");
                recommendations.Add("3. Gather more training data.");
                recommendations.Add("4. Consider simplifying or removing highly nonlinear features:");
                for (int i = 0; i < Math.Min(5, sortedFeatures.Count); i++)
                {
                    recommendations.Add($"   - {sortedFeatures[i]}");
                }
                break;

            case FitType.Underfit:
                recommendations.Add("The model may be underfitting. Consider the following:");
                recommendations.Add("1. Increase model complexity.");
                recommendations.Add("2. Reduce regularization strength.");
                recommendations.Add("3. Add more relevant features or engineer new features.");
                recommendations.Add("4. Consider adding nonlinear transformations for these features:");
                for (int i = sortedFeatures.Count - 1; i >= Math.Max(0, sortedFeatures.Count - 5); i--)
                {
                    recommendations.Add($"   - {sortedFeatures[i]}");
                }
                break;

            case FitType.GoodFit:
                recommendations.Add("The model appears to have a good fit. Consider the following:");
                recommendations.Add("1. Fine-tune hyperparameters for potential improvements.");
                recommendations.Add("2. Validate the model on new, unseen data.");
                recommendations.Add("3. Monitor model performance over time for potential drift.");
                break;
        }

        recommendations.Add("Top 5 most nonlinear features:");
        for (int i = 0; i < Math.Min(5, sortedFeatures.Count); i++)
        {
            recommendations.Add($"   - {sortedFeatures[i]}");
        }

        return recommendations;
    }
}
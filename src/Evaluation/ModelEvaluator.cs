namespace AiDotNet.Evaluation;

public class ModelEvaluator<T> : IModelEvaluator<T>
{
    private readonly INumericOperations<T> _numOps;
    protected readonly PredictionStatsOptions _predictionOptions;

    public ModelEvaluator(PredictionStatsOptions? predictionOptions = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _predictionOptions = predictionOptions ?? new PredictionStatsOptions();
    }

    public ModelEvaluationData<T> EvaluateModel(ModelEvaluationInput<T> input)
    {
        var model = input.Model ?? new VectorModel<T>(Vector<T>.Empty());

        var evaluationData = new ModelEvaluationData<T>
        {
            TrainingSet = CalculateDataSetStats(input.InputData.XTrain, input.InputData.YTrain, model),
            ValidationSet = CalculateDataSetStats(input.InputData.XVal, input.InputData.YVal, model),
            TestSet = CalculateDataSetStats(input.InputData.XTest, input.InputData.YTest, model),
            ModelStats = CalculateModelStats(model, input.InputData.XTrain, input.NormInfo)
        };

        return evaluationData;
    }

    private static Vector<T> PredictWithSymbolicModel(ISymbolicModel<T> model, Matrix<T> X)
    {
        var predictions = new Vector<T>(X.Rows);
        for (int i = 0; i < X.Rows; i++)
        {
            predictions[i] = model.Evaluate(X.GetRow(i));
        }

        return predictions;
    }

    private DataSetStats<T> CalculateDataSetStats(Matrix<T> X, Vector<T> y, ISymbolicModel<T> model)
    {
        var predictions = PredictWithSymbolicModel(model, X);

        return new DataSetStats<T>
        {
            ErrorStats = CalculateErrorStats(y, predictions, X.Columns),
            ActualBasicStats = CalculateBasicStats(y),
            PredictedBasicStats = CalculateBasicStats(predictions),
            PredictionStats = CalculatePredictionStats(y, predictions, X.Columns),
            Predictions = predictions
        };
    }

    private static ErrorStats<T> CalculateErrorStats(Vector<T> actual, Vector<T> predicted, int featureCount)
    {
        return new ErrorStats<T>(new ErrorStatsInputs<T> { Actual = actual, Predicted = predicted, FeatureCount = featureCount });
    }

    private static BasicStats<T> CalculateBasicStats(Vector<T> values)
    {
        return new BasicStats<T>(new BasicStatsInputs<T> { Values = values });
    }

    private PredictionStats<T> CalculatePredictionStats(Vector<T> actual, Vector<T> predicted, int featureCount)
    {
        return new PredictionStats<T>(new PredictionStatsInputs<T> 
        { 
            Actual = actual, 
            Predicted = predicted, 
            NumberOfParameters = featureCount, 
            ConfidenceLevel = _predictionOptions.ConfidenceLevel, 
            LearningCurveSteps = _predictionOptions.LearningCurveSteps 
        });
    }

    private static ModelStats<T> CalculateModelStats(ISymbolicModel<T> model, Matrix<T> xTrain, NormalizationInfo<T> normInfo)
    {
        var predictionModelResult = new PredictionModelResult<T>(model, new OptimizationResult<T>(), normInfo);

        return new ModelStats<T>(new ModelStatsInputs<T>
        {
            XMatrix = xTrain,
            FeatureCount = xTrain.Columns,
            Model = predictionModelResult
        });
    }
}
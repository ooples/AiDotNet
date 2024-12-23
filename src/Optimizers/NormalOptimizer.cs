using AiDotNet.Models.Options;
using AiDotNet.Models.Results;

namespace AiDotNet.Optimizers;

public class NormalOptimizer<T> : OptimizerBase<T>
{
    private readonly Random _random = new();
    private readonly OptimizationAlgorithmOptions _optimizationOptions;

    public NormalOptimizer(OptimizationAlgorithmOptions? optimizationOptions = null) : base(optimizationOptions)
    {
        _optimizationOptions = optimizationOptions ?? new OptimizationAlgorithmOptions();
    }

    public override OptimizationResult<T> Optimize(
        Matrix<T> XTrain,
        Vector<T> yTrain,
        Matrix<T> XVal,
        Vector<T> yVal,
        Matrix<T> XTest,
        Vector<T> yTest,
        IRegression<T> regressionMethod,
        IRegularization<T> regularization,
        INormalizer<T> normalizer,
        NormalizationInfo<T> normInfo,
        IFitnessCalculator<T> fitnessCalculator,
        IFitDetector<T> fitDetector)
    {
        var bestSolution = new Vector<T>(XTrain.Columns, _numOps);
        var bestIntercept = _numOps.Zero;
        T bestFitness = fitnessCalculator.IsHigherScoreBetter ? _numOps.MinValue : _numOps.MaxValue;
        var fitnessHistory = new List<T>();
        var iterationHistory = new List<OptimizationIterationInfo<T>>();
        var bestSelectedFeatures = new List<Vector<T>>();
        var bestFitDetectionResult = new FitDetectorResult<T>();
        var bestTrainingPredictions = new Vector<T>(yTrain.Length, _numOps);
        var bestValidationPredictions = new Vector<T>(yVal.Length, _numOps);
        var bestTestPredictions = new Vector<T>(yTest.Length, _numOps);
        var bestTrainingFeatures = new Matrix<T>(XTrain.Rows, 0, _numOps);
        var bestValidationFeatures = new Matrix<T>(XVal.Rows, 0, _numOps);
        var bestTestFeatures = new Matrix<T>(XTest.Rows, 0, _numOps);
        var bestEvaluationData = new ModelEvaluationData<T>();

        for (int iteration = 0; iteration < _optimizationOptions.MaxIterations; iteration++)
        {
            // Randomly select features
            var selectedFeatures = RandomlySelectFeatures(XTrain.Columns, _options.MinimumFeatures, _options.MaximumFeatures);

            // Create subsets of the data with selected features
            var XTrainSubset = XTrain.SubMatrix(0, XTrain.Rows - 1, selectedFeatures);
            var XValSubset = XVal.SubMatrix(0, XVal.Rows - 1, selectedFeatures);
            var XTestSubset = XTest.SubMatrix(0, XTest.Rows - 1, selectedFeatures);
            var featureCount = selectedFeatures.Count + (regressionMethod.HasIntercept ? 1 : 0);

            var (currentFitnessScore, fitDetectionResult, 
             trainingPredictions, validationPredictions, testPredictions,
             evaluationData) = EvaluateSolution(XTrainSubset, XValSubset, XTestSubset, 
                             yTrain, yVal, yTest, 
                             regressionMethod, normalizer, normInfo, 
                             fitnessCalculator, fitDetector, featureCount);

            UpdateBestSolution(
                currentFitnessScore, regressionMethod.Coefficients, regressionMethod.Intercept,
                fitDetectionResult, trainingPredictions, validationPredictions, testPredictions, evaluationData,
                selectedFeatures, XTrain, XTestSubset, XTrainSubset, XValSubset, fitnessCalculator,
                ref bestFitness, ref bestSolution, ref bestIntercept, ref bestFitDetectionResult,
                ref bestTrainingPredictions, ref bestValidationPredictions, ref bestTestPredictions, ref bestEvaluationData, 
                ref bestSelectedFeatures, ref bestTestFeatures, ref bestTrainingFeatures, ref bestValidationFeatures);

            if (UpdateIterationHistoryAndCheckEarlyStopping(fitnessHistory, iterationHistory, iteration, bestFitness, bestFitDetectionResult, fitnessCalculator))
            {
                break; // Early stopping criteria met, exit the loop
            }
        }

        var trainingResult = OptimizerHelper.CreateDatasetResult(
            bestTrainingPredictions, bestEvaluationData.TrainingErrorStats, bestEvaluationData.TrainingActualBasicStats,
            bestEvaluationData.TrainingPredictedBasicStats, bestEvaluationData.TrainingPredictionStats, bestTrainingFeatures, yTrain);

        var validationResult = OptimizerHelper.CreateDatasetResult(
            bestValidationPredictions, bestEvaluationData.ValidationErrorStats, bestEvaluationData.ValidationActualBasicStats,
            bestEvaluationData.ValidationPredictedBasicStats, bestEvaluationData.ValidationPredictionStats, bestValidationFeatures, yVal);

        var testResult = OptimizerHelper.CreateDatasetResult(
            bestTestPredictions, bestEvaluationData.TestErrorStats, bestEvaluationData.TestActualBasicStats,
            bestEvaluationData.TestPredictedBasicStats, bestEvaluationData.TestPredictionStats, bestTestFeatures, yTest);

        return OptimizerHelper.CreateOptimizationResult(
            bestSolution, bestIntercept, bestFitness, fitnessHistory, bestSelectedFeatures,
            trainingResult, validationResult, testResult, bestFitDetectionResult ?? new FitDetectorResult<T>(),
            iterationHistory.Count, _numOps);
    }

    private List<int> RandomlySelectFeatures(int totalFeatures, int minFeatures, int maxFeatures)
    {
        int numFeatures = _random.Next(minFeatures, Math.Min(maxFeatures, totalFeatures) + 1);
        return [.. Enumerable.Range(0, totalFeatures).OrderBy(x => _random.Next()).Take(numFeatures)];
    }
}
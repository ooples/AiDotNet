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
    IFullModel<T> regressionMethod,
    IRegularization<T> regularization,
    INormalizer<T> normalizer,
    NormalizationInfo<T> normInfo,
    IFitnessCalculator<T> fitnessCalculator,
    IFitDetector<T> fitDetector)
{
    var bestSolution = SymbolicModelFactory<T>.CreateEmptyModel(_options.UseExpressionTrees, XTrain.Columns, _numOps);
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
        var selectedFeatures = RandomlySelectFeatures(XTrain.Columns, _options.MinimumFeatures, _options.MaximumFeatures);
        var XTrainSubset = XTrain.SubMatrix(0, XTrain.Rows - 1, selectedFeatures);
        var XValSubset = XVal.SubMatrix(0, XVal.Rows - 1, selectedFeatures);
        var XTestSubset = XTest.SubMatrix(0, XTest.Rows - 1, selectedFeatures);

        var currentSolution = SymbolicModelFactory<T>.CreateRandomModel(_options.UseExpressionTrees, selectedFeatures.Count, _numOps);

        var (currentFitnessScore, fitDetectionResult, 
            trainingPredictions, validationPredictions, testPredictions,
            evaluationData) = EvaluateSolution(currentSolution,
                XTrainSubset, XValSubset, XTestSubset, yTrain, yVal, yTest,
                normalizer, normInfo, 
                fitnessCalculator, fitDetector);

        var currentResult = new ModelResult<T>
        {
            Solution = currentSolution,
            Fitness = currentFitnessScore,
            FitDetectionResult = fitDetectionResult,
            TrainingPredictions = trainingPredictions,
            ValidationPredictions = validationPredictions,
            TestPredictions = testPredictions,
            EvaluationData = evaluationData,
            SelectedFeatures = selectedFeatures.ToVectorList<T>()
        };

        var bestResult = new ModelResult<T>
        {
            Solution = bestSolution,
            Fitness = bestFitness,
            FitDetectionResult = bestFitDetectionResult,
            TrainingPredictions = bestTrainingPredictions,
            ValidationPredictions = bestValidationPredictions,
            TestPredictions = bestTestPredictions,
            EvaluationData = bestEvaluationData,
            SelectedFeatures = bestSelectedFeatures
        };

        OptimizerHelper.UpdateAndApplyBestSolution(
            currentResult,
            ref bestResult,
            XTrainSubset,
            XTestSubset,
            XValSubset,
            fitnessCalculator
        );

        bestSolution = bestResult.Solution;
        bestFitness = bestResult.Fitness;
        bestFitDetectionResult = bestResult.FitDetectionResult;
        bestTrainingPredictions = bestResult.TrainingPredictions;
        bestValidationPredictions = bestResult.ValidationPredictions;
        bestTestPredictions = bestResult.TestPredictions;
        bestEvaluationData = bestResult.EvaluationData;
        bestSelectedFeatures = bestResult.SelectedFeatures;

        if (UpdateIterationHistoryAndCheckEarlyStopping(fitnessHistory, iterationHistory, iteration, bestFitness, bestFitDetectionResult, fitnessCalculator))
        {
            break;
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

    return new OptimizationResult<T>
    {
        BestSolution = bestSolution,
        BestIntercept = bestIntercept,
        BestFitnessScore = bestFitness,
        FitnessHistory = new Vector<T>(fitnessHistory),
        SelectedFeatures = bestSelectedFeatures,
        TrainingResult = trainingResult,
        ValidationResult = validationResult,
        TestResult = testResult,
        FitDetectionResult = bestFitDetectionResult,
        Iterations = iterationHistory.Count
    };
}

    private List<int> RandomlySelectFeatures(int totalFeatures, int minFeatures, int maxFeatures)
    {
        int numFeatures = _random.Next(minFeatures, Math.Min(maxFeatures, totalFeatures) + 1);
        return [.. Enumerable.Range(0, totalFeatures).OrderBy(x => _random.Next()).Take(numFeatures)];
    }
}
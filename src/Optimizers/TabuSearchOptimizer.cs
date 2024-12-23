using AiDotNet.Models.Results;

namespace AiDotNet.Optimizers;

public class TabuSearchOptimizer<T> : OptimizerBase<T>
{
    private readonly Random _random;
    private readonly TabuSearchOptions _tabuOptions;

    public TabuSearchOptimizer(TabuSearchOptions? options = null) : base(options)
    {
        _random = new Random();
        _tabuOptions = options ?? new TabuSearchOptions();
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
        var currentSolution = InitializeRandomSolution(XTrain.Columns);
        var bestSolution = currentSolution.Copy();
        var tabuList = new Queue<Vector<T>>(_tabuOptions.TabuListSize);
        T bestIntercept = _numOps.Zero;
        T bestFitness = fitnessCalculator.IsHigherScoreBetter ? _numOps.MinValue : _numOps.MaxValue;
        FitDetectorResult<T> bestFitDetectionResult = new();
        Vector<T> bestTrainingPredictions = Vector<T>.Empty();
        Vector<T> bestValidationPredictions = Vector<T>.Empty();
        Vector<T> bestTestPredictions = Vector<T>.Empty();
        ModelEvaluationData<T> bestEvaluationData = new();
        List<Vector<T>> bestSelectedFeatures = [];
        Matrix<T> bestTestFeatures = Matrix<T>.Empty();
        Matrix<T> bestTrainingFeatures = Matrix<T>.Empty();
        Matrix<T> bestValidationFeatures = Matrix<T>.Empty();

        var fitnessHistory = new List<T>();
        var iterationHistory = new List<OptimizationIterationInfo<T>>();

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            var neighbors = GenerateNeighbors(currentSolution);
            var bestNeighbor = neighbors
                .Where(n => !IsTabu(n, tabuList))
                .OrderByDescending(n => EvaluateSolutionFitness(n, XTrain, yTrain, XVal, yVal, XTest, yTest, regressionMethod, regularization, normalizer, normInfo, fitnessCalculator, fitDetector))
                .FirstOrDefault() ?? neighbors.First();

            currentSolution = bestNeighbor;

            var selectedFeatures = OptimizerHelper.GetSelectedFeatures(currentSolution);
            var XTrainSubset = OptimizerHelper.SelectFeatures(XTrain, selectedFeatures);
            var XValSubset = OptimizerHelper.SelectFeatures(XVal, selectedFeatures);
            var XTestSubset = OptimizerHelper.SelectFeatures(XTest, selectedFeatures);

            var (currentFitnessScore, fitDetectionResult, trainingPredictions, validationPredictions, testPredictions, evaluationData) = EvaluateSolution(
                        XTrainSubset, XValSubset, XTestSubset,
                        yTrain, yVal, yTest,
                        regressionMethod, normalizer, normInfo,
                        fitnessCalculator, fitDetector, selectedFeatures.Count);

            UpdateBestSolution(
                    currentFitnessScore, currentSolution, _numOps.Zero, fitDetectionResult,
                    trainingPredictions, validationPredictions, testPredictions, evaluationData,
                    selectedFeatures, XTrain, XTestSubset, XTrainSubset, XValSubset, fitnessCalculator,
                    ref bestFitness, ref bestSolution, ref bestIntercept, ref bestFitDetectionResult,
                    ref bestTrainingPredictions, ref bestValidationPredictions, ref bestTestPredictions, ref bestEvaluationData,
                    ref bestSelectedFeatures, ref bestTestFeatures, ref bestTrainingFeatures, ref bestValidationFeatures);

            UpdateTabuList(tabuList, currentSolution);

            if (UpdateIterationHistoryAndCheckEarlyStopping(fitnessHistory, iterationHistory, iteration, bestFitness, bestFitDetectionResult, fitnessCalculator))
            {
                break; // Early stopping criteria met, exit the loop
            }
        }

        return OptimizerHelper.CreateOptimizationResult(
            bestSolution,
            bestIntercept,
            bestFitness,
            fitnessHistory,
            bestSelectedFeatures,
            new OptimizationResult<T>.DatasetResult
            {
                X = bestTrainingFeatures,
                Y = yTrain,
                Predictions = bestTrainingPredictions,
                ErrorStats = bestEvaluationData.TrainingErrorStats,
                ActualBasicStats = bestEvaluationData.TrainingActualBasicStats,
                PredictedBasicStats = bestEvaluationData.TrainingPredictedBasicStats,
                PredictionStats = bestEvaluationData.TrainingPredictionStats
            },
            new OptimizationResult<T>.DatasetResult
            {
                X = bestValidationFeatures,
                Y = yVal,
                Predictions = bestValidationPredictions,
                ErrorStats = bestEvaluationData.ValidationErrorStats,
                ActualBasicStats = bestEvaluationData.ValidationActualBasicStats,
                PredictedBasicStats = bestEvaluationData.ValidationPredictedBasicStats,
                PredictionStats = bestEvaluationData.ValidationPredictionStats
            },
            new OptimizationResult<T>.DatasetResult
            {
                X = bestTestFeatures,
                Y = yTest,
                Predictions = bestTestPredictions,
                ErrorStats = bestEvaluationData.TestErrorStats,
                ActualBasicStats = bestEvaluationData.TestActualBasicStats,
                PredictedBasicStats = bestEvaluationData.TestPredictedBasicStats,
                PredictionStats = bestEvaluationData.TestPredictionStats
            },
            bestFitDetectionResult,
            fitnessHistory.Count,
            _numOps);
    }

    private Vector<T> InitializeRandomSolution(int dimensions)
    {
        var solution = new Vector<T>(dimensions);
        for (int i = 0; i < dimensions; i++)
        {
            solution[i] = _numOps.FromDouble(_random.NextDouble());
        }
        return solution;
    }

    private List<Vector<T>> GenerateNeighbors(Vector<T> currentSolution)
    {
        var neighbors = new List<Vector<T>>();
        for (int i = 0; i < _tabuOptions.NeighborhoodSize; i++)
        {
            var neighbor = currentSolution.Copy();
            int index = _random.Next(neighbor.Length);
            neighbor[index] = _numOps.Add(neighbor[index], _numOps.FromDouble(_random.NextDouble() * _tabuOptions.PerturbationFactor - _tabuOptions.PerturbationFactor / 2));
            neighbors.Add(neighbor);
        }
        return neighbors;
    }

    private bool IsTabu(Vector<T> solution, Queue<Vector<T>> tabuList)
    {
        return tabuList.Any(tabuSolution => tabuSolution.Equals(solution));
    }

    private void UpdateTabuList(Queue<Vector<T>> tabuList, Vector<T> solution)
    {
        if (tabuList.Count >= _tabuOptions.TabuListSize)
        {
            tabuList.Dequeue();
        }
        tabuList.Enqueue(solution);
    }

    private T EvaluateSolutionFitness(Vector<T> solution, Matrix<T> XTrain, Vector<T> yTrain, Matrix<T> XVal, Vector<T> yVal, Matrix<T> XTest, Vector<T> yTest,
        IRegression<T> regressionMethod, IRegularization<T> regularization, INormalizer<T> normalizer, NormalizationInfo<T> normInfo,
        IFitnessCalculator<T> fitnessCalculator, IFitDetector<T> fitDetector)
    {
        var selectedFeatures = OptimizerHelper.GetSelectedFeatures(solution);
        var XTrainSubset = OptimizerHelper.SelectFeatures(XTrain, selectedFeatures);
        var XValSubset = OptimizerHelper.SelectFeatures(XVal, selectedFeatures);
        var XTestSubset = OptimizerHelper.SelectFeatures(XTest, selectedFeatures);

        var (currentFitnessScore, fitDetectionResult, trainingPredictions, validationPredictions, testPredictions, evaluationData) = EvaluateSolution(
                        XTrainSubset, XValSubset, XTestSubset,
                        yTrain, yVal, yTest,
                        regressionMethod, normalizer, normInfo,
                        fitnessCalculator, fitDetector, selectedFeatures.Count);

        return currentFitnessScore;
    }
}
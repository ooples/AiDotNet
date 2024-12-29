namespace AiDotNet.Optimizers;

public class GradientDescentOptimizer<T> : OptimizerBase<T>
{
    private readonly GradientDescentOptimizerOptions _gdOptions;

    public GradientDescentOptimizer(GradientDescentOptimizerOptions? options = null)
        : base(options)
    {
        _gdOptions = options ?? new GradientDescentOptimizerOptions();
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
        var currentSolution = InitializeRandomSolution(XTrain.Columns);
        T bestFitness = fitnessCalculator.IsHigherScoreBetter ? _numOps.MinValue : _numOps.MaxValue;
        var bestSolution = SymbolicModelFactory<T>.CreateEmptyModel(_options.UseExpressionTrees, XTrain.Columns, _numOps);
        List<T> fitnessHistory = [];
        var bestIntercept = _numOps.Zero;
        var iterationHistory = new List<OptimizationIterationInfo<T>>();

        var bestFitDetectionResult = new FitDetectorResult<T>();
        var bestTrainingPredictions = Vector<T>.Empty();
        var bestValidationPredictions = Vector<T>.Empty();
        var bestTestPredictions = Vector<T>.Empty();
        var bestEvaluationData = new ModelEvaluationData<T>();
        var bestSelectedFeatures = new List<Vector<T>>();
        var bestTestFeatures = Matrix<T>.Empty();
        var bestTrainingFeatures = Matrix<T>.Empty();
        var bestValidationFeatures = Matrix<T>.Empty();

        for (int iteration = 0; iteration < _gdOptions.MaxIterations; iteration++)
        {
            var gradient = CalculateGradient(currentSolution, XTrain, yTrain, regressionMethod, regularization);
            var newSolution = UpdateSolution(currentSolution, gradient);

            var selectedFeatures = OptimizerHelper.GetSelectedFeatures(currentSolution);
            var XTrainSubset = OptimizerHelper.SelectFeatures(XTrain, selectedFeatures);
            var XValSubset = OptimizerHelper.SelectFeatures(XVal, selectedFeatures);
            var XTestSubset = OptimizerHelper.SelectFeatures(XTest, selectedFeatures);

            var (currentFitness, fitDetectionResult, trainingPredictions, validationPredictions, testPredictions, evaluationData) = 
                EvaluateSolution(currentSolution,
                    XTrainSubset, XValSubset, XTestSubset,
                    yTrain, yVal, yTest,
                    normalizer, normInfo,
                    fitnessCalculator, fitDetector
                );

            var currentResult = new ModelResult<T>
            {
                Solution = newSolution,
                Fitness = currentFitness,
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

            fitnessHistory.Add(bestFitness);
            iterationHistory.Add(new OptimizationIterationInfo<T> { Iteration = iteration, Fitness = bestFitness, FitDetectionResult = bestFitDetectionResult });

            if (ShouldEarlyStop(iterationHistory, fitnessCalculator))
            {
                break;
            }

            if (_numOps.LessThan(_numOps.Abs(_numOps.Subtract(bestFitness, currentFitness)), _numOps.FromDouble(_gdOptions.Tolerance)))
            {
                break;
            }
        }

        return OptimizerHelper.CreateOptimizationResult(
            bestSolution,
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
            iterationHistory.Count,
            _numOps
        );
    }

    private ISymbolicModel<T> InitializeRandomSolution(int dimensions)
    {
        return SymbolicModelFactory<T>.CreateRandomModel(_options.UseExpressionTrees, dimensions, _numOps);
    }

    private Vector<T> CalculateGradient(ISymbolicModel<T> solution, Matrix<T> X, Vector<T> y, 
                                    IFullModel<T> regressionMethod, IRegularization<T> regularization)
    {
        Vector<T> gradient = new(solution.Coefficients.Length, _numOps);
        T epsilon = _numOps.FromDouble(1e-8);

        for (int i = 0; i < solution.Coefficients.Length; i++)
        {
            Vector<T> perturbedCoefficientsPlus = solution.Coefficients.Copy();
            perturbedCoefficientsPlus[i] = _numOps.Add(perturbedCoefficientsPlus[i], epsilon);

            Vector<T> perturbedCoefficientsMinus = solution.Coefficients.Copy();
            perturbedCoefficientsMinus[i] = _numOps.Subtract(perturbedCoefficientsMinus[i], epsilon);

            T lossPlus = CalculateLoss(solution.UpdateCoefficients(perturbedCoefficientsPlus), X, y, regressionMethod, regularization);
            T lossMinus = CalculateLoss(solution.UpdateCoefficients(perturbedCoefficientsMinus), X, y, regressionMethod, regularization);

            gradient[i] = _numOps.Divide(_numOps.Subtract(lossPlus, lossMinus), _numOps.Multiply(_numOps.FromDouble(2.0), epsilon));
        }

        return gradient;
    }

    private T CalculateLoss(ISymbolicModel<T> solution, Matrix<T> X, Vector<T> y, 
                        IFullModel<T> regressionMethod, IRegularization<T> regularization)
    {
        Vector<T> predictions = new Vector<T>(X.Rows, _numOps);
        for (int i = 0; i < X.Rows; i++)
        {
            predictions[i] = solution.Evaluate(X.GetRow(i));
        }

        T mse = StatisticsHelper<T>.CalculateMeanSquaredError(predictions, y);
        Vector<T> regularizedCoefficients = regularization.RegularizeCoefficients(solution.Coefficients);
        T regularizationTerm = regularizedCoefficients.Subtract(solution.Coefficients).Transform(_numOps.Abs).Sum();

        return _numOps.Add(mse, regularizationTerm);
    }

    private ISymbolicModel<T> UpdateSolution(ISymbolicModel<T> currentSolution, Vector<T> gradient)
    {
        Vector<T> updatedCoefficients = currentSolution.Coefficients.Subtract(gradient.Multiply(_numOps.FromDouble(_gdOptions.LearningRate)));
        return currentSolution.UpdateCoefficients(updatedCoefficients);
    }
}
using AiDotNet.Models.Results;

namespace AiDotNet.Optimizers;

public class SimulatedAnnealingOptimizer<T> : OptimizerBase<T>
    {
        private readonly Random _random;
        private readonly SimulatedAnnealingOptions _saOptions;

        public SimulatedAnnealingOptimizer(SimulatedAnnealingOptions? options = null) 
            : base(options)
        {
            _random = new Random();
            _saOptions = options ?? new SimulatedAnnealingOptions();
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
            int dimensions = XTrain.Columns;
            var currentSolution = InitializeRandomSolution(dimensions);
            var bestSolution = currentSolution.Copy();
            T temperature = _numOps.FromDouble(_saOptions.InitialTemperature);
            T bestIntercept = _numOps.Zero;
            T bestFitness = fitnessCalculator.IsHigherScoreBetter ? _numOps.MinValue : _numOps.MaxValue;
            FitDetectorResult<T> bestFitDetectionResult = new();
            Vector<T> bestTrainingPredictions = new(yTrain.Length, _numOps);
            Vector<T> bestValidationPredictions = new(yVal.Length, _numOps);
            Vector<T> bestTestPredictions = new(yTest.Length, _numOps);
            ModelEvaluationData<T> bestEvaluationData = new();
            List<Vector<T>> bestSelectedFeatures = [];
            Matrix<T> bestTestFeatures = new(XTest.Rows, XTest.Columns, _numOps);
            Matrix<T> bestTrainingFeatures = new(XTrain.Rows, XTrain.Columns, _numOps);
            Matrix<T> bestValidationFeatures = new(XVal.Rows, XVal.Columns, _numOps);

            var fitnessHistory = new List<T>();
            var iterationHistory = new List<OptimizationIterationInfo<T>>();
            var solutions = new List<Vector<T>>();

            for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
            {
                var newSolution = PerturbSolution(currentSolution);
                var selectedFeatures = OptimizerHelper.GetSelectedFeatures(newSolution);
                var XTrainSubset = OptimizerHelper.SelectFeatures(XTrain, selectedFeatures);
                var XValSubset = OptimizerHelper.SelectFeatures(XVal, selectedFeatures);
                var XTestSubset = OptimizerHelper.SelectFeatures(XTest, selectedFeatures);

                var (currentFitnessScore, fitDetectionResult, 
                     trainingPredictions, validationPredictions, testPredictions, evaluationData) = EvaluateSolution(
                        XTrainSubset, XValSubset, XTestSubset,
                        yTrain, yVal, yTest,
                        regressionMethod, normalizer, normInfo,
                        fitnessCalculator, fitDetector, selectedFeatures.Count);

                if (AcceptSolution(currentFitnessScore, bestFitness, temperature))
                {
                    UpdateBestSolution(
                        currentFitnessScore,
                        newSolution,
                        regressionMethod.Intercept,
                        fitDetectionResult,
                        trainingPredictions,
                        validationPredictions,
                        testPredictions,
                        evaluationData,
                        selectedFeatures,
                        XTrain,
                        XTestSubset,
                        XTrainSubset,
                        XValSubset,
                        fitnessCalculator,
                        ref bestFitness,
                        ref bestSolution,
                        ref bestIntercept,
                        ref bestFitDetectionResult,
                        ref bestTrainingPredictions,
                        ref bestValidationPredictions,
                        ref bestTestPredictions,
                        ref bestEvaluationData,
                        ref bestSelectedFeatures,
                        ref bestTestFeatures,
                        ref bestTrainingFeatures,
                        ref bestValidationFeatures);
                }

                temperature = _numOps.Multiply(temperature, _numOps.FromDouble(_saOptions.CoolingRate));

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
                iterationHistory.Count,
                _numOps
            );
        }

        private Vector<T> InitializeRandomSolution(int dimensions)
        {
            var solution = new Vector<T>(dimensions, _numOps);
            for (int i = 0; i < dimensions; i++)
            {
                solution[i] = _numOps.FromDouble(_random.NextDouble());
            }

            return solution;
        }

        private Vector<T> PerturbSolution(Vector<T> solution)
        {
            var newSolution = solution.Copy();
            int index = _random.Next(solution.Length);
            newSolution[index] = _numOps.FromDouble(_random.NextDouble());

            return newSolution;
        }

        private bool AcceptSolution(T newFitness, T currentFitness, T temperature)
        {
            if (_numOps.GreaterThan(newFitness, currentFitness))
            {
                return true;
            }

            T probability = _numOps.Exp(_numOps.Divide(_numOps.Subtract(newFitness, currentFitness), temperature));
            return _numOps.GreaterThan(_numOps.FromDouble(_random.NextDouble()), probability);
        }
    }
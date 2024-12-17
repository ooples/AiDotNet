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
            ErrorStats<T> bestTrainingErrorStats = ErrorStats<T>.Empty();
            ErrorStats<T> bestValidationErrorStats = ErrorStats<T>.Empty();
            ErrorStats<T> bestTestErrorStats = ErrorStats<T>.Empty();
            BasicStats<T> bestTrainingActualBasicStats = BasicStats<T>.Empty();
            BasicStats<T> bestTrainingPredictedBasicStats = BasicStats<T>.Empty();
            BasicStats<T> bestValidationActualBasicStats = BasicStats<T>.Empty();
            BasicStats<T> bestValidationPredictedBasicStats = BasicStats<T>.Empty();
            BasicStats<T> bestTestActualBasicStats = BasicStats<T>.Empty();
            BasicStats<T> bestTestPredictedBasicStats = BasicStats<T>.Empty();
            PredictionStats<T> bestTrainingPredictionStats = PredictionStats<T>.Empty();
            PredictionStats<T> bestValidationPredictionStats = PredictionStats<T>.Empty();
            PredictionStats<T> bestTestPredictionStats = PredictionStats<T>.Empty();
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
                     trainingPredictions, validationPredictions, testPredictions,
                     trainingErrorStats, validationErrorStats, testErrorStats,
                     trainingActualBasicStats, trainingPredictedBasicStats,
                     validationActualBasicStats, validationPredictedBasicStats,
                     testActualBasicStats, testPredictedBasicStats,
                     trainingPredictionStats, validationPredictionStats, testPredictionStats) = 
                    EvaluateSolution(
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
                        trainingErrorStats,
                        validationErrorStats,
                        testErrorStats,
                        trainingActualBasicStats,
                        trainingPredictedBasicStats,
                        validationActualBasicStats,
                        validationPredictedBasicStats,
                        testActualBasicStats,
                        testPredictedBasicStats,
                        trainingPredictionStats,
                        validationPredictionStats,
                        testPredictionStats,
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
                        ref bestTrainingErrorStats,
                        ref bestValidationErrorStats,
                        ref bestTestErrorStats,
                        ref bestTrainingActualBasicStats,
                        ref bestTrainingPredictedBasicStats,
                        ref bestValidationActualBasicStats,
                        ref bestValidationPredictedBasicStats,
                        ref bestTestActualBasicStats,
                        ref bestTestPredictedBasicStats,
                        ref bestTrainingPredictionStats,
                        ref bestValidationPredictionStats,
                        ref bestTestPredictionStats,
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
                    ErrorStats = bestTrainingErrorStats,
                    ActualBasicStats = bestTrainingActualBasicStats,
                    PredictedBasicStats = bestTrainingPredictedBasicStats,
                    PredictionStats = bestTrainingPredictionStats
                },
                new OptimizationResult<T>.DatasetResult
                {
                    X = bestValidationFeatures,
                    Y = yVal,
                    Predictions = bestValidationPredictions,
                    ErrorStats = bestValidationErrorStats,
                    ActualBasicStats = bestValidationActualBasicStats,
                    PredictedBasicStats = bestValidationPredictedBasicStats,
                    PredictionStats = bestValidationPredictionStats
                },
                new OptimizationResult<T>.DatasetResult
                {
                    X = bestTestFeatures,
                    Y = yTest,
                    Predictions = bestTestPredictions,
                    ErrorStats = bestTestErrorStats,
                    ActualBasicStats = bestTestActualBasicStats,
                    PredictedBasicStats = bestTestPredictedBasicStats,
                    PredictionStats = bestTestPredictionStats
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
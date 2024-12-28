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
            IFullModel<T> regressionMethod,
            IRegularization<T> regularization,
            INormalizer<T> normalizer,
            NormalizationInfo<T> normInfo,
            IFitnessCalculator<T> fitnessCalculator,
            IFitDetector<T> fitDetector)
        {
            int dimensions = XTrain.Columns;
            var currentSolution = InitializeRandomSolution(dimensions);
            var bestSolution = SymbolicModelFactory<T>.CreateEmptyModel(_options.UseExpressionTrees, XTrain.Columns, _numOps);
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
                        newSolution, XTrainSubset, XValSubset, XTestSubset,
                        yTrain, yVal, yTest, normalizer, normInfo,
                        fitnessCalculator, fitDetector);

                if (AcceptSolution(currentFitnessScore, bestFitness, temperature))
                {
                    int featureCount = selectedFeatures.Count;
                    Vector<T>? coefficients = null;
                    T? intercept = default;
                    bool hasIntercept = false;

                    if (regressionMethod is ILinearModel<T> linearModel)
                    {
                        hasIntercept = linearModel.HasIntercept;
                        coefficients = linearModel.Coefficients;
                        intercept = linearModel.Intercept;
                        featureCount += hasIntercept ? 1 : 0;
                    }

                    var currentResult = new ModelResult<T>
                    {
                        Solution = newSolution,
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
                }

                temperature = _numOps.Multiply(temperature, _numOps.FromDouble(_saOptions.CoolingRate));

                if (UpdateIterationHistoryAndCheckEarlyStopping(fitnessHistory, iterationHistory, iteration, bestFitness, bestFitDetectionResult, fitnessCalculator))
                {
                    break; // Early stopping criteria met, exit the loop
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
            if (_options.UseExpressionTrees)
            {
                return SymbolicModelFactory<T>.CreateRandomModel(true, dimensions, _numOps);
            }
            else
            {
                var solution = new Vector<T>(dimensions, _numOps);
                for (int i = 0; i < dimensions; i++)
                {
                    solution[i] = _numOps.FromDouble(_random.NextDouble());
                }
                return new VectorModel<T>(solution, _numOps);
            }
        }

        private ISymbolicModel<T> PerturbSolution(ISymbolicModel<T> solution)
        {
            if (solution is ExpressionTree<T> expressionTree)
            {
                return expressionTree.Mutate(_saOptions.MutationRate, _numOps);
            }
            else if (solution is VectorModel<T> vectorModel)
            {
                var newSolution = vectorModel.Coefficients.Copy();
                for (int i = 0; i < newSolution.Length; i++)
                {
                    if (_random.NextDouble() < _saOptions.MutationRate)
                    {
                        newSolution[i] = _numOps.FromDouble(_random.NextDouble() * 2 - 1); // Random value between -1 and 1
                    }
                }
                return new VectorModel<T>(newSolution, _numOps);
            }
            else
            {
                throw new ArgumentException("Unsupported model type");
            }
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
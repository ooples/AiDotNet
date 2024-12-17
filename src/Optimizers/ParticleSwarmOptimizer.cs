namespace AiDotNet.Optimizers;

public class ParticleSwarmOptimizer<T> : OptimizerBase<T>
{
    private readonly Random _random;
    private readonly ParticleSwarmOptimizationOptions _psoOptions;

    public ParticleSwarmOptimizer(ParticleSwarmOptimizationOptions? options = null)
        : base(options)
    {
        _random = new Random();
        _psoOptions = options ?? new ParticleSwarmOptimizationOptions();
    }

    public override OptimizationResult<T> Optimize(
        Matrix<T> XTrain,
        Vector<T> yTrain,
        Matrix<T> XVal,
        Vector<T> yVal,
        Matrix<T> XTest,
        Vector<T> yTest,
        PredictionModelOptions modelOptions,
        IRegression<T> regressionMethod,
        IRegularization<T> regularization,
        INormalizer<T> normalizer,
        NormalizationInfo<T> normInfo,
        IFitnessCalculator<T> fitnessCalculator,
        IFitDetector<T> fitDetector)
    {
        var swarm = InitializeSwarm(XTrain.Columns, _psoOptions.SwarmSize);
        var velocities = InitializeVelocities(XTrain.Columns, _psoOptions.SwarmSize);

        T bestFitness = _options.MaximizeFitness ? _numOps.MinValue : _numOps.MaxValue;
        Vector<T> bestSolution = new(XTrain.Columns, _numOps);
        T bestIntercept = _numOps.Zero;
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

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            for (int i = 0; i < _psoOptions.SwarmSize; i++)
            {
                var particle = swarm[i];
                var selectedFeatures = OptimizerHelper.GetSelectedFeatures(particle);
                var XTrainSubset = OptimizerHelper.SelectFeatures(XTrain, selectedFeatures);
                var XValSubset = OptimizerHelper.SelectFeatures(XVal, selectedFeatures);
                var XTestSubset = OptimizerHelper.SelectFeatures(XTest, selectedFeatures);

                var (currentFitnessScore, fitDetectionResult, trainingPredictions, validationPredictions, testPredictions,
                    trainingErrorStats, validationErrorStats, testErrorStats,
                    trainingActualBasicStats, trainingPredictedBasicStats,
                    validationActualBasicStats, validationPredictedBasicStats,
                    testActualBasicStats, testPredictedBasicStats,
                    trainingPredictionStats, validationPredictionStats, testPredictionStats) = EvaluateSolution(
                        XTrainSubset, XValSubset, XTestSubset,
                        yTrain, yVal, yTest,
                        regressionMethod, normalizer, normInfo,
                        fitnessCalculator, fitDetector, selectedFeatures.Count);

                UpdateBestSolution(
                    currentFitnessScore, particle, _numOps.Zero, fitDetectionResult,
                    trainingPredictions, validationPredictions, testPredictions,
                    trainingErrorStats, validationErrorStats, testErrorStats,
                    trainingActualBasicStats, trainingPredictedBasicStats,
                    validationActualBasicStats, validationPredictedBasicStats,
                    testActualBasicStats, testPredictedBasicStats,
                    trainingPredictionStats, validationPredictionStats, testPredictionStats,
                    selectedFeatures, XTrain, XTestSubset, XTrainSubset, XValSubset, fitnessCalculator,
                    ref bestFitness, ref bestSolution, ref bestIntercept, ref bestFitDetectionResult,
                    ref bestTrainingPredictions, ref bestValidationPredictions, ref bestTestPredictions,
                    ref bestTrainingErrorStats, ref bestValidationErrorStats, ref bestTestErrorStats,
                    ref bestTrainingActualBasicStats, ref bestTrainingPredictedBasicStats,
                    ref bestValidationActualBasicStats, ref bestValidationPredictedBasicStats,
                    ref bestTestActualBasicStats, ref bestTestPredictedBasicStats,
                    ref bestTrainingPredictionStats, ref bestValidationPredictionStats, ref bestTestPredictionStats,
                    ref bestSelectedFeatures, ref bestTestFeatures, ref bestTrainingFeatures, ref bestValidationFeatures);

                // Update velocity and position
                UpdateParticle(swarm[i], velocities[i], particle, bestSolution);
            }

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
            OptimizerHelper.CreateDatasetResult(bestTrainingPredictions, bestTrainingErrorStats, bestTrainingActualBasicStats, bestTrainingPredictedBasicStats, bestTrainingPredictionStats, bestTrainingFeatures, yTrain),
            OptimizerHelper.CreateDatasetResult(bestValidationPredictions, bestValidationErrorStats, bestValidationActualBasicStats, bestValidationPredictedBasicStats, bestValidationPredictionStats, bestValidationFeatures, yVal),
            OptimizerHelper.CreateDatasetResult(bestTestPredictions, bestTestErrorStats, bestTestActualBasicStats, bestTestPredictedBasicStats, bestTestPredictionStats, bestTestFeatures, yTest),
            bestFitDetectionResult,
            fitnessHistory.Count,
            _numOps);
    }

    private List<Vector<T>> InitializeSwarm(int dimensions, int swarmSize)
    {
        var swarm = new List<Vector<T>>();
        for (int i = 0; i < swarmSize; i++)
        {
            var particle = new Vector<T>(dimensions, _numOps);
            for (int j = 0; j < dimensions; j++)
            {
                particle[j] = _numOps.FromDouble(_random.NextDouble() * 2 - 1); // Random values between -1 and 1
            }
            swarm.Add(particle);
        }

        return swarm;
    }

    private List<Vector<T>> InitializeVelocities(int dimensions, int swarmSize)
    {
        var velocities = new List<Vector<T>>();
        for (int i = 0; i < swarmSize; i++)
        {
            var velocity = new Vector<T>(dimensions, _numOps);
            for (int j = 0; j < dimensions; j++)
            {
                velocity[j] = _numOps.FromDouble(_random.NextDouble() * 0.1 - 0.05); // Small random values
            }
            velocities.Add(velocity);
        }

        return velocities;
    }

    private void UpdateParticle(Vector<T> position, Vector<T> velocity, Vector<T> personalBest, Vector<T> globalBest)
    {
        for (int j = 0; j < position.Length; j++)
        {
            T r1 = _numOps.FromDouble(_random.NextDouble());
            T r2 = _numOps.FromDouble(_random.NextDouble());

            velocity[j] = _numOps.Add(
                _numOps.Multiply(_numOps.FromDouble(_psoOptions.InertiaWeight), velocity[j]),
                _numOps.Add(
                    _numOps.Multiply(_numOps.FromDouble(_psoOptions.CognitiveParameter), _numOps.Multiply(r1, _numOps.Subtract(personalBest[j], position[j]))),
                    _numOps.Multiply(_numOps.FromDouble(_psoOptions.SocialParameter), _numOps.Multiply(r2, _numOps.Subtract(globalBest[j], position[j])))
                )
            );

            position[j] = _numOps.Add(position[j], velocity[j]);
        }
    }
}
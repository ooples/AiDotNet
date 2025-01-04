global using AiDotNet.Models.Inputs;
global using AiDotNet.Evaluation;
global using AiDotNet.Caching;

namespace AiDotNet.Optimizers;

public abstract class OptimizerBase<T> : IOptimizer<T>
{
    protected readonly INumericOperations<T> NumOps;
    protected readonly OptimizationAlgorithmOptions Options;
    protected readonly PredictionStatsOptions _predictionOptions;
    protected readonly ModelStatsOptions _modelStatsOptions;
    protected readonly IModelEvaluator<T> _modelEvaluator;
    protected readonly IFitDetector<T> _fitDetector;
    protected readonly IFitnessCalculator<T> _fitnessCalculator;
    protected readonly List<T> _fitnessList;
    protected readonly List<OptimizationIterationInfo<T>> _iterationHistoryList;
    protected readonly IModelCache<T> ModelCache;
    protected T CurrentLearningRate;
    protected T CurrentMomentum;
    protected int IterationsWithoutImprovement;
    protected int IterationsWithImprovement;

    protected OptimizerBase(
        OptimizationAlgorithmOptions? options = null,
        PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null,
        IModelCache<T>? modelCache = null)
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        Options = options ?? new OptimizationAlgorithmOptions();
        _predictionOptions = predictionOptions ?? new PredictionStatsOptions();
        _modelStatsOptions = modelOptions ?? new ModelStatsOptions();
        _modelEvaluator = modelEvaluator ?? new ModelEvaluator<T>(_predictionOptions);
        _fitDetector = fitDetector ?? new DefaultFitDetector<T>();
        _fitnessCalculator = fitnessCalculator ?? new MeanSquaredErrorFitnessCalculator<T>();
        _fitnessList = [];
        _iterationHistoryList = [];
        ModelCache = modelCache ?? new DefaultModelCache<T>();
        CurrentLearningRate = NumOps.Zero;
        CurrentMomentum = NumOps.Zero;
    }

    protected ISymbolicModel<T> GetCachedSolution(string key)
    {
        return ModelCache.GetCachedModel(key);
    }

    protected void CacheSolution(string key, ISymbolicModel<T> solution)
    {
        ModelCache.CacheModel(key, solution);
    }

    public abstract OptimizationResult<T> Optimize(OptimizationInputData<T> inputData);

    public virtual Task<OptimizationResult<T>> OptimizeAsync(OptimizationInputData<T> inputData)
    {
        return Task.Run(() => Optimize(inputData));
    }

    protected OptimizationStepData<T> PrepareAndEvaluateSolution(ISymbolicModel<T> solution, OptimizationInputData<T> inputData)
    {
        var selectedFeatures = inputData.XTrain.GetColumnVectors(OptimizerHelper.GetSelectedFeatures(solution));
        var XTrainSubset = OptimizerHelper.SelectFeatures(inputData.XTrain, selectedFeatures);
        var XValSubset = OptimizerHelper.SelectFeatures(inputData.XVal, selectedFeatures);
        var XTestSubset = OptimizerHelper.SelectFeatures(inputData.XTest, selectedFeatures);

        var input = new ModelEvaluationInput<T>
        {
            InputData = inputData
        };

        var (currentFitnessScore, fitDetectionResult, evaluationData) = EvaluateSolution(input);
        _fitnessList.Add(currentFitnessScore);

        return new OptimizationStepData<T>
        {
            Solution = solution,
            SelectedFeatures = selectedFeatures,
            XTrainSubset = XTrainSubset,
            XValSubset = XValSubset,
            XTestSubset = XTestSubset,
            FitnessScore = currentFitnessScore,
            FitDetectionResult = fitDetectionResult,
            EvaluationData = evaluationData
        };
    }

    private (T CurrentFitnessScore, FitDetectorResult<T> FitDetectionResult, ModelEvaluationData<T> EvaluationData)
    EvaluateSolution(ModelEvaluationInput<T> input)
    {
        var evaluationData = _modelEvaluator.EvaluateModel(input);
        var fitDetectionResult = _fitDetector.DetectFit(evaluationData);
        var currentFitnessScore = _fitnessCalculator.CalculateFitnessScore(evaluationData);

        return (currentFitnessScore, fitDetectionResult, evaluationData);
    }

    protected OptimizationResult<T> CreateOptimizationResult(OptimizationStepData<T> bestStepData, OptimizationInputData<T> input)
    {
        return OptimizerHelper.CreateOptimizationResult(
            bestStepData.Solution,
            bestStepData.FitnessScore,
            _fitnessList,
            bestStepData.SelectedFeatures,
            new OptimizationResult<T>.DatasetResult
            {
                X = bestStepData.XTrainSubset,
                Y = input.YTrain,
                Predictions = bestStepData.EvaluationData.TrainingSet.Predictions,
                ErrorStats = bestStepData.EvaluationData.TrainingSet.ErrorStats,
                ActualBasicStats = bestStepData.EvaluationData.TrainingSet.ActualBasicStats,
                PredictedBasicStats = bestStepData.EvaluationData.TrainingSet.PredictedBasicStats,
                PredictionStats = bestStepData.EvaluationData.TrainingSet.PredictionStats
            },
            new OptimizationResult<T>.DatasetResult
            {
                X = bestStepData.XValSubset,
                Y = input.YVal,
                Predictions = bestStepData.EvaluationData.ValidationSet.Predictions,
                ErrorStats = bestStepData.EvaluationData.ValidationSet.ErrorStats,
                ActualBasicStats = bestStepData.EvaluationData.ValidationSet.ActualBasicStats,
                PredictedBasicStats = bestStepData.EvaluationData.ValidationSet.PredictedBasicStats,
                PredictionStats = bestStepData.EvaluationData.ValidationSet.PredictionStats
            },
            new OptimizationResult<T>.DatasetResult
            {
                X = bestStepData.XTestSubset,
                Y = input.YTest,
                Predictions = bestStepData.EvaluationData.TestSet.Predictions,
                ErrorStats = bestStepData.EvaluationData.TestSet.ErrorStats,
                ActualBasicStats = bestStepData.EvaluationData.TestSet.ActualBasicStats,
                PredictedBasicStats = bestStepData.EvaluationData.TestSet.PredictedBasicStats,
                PredictionStats = bestStepData.EvaluationData.TestSet.PredictionStats
            },
            bestStepData.FitDetectionResult,
            _iterationHistoryList.Count
        );
    }

    private void UpdateAndApplyBestSolution(ModelResult<T> currentResult, ref ModelResult<T> bestResult)
    {
        if (_fitnessCalculator.IsBetterFitness(currentResult.Fitness, bestResult.Fitness))
        {
            bestResult.Solution = currentResult.Solution;
            bestResult.Fitness = currentResult.Fitness;
            bestResult.FitDetectionResult = currentResult.FitDetectionResult;
            bestResult.EvaluationData = currentResult.EvaluationData;
            bestResult.SelectedFeatures = currentResult.SelectedFeatures;
        }
    }

    protected void UpdateBestSolution(OptimizationStepData<T> currentStepData, ref OptimizationStepData<T> bestStepData)
    {
        var currentResult = new ModelResult<T>
        {
            Solution = currentStepData.Solution ?? new VectorModel<T>(Vector<T>.Empty()),
            Fitness = currentStepData.FitnessScore,
            FitDetectionResult = currentStepData.FitDetectionResult,
            EvaluationData = currentStepData.EvaluationData,
            SelectedFeatures = currentStepData.SelectedFeatures
        };

        var bestResult = new ModelResult<T>
        {
            Solution = bestStepData.Solution ?? new VectorModel<T>(Vector<T>.Empty()),
            Fitness = bestStepData.FitnessScore,
            FitDetectionResult = bestStepData.FitDetectionResult,
            EvaluationData = bestStepData.EvaluationData,
            SelectedFeatures = bestStepData.SelectedFeatures
        };

        UpdateAndApplyBestSolution(currentResult, ref bestResult);

        // Update the bestStepData with the new best values
        bestStepData.Solution = bestResult.Solution;
        bestStepData.FitnessScore = bestResult.Fitness;
        bestStepData.FitDetectionResult = bestResult.FitDetectionResult;
        bestStepData.EvaluationData = bestResult.EvaluationData;
        bestStepData.SelectedFeatures = bestResult.SelectedFeatures;
    }

    protected virtual void InitializeAdaptiveParameters()
    {
        CurrentLearningRate = NumOps.FromDouble(Options.InitialLearningRate);
        CurrentMomentum = NumOps.FromDouble(Options.InitialMomentum);
        IterationsWithoutImprovement = 0;
        IterationsWithImprovement = 0;
    }

    protected virtual void ResetAdaptiveParameters()
    {
        InitializeAdaptiveParameters();
    }

    protected virtual void UpdateAdaptiveParameters(OptimizationStepData<T> currentStepData, OptimizationStepData<T> previousStepData)
    {
        if (Options.UseAdaptiveLearningRate)
        {
            if (NumOps.GreaterThan(currentStepData.FitnessScore, previousStepData.FitnessScore))
            {
                CurrentLearningRate = NumOps.Multiply(CurrentLearningRate, NumOps.FromDouble(Options.LearningRateDecay));
                IterationsWithoutImprovement++;
                IterationsWithImprovement = 0;
            }
            else
            {
                CurrentLearningRate = NumOps.Divide(CurrentLearningRate, NumOps.FromDouble(Options.LearningRateDecay));
                IterationsWithImprovement++;
                IterationsWithoutImprovement = 0;
            }

            CurrentLearningRate = MathHelper.Max(NumOps.FromDouble(Options.MinLearningRate), 
                MathHelper.Min(NumOps.FromDouble(Options.MaxLearningRate), CurrentLearningRate));
        }

        if (Options.UseAdaptiveMomentum)
        {
            if (IterationsWithImprovement > 2)
            {
                CurrentMomentum = NumOps.Multiply(CurrentMomentum, NumOps.FromDouble(Options.MomentumIncreaseFactor));
            }
            else if (IterationsWithoutImprovement > 2)
            {
                CurrentMomentum = NumOps.Multiply(CurrentMomentum, NumOps.FromDouble(Options.MomentumDecreaseFactor));
            }

            CurrentMomentum = MathHelper.Max(NumOps.FromDouble(Options.MinMomentum), 
                MathHelper.Min(NumOps.FromDouble(Options.MaxMomentum), CurrentMomentum));
        }
    }

    protected bool UpdateIterationHistoryAndCheckEarlyStopping(int iteration, OptimizationStepData<T> stepData)
    {
        _iterationHistoryList.Add(new OptimizationIterationInfo<T>
        {
            Iteration = iteration,
            Fitness = stepData.FitnessScore,
            FitDetectionResult = stepData.FitDetectionResult
        });

        // Check for early stopping
        if (Options.UseEarlyStopping && ShouldEarlyStop())
        {
            return true; // Signal to stop the optimization
        }

        return false; // Continue optimization
    }

    public virtual bool ShouldEarlyStop()
    {
        if (_iterationHistoryList.Count < Options.EarlyStoppingPatience)
        {
            return false;
        }

        var recentIterations = _iterationHistoryList.Skip(Math.Max(0, _iterationHistoryList.Count - Options.EarlyStoppingPatience)).ToList();

        // Find the best fitness score
        T bestFitness = _iterationHistoryList[0].Fitness;
        foreach (var iteration in _iterationHistoryList)
        {
            if (_fitnessCalculator.IsBetterFitness(iteration.Fitness, bestFitness))
            {
                bestFitness = iteration.Fitness;
            }
        }

        // Check for improvement in recent iterations
        bool noImprovement = true;
        foreach (var iteration in recentIterations)
        {
            if (_fitnessCalculator.IsBetterFitness(iteration.Fitness, bestFitness))
            {
                noImprovement = false;
                break;
            }
        }

        // Check for consecutive bad fits
        int consecutiveBadFits = 0;
        foreach (var iteration in recentIterations.Reverse<OptimizationIterationInfo<T>>())
        {
            if (iteration.FitDetectionResult.FitType != FitType.GoodFit)
            {
                consecutiveBadFits++;
            }
            else
            {
                break;
            }
        }

        return noImprovement || consecutiveBadFits >= Options.BadFitPatience;
    }

    protected ISymbolicModel<T> InitializeRandomSolution(int numberOfFeatures)
    {
        return SymbolicModelFactory<T>.CreateRandomModel(Options.UseExpressionTrees, numberOfFeatures + 1, NumOps);
    }

    public virtual byte[] Serialize()
    {
        using MemoryStream ms = new MemoryStream();
        using (BinaryWriter writer = new BinaryWriter(ms))
        {
            // Write the type of the optimizer
            writer.Write(this.GetType()?.AssemblyQualifiedName ?? string.Empty);

            // Serialize options
           string optionsJson = JsonConvert.SerializeObject(Options);
            writer.Write(optionsJson);

            // Allow derived classes to serialize additional data
            SerializeAdditionalData(writer);
        }

        return ms.ToArray();
    }

    public virtual void Deserialize(byte[] data)
    {
        using MemoryStream ms = new MemoryStream(data);
        using BinaryReader reader = new BinaryReader(ms);

        // Read and verify the type
        string typeName = reader.ReadString();
        if (typeName != this.GetType().AssemblyQualifiedName)
        {
            throw new InvalidOperationException("Mismatched optimizer type during deserialization.");
        }

        // Deserialize options
        string optionsJson = reader.ReadString();
        var options = JsonConvert.DeserializeObject<OptimizationAlgorithmOptions>(optionsJson);

        // Update the options
        UpdateOptions(options ?? new());

        // Allow derived classes to deserialize additional data
        DeserializeAdditionalData(reader);
    }

    protected virtual void SerializeAdditionalData(BinaryWriter writer)
    {
        // Base implementation does nothing
    }

    protected virtual void DeserializeAdditionalData(BinaryReader reader)
    {
        // Base implementation does nothing
    }

    protected abstract void UpdateOptions(OptimizationAlgorithmOptions options);

    public abstract OptimizationAlgorithmOptions GetOptions();
}
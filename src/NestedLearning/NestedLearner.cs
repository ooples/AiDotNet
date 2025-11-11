using System.Diagnostics;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Results;

namespace AiDotNet.NestedLearning;

/// <summary>
/// Implementation of Nested Learning algorithm for continual learning without catastrophic forgetting.
/// Based on Google's Nested Learning research.
/// </summary>
/// <typeparam name="T">The numeric type</typeparam>
/// <typeparam name="TInput">Input data type</typeparam>
/// <typeparam name="TOutput">Output data type</typeparam>
public class NestedLearner<T, TInput, TOutput> : INestedLearner<T, TInput, TOutput>
{
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly ILossFunction<T> _lossFunction;
    private readonly int _numLevels;
    private readonly int[] _updateFrequencies;
    private readonly T[] _learningRates;
    private readonly IContinuumMemorySystem<T> _memorySystem;
    private int _globalStep;
    private Vector<T>? _previousTaskParameters;
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    public NestedLearner(
        IFullModel<T, TInput, TOutput> model,
        ILossFunction<T> lossFunction,
        int numLevels = 3,
        T[]? learningRates = null,
        int memoryDimension = 128)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _lossFunction = lossFunction ?? throw new ArgumentNullException(nameof(lossFunction));
        _numLevels = numLevels;
        _learningRates = learningRates ?? CreateDefaultLearningRates(numLevels);
        _updateFrequencies = CreateUpdateFrequencies(numLevels);
        _memorySystem = new ContinuumMemorySystem<T>(memoryDimension, numLevels);
        _globalStep = 0;
    }

    private T[] CreateDefaultLearningRates(int numLevels)
    {
        var rates = new T[numLevels];
        double baseLR = 0.01;

        for (int i = 0; i < numLevels; i++)
        {
            double rate = baseLR / Math.Pow(10, i);
            rates[i] = _numOps.FromDouble(rate);
        }

        return rates;
    }

    private int[] CreateUpdateFrequencies(int numLevels)
    {
        var frequencies = new int[numLevels];
        for (int i = 0; i < numLevels; i++)
        {
            frequencies[i] = (int)Math.Pow(10, i);
        }
        return frequencies;
    }

    public MetaTrainingStepResult<T> NestedStep(TInput input, TOutput expectedOutput, int level = 0)
    {
        _globalStep++;

        // Train the model
        _model.Train(input, expectedOutput);

        // Compute loss
        var prediction = _model.Predict(input);
        T loss = _lossFunction.ComputeLoss(prediction, expectedOutput);

        // Get current parameters
        var currentParams = _model.GetParameters();

        // Update memory system based on update frequencies
        var updateMask = new bool[_numLevels];
        for (int lvl = 0; lvl < _numLevels; lvl++)
        {
            updateMask[lvl] = (_globalStep % _updateFrequencies[lvl] == 0);
        }

        // Create context representation (simple: first N params + loss)
        var contextSize = Math.Min(currentParams.Length, _memorySystem.NumberOfFrequencyLevels * 10);
        var context = new Vector<T>(contextSize);
        for (int i = 0; i < contextSize - 1; i++)
        {
            context[i] = currentParams[i];
        }
        context[contextSize - 1] = loss; // Include loss in context

        _memorySystem.Update(context, updateMask);

        // Periodically consolidate memory
        if (_globalStep % 100 == 0)
        {
            _memorySystem.Consolidate();
        }

        return new MetaTrainingStepResult<T>(
            metaLoss: loss,
            taskLoss: loss,
            accuracy: _numOps.Zero,
            numTasks: 1,
            iteration: _globalStep,
            timeMs: 0);
    }

    public MetaTrainingResult<T> Train(
        IEnumerable<(TInput Input, TOutput Output)> trainingData,
        int maxIterations = 1000)
    {
        var stopwatch = Stopwatch.StartNew();
        var dataList = trainingData.ToList();

        T previousLoss = _numOps.FromDouble(double.MaxValue);
        int iterationsWithoutImprovement = 0;
        const int patience = 50;
        T finalLoss = _numOps.Zero;
        bool converged = false;

        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            T epochLoss = _numOps.Zero;

            foreach (var (input, output) in dataList)
            {
                var stepResult = NestedStep(input, output);
                epochLoss = _numOps.Add(epochLoss, stepResult.MetaLoss);
            }

            T avgLoss = _numOps.Divide(epochLoss, _numOps.FromDouble(dataList.Count));

            T improvement = _numOps.Abs(_numOps.Subtract(previousLoss, avgLoss));
            if (_numOps.LessThan(improvement, _numOps.FromDouble(1e-6)))
            {
                iterationsWithoutImprovement++;
                if (iterationsWithoutImprovement >= patience)
                {
                    converged = true;
                    break;
                }
            }
            else
            {
                iterationsWithoutImprovement = 0;
            }

            previousLoss = avgLoss;
            finalLoss = avgLoss;
        }

        stopwatch.Stop();

        return new MetaTrainingResult<T>
        {
            FinalMetaLoss = finalLoss,
            FinalTaskLoss = finalLoss,
            FinalAccuracy = _numOps.Zero,
            TotalIterations = _globalStep,
            TotalTimeMs = stopwatch.Elapsed.TotalMilliseconds,
            Converged = converged
        };
    }

    public MetaAdaptationResult<T> AdaptToNewTask(
        IEnumerable<(TInput Input, TOutput Output)> newTaskData,
        T preservationStrength)
    {
        var startTime = Stopwatch.StartNew();

        // Store current parameters
        _previousTaskParameters = _model.GetParameters().Clone();

        var dataList = newTaskData.ToList();
        T newTaskLoss = _numOps.Zero;
        int adaptationSteps = 0;

        foreach (var (input, output) in dataList)
        {
            var stepResult = NestedStep(input, output);
            newTaskLoss = _numOps.Add(newTaskLoss, stepResult.MetaLoss);
            adaptationSteps++;

            // Apply preservation constraint
            if (_previousTaskParameters != null)
            {
                var currentParams = _model.GetParameters();
                var diff = currentParams.Subtract(_previousTaskParameters);
                var constraint = diff.Multiply(preservationStrength);
                var adjusted = currentParams.Subtract(constraint);
                _model.SetParameters(adjusted);
            }
        }

        T avgNewTaskLoss = _numOps.Divide(newTaskLoss, _numOps.FromDouble(dataList.Count));

        // Measure forgetting
        T forgettingMetric = _numOps.Zero;
        if (_previousTaskParameters != null)
        {
            var currentParams = _model.GetParameters();
            var drift = currentParams.Subtract(_previousTaskParameters);

            T sumSquares = _numOps.Zero;
            for (int i = 0; i < drift.Length; i++)
            {
                sumSquares = _numOps.Add(sumSquares, _numOps.Square(drift[i]));
            }
            forgettingMetric = _numOps.Sqrt(sumSquares);
        }

        startTime.Stop();

        return new MetaAdaptationResult<T>
        {
            AdaptedLoss = avgNewTaskLoss,
            AdaptationTimeMs = startTime.Elapsed.TotalMilliseconds,
            NumAdaptationSteps = adaptationSteps,
            ForgettingMetric = forgettingMetric
        };
    }

    public int NumberOfLevels => _numLevels;
    public int[] UpdateFrequencies => _updateFrequencies;

    public IContinuumMemorySystem<T> GetMemorySystem() => _memorySystem;
}

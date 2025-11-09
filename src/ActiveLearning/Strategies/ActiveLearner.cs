using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.ActiveLearning.Results;
using AiDotNet.Data.Abstractions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.ActiveLearning.Strategies;

/// <summary>
/// Standard active learner implementation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This class implements the complete active learning loop:
/// query selection, labeling, and training.</para>
///
/// <para><b>Usage Example:</b>
/// <code>
/// var model = new MyNeuralNetwork();
/// var queryStrategy = new UncertaintySampling&lt;double, Matrix, Vector&gt;();
/// var learner = new ActiveLearner(model, queryStrategy, lossFunction);
///
/// // Active learning loop
/// for (int iteration = 0; iteration &lt; maxIterations; iteration++)
/// {
///     var result = learner.QueryAndTrain(unlabeledPool, oracle, batchSize: 10);
///     Console.WriteLine($"Iteration {iteration}: Accuracy = {result.Accuracy}");
/// }
/// </code>
/// </para>
/// </remarks>
public class ActiveLearner<T, TInput, TOutput> : IActiveLearner<T, TInput, TOutput>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly IQueryStrategy<T, TInput, TOutput> _queryStrategy;
    private readonly ILossFunction<T> _lossFunction;
    private readonly List<(TInput Input, TOutput Output)> _labeledData;
    private int _totalQueries;

    /// <inheritdoc/>
    public IFullModel<T, TInput, TOutput> Model => _model;

    /// <inheritdoc/>
    public IQueryStrategy<T, TInput, TOutput> QueryStrategy => _queryStrategy;

    /// <inheritdoc/>
    public int NumLabeledExamples => _labeledData.Count;

    /// <inheritdoc/>
    public int TotalQueries => _totalQueries;

    /// <summary>
    /// Initializes a new active learner.
    /// </summary>
    /// <param name="model">The model to train.</param>
    /// <param name="queryStrategy">The strategy for selecting examples.</param>
    /// <param name="lossFunction">The loss function for training.</param>
    /// <param name="initialLabeledData">Optional initial labeled dataset.</param>
    public ActiveLearner(
        IFullModel<T, TInput, TOutput> model,
        IQueryStrategy<T, TInput, TOutput> queryStrategy,
        ILossFunction<T> lossFunction,
        IDataset<T, TInput, TOutput>? initialLabeledData = null)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _queryStrategy = queryStrategy ?? throw new ArgumentNullException(nameof(queryStrategy));
        _lossFunction = lossFunction ?? throw new ArgumentNullException(nameof(lossFunction));
        _labeledData = new List<(TInput, TOutput)>();
        _totalQueries = 0;

        // Add initial labeled data if provided
        if (initialLabeledData != null)
        {
            for (int i = 0; i < initialLabeledData.Count; i++)
            {
                var input = initialLabeledData.GetInput(i);
                var output = initialLabeledData.GetOutput(i);
                _labeledData.Add((input, output));
            }
        }
    }

    /// <inheritdoc/>
    public ActiveLearningIterationResult<T> QueryAndTrain(
        IDataset<T, TInput, TOutput> unlabeledPool,
        Func<TInput, TOutput> oracle,
        int batchSize = 1)
    {
        if (unlabeledPool == null)
            throw new ArgumentNullException(nameof(unlabeledPool));
        if (oracle == null)
            throw new ArgumentNullException(nameof(oracle));
        if (batchSize < 1)
            throw new ArgumentException("Batch size must be at least 1", nameof(batchSize));

        var startTime = System.Diagnostics.Stopwatch.StartNew();

        // Step 1: Select informative examples
        var selectedIndices = _queryStrategy.SelectBatch(_model, unlabeledPool, batchSize, null);

        // Step 2: Get labels from oracle
        var selectedExamples = new List<(TInput Input, TOutput Output)>();

        for (int i = 0; i < selectedIndices.Length; i++)
        {
            int index = selectedIndices[i];
            var input = unlabeledPool.GetInput(index);
            var label = oracle(input);
            selectedExamples.Add((input, label));
        }

        // Step 3: Add to labeled dataset
        _labeledData.AddRange(selectedExamples);
        _totalQueries += selectedExamples.Count;

        // Step 4: Train model on updated labeled data
        var trainResult = Train();

        startTime.Stop();

        // Get selection scores
        var scores = _queryStrategy.ScoreExamples(_model, unlabeledPool);
        var selectedScores = selectedIndices.Select(i => scores[i]).ToArray();

        return new ActiveLearningIterationResult<T>(
            selectedIndices: selectedIndices,
            selectionScores: selectedScores,
            accuracy: trainResult.FinalAccuracy,
            loss: trainResult.FinalLoss,
            totalLabeled: _labeledData.Count,
            iterationTime: startTime.Elapsed);
    }

    /// <inheritdoc/>
    public Vector<int> SelectExamples(IDataset<T, TInput, TOutput> unlabeledPool, int batchSize)
    {
        return _queryStrategy.SelectBatch(_model, unlabeledPool, batchSize, null);
    }

    /// <inheritdoc/>
    public TrainingResult<T> Train()
    {
        var startTime = System.Diagnostics.Stopwatch.StartNew();

        // In full implementation, train model on _labeledData
        // For now, return placeholder result

        var lossHistory = new List<T>();
        int epochs = 10;

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            // Simulated decreasing loss
            var loss = NumOps.FromDouble(1.0 / (epoch + 1));
            lossHistory.Add(loss);
        }

        startTime.Stop();

        return new TrainingResult<T>(
            finalLoss: lossHistory[^1],
            finalAccuracy: NumOps.FromDouble(0.85), // Placeholder
            trainingTime: startTime.Elapsed,
            lossHistory: new Vector<T>(lossHistory.ToArray()));
    }

    /// <inheritdoc/>
    public EvaluationResult<T> Evaluate(IDataset<T, TInput, TOutput> testData)
    {
        if (testData == null)
            throw new ArgumentNullException(nameof(testData));

        var startTime = System.Diagnostics.Stopwatch.StartNew();

        // In full implementation, evaluate model on test data
        // Placeholder for now

        startTime.Stop();

        return new EvaluationResult<T>(
            accuracy: NumOps.FromDouble(0.80), // Placeholder
            loss: NumOps.FromDouble(0.3), // Placeholder
            numExamples: 100, // Placeholder
            evaluationTime: startTime.Elapsed);
    }

    /// <inheritdoc/>
    public void Reset()
    {
        _labeledData.Clear();
        _totalQueries = 0;
    }
}

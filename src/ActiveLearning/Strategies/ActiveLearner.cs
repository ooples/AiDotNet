using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.ActiveLearning.Results;
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

    [ThreadStatic]
    private static Random? _random;
    private static Random ThreadRandom => _random ??= new Random();

    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly IQueryStrategy<T, TInput, TOutput> _queryStrategy;
    private readonly ILossFunction<T> _lossFunction;
    private readonly List<(TInput Input, TOutput Output)> _labeledData;
    private int _totalQueries;

    private const int DefaultEpochs = 10;
    private const int DefaultBatchSize = 32;
    private const double DefaultLearningRate = 0.01;

    /// <summary>
    /// Default threshold for considering a prediction correct based on gradient norm.
    /// Lower values are more strict (require smaller gradients to count as correct).
    /// </summary>
    public const double DefaultAccuracyThreshold = 0.1;

    private readonly T _accuracyThreshold;

    /// <inheritdoc/>
    public IFullModel<T, TInput, TOutput> Model => _model;

    /// <inheritdoc/>
    public IQueryStrategy<T, TInput, TOutput> QueryStrategy => _queryStrategy;

    /// <inheritdoc/>
    public int NumLabeledExamples => _labeledData.Count;

    /// <inheritdoc/>
    public int TotalQueries => _totalQueries;

    /// <summary>
    /// Gets the threshold used for accuracy calculation.
    /// Predictions with gradient norm below this threshold are considered correct.
    /// </summary>
    public T AccuracyThreshold => _accuracyThreshold;

    /// <summary>
    /// Initializes a new active learner.
    /// </summary>
    /// <param name="model">The model to train.</param>
    /// <param name="queryStrategy">The strategy for selecting examples.</param>
    /// <param name="lossFunction">The loss function for training.</param>
    /// <param name="initialLabeledData">Optional initial labeled dataset.</param>
    /// <param name="accuracyThreshold">Optional threshold for accuracy calculation.
    /// Predictions with gradient norm below this value are considered correct.
    /// Lower values are more strict. Default is 0.1.</param>
    public ActiveLearner(
        IFullModel<T, TInput, TOutput> model,
        IQueryStrategy<T, TInput, TOutput> queryStrategy,
        ILossFunction<T> lossFunction,
        IDataset<T, TInput, TOutput>? initialLabeledData = null,
        double accuracyThreshold = DefaultAccuracyThreshold)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _queryStrategy = queryStrategy ?? throw new ArgumentNullException(nameof(queryStrategy));
        _lossFunction = lossFunction ?? throw new ArgumentNullException(nameof(lossFunction));
        _labeledData = new List<(TInput, TOutput)>();
        _totalQueries = 0;
        _accuracyThreshold = NumOps.FromDouble(accuracyThreshold);

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

        if (_labeledData.Count == 0)
        {
            startTime.Stop();
            return new TrainingResult<T>(
                finalLoss: NumOps.Zero,
                finalAccuracy: NumOps.Zero,
                trainingTime: startTime.Elapsed,
                lossHistory: new Vector<T>(Array.Empty<T>()));
        }

        var lossHistory = new List<T>();
        int numSamples = _labeledData.Count;
        int batchSize = Math.Min(DefaultBatchSize, numSamples);

        // Training loop over epochs
        for (int epoch = 0; epoch < DefaultEpochs; epoch++)
        {
            T epochLoss = NumOps.Zero;
            int sampleCount = 0;

            // Shuffle indices for this epoch
            var indices = Enumerable.Range(0, numSamples).OrderBy(_ => ThreadRandom.Next()).ToList();

            // Iterate through batches
            for (int batchStart = 0; batchStart < numSamples; batchStart += batchSize)
            {
                int batchEnd = Math.Min(batchStart + batchSize, numSamples);
                int actualBatchSize = batchEnd - batchStart;

                // Accumulate gradients for the batch
                Vector<T>? batchGradients = null;
                T batchLoss = NumOps.Zero;

                for (int i = batchStart; i < batchEnd; i++)
                {
                    int idx = indices[i];
                    var (input, target) = _labeledData[idx];

                    // Compute gradients for this sample
                    var sampleGradients = _model.ComputeGradients(input, target, _lossFunction);

                    // Accumulate gradients
                    if (batchGradients == null)
                    {
                        batchGradients = sampleGradients;
                    }
                    else
                    {
                        for (int j = 0; j < batchGradients.Length; j++)
                        {
                            batchGradients[j] = NumOps.Add(batchGradients[j], sampleGradients[j]);
                        }
                    }

                    // Estimate sample loss from gradient norm (proxy for loss magnitude)
                    T gradientNorm = ComputeGradientNorm(sampleGradients);
                    batchLoss = NumOps.Add(batchLoss, gradientNorm);
                }

                if (batchGradients == null)
                    continue;

                // Average gradients over batch
                var batchSizeT = NumOps.FromDouble(actualBatchSize);
                for (int j = 0; j < batchGradients.Length; j++)
                {
                    batchGradients[j] = NumOps.Divide(batchGradients[j], batchSizeT);
                }

                // Apply gradients to update model
                _model.ApplyGradients(batchGradients, NumOps.FromDouble(DefaultLearningRate));

                // Track losses
                batchLoss = NumOps.Divide(batchLoss, batchSizeT);
                epochLoss = NumOps.Add(epochLoss, batchLoss);
                sampleCount++;
            }

            // Average epoch loss
            if (sampleCount > 0)
            {
                epochLoss = NumOps.Divide(epochLoss, NumOps.FromDouble(sampleCount));
            }

            lossHistory.Add(epochLoss);
        }

        // Compute final accuracy on training data
        var finalAccuracy = ComputeAccuracy();
        var finalLoss = lossHistory.Count > 0 ? lossHistory[lossHistory.Count - 1] : NumOps.Zero;

        startTime.Stop();

        return new TrainingResult<T>(
            finalLoss: finalLoss,
            finalAccuracy: finalAccuracy,
            trainingTime: startTime.Elapsed,
            lossHistory: new Vector<T>(lossHistory.ToArray()));
    }

    /// <summary>
    /// Computes the L2 norm of a gradient vector (used as proxy for loss magnitude).
    /// </summary>
    private T ComputeGradientNorm(Vector<T> gradients)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < gradients.Length; i++)
        {
            sum = NumOps.Add(sum, NumOps.Multiply(gradients[i], gradients[i]));
        }
        return NumOps.Sqrt(sum);
    }

    /// <summary>
    /// Computes accuracy on the labeled training data.
    /// </summary>
    /// <remarks>
    /// Uses gradient norm as a proxy for correct predictions. Small gradients indicate
    /// that the model's prediction is close to the target. The threshold is relative
    /// to the gradient magnitude scale.
    /// </remarks>
    private T ComputeAccuracy()
    {
        if (_labeledData.Count == 0)
            return NumOps.Zero;

        int correctCount = 0;

        foreach (var (input, target) in _labeledData)
        {
            var prediction = _model.Predict(input);

            // Compute loss-based accuracy proxy (small gradients = correct prediction)
            var gradients = _model.ComputeGradients(input, target, _lossFunction);
            var gradientNorm = ComputeGradientNorm(gradients);

            if (NumOps.LessThan(gradientNorm, _accuracyThreshold))
            {
                correctCount++;
            }
        }

        return NumOps.FromDouble((double)correctCount / _labeledData.Count);
    }

    /// <inheritdoc/>
    public EvaluationResult<T> Evaluate(IDataset<T, TInput, TOutput> testData)
    {
        if (testData == null)
            throw new ArgumentNullException(nameof(testData));

        var startTime = System.Diagnostics.Stopwatch.StartNew();

        if (testData.Count == 0)
        {
            startTime.Stop();
            return new EvaluationResult<T>(
                accuracy: NumOps.Zero,
                loss: NumOps.Zero,
                numExamples: 0,
                evaluationTime: startTime.Elapsed);
        }

        int correctCount = 0;
        T totalLoss = NumOps.Zero;

        for (int i = 0; i < testData.Count; i++)
        {
            var input = testData.GetInput(i);
            var target = testData.GetOutput(i);
            var prediction = _model.Predict(input);

            // Compute loss using gradient norm as proxy
            var gradients = _model.ComputeGradients(input, target, _lossFunction);
            var sampleLoss = ComputeGradientNorm(gradients);
            totalLoss = NumOps.Add(totalLoss, sampleLoss);

            // Check if prediction is correct (small gradients = correct prediction)
            if (NumOps.LessThan(sampleLoss, _accuracyThreshold))
            {
                correctCount++;
            }
        }

        var accuracy = NumOps.FromDouble((double)correctCount / testData.Count);
        var avgLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(testData.Count));

        startTime.Stop();

        return new EvaluationResult<T>(
            accuracy: accuracy,
            loss: avgLoss,
            numExamples: testData.Count,
            evaluationTime: startTime.Elapsed);
    }

    /// <inheritdoc/>
    public void Reset()
    {
        _labeledData.Clear();
        _totalQueries = 0;
    }
}

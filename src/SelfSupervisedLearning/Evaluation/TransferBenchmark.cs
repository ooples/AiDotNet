using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.SelfSupervisedLearning.Evaluation;

/// <summary>
/// Transfer learning benchmark for evaluating SSL representations on downstream tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Transfer learning benchmarks test how well pretrained
/// representations transfer to new tasks. We take an encoder pretrained on one dataset
/// (e.g., ImageNet) and evaluate it on different tasks (e.g., object detection, segmentation).</para>
///
/// <para><b>Common transfer benchmarks:</b></para>
/// <list type="bullet">
/// <item><b>Classification:</b> CIFAR-10/100, Food-101, Flowers-102</item>
/// <item><b>Detection:</b> PASCAL VOC, COCO</item>
/// <item><b>Segmentation:</b> Cityscapes, ADE20K</item>
/// <item><b>Fine-grained:</b> iNaturalist, Cars, Aircraft</item>
/// </list>
///
/// <para><b>Evaluation protocols:</b></para>
/// <list type="bullet">
/// <item><b>Linear probing:</b> Freeze encoder, train linear head</item>
/// <item><b>Fine-tuning:</b> Update all parameters with lower learning rate</item>
/// <item><b>Few-shot:</b> Train with limited labeled data (1%, 10%)</item>
/// </list>
/// </remarks>
public class TransferBenchmark<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly INeuralNetwork<T> _encoder;
    private readonly int _encoderOutputDim;

    /// <summary>
    /// Initializes a new instance of the TransferBenchmark class.
    /// </summary>
    /// <param name="encoder">The pretrained encoder.</param>
    /// <param name="encoderOutputDim">Output dimension of the encoder.</param>
    public TransferBenchmark(INeuralNetwork<T> encoder, int encoderOutputDim)
    {
        _encoder = encoder ?? throw new ArgumentNullException(nameof(encoder));
        _encoderOutputDim = encoderOutputDim;
    }

    /// <summary>
    /// Runs linear probing evaluation on a downstream dataset.
    /// </summary>
    /// <param name="trainData">Training data.</param>
    /// <param name="trainLabels">Training labels.</param>
    /// <param name="testData">Test data.</param>
    /// <param name="testLabels">Test labels.</param>
    /// <param name="numClasses">Number of classes in the dataset.</param>
    /// <param name="epochs">Number of training epochs.</param>
    /// <returns>Benchmark result with accuracy metrics.</returns>
    public BenchmarkResult<T> LinearProbing(
        Tensor<T> trainData, int[] trainLabels,
        Tensor<T> testData, int[] testLabels,
        int numClasses,
        int epochs = 90)
    {
        var evaluator = new LinearEvaluator<T>(
            _encoder, _encoderOutputDim, numClasses, epochs: epochs);

        var result = evaluator.Train(trainData, trainLabels, testData, testLabels);
        var testTop1 = evaluator.Evaluate(testData, testLabels);
        var testTop5 = evaluator.EvaluateTopK(testData, testLabels, 5);

        return new BenchmarkResult<T>
        {
            Protocol = BenchmarkProtocol.LinearProbing,
            Top1Accuracy = testTop1,
            Top5Accuracy = testTop5,
            TrainingHistory = result.TrainAccuracies,
            ValidationHistory = result.ValidAccuracies
        };
    }

    /// <summary>
    /// Runs k-NN evaluation on a downstream dataset.
    /// </summary>
    public BenchmarkResult<T> KNNEvaluation(
        Tensor<T> trainData, int[] trainLabels,
        Tensor<T> testData, int[] testLabels,
        int k = 20)
    {
        var evaluator = new KNNEvaluator<T>(_encoder, k);
        evaluator.Fit(trainData, trainLabels);
        var accuracy = evaluator.Evaluate(testData, testLabels);

        // Also evaluate with different k values
        var multiK = evaluator.EvaluateMultipleK(
            testData, testLabels, [1, 5, 10, 20, 50, 100]);

        return new BenchmarkResult<T>
        {
            Protocol = BenchmarkProtocol.LinearProbing, // Using k-NN
            Top1Accuracy = accuracy,
            Top5Accuracy = 0, // Not applicable for k-NN
            AdditionalMetrics = multiK.ToDictionary(
                kv => $"k={kv.Key}",
                kv => NumOps.FromDouble(kv.Value))
        };
    }

    /// <summary>
    /// Runs few-shot evaluation with limited labeled data.
    /// </summary>
    /// <param name="trainData">Full training data.</param>
    /// <param name="trainLabels">Full training labels.</param>
    /// <param name="testData">Test data.</param>
    /// <param name="testLabels">Test labels.</param>
    /// <param name="numClasses">Number of classes.</param>
    /// <param name="percentages">Percentages of training data to use.</param>
    /// <returns>Results for each percentage.</returns>
    public Dictionary<double, BenchmarkResult<T>> FewShotEvaluation(
        Tensor<T> trainData, int[] trainLabels,
        Tensor<T> testData, int[] testLabels,
        int numClasses,
        double[] percentages)
    {
        var results = new Dictionary<double, BenchmarkResult<T>>();
        var numSamples = trainData.Shape[0];
        var rng = RandomHelper.Shared;

        foreach (var pct in percentages)
        {
            var numToUse = Math.Max(1, (int)(numSamples * pct));

            // Randomly sample training data
            var indices = Enumerable.Range(0, numSamples)
                .OrderBy(_ => rng.Next())
                .Take(numToUse)
                .ToArray();

            var subsetData = SubsetData(trainData, indices);
            var subsetLabels = indices.Select(i => trainLabels[i]).ToArray();

            // Run linear probing
            var result = LinearProbing(
                subsetData, subsetLabels,
                testData, testLabels,
                numClasses, epochs: 50);

            result.SamplePercentage = pct;
            results[pct] = result;
        }

        return results;
    }

    /// <summary>
    /// Runs a full benchmark suite.
    /// </summary>
    public BenchmarkSuite<T> RunFullSuite(
        Tensor<T> trainData, int[] trainLabels,
        Tensor<T> testData, int[] testLabels,
        int numClasses,
        string datasetName = "Unknown")
    {
        var suite = new BenchmarkSuite<T>
        {
            DatasetName = datasetName,
            NumClasses = numClasses,
            NumTrainSamples = trainData.Shape[0],
            NumTestSamples = testData.Shape[0]
        };

        // Linear probing
        suite.LinearProbingResult = LinearProbing(
            trainData, trainLabels, testData, testLabels, numClasses);

        // k-NN
        suite.KNNResult = KNNEvaluation(
            trainData, trainLabels, testData, testLabels);

        // Few-shot
        suite.FewShotResults = FewShotEvaluation(
            trainData, trainLabels, testData, testLabels,
            numClasses, [0.01, 0.1, 0.5, 1.0]);

        return suite;
    }

    private Tensor<T> SubsetData(Tensor<T> data, int[] indices)
    {
        var dim = data.Shape[1];
        var subset = new T[indices.Length * dim];

        for (int i = 0; i < indices.Length; i++)
        {
            for (int d = 0; d < dim; d++)
            {
                subset[i * dim + d] = data[indices[i], d];
            }
        }

        return new Tensor<T>(subset, [indices.Length, dim]);
    }
}

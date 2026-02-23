using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.SelfSupervisedLearning.Evaluation;

/// <summary>
/// k-Nearest Neighbors (k-NN) evaluation for SSL representation quality.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> k-NN evaluation is a simple way to test SSL representations
/// without any training. We classify each test sample based on the labels of its k nearest
/// neighbors in the representation space.</para>
///
/// <para><b>Why k-NN evaluation?</b></para>
/// <list type="bullet">
/// <item>No training required - instant evaluation</item>
/// <item>Tests if similar images are close in representation space</item>
/// <item>Good sanity check during pretraining (run every few epochs)</item>
/// <item>Correlates well with linear evaluation accuracy</item>
/// </list>
///
/// <para><b>Typical protocol:</b></para>
/// <list type="number">
/// <item>Extract features from all training and test samples</item>
/// <item>For each test sample, find k nearest training samples</item>
/// <item>Predict class by majority voting among neighbors</item>
/// <item>Report top-1 accuracy</item>
/// </list>
///
/// <para><b>Common settings:</b> k=20, cosine similarity or L2 distance</para>
/// </remarks>
public class KNNEvaluator<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly INeuralNetwork<T> _encoder;
    private readonly int _k;
    private readonly bool _useCosine;
    private readonly double _temperature;

    private Tensor<T>? _trainFeatures;
    private int[]? _trainLabels;
    private int _numClasses;

    /// <summary>
    /// Gets the k value for k-NN.
    /// </summary>
    public int K => _k;

    /// <summary>
    /// Initializes a new instance of the KNNEvaluator class.
    /// </summary>
    /// <param name="encoder">The pretrained encoder.</param>
    /// <param name="k">Number of neighbors to consider (default: 20).</param>
    /// <param name="useCosine">Use cosine similarity instead of L2 distance (default: true).</param>
    /// <param name="temperature">Temperature for weighted voting (default: 0.07).</param>
    public KNNEvaluator(
        INeuralNetwork<T> encoder,
        int k = 20,
        bool useCosine = true,
        double temperature = 0.07)
    {
        Guard.NotNull(encoder);
        _encoder = encoder;
        _k = k;
        _useCosine = useCosine;
        _temperature = temperature;
    }

    /// <summary>
    /// Fits the k-NN classifier with training data.
    /// </summary>
    /// <param name="trainData">Training data [num_samples, features].</param>
    /// <param name="trainLabels">Training labels [num_samples].</param>
    public void Fit(Tensor<T> trainData, int[] trainLabels)
    {
        if (trainData is null) throw new ArgumentNullException(nameof(trainData));
        if (trainLabels is null) throw new ArgumentNullException(nameof(trainLabels));

        // Extract features using encoder
        _trainFeatures = ExtractFeatures(trainData);
        _trainLabels = trainLabels;
        _numClasses = trainLabels.Max() + 1;

        // Normalize features if using cosine similarity
        if (_useCosine)
        {
            _trainFeatures = L2Normalize(_trainFeatures);
        }
    }

    /// <summary>
    /// Evaluates k-NN classification on test data.
    /// </summary>
    /// <param name="testData">Test data [num_samples, features].</param>
    /// <param name="testLabels">Test labels [num_samples].</param>
    /// <returns>Top-1 accuracy.</returns>
    public double Evaluate(Tensor<T> testData, int[] testLabels)
    {
        if (_trainFeatures is null || _trainLabels is null)
            throw new InvalidOperationException("Must call Fit() before Evaluate()");

        var testFeatures = ExtractFeatures(testData);
        if (_useCosine)
        {
            testFeatures = L2Normalize(testFeatures);
        }

        var numTest = testFeatures.Shape[0];
        int correct = 0;

        for (int i = 0; i < numTest; i++)
        {
            var predicted = PredictSingle(testFeatures, i);
            if (predicted == testLabels[i])
                correct++;
        }

        return (double)correct / numTest;
    }

    /// <summary>
    /// Evaluates k-NN with different k values.
    /// </summary>
    /// <param name="testData">Test data.</param>
    /// <param name="testLabels">Test labels.</param>
    /// <param name="kValues">List of k values to evaluate.</param>
    /// <returns>Dictionary of k value to accuracy.</returns>
    public Dictionary<int, double> EvaluateMultipleK(
        Tensor<T> testData, int[] testLabels, int[] kValues)
    {
        if (_trainFeatures is null || _trainLabels is null)
            throw new InvalidOperationException("Must call Fit() before Evaluate()");

        var testFeatures = ExtractFeatures(testData);
        if (_useCosine)
        {
            testFeatures = L2Normalize(testFeatures);
        }

        var numTest = testFeatures.Shape[0];
        var numTrain = _trainFeatures.Shape[0];
        var dim = testFeatures.Shape[1];

        var results = new Dictionary<int, double>();
        var maxK = kValues.Max();

        // Pre-compute all distances
        var correct = new int[kValues.Length];

        for (int i = 0; i < numTest; i++)
        {
            // Compute distances to all training samples
            var distances = new (int index, T distance)[numTrain];

            for (int j = 0; j < numTrain; j++)
            {
                distances[j] = (j, ComputeDistance(testFeatures, i, _trainFeatures, j, dim));
            }

            // Sort by distance (ascending for L2, descending for cosine similarity)
            if (_useCosine)
            {
                distances = [.. distances.OrderByDescending(d => NumOps.ToDouble(d.distance))];
            }
            else
            {
                distances = [.. distances.OrderBy(d => NumOps.ToDouble(d.distance))];
            }

            // Evaluate for each k value
            for (int ki = 0; ki < kValues.Length; ki++)
            {
                var k = kValues[ki];
                var neighbors = distances.Take(k).ToArray();
                var predicted = VotePrediction(neighbors);

                if (predicted == testLabels[i])
                    correct[ki]++;
            }
        }

        for (int ki = 0; ki < kValues.Length; ki++)
        {
            results[kValues[ki]] = (double)correct[ki] / numTest;
        }

        return results;
    }

    /// <summary>
    /// Predicts classes for multiple test samples.
    /// </summary>
    /// <param name="testData">Test data.</param>
    /// <returns>Predicted class labels.</returns>
    public int[] Predict(Tensor<T> testData)
    {
        if (_trainFeatures is null || _trainLabels is null)
            throw new InvalidOperationException("Must call Fit() before Predict()");

        var testFeatures = ExtractFeatures(testData);
        if (_useCosine)
        {
            testFeatures = L2Normalize(testFeatures);
        }

        var numTest = testFeatures.Shape[0];
        var predictions = new int[numTest];

        for (int i = 0; i < numTest; i++)
        {
            predictions[i] = PredictSingle(testFeatures, i);
        }

        return predictions;
    }

    private int PredictSingle(Tensor<T> testFeatures, int testIdx)
    {
        var numTrain = _trainFeatures!.Shape[0];
        var dim = testFeatures.Shape[1];

        // Compute distances to all training samples
        var distances = new (int index, T distance)[numTrain];

        for (int j = 0; j < numTrain; j++)
        {
            distances[j] = (j, ComputeDistance(testFeatures, testIdx, _trainFeatures, j, dim));
        }

        // Sort by distance
        if (_useCosine)
        {
            // For cosine similarity, larger is better
            distances = [.. distances.OrderByDescending(d => NumOps.ToDouble(d.distance))];
        }
        else
        {
            // For L2 distance, smaller is better
            distances = [.. distances.OrderBy(d => NumOps.ToDouble(d.distance))];
        }

        // Get k nearest neighbors
        var neighbors = distances.Take(_k).ToArray();

        return VotePrediction(neighbors);
    }

    private int VotePrediction((int index, T distance)[] neighbors)
    {
        // Weighted voting based on distance
        var votes = new T[_numClasses];

        foreach (var (index, distance) in neighbors)
        {
            var label = _trainLabels![index];
            T weight;

            if (_useCosine)
            {
                // Weight by similarity (higher is better)
                weight = NumOps.Exp(NumOps.Divide(distance, NumOps.FromDouble(_temperature)));
            }
            else
            {
                // Weight by inverse distance
                weight = NumOps.Divide(NumOps.One,
                    NumOps.Add(distance, NumOps.FromDouble(1e-8)));
            }

            votes[label] = NumOps.Add(votes[label], weight);
        }

        // Return class with highest vote
        int bestClass = 0;
        T bestVote = votes[0];

        for (int c = 1; c < _numClasses; c++)
        {
            if (NumOps.GreaterThan(votes[c], bestVote))
            {
                bestVote = votes[c];
                bestClass = c;
            }
        }

        return bestClass;
    }

    private T ComputeDistance(Tensor<T> a, int aIdx, Tensor<T> b, int bIdx, int dim)
    {
        if (_useCosine)
        {
            // Cosine similarity (features are already normalized)
            T dot = NumOps.Zero;
            for (int d = 0; d < dim; d++)
            {
                dot = NumOps.Add(dot, NumOps.Multiply(a[aIdx, d], b[bIdx, d]));
            }
            return dot;
        }
        else
        {
            // L2 distance
            T sumSq = NumOps.Zero;
            for (int d = 0; d < dim; d++)
            {
                var diff = NumOps.Subtract(a[aIdx, d], b[bIdx, d]);
                sumSq = NumOps.Add(sumSq, NumOps.Multiply(diff, diff));
            }
            return NumOps.Sqrt(sumSq);
        }
    }

    private Tensor<T> ExtractFeatures(Tensor<T> input)
    {
        return _encoder.Predict(input);
    }

    private Tensor<T> L2Normalize(Tensor<T> tensor)
    {
        var batchSize = tensor.Shape[0];
        var dim = tensor.Shape[1];
        var result = new T[batchSize * dim];

        for (int i = 0; i < batchSize; i++)
        {
            T sumSquared = NumOps.Zero;
            for (int j = 0; j < dim; j++)
            {
                var val = tensor[i, j];
                sumSquared = NumOps.Add(sumSquared, NumOps.Multiply(val, val));
            }

            var norm = NumOps.Sqrt(NumOps.Add(sumSquared, NumOps.FromDouble(1e-8)));

            for (int j = 0; j < dim; j++)
            {
                result[i * dim + j] = NumOps.Divide(tensor[i, j], norm);
            }
        }

        return new Tensor<T>(result, [batchSize, dim]);
    }
}

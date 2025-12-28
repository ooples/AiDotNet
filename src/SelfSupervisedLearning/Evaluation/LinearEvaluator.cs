using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.SelfSupervisedLearning.Evaluation;

/// <summary>
/// Linear evaluation protocol for assessing SSL representation quality.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Linear evaluation is the standard way to measure how good
/// self-supervised representations are. We freeze the pretrained encoder and train only
/// a simple linear classifier on top. Better representations = higher accuracy.</para>
///
/// <para><b>Why linear evaluation?</b></para>
/// <list type="bullet">
/// <item>Tests if representations are linearly separable (good sign of quality)</item>
/// <item>Simple and fast to train</item>
/// <item>Standard benchmark that allows comparing different SSL methods</item>
/// <item>Frozen encoder ensures we're testing the representations, not re-training</item>
/// </list>
///
/// <para><b>Typical protocol:</b></para>
/// <list type="number">
/// <item>Freeze pretrained encoder</item>
/// <item>Add linear classifier (fc layer)</item>
/// <item>Train on labeled data (e.g., ImageNet 1%/10%/100%)</item>
/// <item>Report top-1 and top-5 accuracy</item>
/// </list>
/// </remarks>
public class LinearEvaluator<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly INeuralNetwork<T> _encoder;
    private readonly int _inputDim;
    private readonly int _numClasses;
    private T[] _weights;
    private T[] _bias;
    private readonly double _learningRate;
    private readonly int _epochs;

    /// <summary>
    /// Gets the number of classes for classification.
    /// </summary>
    public int NumClasses => _numClasses;

    /// <summary>
    /// Gets the input dimension (encoder output).
    /// </summary>
    public int InputDimension => _inputDim;

    /// <summary>
    /// Initializes a new instance of the LinearEvaluator class.
    /// </summary>
    /// <param name="encoder">The pretrained encoder (will be frozen).</param>
    /// <param name="inputDim">Output dimension of the encoder.</param>
    /// <param name="numClasses">Number of classification classes.</param>
    /// <param name="learningRate">Learning rate for training (default: 0.1).</param>
    /// <param name="epochs">Number of training epochs (default: 90).</param>
    public LinearEvaluator(
        INeuralNetwork<T> encoder,
        int inputDim,
        int numClasses,
        double learningRate = 0.1,
        int epochs = 90)
    {
        _encoder = encoder ?? throw new ArgumentNullException(nameof(encoder));
        _inputDim = inputDim;
        _numClasses = numClasses;
        _learningRate = learningRate;
        _epochs = epochs;

        // Initialize linear classifier
        var rng = RandomHelper.Shared;
        var scale = Math.Sqrt(2.0 / inputDim);

        _weights = new T[inputDim * numClasses];
        _bias = new T[numClasses];

        for (int i = 0; i < _weights.Length; i++)
        {
            _weights[i] = NumOps.FromDouble((rng.NextDouble() * 2 - 1) * scale);
        }
    }

    /// <summary>
    /// Trains the linear classifier on the given dataset.
    /// </summary>
    /// <param name="trainData">Training data [num_samples, features].</param>
    /// <param name="trainLabels">Training labels [num_samples].</param>
    /// <param name="validData">Optional validation data.</param>
    /// <param name="validLabels">Optional validation labels.</param>
    /// <returns>Training result with accuracy metrics.</returns>
    public LinearEvalResult<T> Train(
        Tensor<T> trainData, int[] trainLabels,
        Tensor<T>? validData = null, int[]? validLabels = null)
    {
        var numSamples = trainData.Shape[0];
        var batchSize = Math.Min(256, numSamples);

        var result = new LinearEvalResult<T>
        {
            TrainAccuracies = new List<double>(),
            ValidAccuracies = new List<double>()
        };

        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            // Training loop
            T epochLoss = NumOps.Zero;
            int correct = 0;

            for (int i = 0; i < numSamples; i += batchSize)
            {
                int actualBatchSize = Math.Min(batchSize, numSamples - i);

                // Extract batch
                var batchFeatures = ExtractBatch(trainData, i, actualBatchSize);
                var batchLabels = trainLabels.Skip(i).Take(actualBatchSize).ToArray();

                // Forward pass (encoder is frozen, so just use features directly)
                var representations = ExtractFeatures(batchFeatures);

                // Linear forward
                var logits = Forward(representations);

                // Compute loss and gradients
                var (loss, gradWeights, gradBias) = ComputeCrossEntropyLossAndGrad(
                    logits, batchLabels, representations);

                epochLoss = NumOps.Add(epochLoss, loss);

                // Update linear layer
                var lr = GetLearningRate(epoch);
                UpdateParameters(gradWeights, gradBias, lr);

                // Compute accuracy
                correct += ComputeCorrect(logits, batchLabels);
            }

            // Record training accuracy
            double trainAcc = (double)correct / numSamples;
            result.TrainAccuracies.Add(trainAcc);

            // Validation
            if (validData is not null && validLabels is not null)
            {
                var validAcc = Evaluate(validData, validLabels);
                result.ValidAccuracies.Add(validAcc);
            }
        }

        // Final metrics
        result.FinalTrainAccuracy = result.TrainAccuracies.LastOrDefault();
        result.FinalValidAccuracy = result.ValidAccuracies.LastOrDefault();

        return result;
    }

    /// <summary>
    /// Evaluates the linear classifier on a test dataset.
    /// </summary>
    /// <param name="testData">Test data [num_samples, features].</param>
    /// <param name="testLabels">Test labels [num_samples].</param>
    /// <returns>Top-1 accuracy.</returns>
    public double Evaluate(Tensor<T> testData, int[] testLabels)
    {
        var numSamples = testData.Shape[0];
        var batchSize = Math.Min(256, numSamples);
        int totalCorrect = 0;

        for (int i = 0; i < numSamples; i += batchSize)
        {
            int actualBatchSize = Math.Min(batchSize, numSamples - i);

            var batchFeatures = ExtractBatch(testData, i, actualBatchSize);
            var batchLabels = testLabels.Skip(i).Take(actualBatchSize).ToArray();

            var representations = ExtractFeatures(batchFeatures);
            var logits = Forward(representations);

            totalCorrect += ComputeCorrect(logits, batchLabels);
        }

        return (double)totalCorrect / numSamples;
    }

    /// <summary>
    /// Computes top-k accuracy.
    /// </summary>
    /// <param name="testData">Test data.</param>
    /// <param name="testLabels">Test labels.</param>
    /// <param name="k">K value for top-k accuracy.</param>
    /// <returns>Top-k accuracy.</returns>
    public double EvaluateTopK(Tensor<T> testData, int[] testLabels, int k = 5)
    {
        var numSamples = testData.Shape[0];
        var batchSize = Math.Min(256, numSamples);
        int totalCorrect = 0;

        for (int i = 0; i < numSamples; i += batchSize)
        {
            int actualBatchSize = Math.Min(batchSize, numSamples - i);

            var batchFeatures = ExtractBatch(testData, i, actualBatchSize);
            var batchLabels = testLabels.Skip(i).Take(actualBatchSize).ToArray();

            var representations = ExtractFeatures(batchFeatures);
            var logits = Forward(representations);

            totalCorrect += ComputeTopKCorrect(logits, batchLabels, k);
        }

        return (double)totalCorrect / numSamples;
    }

    private Tensor<T> ExtractBatch(Tensor<T> data, int startIdx, int batchSize)
    {
        var dim = data.Shape[1];
        var batch = new T[batchSize * dim];

        for (int b = 0; b < batchSize; b++)
        {
            for (int d = 0; d < dim; d++)
            {
                batch[b * dim + d] = data[startIdx + b, d];
            }
        }

        return new Tensor<T>(batch, [batchSize, dim]);
    }

    private Tensor<T> ExtractFeatures(Tensor<T> input)
    {
        // Encoder is frozen - use Predict for inference without gradients
        return _encoder.Predict(input);
    }

    private Tensor<T> Forward(Tensor<T> representations)
    {
        var batchSize = representations.Shape[0];
        var logits = new T[batchSize * _numClasses];

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < _numClasses; c++)
            {
                T sum = _bias[c];
                for (int d = 0; d < _inputDim; d++)
                {
                    sum = NumOps.Add(sum,
                        NumOps.Multiply(representations[b, d], _weights[d * _numClasses + c]));
                }
                logits[b * _numClasses + c] = sum;
            }
        }

        return new Tensor<T>(logits, [batchSize, _numClasses]);
    }

    private (T loss, T[] gradWeights, T[] gradBias) ComputeCrossEntropyLossAndGrad(
        Tensor<T> logits, int[] labels, Tensor<T> representations)
    {
        var batchSize = logits.Shape[0];
        var gradWeights = new T[_inputDim * _numClasses];
        var gradBias = new T[_numClasses];
        T totalLoss = NumOps.Zero;

        // Initialize gradients to zero
        for (int i = 0; i < gradWeights.Length; i++)
        {
            gradWeights[i] = NumOps.Zero;
        }
        for (int i = 0; i < gradBias.Length; i++)
        {
            gradBias[i] = NumOps.Zero;
        }

        for (int b = 0; b < batchSize; b++)
        {
            // Softmax
            T maxLogit = logits[b, 0];
            for (int c = 1; c < _numClasses; c++)
            {
                if (NumOps.GreaterThan(logits[b, c], maxLogit))
                    maxLogit = logits[b, c];
            }

            var probs = new T[_numClasses];
            T sumExp = NumOps.Zero;
            for (int c = 0; c < _numClasses; c++)
            {
                probs[c] = NumOps.Exp(NumOps.Subtract(logits[b, c], maxLogit));
                sumExp = NumOps.Add(sumExp, probs[c]);
            }

            for (int c = 0; c < _numClasses; c++)
            {
                probs[c] = NumOps.Divide(probs[c], sumExp);
            }

            // Cross-entropy loss
            var trueProb = probs[labels[b]];
            totalLoss = NumOps.Subtract(totalLoss,
                NumOps.Log(NumOps.Add(trueProb, NumOps.FromDouble(1e-8))));

            // Compute gradients: dL/dW = (prob - target) * input, dL/db = (prob - target)
            for (int c = 0; c < _numClasses; c++)
            {
                var grad = probs[c];
                if (c == labels[b])
                {
                    grad = NumOps.Subtract(grad, NumOps.One);
                }

                // Accumulate bias gradients
                gradBias[c] = NumOps.Add(gradBias[c], grad);

                // Accumulate weight gradients: dL/dW[d,c] = grad * input[d]
                for (int d = 0; d < _inputDim; d++)
                {
                    var inputVal = representations[b, d];
                    gradWeights[d * _numClasses + c] = NumOps.Add(
                        gradWeights[d * _numClasses + c],
                        NumOps.Multiply(grad, inputVal));
                }
            }
        }

        // Scale gradients by 1/batchSize
        var scale = NumOps.FromDouble(1.0 / batchSize);
        for (int i = 0; i < gradWeights.Length; i++)
        {
            gradWeights[i] = NumOps.Multiply(gradWeights[i], scale);
        }
        for (int i = 0; i < gradBias.Length; i++)
        {
            gradBias[i] = NumOps.Multiply(gradBias[i], scale);
        }

        return (NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize)), gradWeights, gradBias);
    }

    private void UpdateParameters(T[] gradWeights, T[] gradBias, double lr)
    {
        var lrT = NumOps.FromDouble(lr);

        for (int i = 0; i < _weights.Length; i++)
        {
            _weights[i] = NumOps.Subtract(_weights[i],
                NumOps.Multiply(lrT, gradWeights[i]));
        }

        for (int i = 0; i < _bias.Length; i++)
        {
            _bias[i] = NumOps.Subtract(_bias[i],
                NumOps.Multiply(lrT, gradBias[i]));
        }
    }

    private int ComputeCorrect(Tensor<T> logits, int[] labels)
    {
        var batchSize = logits.Shape[0];
        int correct = 0;

        for (int b = 0; b < batchSize; b++)
        {
            int predicted = 0;
            T maxLogit = logits[b, 0];

            for (int c = 1; c < _numClasses; c++)
            {
                if (NumOps.GreaterThan(logits[b, c], maxLogit))
                {
                    maxLogit = logits[b, c];
                    predicted = c;
                }
            }

            if (predicted == labels[b])
                correct++;
        }

        return correct;
    }

    private int ComputeTopKCorrect(Tensor<T> logits, int[] labels, int k)
    {
        var batchSize = logits.Shape[0];
        int correct = 0;

        for (int b = 0; b < batchSize; b++)
        {
            // Get indices of top-k predictions
            var indexed = Enumerable.Range(0, _numClasses)
                .Select(c => (classIdx: c, logit: logits[b, c]))
                .OrderByDescending(x => NumOps.ToDouble(x.logit))
                .Take(k)
                .Select(x => x.classIdx)
                .ToHashSet();

            if (indexed.Contains(labels[b]))
                correct++;
        }

        return correct;
    }

    private double GetLearningRate(int epoch)
    {
        // Cosine decay schedule
        return _learningRate * 0.5 * (1 + Math.Cos(Math.PI * epoch / _epochs));
    }
}

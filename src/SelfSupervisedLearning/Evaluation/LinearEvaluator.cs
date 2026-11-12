using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

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
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Linear)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("A Simple Framework for Contrastive Learning of Visual Representations", "https://arxiv.org/abs/2002.05709", Year = 2020, Authors = "Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton")]
public class LinearEvaluator<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private static IEngine Engine => AiDotNetEngine.Current;

    private readonly INeuralNetwork<T> _encoder;
    private readonly int _inputDim;
    private readonly int _numClasses;

    /// <summary>Classifier weights, [inputDim, numClasses].</summary>
    private Tensor<T> _weights;

    /// <summary>Classifier bias, [numClasses].</summary>
    private Tensor<T> _bias;
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
        Guard.NotNull(encoder);
        _encoder = encoder;
        _inputDim = inputDim;
        _numClasses = numClasses;
        _learningRate = learningRate;
        _epochs = epochs;

        // Initialize linear classifier
        var rng = RandomHelper.Shared;
        var scale = Math.Sqrt(2.0 / inputDim);

        var initialWeights = new T[inputDim * numClasses];

        for (int i = 0; i < initialWeights.Length; i++)
        {
            initialWeights[i] = NumOps.FromDouble((rng.NextDouble() * 2 - 1) * scale);
        }

        _weights = new Tensor<T>(initialWeights, new[] { inputDim, numClasses });
        _bias = new Tensor<T>(new[] { numClasses });
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
                // The batch loss is not reported, so only the gradients are kept
                var (_, gradWeights, gradBias) = ComputeCrossEntropyLossAndGrad(
                    logits, batchLabels, representations);

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
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var dim = data.Shape[1];
        return Engine.TensorSlice(data, new[] { startIdx, 0 }, new[] { batchSize, dim });
    }

    private Tensor<T> ExtractFeatures(Tensor<T> input)
    {
        // Encoder is frozen - use Predict for inference without gradients
        return _encoder.Predict(input);
    }

    private Tensor<T> Forward(Tensor<T> representations)
    {
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        // logits = x[:, :inputDim] W + b
        var inputs = InputColumns(representations);
        return Engine.TensorAdd(Engine.TensorMatMul(inputs, _weights), _bias.Reshape(1, _numClasses));
    }

    /// <summary>
    /// The first <c>inputDim</c> feature columns of a [batch, features] tensor.
    /// </summary>
    private Tensor<T> InputColumns(Tensor<T> representations)
    {
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var batchSize = representations.Shape[0];
        return representations.Rank == 2 && representations.Shape[1] == _inputDim
            ? representations
            : Engine.TensorSlice(representations, new[] { 0, 0 }, new[] { batchSize, _inputDim });
    }

    private (T loss, Tensor<T> gradWeights, Tensor<T> gradBias) ComputeCrossEntropyLossAndGrad(
        Tensor<T> logits, int[] labels, Tensor<T> representations)
    {
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var batchSize = logits.Shape[0];

        // Softmax per row (max-shifted)
        var probs = Engine.TensorSoftmax(logits, 1);
        var targets = Engine.TensorOneHot<T>(new Tensor<int>(labels, new[] { labels.Length }), _numClasses);

        // Cross-entropy loss: -sum_b log(p[b, label_b] + 1e-8)
        var trueProbs = Engine.ReduceSum(Engine.TensorMultiply(probs, targets), new[] { 1 }, keepDims: false);
        T totalLoss = NumOps.Negate(Engine.TensorSum(
            Engine.TensorLog(Engine.TensorAddScalar(trueProbs, NumOps.FromDouble(1e-8)))));

        // Compute gradients: dL/dW = x^T (prob - target), dL/db = sum_b (prob - target), scaled by 1/batchSize
        var scale = NumOps.FromDouble(1.0 / batchSize);
        var grad = Engine.TensorSubtract(probs, targets);
        var inputs = InputColumns(representations);
        var gradWeights = Engine.TensorMultiplyScalar(
            Engine.TensorMatMul(Engine.TensorTranspose(inputs), grad), scale);
        var gradBias = Engine.TensorMultiplyScalar(
            Engine.ReduceSum(grad, new[] { 0 }, keepDims: false), scale);

        return (NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize)), gradWeights, gradBias);
    }

    private void UpdateParameters(Tensor<T> gradWeights, Tensor<T> gradBias, double lr)
    {
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var lrT = NumOps.FromDouble(lr);

        _weights = Engine.TensorSubtract(_weights, Engine.TensorMultiplyScalar(gradWeights, lrT));
        _bias = Engine.TensorSubtract(_bias, Engine.TensorMultiplyScalar(gradBias, lrT));
    }

    private int ComputeCorrect(Tensor<T> logits, int[] labels)
    {
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        // Predicted class per row is the first index of the row maximum
        var predicted = Engine.TensorArgMax(logits, 1).ToArray();
        // labels[b] is indexed per row, so a short label array throws exactly as master did
        return predicted.Select((p, b) => p == labels[b] ? 1 : 0).Sum();
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

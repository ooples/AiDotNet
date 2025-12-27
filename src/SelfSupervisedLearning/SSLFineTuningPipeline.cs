using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.SelfSupervisedLearning;

/// <summary>
/// Pipeline for fine-tuning SSL pretrained encoders on downstream tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> After SSL pretraining, you typically want to fine-tune
/// the encoder on a specific task with labeled data. This pipeline handles the
/// fine-tuning process with proper learning rate schedules and evaluation.</para>
///
/// <para><b>Fine-tuning strategies:</b></para>
/// <list type="bullet">
/// <item><b>Full fine-tuning:</b> Update all parameters</item>
/// <item><b>Linear probing:</b> Freeze encoder, train only classifier</item>
/// <item><b>Gradual unfreezing:</b> Unfreeze layers progressively</item>
/// </list>
/// </remarks>
public class SSLFineTuningPipeline<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly INeuralNetwork<T> _encoder;
    private readonly int _encoderOutputDim;
    private readonly int _numClasses;

    private FineTuningConfig _config;
    private T[] _classifierWeights;
    private T[] _classifierBias;

    /// <summary>
    /// Event raised for progress updates.
    /// </summary>
    public event Action<int, int, double>? OnProgress;

    /// <summary>
    /// Initializes a new fine-tuning pipeline.
    /// </summary>
    /// <param name="encoder">Pretrained encoder to fine-tune.</param>
    /// <param name="encoderOutputDim">Output dimension of the encoder.</param>
    /// <param name="numClasses">Number of classes for classification.</param>
    public SSLFineTuningPipeline(
        INeuralNetwork<T> encoder,
        int encoderOutputDim,
        int numClasses)
    {
        _encoder = encoder ?? throw new ArgumentNullException(nameof(encoder));
        _encoderOutputDim = encoderOutputDim;
        _numClasses = numClasses;
        _config = new FineTuningConfig();

        // Initialize classifier
        var rng = RandomHelper.Shared;
        var scale = Math.Sqrt(2.0 / encoderOutputDim);

        _classifierWeights = new T[encoderOutputDim * numClasses];
        _classifierBias = new T[numClasses];

        for (int i = 0; i < _classifierWeights.Length; i++)
        {
            _classifierWeights[i] = NumOps.FromDouble((rng.NextDouble() * 2 - 1) * scale);
        }
    }

    /// <summary>
    /// Configures fine-tuning parameters.
    /// </summary>
    public SSLFineTuningPipeline<T> WithConfig(Action<FineTuningConfig> configure)
    {
        configure(_config);
        return this;
    }

    /// <summary>
    /// Sets the fine-tuning strategy.
    /// </summary>
    public SSLFineTuningPipeline<T> WithStrategy(FineTuningStrategy strategy)
    {
        _config.Strategy = strategy;
        return this;
    }

    /// <summary>
    /// Fine-tunes the model on labeled data.
    /// </summary>
    /// <param name="trainData">Training data.</param>
    /// <param name="trainLabels">Training labels.</param>
    /// <param name="validData">Optional validation data.</param>
    /// <param name="validLabels">Optional validation labels.</param>
    /// <returns>Fine-tuning result with accuracy.</returns>
    public FineTuningResult<T> FineTune(
        Tensor<T> trainData, int[] trainLabels,
        Tensor<T>? validData = null, int[]? validLabels = null)
    {
        var result = new FineTuningResult<T>
        {
            TrainAccuracies = [],
            ValidAccuracies = []
        };

        var epochs = _config.Epochs ?? 100;
        var batchSize = _config.BatchSize ?? 256;
        var numSamples = trainData.Shape[0];

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            var lr = GetLearningRate(epoch, epochs);
            var freezeEncoder = ShouldFreezeEncoder(epoch, epochs);

            T epochLoss = NumOps.Zero;
            int correct = 0;

            for (int i = 0; i < numSamples; i += batchSize)
            {
                int actualBatch = Math.Min(batchSize, numSamples - i);

                var batchData = ExtractBatch(trainData, i, actualBatch);
                var batchLabels = trainLabels.Skip(i).Take(actualBatch).ToArray();

                // Forward pass
                var features = _encoder.ForwardWithMemory(batchData);
                var logits = ComputeLogits(features);

                // Compute loss
                var loss = ComputeCrossEntropyLoss(logits, batchLabels);
                epochLoss = NumOps.Add(epochLoss, loss);

                // Backward pass
                var gradLogits = ComputeGradients(logits, batchLabels);

                // Update classifier
                UpdateClassifier(features, gradLogits, lr);

                // Update encoder if not frozen
                if (!freezeEncoder)
                {
                    var gradFeatures = ComputeFeatureGradients(gradLogits);
                    _encoder.Backpropagate(gradFeatures);
                    UpdateEncoder(lr * _config.EncoderLRMultiplier);
                }

                correct += ComputeCorrect(logits, batchLabels);
            }

            var trainAcc = (double)correct / numSamples;
            result.TrainAccuracies.Add(trainAcc);

            // Validation
            if (validData is not null && validLabels is not null)
            {
                var validAcc = Evaluate(validData, validLabels);
                result.ValidAccuracies.Add(validAcc);

                if (validAcc > result.BestValidAccuracy)
                {
                    result.BestValidAccuracy = validAcc;
                    result.BestEpoch = epoch;
                }
            }

            OnProgress?.Invoke(epoch, epochs, trainAcc);
        }

        result.FinalTrainAccuracy = result.TrainAccuracies.LastOrDefault();
        result.FinalValidAccuracy = result.ValidAccuracies.LastOrDefault();

        return result;
    }

    /// <summary>
    /// Evaluates the model on test data.
    /// </summary>
    public double Evaluate(Tensor<T> testData, int[] testLabels)
    {
        var numSamples = testData.Shape[0];
        var batchSize = 256;
        int correct = 0;

        for (int i = 0; i < numSamples; i += batchSize)
        {
            int actualBatch = Math.Min(batchSize, numSamples - i);

            var batchData = ExtractBatch(testData, i, actualBatch);
            var batchLabels = testLabels.Skip(i).Take(actualBatch).ToArray();

            var features = _encoder.Predict(batchData);
            var logits = ComputeLogits(features);

            correct += ComputeCorrect(logits, batchLabels);
        }

        return (double)correct / numSamples;
    }

    private Tensor<T> ComputeLogits(Tensor<T> features)
    {
        var batchSize = features.Shape[0];
        var logits = new T[batchSize * _numClasses];

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < _numClasses; c++)
            {
                T sum = _classifierBias[c];
                for (int d = 0; d < _encoderOutputDim; d++)
                {
                    sum = NumOps.Add(sum,
                        NumOps.Multiply(features[b, d], _classifierWeights[d * _numClasses + c]));
                }
                logits[b * _numClasses + c] = sum;
            }
        }

        return new Tensor<T>(logits, [batchSize, _numClasses]);
    }

    private T ComputeCrossEntropyLoss(Tensor<T> logits, int[] labels)
    {
        var batchSize = logits.Shape[0];
        T totalLoss = NumOps.Zero;

        for (int b = 0; b < batchSize; b++)
        {
            // Softmax
            T maxLogit = logits[b, 0];
            for (int c = 1; c < _numClasses; c++)
            {
                if (NumOps.GreaterThan(logits[b, c], maxLogit))
                    maxLogit = logits[b, c];
            }

            T sumExp = NumOps.Zero;
            for (int c = 0; c < _numClasses; c++)
            {
                sumExp = NumOps.Add(sumExp, NumOps.Exp(NumOps.Subtract(logits[b, c], maxLogit)));
            }

            var trueLogit = logits[b, labels[b]];
            var logProb = NumOps.Subtract(NumOps.Subtract(trueLogit, maxLogit), NumOps.Log(sumExp));
            totalLoss = NumOps.Subtract(totalLoss, logProb);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    private Tensor<T> ComputeGradients(Tensor<T> logits, int[] labels)
    {
        var batchSize = logits.Shape[0];
        var grads = new T[batchSize * _numClasses];

        for (int b = 0; b < batchSize; b++)
        {
            // Softmax
            T maxLogit = logits[b, 0];
            for (int c = 1; c < _numClasses; c++)
            {
                if (NumOps.GreaterThan(logits[b, c], maxLogit))
                    maxLogit = logits[b, c];
            }

            T sumExp = NumOps.Zero;
            var probs = new T[_numClasses];
            for (int c = 0; c < _numClasses; c++)
            {
                probs[c] = NumOps.Exp(NumOps.Subtract(logits[b, c], maxLogit));
                sumExp = NumOps.Add(sumExp, probs[c]);
            }

            for (int c = 0; c < _numClasses; c++)
            {
                probs[c] = NumOps.Divide(probs[c], sumExp);
                grads[b * _numClasses + c] = probs[c];
                if (c == labels[b])
                {
                    grads[b * _numClasses + c] = NumOps.Subtract(grads[b * _numClasses + c], NumOps.One);
                }
            }
        }

        return new Tensor<T>(grads, [batchSize, _numClasses]);
    }

    private void UpdateClassifier(Tensor<T> features, Tensor<T> gradLogits, double lr)
    {
        var batchSize = features.Shape[0];
        var lrT = NumOps.FromDouble(lr / batchSize);

        for (int d = 0; d < _encoderOutputDim; d++)
        {
            for (int c = 0; c < _numClasses; c++)
            {
                T grad = NumOps.Zero;
                for (int b = 0; b < batchSize; b++)
                {
                    grad = NumOps.Add(grad, NumOps.Multiply(features[b, d], gradLogits[b, c]));
                }
                _classifierWeights[d * _numClasses + c] = NumOps.Subtract(
                    _classifierWeights[d * _numClasses + c],
                    NumOps.Multiply(lrT, grad));
            }
        }

        for (int c = 0; c < _numClasses; c++)
        {
            T grad = NumOps.Zero;
            for (int b = 0; b < batchSize; b++)
            {
                grad = NumOps.Add(grad, gradLogits[b, c]);
            }
            _classifierBias[c] = NumOps.Subtract(
                _classifierBias[c],
                NumOps.Multiply(lrT, grad));
        }
    }

    private Tensor<T> ComputeFeatureGradients(Tensor<T> gradLogits)
    {
        var batchSize = gradLogits.Shape[0];
        var gradFeatures = new T[batchSize * _encoderOutputDim];

        for (int b = 0; b < batchSize; b++)
        {
            for (int d = 0; d < _encoderOutputDim; d++)
            {
                T grad = NumOps.Zero;
                for (int c = 0; c < _numClasses; c++)
                {
                    grad = NumOps.Add(grad,
                        NumOps.Multiply(gradLogits[b, c], _classifierWeights[d * _numClasses + c]));
                }
                gradFeatures[b * _encoderOutputDim + d] = grad;
            }
        }

        return new Tensor<T>(gradFeatures, [batchSize, _encoderOutputDim]);
    }

    private void UpdateEncoder(double lr)
    {
        var lrT = NumOps.FromDouble(lr);
        var grads = _encoder.GetParameterGradients();
        var pars = _encoder.GetParameters();
        var newParams = new T[pars.Length];

        for (int i = 0; i < pars.Length; i++)
        {
            newParams[i] = NumOps.Subtract(pars[i], NumOps.Multiply(lrT, grads[i]));
        }

        _encoder.UpdateParameters(new Vector<T>(newParams));
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

    private double GetLearningRate(int epoch, int totalEpochs)
    {
        var baseLr = _config.LearningRate ?? 0.01;
        return baseLr * 0.5 * (1 + Math.Cos(Math.PI * epoch / totalEpochs));
    }

    private bool ShouldFreezeEncoder(int epoch, int totalEpochs)
    {
        return _config.Strategy switch
        {
            FineTuningStrategy.LinearProbing => true,
            FineTuningStrategy.GradualUnfreezing => epoch < totalEpochs / 4,
            _ => false
        };
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
}

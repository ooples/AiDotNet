using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// TabM implementation for classification tasks.
/// </summary>
/// <remarks>
/// <para>
/// TabMClassifier uses the TabM architecture with BatchEnsemble layers for multi-class
/// classification. It averages predictions across ensemble members and uses softmax
/// for probability outputs.
/// </para>
/// <para>
/// <b>For Beginners:</b> Use this model when you want to classify data into categories
/// with the benefits of an ensemble (multiple models voting).
///
/// How predictions work:
/// 1. Input passes through shared layers with per-member modulation
/// 2. Each ensemble member produces class logits
/// 3. Logits are averaged across members
/// 4. Softmax converts to probabilities
///
/// Benefits over single models:
/// - Reduced overfitting through ensemble averaging
/// - Better calibrated probability estimates
/// - More robust to noisy features
/// - Comparable speed to single models (parameter sharing)
///
/// Example:
/// <code>
/// var options = new TabMOptions&lt;double&gt; { NumEnsembleMembers = 4, HiddenDimensions = [256, 128] };
/// var classifier = new TabMClassifier&lt;double&gt;(10, 3, options);
///
/// var input = new Tensor&lt;double&gt;([32, 10]); // batch of 32 samples
/// var probabilities = classifier.PredictProbabilities(input);
/// var predictions = classifier.Predict(input);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TabMClassifier<T> : TabMBase<T>
{
    private readonly int _numClasses;
    private readonly BatchEnsembleLayer<T> _classificationHead;

    // Cache for backward pass
    private Tensor<T>? _backboneOutputCache;
    private Tensor<T>? _logitsCache;
    private Tensor<T>? _probabilitiesCache;

    /// <summary>
    /// Gets the number of output classes.
    /// </summary>
    public int NumClasses => _numClasses;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount => base.ParameterCount + _classificationHead.ParameterCount;

    /// <summary>
    /// Initializes a new instance of the TabMClassifier class.
    /// </summary>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="numClasses">Number of output classes.</param>
    /// <param name="options">Model configuration options.</param>
    public TabMClassifier(
        int numFeatures,
        int numClasses,
        TabMOptions<T>? options = null)
        : base(numFeatures, options)
    {
        if (numClasses < 2)
        {
            throw new ArgumentException("Number of classes must be at least 2", nameof(numClasses));
        }

        _numClasses = numClasses;

        // Classification head: BatchEnsemble layer to class logits
        _classificationHead = new BatchEnsembleLayer<T>(
            GetLastHiddenDim(),
            numClasses,
            Options.NumEnsembleMembers,
            Options.UseBias,
            Options.RankInitScale);
    }

    /// <summary>
    /// Performs the forward pass to get class logits (per member).
    /// </summary>
    /// <param name="features">Input features tensor [batch_size, num_features].</param>
    /// <returns>Class logits tensor [batch_size * num_members, num_classes].</returns>
    public Tensor<T> Forward(Tensor<T> features)
    {
        // Forward through backbone
        var backboneOutput = ForwardBackbone(features);
        _backboneOutputCache = backboneOutput;

        // Forward through classification head
        var logits = _classificationHead.Forward(backboneOutput);
        _logitsCache = logits;

        return logits;
    }

    /// <summary>
    /// Predicts class probabilities (averaged across ensemble members).
    /// </summary>
    /// <param name="features">Input features tensor [batch_size, num_features].</param>
    /// <returns>Probability tensor [batch_size, num_classes].</returns>
    public Tensor<T> PredictProbabilities(Tensor<T> features)
    {
        var memberLogits = Forward(features);

        // Average logits across members first, then softmax
        var avgLogits = AverageMemberOutputs(memberLogits, _numClasses);

        // Apply softmax
        var probabilities = ApplySoftmax(avgLogits);
        _probabilitiesCache = probabilities;

        return probabilities;
    }

    /// <summary>
    /// Predicts the most likely class for each sample.
    /// </summary>
    /// <param name="features">Input features tensor [batch_size, num_features].</param>
    /// <returns>Predicted class indices [batch_size].</returns>
    public int[] Predict(Tensor<T> features)
    {
        var probabilities = PredictProbabilities(features);
        int batchSize = probabilities.Shape[0];
        var predictions = new int[batchSize];

        for (int b = 0; b < batchSize; b++)
        {
            int maxIdx = 0;
            var maxProb = probabilities[b * _numClasses + 0];

            for (int c = 1; c < _numClasses; c++)
            {
                var prob = probabilities[b * _numClasses + c];
                if (NumOps.Compare(prob, maxProb) > 0)
                {
                    maxProb = prob;
                    maxIdx = c;
                }
            }

            predictions[b] = maxIdx;
        }

        return predictions;
    }

    /// <summary>
    /// Computes the cross-entropy loss.
    /// </summary>
    /// <param name="probabilities">Predicted probabilities [batch_size, num_classes].</param>
    /// <param name="targets">Target class indices [batch_size].</param>
    /// <returns>The average cross-entropy loss.</returns>
    public T ComputeCrossEntropyLoss(Tensor<T> probabilities, int[] targets)
    {
        int batchSize = probabilities.Shape[0];
        var totalLoss = NumOps.Zero;
        var epsilon = NumOps.FromDouble(1e-15);

        for (int b = 0; b < batchSize; b++)
        {
            int targetClass = targets[b];
            var prob = probabilities[b * _numClasses + targetClass];
            var safeProb = NumOps.Add(prob, epsilon);
            var loss = NumOps.Negate(NumOps.Log(safeProb));
            totalLoss = NumOps.Add(totalLoss, loss);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Performs the backward pass for the classification loss.
    /// </summary>
    /// <param name="targets">Target class indices [batch_size].</param>
    /// <returns>Gradient with respect to input features [batch_size, num_features].</returns>
    public Tensor<T> Backward(int[] targets)
    {
        if (_probabilitiesCache == null || _logitsCache == null || _backboneOutputCache == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        int batchSize = _probabilitiesCache.Shape[0];
        int expandedBatchSize = batchSize * Options.NumEnsembleMembers;

        // Gradient of cross-entropy + softmax w.r.t averaged logits
        var avgLogitsGrad = new Tensor<T>([batchSize, _numClasses]);
        var scale = NumOps.FromDouble(1.0 / batchSize);

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < _numClasses; c++)
            {
                var prob = _probabilitiesCache[b * _numClasses + c];
                var oneHot = targets[b] == c ? NumOps.One : NumOps.Zero;
                avgLogitsGrad[b * _numClasses + c] = NumOps.Multiply(
                    NumOps.Subtract(prob, oneHot), scale);
            }
        }

        // Expand gradient back to all members (each member gets the same gradient / num_members)
        var memberLogitsGrad = new Tensor<T>([expandedBatchSize, _numClasses]);
        var memberScale = NumOps.FromDouble(1.0 / Options.NumEnsembleMembers);

        for (int b = 0; b < batchSize; b++)
        {
            for (int m = 0; m < Options.NumEnsembleMembers; m++)
            {
                for (int c = 0; c < _numClasses; c++)
                {
                    memberLogitsGrad[(b * Options.NumEnsembleMembers + m) * _numClasses + c] =
                        NumOps.Multiply(avgLogitsGrad[b * _numClasses + c], memberScale);
                }
            }
        }

        // Backward through classification head
        var backboneGrad = _classificationHead.Backward(memberLogitsGrad);

        // Backward through backbone
        return BackwardBackbone(backboneGrad);
    }

    /// <summary>
    /// Performs a single training step.
    /// </summary>
    /// <param name="features">Input features tensor [batch_size, num_features].</param>
    /// <param name="targets">Target class indices [batch_size].</param>
    /// <param name="learningRate">The learning rate.</param>
    /// <returns>The training loss for this step.</returns>
    public T TrainStep(Tensor<T> features, int[] targets, T learningRate)
    {
        // Forward pass
        var probabilities = PredictProbabilities(features);

        // Compute loss
        var loss = ComputeCrossEntropyLoss(probabilities, targets);

        // Backward pass
        _ = Backward(targets);

        // Update parameters
        UpdateParameters(learningRate);

        // Reset for next iteration
        ResetState();

        return loss;
    }

    /// <summary>
    /// Computes classification accuracy.
    /// </summary>
    public T ComputeAccuracy(Tensor<T> features, int[] targets)
    {
        var predictions = Predict(features);
        int correct = 0;

        for (int i = 0; i < predictions.Length; i++)
        {
            if (predictions[i] == targets[i])
            {
                correct++;
            }
        }

        return NumOps.FromDouble((double)correct / predictions.Length);
    }

    /// <summary>
    /// Computes the macro F1 score across all classes.
    /// </summary>
    public T ComputeF1Score(Tensor<T> features, int[] targets)
    {
        var predictions = Predict(features);
        var totalF1 = NumOps.Zero;

        for (int c = 0; c < _numClasses; c++)
        {
            int tp = 0, fp = 0, fn = 0;

            for (int i = 0; i < predictions.Length; i++)
            {
                if (predictions[i] == c && targets[i] == c) tp++;
                else if (predictions[i] == c && targets[i] != c) fp++;
                else if (predictions[i] != c && targets[i] == c) fn++;
            }

            double precision = tp + fp > 0 ? (double)tp / (tp + fp) : 0.0;
            double recall = tp + fn > 0 ? (double)tp / (tp + fn) : 0.0;
            double f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0.0;

            totalF1 = NumOps.Add(totalF1, NumOps.FromDouble(f1));
        }

        return NumOps.Divide(totalF1, NumOps.FromDouble(_numClasses));
    }

    /// <summary>
    /// Gets predictive uncertainty as the entropy of the averaged predictions.
    /// </summary>
    /// <param name="features">Input features tensor [batch_size, num_features].</param>
    /// <returns>Entropy values [batch_size].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Uncertainty tells you how confident the model is:
    /// - Low entropy (near 0): Model is confident (probabilities peaked at one class)
    /// - High entropy: Model is uncertain (probabilities spread across classes)
    ///
    /// This can be useful for:
    /// - Identifying samples that need human review
    /// - Active learning (selecting samples to label)
    /// - Detecting out-of-distribution samples
    /// </para>
    /// </remarks>
    public Vector<T> GetPredictiveUncertainty(Tensor<T> features)
    {
        var probabilities = PredictProbabilities(features);
        int batchSize = probabilities.Shape[0];
        var uncertainty = new Vector<T>(batchSize);
        var epsilon = NumOps.FromDouble(1e-15);

        for (int b = 0; b < batchSize; b++)
        {
            var entropy = NumOps.Zero;
            for (int c = 0; c < _numClasses; c++)
            {
                var p = probabilities[b * _numClasses + c];
                var pSafe = NumOps.Add(p, epsilon);
                entropy = NumOps.Subtract(entropy,
                    NumOps.Multiply(p, NumOps.Log(pSafe)));
            }
            uncertainty[b] = entropy;
        }

        return uncertainty;
    }

    /// <summary>
    /// Applies softmax to convert logits to probabilities.
    /// </summary>
    private Tensor<T> ApplySoftmax(Tensor<T> logits)
    {
        int batchSize = logits.Shape[0];
        int numClasses = logits.Shape[1];
        var probabilities = new Tensor<T>(logits.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            // Find max for numerical stability
            var maxLogit = logits[b * numClasses + 0];
            for (int c = 1; c < numClasses; c++)
            {
                var logit = logits[b * numClasses + c];
                if (NumOps.Compare(logit, maxLogit) > 0)
                {
                    maxLogit = logit;
                }
            }

            // Compute exp(logits - max) and sum
            var sumExp = NumOps.Zero;
            for (int c = 0; c < numClasses; c++)
            {
                var shiftedLogit = NumOps.Subtract(logits[b * numClasses + c], maxLogit);
                var expVal = NumOps.Exp(shiftedLogit);
                probabilities[b * numClasses + c] = expVal;
                sumExp = NumOps.Add(sumExp, expVal);
            }

            // Normalize
            for (int c = 0; c < numClasses; c++)
            {
                probabilities[b * numClasses + c] = NumOps.Divide(
                    probabilities[b * numClasses + c], sumExp);
            }
        }

        return probabilities;
    }

    /// <summary>
    /// Updates all parameters including the classification head.
    /// </summary>
    public override void UpdateParameters(T learningRate)
    {
        base.UpdateParameters(learningRate);
        _classificationHead.UpdateParameters(learningRate);
    }

    /// <summary>
    /// Gets all parameters including the classification head.
    /// </summary>
    public override Vector<T> GetParameters()
    {
        var baseParams = base.GetParameters();
        var headParams = _classificationHead.GetParameters();

        var allParams = new T[baseParams.Length + headParams.Length];
        for (int i = 0; i < baseParams.Length; i++)
        {
            allParams[i] = baseParams[i];
        }
        for (int i = 0; i < headParams.Length; i++)
        {
            allParams[baseParams.Length + i] = headParams[i];
        }

        return new Vector<T>(allParams);
    }

    /// <summary>
    /// Resets internal state.
    /// </summary>
    public override void ResetState()
    {
        base.ResetState();
        _backboneOutputCache = null;
        _logitsCache = null;
        _probabilitiesCache = null;
        _classificationHead.ResetGradients();
    }
}

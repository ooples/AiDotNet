using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// TabNet implementation for classification tasks.
/// </summary>
/// <remarks>
/// <para>
/// TabNetClassifier extends the TabNet architecture for predicting discrete class labels.
/// It uses softmax activation on the output layer to produce class probabilities.
/// </para>
/// <para>
/// <b>For Beginners:</b> Use TabNetClassifier when you want to categorize data into
/// different classes (groups) from tabular data.
///
/// Example use cases:
/// - Customer churn prediction (will stay / will leave)
/// - Fraud detection (legitimate / fraudulent transaction)
/// - Medical diagnosis (healthy / disease A / disease B / ...)
/// - Credit risk assessment (low / medium / high risk)
///
/// Key features:
/// - **Automatic Feature Selection**: Learns which features matter for classification
/// - **Interpretability**: See which features drove each classification decision
/// - **Multi-class Support**: Works for binary (2 classes) or multi-class problems
/// - **Probability Outputs**: Returns confidence scores for each class
///
/// Basic usage:
/// <code>
/// var options = new TabNetOptions&lt;double&gt;
/// {
///     NumDecisionSteps = 5,
///     FeatureDimension = 64
/// };
/// var model = new TabNetClassifier&lt;double&gt;(inputFeatures: 10, numClasses: 3, options: options);
///
/// // Training
/// var probabilities = model.Forward(inputBatch);
/// var loss = model.ComputeCrossEntropyLoss(probabilities, targets);
/// var gradient = model.ComputeCrossEntropyGradient(probabilities, targets);
/// model.Backward(gradient);
/// model.UpdateParameters(learningRate);
///
/// // Prediction
/// var predictions = model.Predict(testBatch); // Returns class indices
/// var probs = model.PredictProbabilities(testBatch); // Returns class probabilities
/// </code>
/// </para>
/// <para>
/// Reference: "TabNet: Attentive Interpretable Tabular Learning" (Arik &amp; Pfister, AAAI 2021)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TabNetClassifier<T> : TabNetBase<T>
{
    private readonly int _numClasses;

    /// <summary>
    /// Initializes a new instance of the TabNetClassifier class.
    /// </summary>
    /// <param name="inputFeatures">Number of input features (columns in your data).</param>
    /// <param name="numClasses">Number of classes to predict.</param>
    /// <param name="options">TabNet configuration options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates a TabNet model that predicts class labels.
    ///
    /// For binary classification (yes/no, true/false):
    /// <code>
    /// // 10 features, 2 classes (churn / no churn)
    /// var model = new TabNetClassifier&lt;double&gt;(10, 2);
    /// </code>
    ///
    /// For multi-class classification:
    /// <code>
    /// // 15 features, 5 classes (product categories)
    /// var model = new TabNetClassifier&lt;double&gt;(15, 5);
    /// </code>
    /// </para>
    /// </remarks>
    public TabNetClassifier(int inputFeatures, int numClasses, TabNetOptions<T>? options = null)
        : base(inputFeatures, numClasses, options)
    {
        _numClasses = numClasses;
    }

    /// <summary>
    /// Performs forward pass with softmax activation.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch_size, input_features].</param>
    /// <returns>Class probabilities of shape [batch_size, num_classes].</returns>
    /// <remarks>
    /// <para>
    /// The output passes through softmax to convert raw scores into probabilities:
    /// - All outputs are between 0 and 1
    /// - Outputs sum to 1 across classes
    /// - Higher values indicate higher confidence for that class
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This returns the probability for each class.
    ///
    /// For example, with 3 classes:
    /// - Output [0.8, 0.15, 0.05] means:
    ///   - 80% confident it's class 0
    ///   - 15% confident it's class 1
    ///   - 5% confident it's class 2
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        var logits = base.Forward(input);
        return ApplySoftmax(logits);
    }

    /// <summary>
    /// Gets the raw logits without softmax.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Raw logits before softmax.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Logits are the raw scores before converting to probabilities.
    ///
    /// You might need logits when:
    /// - Using certain loss functions (some expect logits, not probabilities)
    /// - Debugging the model
    /// - Implementing custom classification logic
    /// </para>
    /// </remarks>
    public Tensor<T> ForwardLogits(Tensor<T> input)
    {
        return base.Forward(input);
    }

    /// <summary>
    /// Predicts class labels (indices of highest probability classes).
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Array of predicted class indices for each sample.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This returns the predicted class for each sample.
    ///
    /// Example:
    /// <code>
    /// var predictions = model.Predict(testBatch);
    /// // predictions[0] = 2 means the first sample is predicted as class 2
    /// // predictions[1] = 0 means the second sample is predicted as class 0
    /// </code>
    /// </para>
    /// </remarks>
    public int[] Predict(Tensor<T> input)
    {
        var probabilities = Forward(input);
        int batchSize = probabilities.Shape[0];
        var predictions = new int[batchSize];

        for (int b = 0; b < batchSize; b++)
        {
            var maxProb = NumOps.FromDouble(double.NegativeInfinity);
            int maxIdx = 0;

            for (int c = 0; c < _numClasses; c++)
            {
                var prob = probabilities[b * _numClasses + c];
                if (NumOps.GreaterThan(prob, maxProb))
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
    /// Gets class probabilities for each sample.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Class probabilities tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Same as Forward(), returns probabilities for each class.
    ///
    /// Useful for:
    /// - Getting confidence scores for predictions
    /// - Thresholding decisions (e.g., only accept if confidence > 90%)
    /// - Analyzing uncertain cases
    /// </para>
    /// </remarks>
    public Tensor<T> PredictProbabilities(Tensor<T> input)
    {
        return Forward(input);
    }

    /// <summary>
    /// Applies softmax activation to convert logits to probabilities.
    /// </summary>
    private Tensor<T> ApplySoftmax(Tensor<T> logits)
    {
        int batchSize = logits.Shape[0];
        var probabilities = new Tensor<T>(logits.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            // Find max for numerical stability
            var maxLogit = NumOps.FromDouble(double.NegativeInfinity);
            for (int c = 0; c < _numClasses; c++)
            {
                var logit = logits[b * _numClasses + c];
                if (NumOps.GreaterThan(logit, maxLogit))
                {
                    maxLogit = logit;
                }
            }

            // Compute exp(logit - max) and sum
            var expSum = NumOps.Zero;
            for (int c = 0; c < _numClasses; c++)
            {
                var shiftedLogit = NumOps.Subtract(logits[b * _numClasses + c], maxLogit);
                var expValue = NumOps.Exp(shiftedLogit);
                probabilities[b * _numClasses + c] = expValue;
                expSum = NumOps.Add(expSum, expValue);
            }

            // Normalize
            for (int c = 0; c < _numClasses; c++)
            {
                probabilities[b * _numClasses + c] = NumOps.Divide(
                    probabilities[b * _numClasses + c], expSum);
            }
        }

        return probabilities;
    }

    /// <summary>
    /// Computes the cross-entropy loss for classification.
    /// </summary>
    /// <param name="probabilities">Model output probabilities.</param>
    /// <param name="targets">Ground truth class indices (as integers encoded in T).</param>
    /// <returns>The cross-entropy loss value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Cross-entropy measures how well probabilities match true labels.
    ///
    /// It works by:
    /// 1. Looking at the probability assigned to the correct class
    /// 2. Taking the negative log of that probability
    /// 3. Averaging across all samples
    ///
    /// Lower cross-entropy = better predictions.
    /// - If model is 100% confident and correct: loss ≈ 0
    /// - If model is 100% confident and wrong: loss → ∞
    ///
    /// Cross-entropy encourages the model to be both accurate AND confident.
    /// </para>
    /// </remarks>
    public T ComputeCrossEntropyLoss(Tensor<T> probabilities, Tensor<T> targets)
    {
        int batchSize = probabilities.Shape[0];
        var totalLoss = NumOps.Zero;
        var epsilon = NumOps.FromDouble(1e-15); // For numerical stability

        for (int b = 0; b < batchSize; b++)
        {
            // Get the true class index
            int trueClass = (int)NumOps.ToDouble(targets[b]);

            // Get the probability of the true class
            var prob = probabilities[b * _numClasses + trueClass];

            // Add epsilon for numerical stability
            prob = NumOps.Add(prob, epsilon);

            // Cross-entropy: -log(prob of true class)
            var loss = NumOps.Negate(NumOps.Log(prob));
            totalLoss = NumOps.Add(totalLoss, loss);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Computes cross-entropy loss with one-hot encoded targets.
    /// </summary>
    /// <param name="probabilities">Model output probabilities.</param>
    /// <param name="oneHotTargets">One-hot encoded targets [batch_size, num_classes].</param>
    /// <returns>The cross-entropy loss value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when your targets are one-hot encoded.
    ///
    /// One-hot encoding represents classes as vectors:
    /// - Class 0: [1, 0, 0]
    /// - Class 1: [0, 1, 0]
    /// - Class 2: [0, 0, 1]
    ///
    /// This format is useful when you have soft labels (e.g., [0.8, 0.1, 0.1])
    /// or when working with label smoothing.
    /// </para>
    /// </remarks>
    public T ComputeCrossEntropyLossOneHot(Tensor<T> probabilities, Tensor<T> oneHotTargets)
    {
        int batchSize = probabilities.Shape[0];
        var totalLoss = NumOps.Zero;
        var epsilon = NumOps.FromDouble(1e-15);

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < _numClasses; c++)
            {
                var target = oneHotTargets[b * _numClasses + c];
                var prob = NumOps.Add(probabilities[b * _numClasses + c], epsilon);
                // -target * log(prob)
                var term = NumOps.Multiply(NumOps.Negate(target), NumOps.Log(prob));
                totalLoss = NumOps.Add(totalLoss, term);
            }
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Computes the gradient of cross-entropy loss with softmax.
    /// </summary>
    /// <param name="probabilities">Model output probabilities.</param>
    /// <param name="targets">Ground truth class indices.</param>
    /// <returns>Gradient tensor for backpropagation.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This computes how to adjust the output to improve predictions.
    ///
    /// For softmax + cross-entropy, the gradient has a beautiful simple form:
    /// gradient = probability - one_hot_target
    ///
    /// This means:
    /// - For the correct class: gradient = prob - 1 (push probability toward 1)
    /// - For wrong classes: gradient = prob - 0 (push probability toward 0)
    /// </para>
    /// </remarks>
    public Tensor<T> ComputeCrossEntropyGradient(Tensor<T> probabilities, Tensor<T> targets)
    {
        int batchSize = probabilities.Shape[0];
        var gradient = new Tensor<T>(probabilities.Shape);
        var scale = NumOps.FromDouble(1.0 / batchSize);

        for (int b = 0; b < batchSize; b++)
        {
            int trueClass = (int)NumOps.ToDouble(targets[b]);

            for (int c = 0; c < _numClasses; c++)
            {
                var prob = probabilities[b * _numClasses + c];
                var target = (c == trueClass) ? NumOps.One : NumOps.Zero;

                // gradient = (prob - target) / batch_size
                gradient[b * _numClasses + c] = NumOps.Multiply(
                    NumOps.Subtract(prob, target), scale);
            }
        }

        return gradient;
    }

    /// <summary>
    /// Computes the gradient with one-hot encoded targets.
    /// </summary>
    public Tensor<T> ComputeCrossEntropyGradientOneHot(Tensor<T> probabilities, Tensor<T> oneHotTargets)
    {
        int batchSize = probabilities.Shape[0];
        var gradient = new Tensor<T>(probabilities.Shape);
        var scale = NumOps.FromDouble(1.0 / batchSize);

        for (int i = 0; i < probabilities.Length; i++)
        {
            gradient[i] = NumOps.Multiply(
                NumOps.Subtract(probabilities[i], oneHotTargets[i]), scale);
        }

        return gradient;
    }

    /// <summary>
    /// Computes the total loss including sparsity regularization.
    /// </summary>
    public T ComputeTotalLoss(Tensor<T> probabilities, Tensor<T> targets)
    {
        var crossEntropyLoss = ComputeCrossEntropyLoss(probabilities, targets);
        var sparsityLoss = ComputeSparsityLoss();
        return NumOps.Add(crossEntropyLoss, sparsityLoss);
    }

    /// <summary>
    /// Performs a single training step.
    /// </summary>
    /// <param name="input">Input batch.</param>
    /// <param name="targets">Target class indices.</param>
    /// <param name="learningRate">Learning rate for parameter updates.</param>
    /// <returns>The total loss for this batch.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Convenience method for one complete training iteration.
    ///
    /// Performs:
    /// 1. Forward pass → probabilities
    /// 2. Loss computation
    /// 3. Backward pass → gradients
    /// 4. Parameter update
    ///
    /// Call repeatedly with batches to train the model.
    /// </para>
    /// </remarks>
    public T TrainStep(Tensor<T> input, Tensor<T> targets, T learningRate)
    {
        // Forward pass
        var probabilities = Forward(input);

        // Compute loss
        var loss = ComputeTotalLoss(probabilities, targets);

        // Compute gradient (using logits for proper gradient flow)
        var lossGradient = ComputeCrossEntropyGradient(probabilities, targets);

        // Backward pass
        Backward(lossGradient);

        // Update parameters
        UpdateParameters(learningRate);

        return loss;
    }

    /// <summary>
    /// Computes the accuracy (percentage of correct predictions).
    /// </summary>
    /// <param name="predictions">Predicted class indices.</param>
    /// <param name="targets">True class indices.</param>
    /// <returns>Accuracy as a value between 0 and 1.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Accuracy is the simplest classification metric.
    ///
    /// Accuracy = (number of correct predictions) / (total predictions)
    ///
    /// For example:
    /// - 90 correct out of 100 → accuracy = 0.9 (90%)
    ///
    /// Note: Accuracy can be misleading with imbalanced classes.
    /// If 95% of data is class A, predicting "A" always gives 95% accuracy
    /// but the model hasn't learned anything useful.
    /// </para>
    /// </remarks>
    public T ComputeAccuracy(int[] predictions, Tensor<T> targets)
    {
        int correct = 0;
        int total = predictions.Length;

        for (int i = 0; i < total; i++)
        {
            int trueClass = (int)NumOps.ToDouble(targets[i]);
            if (predictions[i] == trueClass)
            {
                correct++;
            }
        }

        return NumOps.FromDouble((double)correct / total);
    }

    /// <summary>
    /// Computes the F1 score for binary classification.
    /// </summary>
    /// <param name="predictions">Predicted class indices.</param>
    /// <param name="targets">True class indices.</param>
    /// <param name="positiveClass">Which class is considered "positive" (default: 1).</param>
    /// <returns>F1 score between 0 and 1.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> F1 score balances precision and recall.
    ///
    /// - Precision: Of all predicted positives, how many were actually positive?
    /// - Recall: Of all actual positives, how many did we find?
    /// - F1: Harmonic mean of precision and recall
    ///
    /// F1 is especially useful for imbalanced datasets where accuracy is misleading.
    ///
    /// F1 = 2 * (precision * recall) / (precision + recall)
    /// </para>
    /// </remarks>
    public T ComputeF1Score(int[] predictions, Tensor<T> targets, int positiveClass = 1)
    {
        int truePositive = 0;
        int falsePositive = 0;
        int falseNegative = 0;

        for (int i = 0; i < predictions.Length; i++)
        {
            int trueClass = (int)NumOps.ToDouble(targets[i]);
            int predClass = predictions[i];

            if (predClass == positiveClass && trueClass == positiveClass)
            {
                truePositive++;
            }
            else if (predClass == positiveClass && trueClass != positiveClass)
            {
                falsePositive++;
            }
            else if (predClass != positiveClass && trueClass == positiveClass)
            {
                falseNegative++;
            }
        }

        if (truePositive == 0)
        {
            return NumOps.Zero;
        }

        double precision = (double)truePositive / (truePositive + falsePositive);
        double recall = (double)truePositive / (truePositive + falseNegative);
        double f1 = 2 * precision * recall / (precision + recall);

        return NumOps.FromDouble(f1);
    }
}

using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// FT-Transformer implementation for classification tasks.
/// </summary>
/// <remarks>
/// <para>
/// FTTransformerClassifier applies the FT-Transformer architecture to multi-class
/// classification problems. It uses the [CLS] token output with a linear classification head
/// and softmax activation.
/// </para>
/// <para>
/// <b>For Beginners:</b> Use this model when you want to predict categories
/// (like spam/not spam, customer segments, product categories, etc.).
///
/// How it works:
/// 1. Features are tokenized and processed by transformer layers
/// 2. The [CLS] token captures information from all features
/// 3. A linear layer maps the [CLS] representation to class logits
/// 4. Softmax converts logits to probabilities
///
/// Example:
/// <code>
/// // Create classifier for 10 features predicting 3 classes
/// var options = new FTTransformerOptions&lt;double&gt; { EmbeddingDimension = 128, NumLayers = 2 };
/// var classifier = new FTTransformerClassifier&lt;double&gt;(10, 3, options);
///
/// // Forward pass
/// var input = new Tensor&lt;double&gt;([32, 10]); // batch of 32 samples, 10 features
/// var probabilities = classifier.PredictProbabilities(input);
/// var predictions = classifier.Predict(input);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class FTTransformerClassifier<T> : FTTransformerBase<T>
{
    private readonly int _numClasses;
    private readonly FullyConnectedLayer<T> _classificationHead;

    // Cache for backward pass
    private Tensor<T>? _clsOutputCache;
    private Tensor<T>? _logitsCache;
    private Tensor<T>? _probabilitiesCache;

    /// <summary>
    /// Gets the number of output classes.
    /// </summary>
    public int NumClasses => _numClasses;

    /// <summary>
    /// Gets the total number of trainable parameters including the classification head.
    /// </summary>
    public override int ParameterCount => base.ParameterCount + _classificationHead.ParameterCount;

    /// <summary>
    /// Initializes a new instance of the FTTransformerClassifier class.
    /// </summary>
    /// <param name="numNumericalFeatures">Number of numerical input features.</param>
    /// <param name="numClasses">Number of output classes.</param>
    /// <param name="options">Model configuration options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creating a classifier:
    /// - numNumericalFeatures: How many number columns in your input data
    /// - numClasses: How many categories to predict (2 for binary, more for multi-class)
    /// - options: Model configuration (see FTTransformerOptions for details)
    ///
    /// For binary classification, numClasses should be 2.
    /// </para>
    /// </remarks>
    public FTTransformerClassifier(
        int numNumericalFeatures,
        int numClasses,
        FTTransformerOptions<T>? options = null)
        : base(numNumericalFeatures, options)
    {
        if (numClasses < 2)
        {
            throw new ArgumentException("Number of classes must be at least 2", nameof(numClasses));
        }

        _numClasses = numClasses;

        // Classification head: Linear layer from embedding dimension to number of classes
        _classificationHead = new FullyConnectedLayer<T>(
            Options.EmbeddingDimension,
            numClasses,
            (IActivationFunction<T>?)null);  // No activation, softmax applied separately
    }

    /// <summary>
    /// Performs the forward pass to get class logits.
    /// </summary>
    /// <param name="numericalFeatures">Numerical features tensor [batch_size, num_numerical].</param>
    /// <param name="categoricalIndices">Categorical feature indices [batch_size, num_categorical] or null.</param>
    /// <returns>Class logits tensor [batch_size, num_classes].</returns>
    public Tensor<T> Forward(Tensor<T> numericalFeatures, Matrix<int>? categoricalIndices = null)
    {
        // Get [CLS] representation from backbone
        var clsOutput = ForwardBackbone(numericalFeatures, categoricalIndices);
        _clsOutputCache = clsOutput;

        // Apply classification head
        var logits = _classificationHead.Forward(clsOutput);
        _logitsCache = logits;

        return logits;
    }

    /// <summary>
    /// Performs the forward pass with numerical features only.
    /// </summary>
    /// <param name="numericalFeatures">Numerical features tensor [batch_size, num_numerical].</param>
    /// <returns>Class logits tensor [batch_size, num_classes].</returns>
    public Tensor<T> Forward(Tensor<T> numericalFeatures)
    {
        return Forward(numericalFeatures, null);
    }

    /// <summary>
    /// Predicts class probabilities using softmax.
    /// </summary>
    /// <param name="numericalFeatures">Numerical features tensor [batch_size, num_numerical].</param>
    /// <param name="categoricalIndices">Categorical feature indices [batch_size, num_categorical] or null.</param>
    /// <returns>Probability tensor [batch_size, num_classes].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This returns the probability of each class:
    /// - Output values are between 0 and 1
    /// - Values in each row sum to 1
    /// - Higher values indicate more confident predictions
    ///
    /// Example output for 3 classes: [0.7, 0.2, 0.1] means:
    /// - 70% chance of class 0
    /// - 20% chance of class 1
    /// - 10% chance of class 2
    /// </para>
    /// </remarks>
    public Tensor<T> PredictProbabilities(Tensor<T> numericalFeatures, Matrix<int>? categoricalIndices = null)
    {
        var logits = Forward(numericalFeatures, categoricalIndices);
        var probabilities = ApplySoftmax(logits);
        _probabilitiesCache = probabilities;
        return probabilities;
    }

    /// <summary>
    /// Predicts class probabilities with numerical features only.
    /// </summary>
    public Tensor<T> PredictProbabilities(Tensor<T> numericalFeatures)
    {
        return PredictProbabilities(numericalFeatures, null);
    }

    /// <summary>
    /// Predicts the most likely class for each sample.
    /// </summary>
    /// <param name="numericalFeatures">Numerical features tensor [batch_size, num_numerical].</param>
    /// <param name="categoricalIndices">Categorical feature indices [batch_size, num_categorical] or null.</param>
    /// <returns>Predicted class indices [batch_size].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This returns the most likely class for each sample:
    /// - Output is an array of class indices (0, 1, 2, etc.)
    /// - The class with highest probability is selected
    ///
    /// Example: If you have 3 samples and 4 classes, output might be [2, 0, 1]
    /// meaning sample 0 is class 2, sample 1 is class 0, sample 2 is class 1.
    /// </para>
    /// </remarks>
    public int[] Predict(Tensor<T> numericalFeatures, Matrix<int>? categoricalIndices = null)
    {
        var probabilities = PredictProbabilities(numericalFeatures, categoricalIndices);
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
    /// Predicts class indices with numerical features only.
    /// </summary>
    public int[] Predict(Tensor<T> numericalFeatures)
    {
        return Predict(numericalFeatures, null);
    }

    /// <summary>
    /// Computes the cross-entropy loss.
    /// </summary>
    /// <param name="probabilities">Predicted probabilities [batch_size, num_classes].</param>
    /// <param name="targets">Target class indices [batch_size].</param>
    /// <returns>The average cross-entropy loss.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Cross-entropy loss measures how wrong the predictions are:
    /// - Lower loss = better predictions
    /// - Loss of 0 = perfect predictions
    /// - Penalizes confident wrong predictions more heavily
    ///
    /// Formula: -log(probability of correct class)
    /// </para>
    /// </remarks>
    public T ComputeCrossEntropyLoss(Tensor<T> probabilities, int[] targets)
    {
        int batchSize = probabilities.Shape[0];
        var totalLoss = NumOps.Zero;
        var epsilon = NumOps.FromDouble(1e-15);  // Numerical stability

        for (int b = 0; b < batchSize; b++)
        {
            int targetClass = targets[b];
            var prob = probabilities[b * _numClasses + targetClass];
            var safeProp = NumOps.Add(prob, epsilon);  // Avoid log(0)
            var loss = NumOps.Negate(NumOps.Log(safeProp));
            totalLoss = NumOps.Add(totalLoss, loss);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Performs the backward pass for the classification loss.
    /// </summary>
    /// <param name="targets">Target class indices [batch_size].</param>
    /// <returns>Gradient with respect to numerical input [batch_size, num_numerical].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The backward pass computes how to adjust the model weights
    /// to reduce the loss. It propagates gradients backward through:
    /// 1. Softmax (probability computation)
    /// 2. Classification head
    /// 3. Transformer layers
    /// 4. Feature tokenizer
    /// </para>
    /// </remarks>
    public Tensor<T> Backward(int[] targets)
    {
        if (_probabilitiesCache == null || _logitsCache == null || _clsOutputCache == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        int batchSize = _probabilitiesCache.Shape[0];

        // Gradient of cross-entropy + softmax: probabilities - one_hot(targets)
        var logitsGrad = new Tensor<T>(_logitsCache.Shape);
        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < _numClasses; c++)
            {
                var prob = _probabilitiesCache[b * _numClasses + c];
                var oneHot = targets[b] == c ? NumOps.One : NumOps.Zero;
                logitsGrad[b * _numClasses + c] = NumOps.Subtract(prob, oneHot);
            }
        }

        // Scale by 1/batch_size
        var scale = NumOps.FromDouble(1.0 / batchSize);
        for (int i = 0; i < logitsGrad.Length; i++)
        {
            logitsGrad[i] = NumOps.Multiply(logitsGrad[i], scale);
        }

        // Backward through classification head
        var clsGrad = _classificationHead.Backward(logitsGrad);

        // Backward through backbone
        return BackwardBackbone(clsGrad);
    }

    /// <summary>
    /// Performs a single training step.
    /// </summary>
    /// <param name="numericalFeatures">Numerical features tensor [batch_size, num_numerical].</param>
    /// <param name="targets">Target class indices [batch_size].</param>
    /// <param name="learningRate">The learning rate.</param>
    /// <param name="categoricalIndices">Categorical feature indices [batch_size, num_categorical] or null.</param>
    /// <returns>The training loss for this step.</returns>
    public T TrainStep(
        Tensor<T> numericalFeatures,
        int[] targets,
        T learningRate,
        Matrix<int>? categoricalIndices = null)
    {
        // Forward pass
        var probabilities = PredictProbabilities(numericalFeatures, categoricalIndices);

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
    /// <param name="numericalFeatures">Numerical features tensor [batch_size, num_numerical].</param>
    /// <param name="targets">Target class indices [batch_size].</param>
    /// <param name="categoricalIndices">Categorical feature indices [batch_size, num_categorical] or null.</param>
    /// <returns>Accuracy as a value between 0 and 1.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Accuracy is the fraction of correct predictions:
    /// - Accuracy = (number correct) / (total samples)
    /// - 1.0 = 100% correct, 0.0 = 0% correct
    /// </para>
    /// </remarks>
    public T ComputeAccuracy(Tensor<T> numericalFeatures, int[] targets, Matrix<int>? categoricalIndices = null)
    {
        var predictions = Predict(numericalFeatures, categoricalIndices);
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
    /// <param name="numericalFeatures">Numerical features tensor [batch_size, num_numerical].</param>
    /// <param name="targets">Target class indices [batch_size].</param>
    /// <param name="categoricalIndices">Categorical feature indices [batch_size, num_categorical] or null.</param>
    /// <returns>The macro F1 score.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> F1 score balances precision and recall:
    /// - F1 = 2 * (precision * recall) / (precision + recall)
    /// - Macro F1 averages F1 scores across all classes
    /// - Good for imbalanced datasets where accuracy can be misleading
    /// </para>
    /// </remarks>
    public T ComputeF1Score(Tensor<T> numericalFeatures, int[] targets, Matrix<int>? categoricalIndices = null)
    {
        var predictions = Predict(numericalFeatures, categoricalIndices);
        var totalF1 = NumOps.Zero;

        for (int c = 0; c < _numClasses; c++)
        {
            int truePositives = 0;
            int falsePositives = 0;
            int falseNegatives = 0;

            for (int i = 0; i < predictions.Length; i++)
            {
                if (predictions[i] == c && targets[i] == c)
                    truePositives++;
                else if (predictions[i] == c && targets[i] != c)
                    falsePositives++;
                else if (predictions[i] != c && targets[i] == c)
                    falseNegatives++;
            }

            double precision = truePositives + falsePositives > 0
                ? (double)truePositives / (truePositives + falsePositives) : 0.0;
            double recall = truePositives + falseNegatives > 0
                ? (double)truePositives / (truePositives + falseNegatives) : 0.0;
            double f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0.0;

            totalF1 = NumOps.Add(totalF1, NumOps.FromDouble(f1));
        }

        return NumOps.Divide(totalF1, NumOps.FromDouble(_numClasses));
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
    /// Sets all parameters including the classification head.
    /// </summary>
    public override void SetParameters(Vector<T> parameters)
    {
        int baseCount = base.ParameterCount - _classificationHead.ParameterCount;
        var baseParams = new Vector<T>(baseCount);
        for (int i = 0; i < baseCount; i++)
        {
            baseParams[i] = parameters[i];
        }
        base.SetParameters(baseParams);

        int headCount = _classificationHead.ParameterCount;
        var headParams = new Vector<T>(headCount);
        for (int i = 0; i < headCount; i++)
        {
            headParams[i] = parameters[baseCount + i];
        }
        _classificationHead.SetParameters(headParams);
    }

    /// <summary>
    /// Resets internal state.
    /// </summary>
    public override void ResetState()
    {
        base.ResetState();
        _clsOutputCache = null;
        _logitsCache = null;
        _probabilitiesCache = null;
        _classificationHead.ResetState();
    }
}

using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Classification.MultiLabel;

/// <summary>
/// Implements the Classifier Chain approach for multi-label classification.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Classifier Chain is an extension of Binary Relevance that models label dependencies by training
/// classifiers in a chain, where each classifier uses the predictions of previous classifiers as
/// additional features.
/// </para>
/// <para>
/// <b>For Beginners:</b> Classifier Chain improves on Binary Relevance by capturing label correlations:
///
/// Consider predicting movie genres:
/// 1. First classifier asks: "Is this a sequel?" (uses only movie features)
/// 2. Second classifier asks: "Is this action?" (uses movie features + "is sequel" prediction)
/// 3. Third classifier asks: "Is this comedy?" (uses movie features + "is sequel" + "is action")
/// 4. And so on...
///
/// By including previous predictions as features, each classifier can learn from label dependencies.
/// For example, if a movie is classified as "sequel", the next classifier knows this and can adjust
/// its predictions accordingly (sequels are more likely to be action movies).
///
/// Pros:
/// - Captures label dependencies
/// - Still relatively simple to implement
/// - Often outperforms Binary Relevance
///
/// Cons:
/// - Chain order matters (different orders can give different results)
/// - Errors can propagate through the chain
/// - Order selection can be tricky
///
/// The chain order can be specified manually or determined randomly. For best results, consider
/// training multiple chains with different orders and combining their predictions (Ensemble of Classifier Chains).
/// </para>
/// </remarks>
public class ClassifierChainClassifier<T> : MultiLabelClassifierBase<T>
{
    #region Fields

    /// <summary>
    /// Factory function to create binary classifiers for each position in the chain.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This function creates new binary classifiers. The Classifier Chain
    /// creates one classifier per label, but unlike Binary Relevance, each classifier receives
    /// additional features from previous predictions in the chain.
    /// </para>
    /// </remarks>
    private readonly Func<IClassifier<T>> _classifierFactory;

    /// <summary>
    /// The trained binary classifiers in chain order.
    /// </summary>
    private IClassifier<T>[]? _chainClassifiers;

    /// <summary>
    /// The order of labels in the chain (indices into the label array).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This determines which label is predicted first, second, etc.
    /// The order matters because later classifiers use earlier predictions as features.
    /// For example, if order is [2, 0, 1], then label 2 is predicted first (with no
    /// additional features), then label 0 (with label 2's prediction), then label 1
    /// (with labels 2 and 0's predictions).
    /// </para>
    /// </remarks>
    private int[]? _chainOrder;

    /// <summary>
    /// Random number generator for shuffling chain order.
    /// </summary>
    private readonly Random _random;

    /// <summary>
    /// Whether to use a random chain order.
    /// </summary>
    private readonly bool _useRandomOrder;

    /// <summary>
    /// User-specified chain order (null means use natural or random order).
    /// </summary>
    private readonly int[]? _specifiedOrder;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the ClassifierChainClassifier class with a classifier factory.
    /// </summary>
    /// <param name="classifierFactory">A function that creates binary classifiers. Called once per label.</param>
    /// <param name="chainOrder">The order of labels in the chain. If null, uses natural order (0, 1, 2, ...).</param>
    /// <param name="useRandomOrder">If true and chainOrder is null, uses a random order.</param>
    /// <param name="seed">Random seed for reproducibility when using random order.</param>
    /// <param name="options">Configuration options for the classifier.</param>
    /// <param name="regularization">Regularization method to prevent overfitting.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> To create a Classifier Chain:
    ///
    /// 1. Provide a factory function that creates binary classifiers
    /// 2. Optionally specify the chain order (which label to predict first, second, etc.)
    ///
    /// Example usage:
    /// <code>
    /// // Natural order (0, 1, 2, ...)
    /// var cc = new ClassifierChainClassifier&lt;double&gt;(() => new LogisticRegression&lt;double&gt;());
    ///
    /// // Specific order
    /// var cc = new ClassifierChainClassifier&lt;double&gt;(
    ///     () => new LogisticRegression&lt;double&gt;(),
    ///     chainOrder: new int[] { 2, 0, 1 });  // Predict label 2 first, then 0, then 1
    ///
    /// // Random order
    /// var cc = new ClassifierChainClassifier&lt;double&gt;(
    ///     () => new LogisticRegression&lt;double&gt;(),
    ///     useRandomOrder: true);
    /// </code>
    /// </para>
    /// </remarks>
    public ClassifierChainClassifier(
        Func<IClassifier<T>> classifierFactory,
        int[]? chainOrder = null,
        bool useRandomOrder = false,
        int? seed = null,
        ClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _classifierFactory = classifierFactory ?? throw new ArgumentNullException(nameof(classifierFactory));
        _specifiedOrder = chainOrder;
        _useRandomOrder = useRandomOrder;
        _random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    #endregion

    #region Training

    /// <summary>
    /// Core implementation of multi-label training using Classifier Chain.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The multi-label target matrix.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method trains classifiers in a chain:
    ///
    /// 1. Determine the chain order (specified, random, or natural)
    /// 2. For each position in the chain:
    ///    a. Create the augmented feature matrix (original features + previous labels)
    ///    b. Train a binary classifier for that label
    ///    c. Store the classifier
    ///
    /// The key difference from Binary Relevance is the augmented features - each classifier
    /// after the first gets additional features representing previous labels in the chain.
    /// </para>
    /// </remarks>
    protected override void TrainMultiLabelCore(Matrix<T> x, Matrix<T> y)
    {
        // Determine chain order
        _chainOrder = DetermineChainOrder();

        _chainClassifiers = new IClassifier<T>[NumLabels];

        // Train classifiers in chain order
        for (int chainPosition = 0; chainPosition < NumLabels; chainPosition++)
        {
            int labelIndex = _chainOrder[chainPosition];

            // Create augmented features: original features + previous label values
            var augmentedX = CreateAugmentedFeatures(x, y, chainPosition);

            // Extract binary labels for this label
            var binaryLabels = new Vector<T>(y.Rows);
            for (int i = 0; i < y.Rows; i++)
            {
                binaryLabels[i] = y[i, labelIndex];
            }

            // Train classifier
            var classifier = _classifierFactory();
            classifier.Train(augmentedX, binaryLabels);
            _chainClassifiers[chainPosition] = classifier;
        }
    }

    /// <summary>
    /// Determines the order of labels in the chain.
    /// </summary>
    /// <returns>An array of label indices in chain order.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method decides which label to predict first, second, etc.
    /// The order can be:
    /// - User-specified: exactly what you requested
    /// - Random: shuffled randomly (can help avoid biases from natural order)
    /// - Natural: 0, 1, 2, 3, ... (simple default)
    /// </para>
    /// </remarks>
    private int[] DetermineChainOrder()
    {
        if (_specifiedOrder is not null)
        {
            // Validate specified order
            if (_specifiedOrder.Length != NumLabels)
            {
                throw new ArgumentException(
                    $"Specified chain order has {_specifiedOrder.Length} elements but there are {NumLabels} labels.");
            }

            var seen = new HashSet<int>();
            foreach (int idx in _specifiedOrder)
            {
                if (idx < 0 || idx >= NumLabels)
                {
                    throw new ArgumentException($"Invalid label index {idx} in chain order.");
                }
                if (!seen.Add(idx))
                {
                    throw new ArgumentException($"Duplicate label index {idx} in chain order.");
                }
            }

            return _specifiedOrder;
        }

        var order = new int[NumLabels];
        for (int i = 0; i < NumLabels; i++)
        {
            order[i] = i;
        }

        if (_useRandomOrder)
        {
            // Fisher-Yates shuffle
            for (int i = order.Length - 1; i > 0; i--)
            {
                int j = _random.Next(i + 1);
                (order[i], order[j]) = (order[j], order[i]);
            }
        }

        return order;
    }

    /// <summary>
    /// Creates augmented features for a position in the chain.
    /// </summary>
    /// <param name="x">Original features.</param>
    /// <param name="y">Multi-label targets (used for getting previous labels during training).</param>
    /// <param name="chainPosition">Current position in the chain.</param>
    /// <returns>Augmented feature matrix with original features + previous label values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method adds previous label predictions as new features.
    /// For example, if we're at chain position 2 (third classifier):
    /// - Original features: [f1, f2, f3, f4]
    /// - Previous labels: [label0, label1]
    /// - Augmented features: [f1, f2, f3, f4, label0, label1]
    ///
    /// During training, we use the actual label values from y.
    /// During prediction, we use the predicted values from previous classifiers.
    /// </para>
    /// </remarks>
    private Matrix<T> CreateAugmentedFeatures(Matrix<T> x, Matrix<T>? y, int chainPosition)
    {
        int numAugmentedFeatures = x.Columns + chainPosition;
        var augmented = new Matrix<T>(x.Rows, numAugmentedFeatures);

        // Copy original features
        for (int i = 0; i < x.Rows; i++)
        {
            for (int j = 0; j < x.Columns; j++)
            {
                augmented[i, j] = x[i, j];
            }
        }

        // Add previous label values (during training, use actual labels from y)
        if (y is not null && chainPosition > 0)
        {
            for (int i = 0; i < x.Rows; i++)
            {
                for (int p = 0; p < chainPosition; p++)
                {
                    int prevLabelIndex = _chainOrder![p];
                    augmented[i, x.Columns + p] = y[i, prevLabelIndex];
                }
            }
        }

        return augmented;
    }

    /// <summary>
    /// Creates augmented features for prediction using previous predictions.
    /// </summary>
    /// <param name="x">Original features.</param>
    /// <param name="previousPredictions">Predictions from previous classifiers in the chain.</param>
    /// <param name="chainPosition">Current position in the chain.</param>
    /// <returns>Augmented feature matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> During prediction, we don't have actual labels. Instead, we use
    /// the predictions from previous classifiers in the chain as the additional features.
    /// </para>
    /// </remarks>
    private Matrix<T> CreateAugmentedFeaturesForPrediction(Matrix<T> x, Matrix<T> previousPredictions, int chainPosition)
    {
        int numAugmentedFeatures = x.Columns + chainPosition;
        var augmented = new Matrix<T>(x.Rows, numAugmentedFeatures);

        // Copy original features
        for (int i = 0; i < x.Rows; i++)
        {
            for (int j = 0; j < x.Columns; j++)
            {
                augmented[i, j] = x[i, j];
            }
        }

        // Add previous predictions
        if (chainPosition > 0)
        {
            for (int i = 0; i < x.Rows; i++)
            {
                for (int p = 0; p < chainPosition; p++)
                {
                    augmented[i, x.Columns + p] = previousPredictions[i, p];
                }
            }
        }

        return augmented;
    }

    #endregion

    #region Prediction

    /// <summary>
    /// Predicts probabilities for each label for each sample.
    /// </summary>
    /// <param name="input">The input feature matrix.</param>
    /// <returns>A probability matrix where each row is a sample and each column is the probability of that label.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method predicts labels following the chain:
    ///
    /// 1. For the first classifier in the chain, predict using only original features
    /// 2. For each subsequent classifier:
    ///    a. Add the predictions from previous classifiers as additional features
    ///    b. Predict using the augmented features
    ///
    /// The predictions propagate through the chain, allowing later classifiers to benefit
    /// from the predictions of earlier ones.
    /// </para>
    /// </remarks>
    public override Matrix<T> PredictMultiLabelProbabilities(Matrix<T> input)
    {
        if (_chainClassifiers is null || _chainOrder is null)
        {
            throw new InvalidOperationException("Model must be trained before making predictions.");
        }

        var probabilities = new Matrix<T>(input.Rows, NumLabels);
        var chainPredictions = new Matrix<T>(input.Rows, NumLabels);

        // Predict in chain order
        for (int chainPosition = 0; chainPosition < NumLabels; chainPosition++)
        {
            int labelIndex = _chainOrder[chainPosition];
            var classifier = _chainClassifiers[chainPosition];

            // Create augmented features with previous predictions
            var augmentedX = CreateAugmentedFeaturesForPrediction(input, chainPredictions, chainPosition);

            // Get predictions
            if (classifier is IProbabilisticClassifier<T> probabilisticClassifier)
            {
                var labelProbs = probabilisticClassifier.PredictProbabilities(augmentedX);

                for (int i = 0; i < input.Rows; i++)
                {
                    // Probability of positive class
                    T prob = labelProbs.Columns > 1 ? labelProbs[i, 1] : labelProbs[i, 0];
                    probabilities[i, labelIndex] = prob;
                    chainPredictions[i, chainPosition] = prob;
                }
            }
            else
            {
                var predictions = classifier.Predict(augmentedX);
                for (int i = 0; i < input.Rows; i++)
                {
                    T prob = NumOps.Compare(predictions[i], NumOps.One) >= 0 ? NumOps.One : NumOps.Zero;
                    probabilities[i, labelIndex] = prob;
                    chainPredictions[i, chainPosition] = prob;
                }
            }
        }

        return probabilities;
    }

    #endregion

    #region Abstract Method Implementations

    /// <summary>
    /// Gets the model type identifier.
    /// </summary>
    /// <returns>The ModelType enum value for Classifier Chain.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This identifies what kind of model this is within the
    /// AiDotNet library's type system.
    /// </para>
    /// </remarks>
    protected override ModelType GetModelType()
    {
        return ModelType.ClassifierChain;
    }

    /// <summary>
    /// Gets all learnable parameters of the model as a single vector.
    /// </summary>
    /// <returns>A concatenated vector of all chain classifiers' parameters.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Classifier Chain contains multiple classifiers in a chain.
    /// This method collects all parameters from all classifiers into a single vector.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        if (_chainClassifiers is null || _chainClassifiers.Length == 0)
        {
            return new Vector<T>(0);
        }

        var allParams = new List<T>();
        foreach (var classifier in _chainClassifiers)
        {
            var classifierParams = classifier.GetParameters();
            for (int i = 0; i < classifierParams.Length; i++)
            {
                allParams.Add(classifierParams[i]);
            }
        }

        return new Vector<T>(allParams.ToArray());
    }

    /// <summary>
    /// Creates a new instance of the model with the specified parameters.
    /// </summary>
    /// <param name="parameters">The parameters to use.</param>
    /// <returns>A new instance of the classifier.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates a new Classifier Chain and distributes
    /// the provided parameters among its chain classifiers.
    /// </para>
    /// </remarks>
    public override IFullModel<T, Matrix<T>, Matrix<T>> WithParameters(Vector<T> parameters)
    {
        var clone = (ClassifierChainClassifier<T>)Clone();
        clone.SetParameters(parameters);
        return clone;
    }

    /// <summary>
    /// Sets the parameters of this model.
    /// </summary>
    /// <param name="parameters">The parameters to set.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This distributes the provided parameters among all chain classifiers.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        if (_chainClassifiers is null || _chainClassifiers.Length == 0)
        {
            return;
        }

        int paramIndex = 0;
        foreach (var classifier in _chainClassifiers)
        {
            var classifierParams = classifier.GetParameters();
            var newParams = new Vector<T>(classifierParams.Length);

            for (int i = 0; i < classifierParams.Length && paramIndex < parameters.Length; i++)
            {
                newParams[i] = parameters[paramIndex++];
            }

            classifier.SetParameters(newParams);
        }
    }

    /// <summary>
    /// Computes gradients for gradient-based optimization.
    /// </summary>
    /// <param name="input">The input features.</param>
    /// <param name="target">The target labels.</param>
    /// <param name="lossFunction">The loss function (optional).</param>
    /// <returns>A concatenated gradient vector from all chain classifiers.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This computes gradients for each chain classifier and
    /// concatenates them into a single vector.
    /// </para>
    /// </remarks>
    public override Vector<T> ComputeGradients(Matrix<T> input, Matrix<T> target, ILossFunction<T>? lossFunction = null)
    {
        if (_chainClassifiers is null || _chainClassifiers.Length == 0 || _chainOrder is null)
        {
            return new Vector<T>(0);
        }

        var allGradients = new List<T>();
        for (int chainIdx = 0; chainIdx < _chainClassifiers.Length; chainIdx++)
        {
            var classifier = _chainClassifiers[chainIdx];
            var labelIdx = _chainOrder[chainIdx];

            // Extract binary labels for this label
            var binaryTarget = new Vector<T>(target.Rows);
            for (int i = 0; i < target.Rows; i++)
            {
                binaryTarget[i] = target[i, labelIdx];
            }

            // Use augmented features to match training dimensions
            var augmentedX = CreateAugmentedFeatures(input, target, chainIdx);
            var gradients = classifier.ComputeGradients(augmentedX, binaryTarget, lossFunction);
            for (int i = 0; i < gradients.Length; i++)
            {
                allGradients.Add(gradients[i]);
            }
        }

        return new Vector<T>(allGradients.ToArray());
    }

    /// <summary>
    /// Applies gradients to update model parameters.
    /// </summary>
    /// <param name="gradients">The gradients to apply.</param>
    /// <param name="learningRate">The learning rate.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This distributes the gradient updates to each chain classifier.
    /// </para>
    /// </remarks>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        if (_chainClassifiers is null || _chainClassifiers.Length == 0)
        {
            return;
        }

        int gradIndex = 0;
        foreach (var classifier in _chainClassifiers)
        {
            var classifierParams = classifier.GetParameters();
            var classifierGradients = new Vector<T>(classifierParams.Length);

            for (int i = 0; i < classifierParams.Length && gradIndex < gradients.Length; i++)
            {
                classifierGradients[i] = gradients[gradIndex++];
            }

            classifier.ApplyGradients(classifierGradients, learningRate);
        }
    }

    /// <summary>
    /// Creates a new instance of this classifier with default configuration.
    /// </summary>
    /// <returns>A new ClassifierChainClassifier instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is used internally for operations like cloning or serialization.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Matrix<T>> CreateNewInstance()
    {
        return new ClassifierChainClassifier<T>(_classifierFactory, _specifiedOrder, _useRandomOrder, null, Options, Regularization);
    }

    /// <summary>
    /// Creates a deep copy of this classifier.
    /// </summary>
    /// <returns>A new instance with the same parameters and state.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Cloning creates an independent copy of the classifier,
    /// including all its chain classifiers and the chain order.
    /// </para>
    /// </remarks>
    public override IFullModel<T, Matrix<T>, Matrix<T>> Clone()
    {
        var clone = new ClassifierChainClassifier<T>(
            _classifierFactory, _specifiedOrder, _useRandomOrder, _random.Next(), Options, Regularization);
        clone.NumLabels = NumLabels;
        clone.NumFeatures = NumFeatures;
        clone.NumClasses = NumClasses;
        clone.TaskType = TaskType;
        clone._chainOrder = _chainOrder?.ToArray();

        if (_chainClassifiers is not null)
        {
            clone._chainClassifiers = new IClassifier<T>[_chainClassifiers.Length];
            for (int i = 0; i < _chainClassifiers.Length; i++)
            {
                clone._chainClassifiers[i] = (IClassifier<T>)_chainClassifiers[i].Clone();
            }
        }

        return clone;
    }

    #endregion

    #region Properties

    /// <summary>
    /// Gets the chain order used for training and prediction.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This tells you which label is predicted first, second, etc.
    /// Useful for understanding how predictions flow through the chain.
    /// </para>
    /// </remarks>
    public int[]? ChainOrder => _chainOrder?.ToArray();

    #endregion
}

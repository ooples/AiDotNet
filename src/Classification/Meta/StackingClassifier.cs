using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Classification.Meta;

/// <summary>
/// Stacking classifier that uses predictions from base classifiers as features for a meta-classifier.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Stacking trains multiple base classifiers and then uses their predictions
/// as features to train a final meta-classifier.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Stacking is a sophisticated ensemble method:
///
/// 1. Train multiple base classifiers
/// 2. Get predictions from each on training data (using cross-validation)
/// 3. Use these predictions as features for a "meta" classifier
/// 4. Train the meta-classifier on these stacked predictions
///
/// For prediction:
/// 1. Get predictions from all base classifiers
/// 2. Stack them as features
/// 3. Feed to meta-classifier for final prediction
///
/// Benefits:
/// - Can combine very different types of classifiers
/// - Often achieves better accuracy than individual classifiers
/// - The meta-classifier learns which base classifiers to trust
///
/// Considerations:
/// - More complex to implement
/// - Risk of overfitting if not using cross-validation
/// - Computationally expensive
/// </para>
/// </remarks>
public class StackingClassifier<T> : MetaClassifierBase<T>
{
    /// <summary>
    /// Gets the stacking-specific options.
    /// </summary>
    protected new StackingClassifierOptions<T> Options => (StackingClassifierOptions<T>)base.Options;

    /// <summary>
    /// The base estimators.
    /// </summary>
    private List<IClassifier<T>>? _estimators;

    /// <summary>
    /// The meta-classifier (final estimator).
    /// </summary>
    private IClassifier<T>? _finalEstimator;

    /// <summary>
    /// Factory for creating final estimator.
    /// </summary>
    private readonly Func<IClassifier<T>>? _finalEstimatorFactory;

    /// <summary>
    /// Initializes a new instance of the StackingClassifier class.
    /// </summary>
    /// <param name="estimators">List of base classifiers.</param>
    /// <param name="finalEstimator">The meta-classifier for final predictions.</param>
    /// <param name="options">Configuration options for the classifier.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public StackingClassifier(
        IEnumerable<IClassifier<T>> estimators,
        Func<IClassifier<T>> finalEstimator,
        StackingClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options ?? new StackingClassifierOptions<T>(), null, regularization)
    {
        _estimators = estimators.ToList();
        _finalEstimatorFactory = finalEstimator;
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>
    protected override ModelType GetModelType() => ModelType.StackingClassifier;

    /// <summary>
    /// Trains the stacking classifier.
    /// </summary>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("Number of samples in X must match length of y.");
        }

        if (_estimators is null || _estimators.Count == 0)
        {
            throw new InvalidOperationException("No estimators provided for stacking.");
        }

        if (_finalEstimatorFactory is null)
        {
            throw new InvalidOperationException("Final estimator factory is not set.");
        }

        NumFeatures = x.Columns;
        ClassLabels = ExtractClassLabels(y);
        NumClasses = ClassLabels.Length;
        TaskType = InferTaskType(y);

        int n = x.Rows;
        int numEstimators = _estimators.Count;

        // Determine number of meta-features per estimator
        int metaFeaturesPerEstimator = Options.UseProbabilities ? NumClasses : 1;
        int totalMetaFeatures = numEstimators * metaFeaturesPerEstimator;

        // If passthrough, include original features
        if (Options.Passthrough)
        {
            totalMetaFeatures += NumFeatures;
        }

        // Create meta-features using cross-validation
        var metaFeatures = new Matrix<T>(n, totalMetaFeatures);

        if (Options.CrossValidationFolds > 1)
        {
            // Use cross-validation to generate out-of-fold predictions
            CreateCrossValidatedMetaFeatures(x, y, metaFeatures);
        }
        else
        {
            // Simple approach: train on all data and predict on all data (prone to overfitting)
            CreateSimpleMetaFeatures(x, y, metaFeatures);
        }

        // Train base estimators on full training data (for prediction time)
        foreach (var estimator in _estimators)
        {
            estimator.Train(x, y);
        }

        // Train final estimator on meta-features
        _finalEstimator = _finalEstimatorFactory();
        _finalEstimator.Train(metaFeatures, y);
    }

    /// <summary>
    /// Creates meta-features using cross-validation.
    /// </summary>
    private void CreateCrossValidatedMetaFeatures(Matrix<T> x, Vector<T> y, Matrix<T> metaFeatures)
    {
        int n = x.Rows;
        int numFolds = Options.CrossValidationFolds;
        int numEstimators = _estimators!.Count;
        int metaFeaturesPerEstimator = Options.UseProbabilities ? NumClasses : 1;

        var random = Options.RandomState.HasValue
            ? RandomHelper.CreateSeededRandom(Options.RandomState.Value)
            : RandomHelper.CreateSecureRandom();

        // Create fold assignments
        var foldAssignments = new int[n];
        for (int i = 0; i < n; i++)
        {
            foldAssignments[i] = i % numFolds;
        }

        // Shuffle fold assignments
        for (int i = n - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (foldAssignments[i], foldAssignments[j]) = (foldAssignments[j], foldAssignments[i]);
        }

        // Process each fold
        for (int fold = 0; fold < numFolds; fold++)
        {
            // Count samples in train and test
            int trainCount = 0;
            int testCount = 0;
            for (int i = 0; i < n; i++)
            {
                if (foldAssignments[i] == fold)
                    testCount++;
                else
                    trainCount++;
            }

            // Create train and test splits
            var xTrain = new Matrix<T>(trainCount, NumFeatures);
            var yTrain = new Vector<T>(trainCount);
            var xTest = new Matrix<T>(testCount, NumFeatures);
            var testIndices = new int[testCount];

            int trainIdx = 0;
            int testIdx = 0;
            for (int i = 0; i < n; i++)
            {
                if (foldAssignments[i] == fold)
                {
                    for (int j = 0; j < NumFeatures; j++)
                    {
                        xTest[testIdx, j] = x[i, j];
                    }
                    testIndices[testIdx] = i;
                    testIdx++;
                }
                else
                {
                    for (int j = 0; j < NumFeatures; j++)
                    {
                        xTrain[trainIdx, j] = x[i, j];
                    }
                    yTrain[trainIdx] = y[i];
                    trainIdx++;
                }
            }

            // Train each estimator and get predictions for test fold
            for (int e = 0; e < numEstimators; e++)
            {
                // Clone and train estimator on training data
                IClassifier<T> foldEstimator;
                if (_estimators[e] is IFullModel<T, Matrix<T>, Vector<T>> fullModel)
                {
                    foldEstimator = (IClassifier<T>)fullModel.Clone();
                }
                else
                {
                    foldEstimator = _estimators[e];
                }
                foldEstimator.Train(xTrain, yTrain);

                // Get predictions for test fold
                if (Options.UseProbabilities && foldEstimator is IProbabilisticClassifier<T> probClassifier)
                {
                    var probs = probClassifier.PredictProbabilities(xTest);
                    for (int t = 0; t < testCount; t++)
                    {
                        int origIdx = testIndices[t];
                        for (int c = 0; c < NumClasses; c++)
                        {
                            metaFeatures[origIdx, e * metaFeaturesPerEstimator + c] = probs[t, c];
                        }
                    }
                }
                else
                {
                    var preds = foldEstimator.Predict(xTest);
                    for (int t = 0; t < testCount; t++)
                    {
                        int origIdx = testIndices[t];
                        metaFeatures[origIdx, e * metaFeaturesPerEstimator] = preds[t];
                    }
                }
            }
        }

        // Add passthrough features if enabled
        if (Options.Passthrough)
        {
            int offset = numEstimators * metaFeaturesPerEstimator;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < NumFeatures; j++)
                {
                    metaFeatures[i, offset + j] = x[i, j];
                }
            }
        }
    }

    /// <summary>
    /// Creates meta-features without cross-validation.
    /// </summary>
    private void CreateSimpleMetaFeatures(Matrix<T> x, Vector<T> y, Matrix<T> metaFeatures)
    {
        int n = x.Rows;
        int numEstimators = _estimators!.Count;
        int metaFeaturesPerEstimator = Options.UseProbabilities ? NumClasses : 1;

        // Train each estimator and get predictions
        for (int e = 0; e < numEstimators; e++)
        {
            _estimators[e].Train(x, y);

            if (Options.UseProbabilities && _estimators[e] is IProbabilisticClassifier<T> probClassifier)
            {
                var probs = probClassifier.PredictProbabilities(x);
                for (int i = 0; i < n; i++)
                {
                    for (int c = 0; c < NumClasses; c++)
                    {
                        metaFeatures[i, e * metaFeaturesPerEstimator + c] = probs[i, c];
                    }
                }
            }
            else
            {
                var preds = _estimators[e].Predict(x);
                for (int i = 0; i < n; i++)
                {
                    metaFeatures[i, e * metaFeaturesPerEstimator] = preds[i];
                }
            }
        }

        // Add passthrough features
        if (Options.Passthrough)
        {
            int offset = numEstimators * metaFeaturesPerEstimator;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < NumFeatures; j++)
                {
                    metaFeatures[i, offset + j] = x[i, j];
                }
            }
        }
    }

    /// <summary>
    /// Creates meta-features for prediction.
    /// </summary>
    private Matrix<T> CreatePredictionMetaFeatures(Matrix<T> input)
    {
        int n = input.Rows;
        int numEstimators = _estimators!.Count;
        int metaFeaturesPerEstimator = Options.UseProbabilities ? NumClasses : 1;
        int totalMetaFeatures = numEstimators * metaFeaturesPerEstimator;

        if (Options.Passthrough)
        {
            totalMetaFeatures += NumFeatures;
        }

        var metaFeatures = new Matrix<T>(n, totalMetaFeatures);

        // Get predictions from each estimator
        for (int e = 0; e < numEstimators; e++)
        {
            if (Options.UseProbabilities && _estimators[e] is IProbabilisticClassifier<T> probClassifier)
            {
                var probs = probClassifier.PredictProbabilities(input);
                for (int i = 0; i < n; i++)
                {
                    for (int c = 0; c < NumClasses; c++)
                    {
                        metaFeatures[i, e * metaFeaturesPerEstimator + c] = probs[i, c];
                    }
                }
            }
            else
            {
                var preds = _estimators[e].Predict(input);
                for (int i = 0; i < n; i++)
                {
                    metaFeatures[i, e * metaFeaturesPerEstimator] = preds[i];
                }
            }
        }

        // Add passthrough features
        if (Options.Passthrough)
        {
            int offset = numEstimators * metaFeaturesPerEstimator;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < NumFeatures; j++)
                {
                    metaFeatures[i, offset + j] = input[i, j];
                }
            }
        }

        return metaFeatures;
    }

    /// <inheritdoc/>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (_estimators is null || _finalEstimator is null || ClassLabels is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var metaFeatures = CreatePredictionMetaFeatures(input);
        return _finalEstimator.Predict(metaFeatures);
    }

    /// <inheritdoc/>
    public override Matrix<T> PredictProbabilities(Matrix<T> input)
    {
        if (_estimators is null || _finalEstimator is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var metaFeatures = CreatePredictionMetaFeatures(input);

        if (_finalEstimator is IProbabilisticClassifier<T> probClassifier)
        {
            return probClassifier.PredictProbabilities(metaFeatures);
        }

        // Fall back to hard predictions
        var preds = _finalEstimator.Predict(metaFeatures);
        var probs = new Matrix<T>(input.Rows, NumClasses);

        for (int i = 0; i < input.Rows; i++)
        {
            for (int c = 0; c < NumClasses; c++)
            {
                if (NumOps.Compare(preds[i], ClassLabels![c]) == 0)
                {
                    probs[i, c] = NumOps.One;
                }
            }
        }

        return probs;
    }

    /// <inheritdoc/>
    public override Matrix<T> PredictLogProbabilities(Matrix<T> input)
    {
        var probs = PredictProbabilities(input);
        var logProbs = new Matrix<T>(input.Rows, NumClasses);

        for (int i = 0; i < input.Rows; i++)
        {
            for (int c = 0; c < NumClasses; c++)
            {
                T p = probs[i, c];
                T minP = NumOps.FromDouble(1e-15);
                if (NumOps.Compare(p, minP) < 0)
                {
                    p = minP;
                }
                logProbs[i, c] = NumOps.Log(p);
            }
        }

        return logProbs;
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        if (_finalEstimatorFactory is null)
        {
            throw new InvalidOperationException("Final estimator factory is not set.");
        }

        var newEstimators = new List<IClassifier<T>>();

        if (_estimators is not null)
        {
            foreach (var est in _estimators)
            {
                if (est is IFullModel<T, Matrix<T>, Vector<T>> fullModel)
                {
                    newEstimators.Add((IClassifier<T>)fullModel.Clone());
                }
                else
                {
                    newEstimators.Add(est);
                }
            }
        }

        return new StackingClassifier<T>(newEstimators, _finalEstimatorFactory, new StackingClassifierOptions<T>
        {
            CrossValidationFolds = Options.CrossValidationFolds,
            UseProbabilities = Options.UseProbabilities,
            Passthrough = Options.Passthrough,
            RandomState = Options.RandomState
        });
    }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = (StackingClassifier<T>)CreateNewInstance();

        clone.NumFeatures = NumFeatures;
        clone.NumClasses = NumClasses;
        clone.TaskType = TaskType;

        if (ClassLabels is not null)
        {
            clone.ClassLabels = new Vector<T>(ClassLabels.Length);
            for (int i = 0; i < ClassLabels.Length; i++)
            {
                clone.ClassLabels[i] = ClassLabels[i];
            }
        }

        if (_finalEstimator is IFullModel<T, Matrix<T>, Vector<T>> finalFullModel)
        {
            clone._finalEstimator = (IClassifier<T>)finalFullModel.Clone();
        }

        // Clone trained base estimators
        if (_estimators is not null)
        {
            clone._estimators = new List<IClassifier<T>>(_estimators.Count);
            for (int e = 0; e < _estimators.Count; e++)
            {
                if (_estimators[e] is IFullModel<T, Matrix<T>, Vector<T>> fullModel)
                {
                    clone._estimators.Add((IClassifier<T>)fullModel.Clone());
                }
                else
                {
                    // Cannot clone, just reference the same instance
                    clone._estimators.Add(_estimators[e]);
                }
            }
        }

        return clone;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["NumEstimators"] = _estimators?.Count ?? 0;
        metadata.AdditionalInfo["CrossValidationFolds"] = Options.CrossValidationFolds;
        metadata.AdditionalInfo["UseProbabilities"] = Options.UseProbabilities;
        metadata.AdditionalInfo["Passthrough"] = Options.Passthrough;
        return metadata;
    }
}

/// <summary>
/// Configuration options for Stacking classifier.
/// </summary>
/// <typeparam name="T">The data type used for calculations.</typeparam>
public class StackingClassifierOptions<T> : MetaClassifierOptions<T>
{
    /// <summary>
    /// Gets or sets the number of cross-validation folds.
    /// </summary>
    /// <value>Number of folds. Use 1 for no cross-validation. Default is 5.</value>
    public int CrossValidationFolds { get; set; } = 5;

    /// <summary>
    /// Gets or sets whether to use probability predictions as meta-features.
    /// </summary>
    /// <value>True to use probabilities, false to use class predictions. Default is true.</value>
    public bool UseProbabilities { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to pass original features to the final estimator.
    /// </summary>
    /// <value>True to include original features. Default is false.</value>
    public bool Passthrough { get; set; } = false;

    /// <summary>
    /// Gets or sets the random state for reproducibility.
    /// </summary>
    public int? RandomState { get; set; }
}

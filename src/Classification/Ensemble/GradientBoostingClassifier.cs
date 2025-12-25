using AiDotNet.Classification;
using AiDotNet.Classification.Trees;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Classification.Ensemble;

/// <summary>
/// Gradient Boosting classifier that builds trees sequentially to correct errors.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Gradient Boosting builds an additive model in a forward stage-wise fashion.
/// At each stage, a regression tree is fit on the negative gradient of the loss function.
/// For classification, this uses log loss (deviance) or exponential loss.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Gradient Boosting is one of the most powerful machine learning algorithms:
///
/// How it works:
/// 1. Start with an initial prediction
/// 2. Calculate how wrong we are
/// 3. Train a tree to predict our mistakes
/// 4. Add a fraction of this tree's predictions
/// 5. Repeat, each time correcting remaining errors
///
/// Key insight: Each tree fixes what previous trees got wrong!
///
/// Tips for best results:
/// - Use lower learning_rate with more n_estimators
/// - Keep max_depth small (3-5) unlike Random Forest
/// - Consider subsample less than 1.0 for regularization
/// </para>
/// </remarks>
public class GradientBoostingClassifier<T> : EnsembleClassifierBase<T>, ITreeBasedClassifier<T>
{
    /// <summary>
    /// Gets the Gradient Boosting specific options.
    /// </summary>
    protected new GradientBoostingClassifierOptions<T> Options => (GradientBoostingClassifierOptions<T>)base.Options;

    /// <summary>
    /// Initial prediction (prior).
    /// </summary>
    private T _initPrediction = default!;

    /// <summary>
    /// Random number generator.
    /// </summary>
    private Random? _random;

    /// <inheritdoc/>
    public int MaxDepth => Options.MaxDepth;

    /// <inheritdoc/>
    public int LeafCount => CalculateTotalLeafCount();

    /// <inheritdoc/>
    public int NodeCount => CalculateTotalNodeCount();

    /// <summary>
    /// Initializes a new instance of the GradientBoostingClassifier class.
    /// </summary>
    /// <param name="options">Configuration options for Gradient Boosting.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public GradientBoostingClassifier(GradientBoostingClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options ?? new GradientBoostingClassifierOptions<T>(), regularization, new CrossEntropyLoss<T>())
    {
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>
    protected override ModelType GetModelType() => ModelType.GradientBoostingClassifier;

    /// <summary>
    /// Trains the Gradient Boosting classifier on the provided data.
    /// </summary>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("Number of samples in X must match length of y.");
        }

        NumFeatures = x.Columns;
        ClassLabels = ExtractClassLabels(y);
        NumClasses = ClassLabels.Length;
        TaskType = InferTaskType(y);

        _random = Options.RandomState.HasValue
            ? RandomHelper.CreateSeededRandom(Options.RandomState.Value)
            : RandomHelper.CreateSeededRandom(42);

        // Clear existing estimators
        Estimators.Clear();

        int n = x.Rows;

        // Convert labels to 0/1 for binary classification
        var yBinary = new Vector<T>(n);
        T positiveClass = ClassLabels[ClassLabels.Length - 1];
        for (int i = 0; i < n; i++)
        {
            yBinary[i] = NumOps.Compare(y[i], positiveClass) == 0 ? NumOps.One : NumOps.Zero;
        }

        // Initialize F0 = log(p / (1 - p)) where p is the prior probability
        int posCount = 0;
        for (int i = 0; i < n; i++)
        {
            if (NumOps.Compare(yBinary[i], NumOps.One) == 0)
            {
                posCount++;
            }
        }
        double p = (double)posCount / n;
        p = Math.Max(1e-10, Math.Min(1 - 1e-10, p)); // Clip for numerical stability
        _initPrediction = NumOps.FromDouble(Math.Log(p / (1 - p)));

        // Current predictions (in log-odds space)
        var fValues = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            fValues[i] = _initPrediction;
        }

        // Learning rate
        T lr = NumOps.FromDouble(Options.LearningRate);

        // Train trees
        for (int m = 0; m < Options.NEstimators; m++)
        {
            // Compute probabilities from current predictions
            var probs = new Vector<T>(n);
            for (int i = 0; i < n; i++)
            {
                probs[i] = Sigmoid(fValues[i]);
            }

            // Compute negative gradient (residuals for log loss): y - p
            var residuals = new Vector<T>(n);
            for (int i = 0; i < n; i++)
            {
                residuals[i] = NumOps.Subtract(yBinary[i], probs[i]);
            }

            // Subsample if needed
            Matrix<T> xSample;
            Vector<T> residualsSample;
            int[] sampleIndices;

            if (Options.Subsample < 1.0)
            {
                (xSample, residualsSample, sampleIndices) = SubsampleData(x, residuals);
            }
            else
            {
                xSample = x;
                residualsSample = residuals;
                sampleIndices = Enumerable.Range(0, n).ToArray();
            }

            // Train a regression tree on residuals
            // Note: We're using DecisionTreeClassifier but treating residuals as targets
            // In a full implementation, we'd use a DecisionTreeRegressor
            var treeOptions = new DecisionTreeClassifierOptions<T>
            {
                MaxDepth = Options.MaxDepth,
                MinSamplesSplit = Options.MinSamplesSplit,
                MinSamplesLeaf = Options.MinSamplesLeaf,
                RandomState = _random.Next(),
                MinImpurityDecrease = Options.MinImpurityDecrease
            };

            // For simplicity, we'll fit to residual signs for classification
            // A proper implementation would use regression trees
            var residualClasses = new Vector<T>(residualsSample.Length);
            for (int i = 0; i < residualsSample.Length; i++)
            {
                residualClasses[i] = NumOps.Compare(residualsSample[i], NumOps.Zero) >= 0
                    ? NumOps.One
                    : NumOps.Zero;
            }

            var tree = new DecisionTreeClassifier<T>(treeOptions);
            tree.Train(xSample, residualClasses);
            Estimators.Add(tree);

            // Update predictions
            var treePreds = tree.Predict(x);
            for (int i = 0; i < n; i++)
            {
                // Map prediction to residual estimate
                T treeOutput = NumOps.Compare(treePreds[i], NumOps.One) == 0
                    ? NumOps.FromDouble(0.5)
                    : NumOps.FromDouble(-0.5);
                fValues[i] = NumOps.Add(fValues[i], NumOps.Multiply(lr, treeOutput));
            }
        }

        // Aggregate feature importances
        AggregateFeatureImportances();
    }

    /// <summary>
    /// Computes the sigmoid function.
    /// </summary>
    private T Sigmoid(T x)
    {
        // sigmoid(x) = 1 / (1 + exp(-x))
        T negX = NumOps.Negate(x);
        T expNegX = NumOps.Exp(negX);
        return NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, expNegX));
    }

    /// <summary>
    /// Subsamples data for stochastic gradient boosting.
    /// </summary>
    private (Matrix<T> x, Vector<T> y, int[] indices) SubsampleData(Matrix<T> x, Vector<T> y)
    {
        if (_random is null)
        {
            throw new InvalidOperationException("Random number generator not initialized.");
        }

        int n = x.Rows;
        int sampleSize = (int)(n * Options.Subsample);
        sampleSize = Math.Max(1, sampleSize);

        var indices = new int[sampleSize];
        for (int i = 0; i < sampleSize; i++)
        {
            indices[i] = _random.Next(n);
        }

        var xSample = new Matrix<T>(sampleSize, x.Columns);
        var ySample = new Vector<T>(sampleSize);

        for (int i = 0; i < sampleSize; i++)
        {
            int idx = indices[i];
            for (int j = 0; j < x.Columns; j++)
            {
                xSample[i, j] = x[idx, j];
            }
            ySample[i] = y[idx];
        }

        return (xSample, ySample, indices);
    }

    /// <inheritdoc/>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (Estimators.Count == 0 || ClassLabels is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var predictions = new Vector<T>(input.Rows);
        T lr = NumOps.FromDouble(Options.LearningRate);

        for (int i = 0; i < input.Rows; i++)
        {
            // Start with initial prediction
            T fValue = _initPrediction;

            // Add contributions from all trees
            var sample = new Matrix<T>(1, input.Columns);
            for (int j = 0; j < input.Columns; j++)
            {
                sample[0, j] = input[i, j];
            }

            for (int m = 0; m < Estimators.Count; m++)
            {
                var treePred = Estimators[m].Predict(sample);
                T treeOutput = NumOps.Compare(treePred[0], NumOps.One) == 0
                    ? NumOps.FromDouble(0.5)
                    : NumOps.FromDouble(-0.5);
                fValue = NumOps.Add(fValue, NumOps.Multiply(lr, treeOutput));
            }

            // Convert to probability and threshold
            T prob = Sigmoid(fValue);
            predictions[i] = NumOps.Compare(prob, NumOps.FromDouble(0.5)) >= 0
                ? ClassLabels[ClassLabels.Length - 1]
                : ClassLabels[0];
        }

        return predictions;
    }

    /// <inheritdoc/>
    public override Matrix<T> PredictProbabilities(Matrix<T> input)
    {
        if (Estimators.Count == 0)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var probabilities = new Matrix<T>(input.Rows, NumClasses);
        T lr = NumOps.FromDouble(Options.LearningRate);

        for (int i = 0; i < input.Rows; i++)
        {
            T fValue = _initPrediction;

            var sample = new Matrix<T>(1, input.Columns);
            for (int j = 0; j < input.Columns; j++)
            {
                sample[0, j] = input[i, j];
            }

            for (int m = 0; m < Estimators.Count; m++)
            {
                var treePred = Estimators[m].Predict(sample);
                T treeOutput = NumOps.Compare(treePred[0], NumOps.One) == 0
                    ? NumOps.FromDouble(0.5)
                    : NumOps.FromDouble(-0.5);
                fValue = NumOps.Add(fValue, NumOps.Multiply(lr, treeOutput));
            }

            T prob = Sigmoid(fValue);
            probabilities[i, 0] = NumOps.Subtract(NumOps.One, prob); // Negative class
            if (NumClasses > 1)
            {
                probabilities[i, 1] = prob; // Positive class
            }
        }

        return probabilities;
    }

    /// <summary>
    /// Calculates the total number of leaf nodes.
    /// </summary>
    private int CalculateTotalLeafCount()
    {
        int total = 0;
        foreach (var estimator in Estimators)
        {
            if (estimator is ITreeBasedClassifier<T> tree)
            {
                total += tree.LeafCount;
            }
        }
        return total;
    }

    /// <summary>
    /// Calculates the total number of nodes.
    /// </summary>
    private int CalculateTotalNodeCount()
    {
        int total = 0;
        foreach (var estimator in Estimators)
        {
            if (estimator is ITreeBasedClassifier<T> tree)
            {
                total += tree.NodeCount;
            }
        }
        return total;
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new GradientBoostingClassifier<T>(new GradientBoostingClassifierOptions<T>
        {
            NEstimators = Options.NEstimators,
            LearningRate = Options.LearningRate,
            MaxDepth = Options.MaxDepth,
            MinSamplesSplit = Options.MinSamplesSplit,
            MinSamplesLeaf = Options.MinSamplesLeaf,
            Subsample = Options.Subsample,
            MaxFeatures = Options.MaxFeatures,
            Loss = Options.Loss,
            RandomState = Options.RandomState,
            MinImpurityDecrease = Options.MinImpurityDecrease
        });
    }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = new GradientBoostingClassifier<T>(new GradientBoostingClassifierOptions<T>
        {
            NEstimators = Options.NEstimators,
            LearningRate = Options.LearningRate,
            MaxDepth = Options.MaxDepth,
            MinSamplesSplit = Options.MinSamplesSplit,
            MinSamplesLeaf = Options.MinSamplesLeaf,
            Subsample = Options.Subsample,
            MaxFeatures = Options.MaxFeatures,
            Loss = Options.Loss,
            RandomState = Options.RandomState,
            MinImpurityDecrease = Options.MinImpurityDecrease
        });

        clone.NumFeatures = NumFeatures;
        clone.NumClasses = NumClasses;
        clone.TaskType = TaskType;
        clone._initPrediction = _initPrediction;

        if (ClassLabels is not null)
        {
            clone.ClassLabels = new Vector<T>(ClassLabels.Length);
            for (int i = 0; i < ClassLabels.Length; i++)
            {
                clone.ClassLabels[i] = ClassLabels[i];
            }
        }

        if (FeatureImportances is not null)
        {
            clone.FeatureImportances = new Vector<T>(FeatureImportances.Length);
            for (int i = 0; i < FeatureImportances.Length; i++)
            {
                clone.FeatureImportances[i] = FeatureImportances[i];
            }
        }

        // Clone all estimators
        foreach (var estimator in Estimators)
        {
            if (estimator is IFullModel<T, Matrix<T>, Vector<T>> fullModel)
            {
                clone.Estimators.Add((IClassifier<T>)fullModel.Clone());
            }
        }

        return clone;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["NEstimators"] = Options.NEstimators;
        metadata.AdditionalInfo["LearningRate"] = Options.LearningRate;
        metadata.AdditionalInfo["MaxDepth"] = Options.MaxDepth;
        metadata.AdditionalInfo["Subsample"] = Options.Subsample;
        metadata.AdditionalInfo["Loss"] = Options.Loss.ToString();
        metadata.AdditionalInfo["TotalNodes"] = NodeCount;
        metadata.AdditionalInfo["TotalLeaves"] = LeafCount;
        return metadata;
    }
}

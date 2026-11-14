using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Classification;
using AiDotNet.Enums;
using AiDotNet.Classification.Trees;
using AiDotNet.Models.Options;
using AiDotNet.Regularization;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

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
/// <example>
/// <code>
/// // Create gradient boosting classifier
/// var options = new GradientBoostingClassifierOptions&lt;double&gt;();
///
/// // Prepare training data
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1.0, 1.1 }, { 1.2, 0.9 }, { 0.8, 1.0 }, { 5.0, 5.1 }, { 5.2, 4.9 }, { 4.8, 5.0 } });
/// var labels = new Vector&lt;double&gt;(new double[] { 0, 0, 0, 1, 1, 1 });
///
/// // Train by sequentially fitting trees to negative gradients of loss
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new GradientBoostingClassifier&lt;double&gt;(options))
///     .Build(features, labels);
///
/// // Predict class for new sample
/// var newSample = new Matrix&lt;double&gt;(new double[,] { { 1.1, 1.0 } });
/// var prediction = result.Predict(newSample);
/// // Result is available in the returned value
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelCategory(ModelCategory.Classifier)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.BinaryClassification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Greedy Function Approximation: A Gradient Boosting Machine",
    "https://doi.org/10.1214/aos/1013203451",
    Year = 2001,
    Authors = "Jerome H. Friedman")]
public partial class GradientBoostingClassifier<T> : EnsembleClassifierBase<T>, ITreeBasedClassifier<T>
{
    /// <summary>
    /// Gets the Gradient Boosting specific options.
    /// </summary>
    protected new GradientBoostingClassifierOptions<T> Options => (GradientBoostingClassifierOptions<T>)base.Options;

    /// <summary>
    /// Initial prediction (prior).
    /// </summary>

    private T _initPrediction;


    /// <summary>
    /// Random number generator.
    /// </summary>
    private Random? _random;

    /// <summary>
    /// Mean residual values for each tree's leaf predictions.
    /// Each entry contains [meanForClass0, meanForClass1] for the corresponding tree.
    /// </summary>
    private readonly List<T[]> _leafResidualMeans = new();

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
        : base(options ?? new GradientBoostingClassifierOptions<T>(), regularization, new CrossEntropyWithLogitsLoss<T>())
    {
        _initPrediction = NumOps.Zero;
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>

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

        _random = Options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(Options.Seed.Value)
            : RandomHelper.CreateSeededRandom(42);

        // Clear existing estimators and residual means
        Estimators.Clear();
        _leafResidualMeans.Clear();

        int n = x.Rows;

        var engine = Engine;

        // Convert labels to 0/1 for binary classification
        T positiveClass = ClassLabels[ClassLabels.Length - 1];
        var yBinary = engine.TensorEquals(y.AsTensor(), positiveClass).ToVector();

        // Initialize F0 = log(p / (1 - p)) where p is the prior probability
        int posCount = engine.TensorCountNonzero(yBinary.AsTensor());
        double p = (double)posCount / n;
        p = Math.Max(1e-10, Math.Min(1 - 1e-10, p)); // Clip for numerical stability
        _initPrediction = NumOps.FromDouble(Math.Log(p / (1 - p)));

        // Current predictions (in log-odds space)
        var fValues = FilledVector(n, _initPrediction);

        // Learning rate
        T lr = NumOps.FromDouble(Options.LearningRate);

        // Train trees
        for (int m = 0; m < Options.NEstimators; m++)
        {
            // Compute probabilities from current predictions
            var probs = engine.Sigmoid(fValues);

            // Compute negative gradient (residuals for log loss): y - p
            var residuals = engine.Subtract(yBinary, probs);

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
                Seed = _random.Next(),
                MinImpurityDecrease = Options.MinImpurityDecrease
            };

            // Fit to residual signs to determine tree structure (1 where residual >= 0)
            var residualClasses = engine.ScalarMinusTensor(NumOps.One,
                engine.TensorLessThan(residualsSample.AsTensor(), NumOps.Zero)).ToVector();

            var tree = new DecisionTreeClassifier<T>(treeOptions);
            tree.Train(xSample, residualClasses);
            Estimators.Add(tree);

            // Compute Newton-Raphson leaf values per leaf region
            // For log loss: leaf_value = sum(residuals) / sum(p * (1 - p))
            // where residuals = y - p (negative gradient) and p*(1-p) is the Hessian
            var samplePreds = tree.Predict(xSample);

            // Hessian p(1 - p) of each sampled point (clamped for numerical stability)
            var sampleProbs = SelectElements(probs, sampleIndices).AsTensor();
            var hessian = engine.TensorClampMin(
                engine.TensorMultiply(sampleProbs, engine.ScalarMinusTensor(NumOps.One, sampleProbs)), Tolerance(1e-10));

            // Sums of residuals and hessians in each leaf region (tree output 1 vs anything else)
            var inLeafOne = engine.TensorEquals(samplePreds.AsTensor(), NumOps.One);
            var inLeafZero = engine.ScalarMinusTensor(NumOps.One, inLeafOne);
            var r = residualsSample.AsTensor();
            var (leafVal0, leafVal1) = NewtonLeafValues(
                engine.TensorSum(engine.TensorMultiply(r, inLeafZero)), engine.TensorSum(engine.TensorMultiply(hessian, inLeafZero)),
                engine.TensorSum(engine.TensorMultiply(r, inLeafOne)), engine.TensorSum(engine.TensorMultiply(hessian, inLeafOne)));

            _leafResidualMeans.Add(new[] { leafVal0, leafVal1 });

            // Update predictions using Newton-step leaf values
            fValues = engine.Add(fValues, engine.Multiply(TreeOutput(tree.Predict(x), leafVal0, leafVal1), lr));
        }

        // Aggregate feature importances
        AggregateFeatureImportances();
    }

    /// <summary>
    /// Newton-step leaf values sum(residual) / sum(hessian) for the two leaf regions (zero when the hessian is tiny).
    /// </summary>
    private (T LeafZero, T LeafOne) NewtonLeafValues(T sumResid0, T sumHessian0, T sumResid1, T sumHessian1)
    {
        // Newton step: sum(gradient) / sum(hessian) per leaf region
        T minHessian = Tolerance(1e-10);
        T leafVal0 = NumOps.Compare(sumHessian0, minHessian) > 0 ? NumOps.Divide(sumResid0, sumHessian0) : NumOps.Zero;
        T leafVal1 = NumOps.Compare(sumHessian1, minHessian) > 0 ? NumOps.Divide(sumResid1, sumHessian1) : NumOps.Zero;
        return (leafVal0, leafVal1);
    }

    /// <summary>
    /// Log-odds of every sample: init + lr * sum over trees of the leaf value selected by the tree's 0/1 output.
    /// </summary>
    private Vector<T> ComputeLogOdds(Matrix<T> input)
    {
        var engine = Engine;
        int n = input.Rows;
        T lr = NumOps.FromDouble(Options.LearningRate);
        var fValues = FilledVector(n, _initPrediction);
        for (int m = 0; m < Estimators.Count; m++)
        {
            fValues = engine.Add(fValues, engine.Multiply(TreeOutput(Estimators[m].Predict(input), _leafResidualMeans[m][0],
                _leafResidualMeans[m][1]), lr));
        }

        return fValues;
    }

    /// <summary>
    /// leafOne where the tree predicts class 1, leafZero elsewhere.
    /// </summary>
    private Vector<T> TreeOutput(Vector<T> treePredictions, T leafZero, T leafOne)
    {
        int n = treePredictions.Length;
        var isOne = Engine.TensorEquals(treePredictions.AsTensor(), NumOps.One);
        return Engine.TensorWhere(isOne, FilledVector(n, leafOne).AsTensor(), FilledVector(n, leafZero).AsTensor()).ToVector();
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

        var (xSample, ySample) = SelectSamples(x, y, indices);
        return (xSample, ySample, indices);
    }

    /// <inheritdoc/>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (Estimators.Count == 0 || ClassLabels is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        if (input.Rows == 0)
        {
            return new Vector<T>(0);
        }

        // Convert to probability and threshold at 0.5
        return PredictBinaryLabels(BinaryProbabilitiesFromLogOdds(ComputeLogOdds(input)));
    }

    /// <inheritdoc/>
    public override Matrix<T> PredictProbabilities(Matrix<T> input)
    {
        if (Estimators.Count == 0)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        int n = input.Rows;
        if (n == 0)
        {
            return new Matrix<T>(0, NumClasses);
        }

        // Column 0 = negative class, column 1 = positive class, any further columns stay zero
        var engine = Engine;
        var prob = engine.Sigmoid(ComputeLogOdds(input));
        var columns = new List<Vector<T>> { engine.ScalarMinusTensor(NumOps.One, prob.AsTensor()).ToVector() };
        if (NumClasses > 1)
        {
            columns.Add(prob);
        }

        for (int c = 2; c < NumClasses; c++)
        {
            columns.Add(new Vector<T>(n));
        }

        return StackColumns(columns);
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

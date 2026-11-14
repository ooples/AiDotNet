using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Classification;
using AiDotNet.Classification.Trees;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Classification.Ensemble;

/// <summary>
/// AdaBoost (Adaptive Boosting) classifier that combines weak learners.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// AdaBoost iteratively trains weak classifiers on re-weighted versions of the data,
/// where incorrectly classified samples receive higher weights in subsequent iterations.
/// The final prediction is a weighted vote of all weak learners.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// AdaBoost works like a learning system that focuses on its mistakes:
///
/// 1. Train a simple classifier on the data
/// 2. See which samples were misclassified
/// 3. Give those samples higher importance
/// 4. Train another classifier with the new importance weights
/// 5. Repeat many times
/// 6. Combine all classifiers with voting
///
/// This creates a powerful classifier from many weak ones.
/// </para>
/// </remarks>
/// <para><b>Recommended:</b> Use <c>AiModelBuilder</c> for the simplest entry point.</para>
/// <example>
/// <code>
/// // Create AdaBoost classifier that combines weak learners
/// var options = new AdaBoostClassifierOptions&lt;double&gt;();
///
/// // Prepare training data
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1.0, 1.1 }, { 1.2, 0.9 }, { 0.8, 1.0 }, { 5.0, 5.1 }, { 5.2, 4.9 }, { 4.8, 5.0 } });
/// var labels = new Vector&lt;double&gt;(new double[] { 0, 0, 0, 1, 1, 1 });
///
/// // Train by iteratively re-weighting misclassified samples
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new AdaBoostClassifier&lt;double&gt;(options))
///     .Build(features, labels);
///
/// // Predict using weighted vote of all weak learners
/// var newSample = new Matrix&lt;double&gt;(new double[,] { { 1.1, 1.0 } });
/// var prediction = result.Predict(newSample);
/// // Result is available in the returned value
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelCategory(ModelCategory.DecisionTree)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting", "https://doi.org/10.1006/jcss.1997.1504", Year = 1997, Authors = "Yoav Freund, Robert E. Schapire")]
public partial class AdaBoostClassifier<T> : EnsembleClassifierBase<T>
{
    /// <summary>
    /// Gets the AdaBoost specific options.
    /// </summary>
    protected new AdaBoostClassifierOptions<T> Options => (AdaBoostClassifierOptions<T>)base.Options;

    /// <summary>
    /// Weights for each estimator (based on their accuracy).
    /// </summary>
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _estimatorWeights;

    /// <summary>
    /// Random number generator.
    /// </summary>
    private Random? _random;

    /// <summary>
    /// Initializes a new instance of the AdaBoostClassifier class.
    /// </summary>
    /// <param name="options">Configuration options for AdaBoost.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public AdaBoostClassifier(AdaBoostClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        // PR #1404 review: AdaBoostClassifier outputs probabilities via
        // PredictProbabilities() (weighted normalized votes). Pass the
        // probability-input cross-entropy so any future code path that
        // evaluates DefaultLossFunction against the classifier's outputs
        // gets the right gradient. AdaBoost's own training doesn't consume
        // this loss but design-time consistency matters.
        : base(options ?? new AdaBoostClassifierOptions<T>(), regularization, new CrossEntropyLoss<T>())
    {
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>

    /// <summary>
    /// Trains the AdaBoost classifier on the provided data.
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

        // Clear existing estimators
        Estimators.Clear();
        var estimatorWeights = new T[Options.NEstimators];
        var engine = Engine;

        // Initialize sample weights uniformly
        int n = x.Rows;
        var sampleWeights = FilledVector(n, NumOps.Divide(NumOps.One, NumOps.FromDouble(n)));

        // Train estimators
        for (int m = 0; m < Options.NEstimators; m++)
        {
            // Create and train a weak learner (decision stump)
            var stumpOptions = new DecisionTreeClassifierOptions<T>
            {
                MaxDepth = 1, // Decision stump
                Seed = _random.Next()
            };

            var stump = new DecisionTreeClassifier<T>(stumpOptions);

            // Sample with replacement based on weights
            var (xSample, ySample) = SampleWithWeights(x, y, sampleWeights);
            stump.Train(xSample, ySample);

            // Get predictions on full training set
            var predictions = stump.Predict(x);

            // Calculate weighted error (1 where the prediction is wrong)
            var wrong = engine.TensorNotEquals(predictions.AsTensor(), y.AsTensor());
            T error = engine.TensorSum(engine.TensorMultiply(wrong, sampleWeights.AsTensor()));

            // Check if error is too high (random guessing or worse); otherwise
            // alpha = learningRate * 0.5 * ln((1 - error) / error)
            if (!TryComputeEstimatorWeight(error, out T alpha))
            {
                // Skip this estimator - it's not better than random
                continue;
            }

            estimatorWeights[Estimators.Count] = alpha;
            Estimators.Add(stump);

            // Update sample weights: w *= exp(alpha * indicator), indicator = -1 when correct, +1 when wrong
            var indicator = engine.TensorSubtractScalar(engine.TensorMultiplyScalar(wrong, NumOps.FromDouble(2)), NumOps.One);
            var updated = engine.TensorMultiply(sampleWeights.AsTensor(),
                engine.TensorExp(engine.TensorMultiplyScalar(indicator, alpha)));

            // Normalize weights
            sampleWeights = engine.TensorDivideScalar(updated, engine.TensorSum(updated)).ToVector();
        }

        _estimatorWeights = Vector<T>.Wrap(estimatorWeights);

        // Aggregate feature importances
        AggregateFeatureImportances();
    }

    /// <summary>
    /// Samples data with replacement based on weights.
    /// </summary>
    private (Matrix<T> x, Vector<T> y) SampleWithWeights(Matrix<T> x, Vector<T> y, Vector<T> weights)
    {
        if (_random is null)
        {
            throw new InvalidOperationException("Random number generator not initialized.");
        }

        int n = x.Rows;
        var indices = new int[n];

        // Build cumulative distribution
        var cumWeights = new double[n];
        cumWeights[0] = NumOps.ToDouble(weights[0]);
        for (int i = 1; i < n; i++)
        {
            cumWeights[i] = cumWeights[i - 1] + NumOps.ToDouble(weights[i]);
        }

        // Sample indices
        for (int i = 0; i < n; i++)
        {
            double r = _random.NextDouble();
            int idx = Array.BinarySearch(cumWeights, r);
            if (idx < 0) idx = ~idx;
            if (idx >= n) idx = n - 1;
            indices[i] = idx;
        }

        // Create sampled data
        return SelectSamples(x, y, indices);
    }

    /// <summary>
    /// alpha = learningRate * 0.5 * ln((1 - error) / (error + 1e-10)); false when error &gt;= 0.5.
    /// </summary>
    private bool TryComputeEstimatorWeight(T error, out T alpha)
    {
        alpha = NumOps.Zero;
        if (NumOps.ToDouble(error) >= 0.5)
        {
            return false;
        }

        T oneMinusError = NumOps.Subtract(NumOps.One, error);
        T ratio = NumOps.Divide(oneMinusError, NumOps.Add(error, Tolerance(1e-10)));
        alpha = NumOps.Multiply(NumOps.Multiply(NumOps.FromDouble(0.5), NumOps.Log(ratio)),
            NumOps.FromDouble(Options.LearningRate));
        return true;
    }

    /// <inheritdoc/>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (Estimators.Count == 0 || _estimatorWeights is null || ClassLabels is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        if (input.Rows == 0)
        {
            return new Vector<T>(0);
        }

        // Class with the maximum weighted vote
        return ClassIndicesToLabels(ArgMaxRows(ComputeWeightedVotes(input).ToMatrix()));
    }

    /// <inheritdoc/>
    public override Matrix<T> PredictProbabilities(Matrix<T> input)
    {
        if (Estimators.Count == 0 || _estimatorWeights is null || ClassLabels is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        if (input.Rows == 0)
        {
            return new Matrix<T>(0, NumClasses);
        }

        var engine = Engine;
        var votes = ComputeWeightedVotes(input);
        T totalWeight = engine.TensorSum(engine.TensorAbs(_estimatorWeights.GetSubVector(0, Estimators.Count).AsTensor()));

        // Normalize to get probabilities (non-negative); uniform when no estimator carries weight
        if (NumOps.Compare(totalWeight, NumOps.Zero) > 0)
        {
            return engine.TensorClampMin(engine.TensorDivideScalar(votes, totalWeight), NumOps.Zero).ToMatrix();
        }

        var uniform = new Tensor<T>(new[] { input.Rows, NumClasses });
        engine.TensorFill(uniform, NumOps.Divide(NumOps.One, NumOps.FromDouble(NumClasses)));
        return uniform.ToMatrix();
    }

    /// <summary>
    /// votes[i, c] = sum of the weights of the estimators that predict class c for sample i.
    /// </summary>
    private Tensor<T> ComputeWeightedVotes(Matrix<T> input)
    {
        var engine = Engine;
        var weights = _estimatorWeights ?? throw new InvalidOperationException("Model has not been trained.");
        var weightValues = weights.ToArray();
        var votes = new Tensor<T>(new[] { input.Rows, NumClasses });
        for (int m = 0; m < Estimators.Count; m++)
        {
            var predicted = ClassMembership(Estimators[m].Predict(input));
            votes = engine.TensorAdd(votes, engine.TensorMultiplyScalar(predicted, weightValues[m]));
        }

        return votes;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["NEstimators"] = Options.NEstimators;
        metadata.AdditionalInfo["LearningRate"] = Options.LearningRate;
        metadata.AdditionalInfo["Algorithm"] = Options.Algorithm.ToString();
        metadata.AdditionalInfo["ActualEstimators"] = Estimators.Count;
        return metadata;
    }
}

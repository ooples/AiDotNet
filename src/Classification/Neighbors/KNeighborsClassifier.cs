using AiDotNet.Attributes;
using AiDotNet.Classification;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;

namespace AiDotNet.Classification.Neighbors;

/// <summary>
/// K-Nearest Neighbors classifier that predicts based on the majority class of nearest neighbors.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// K-Nearest Neighbors (KNN) is a non-parametric, instance-based learning algorithm.
/// It stores all training data and classifies new samples by finding the k closest
/// training samples and voting on their class labels.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// KNN is one of the simplest machine learning algorithms. Think of it as "you are the
/// company you keep." To classify something new:
///
/// 1. Find the k most similar training examples
/// 2. Look at their classes
/// 3. Return the most common class
///
/// Example: Classifying a fruit by its weight and color:
/// - New fruit: 150g, red
/// - 3 nearest neighbors: Apple (160g, red), Apple (145g, red), Cherry (10g, red)
/// - Wait! Cherry is much smaller, so it's not really "near"
/// - Actual 3 nearest: Apple, Apple, Orange -> Predicted: Apple
///
/// This is why feature scaling is important for KNN!
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var newSample = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// var labels = new Vector&lt;double&gt;(new double[] { 0.0, 1.0, 0.0, 1.0 });
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// // Create a KNN classifier with k=5 neighbors
/// // Use AiModelBuilder facade for K-nearest-neighbors classification
/// var builder = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new KNeighborsClassifier&lt;double&gt;(new KNeighborsOptions&lt;double&gt; { NNeighbors = 5 }));
///
/// var result = builder.Build(features, labels);
/// var prediction = result.Predict(newSample);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.InstanceBased)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("Nearest Neighbor Pattern Classification", "https://doi.org/10.1109/TIT.1967.1053964")]
public partial class KNeighborsClassifier<T> : ProbabilisticClassifierBase<T>
{

    // A lazy learner: it stores training data, not parameters, as its own comment says.
    /// <summary>
    /// Gets the KNN specific options.
    /// </summary>
    protected new KNeighborsOptions<T> Options => (KNeighborsOptions<T>)base.Options;

    /// <summary>
    /// Stored training features.
    /// </summary>
    [Buffer]
    private Matrix<T>? _xTrain;

    /// <summary>
    /// Stored training labels.
    /// </summary>
    [Buffer]
    private Vector<T>? _yTrain;

    /// <summary>
    /// Initializes a new instance of the KNeighborsClassifier class.
    /// </summary>
    /// <param name="options">Configuration options for the KNN classifier.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public KNeighborsClassifier(KNeighborsOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options ?? new KNeighborsOptions<T>(), regularization, new CrossEntropyWithLogitsLoss<T>())
    {
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>

    /// <summary>
    /// Trains the KNN classifier by storing the training data.
    /// </summary>
    /// <remarks>
    /// KNN is a lazy learner - it doesn't actually build a model during training.
    /// It simply stores the training data for use during prediction.
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("Number of samples in X must match length of y.");
        }

        if (Options.NNeighbors <= 0)
        {
            throw new ArgumentException($"n_neighbors ({Options.NNeighbors}) must be greater than 0.");
        }

        if (Options.NNeighbors > x.Rows)
        {
            throw new ArgumentException($"n_neighbors ({Options.NNeighbors}) cannot be greater than number of training samples ({x.Rows}).");
        }

        NumFeatures = x.Columns;
        ClassLabels = ExtractClassLabels(y);
        NumClasses = ClassLabels.Length;
        TaskType = InferTaskType(y);

        // Store training data
        _xTrain = x.Clone();
        _yTrain = y.Clone();
    }

    /// <inheritdoc/>
    public override Matrix<T> PredictProbabilities(Matrix<T> input)
    {
        if (_xTrain == null || _yTrain == null)
        {
            throw new InvalidOperationException("Model must be trained before prediction.");
        }

        if (input.Rows == 0 || NumClasses == 0)
        {
            return new Matrix<T>(input.Rows, NumClasses);
        }

        // Process query rows in blocks so the [rows, train, features] distance tensor stays bounded
        int perRow = Math.Max(1, _xTrain.Rows * Math.Max(1, _xTrain.Columns));
        int batch = Math.Max(1, (1 << 22) / perRow);
        var queries = Tensor<T>.FromMatrix(input);
        var blocks = new List<Tensor<T>>();
        for (int start = 0; start < input.Rows; start += batch)
        {
            int rows = Math.Min(batch, input.Rows - start);
            blocks.Add(PredictBlockProbabilities(queries.Slice(0, start, start + rows)));
        }

        var probabilities = blocks.Count == 1 ? blocks[0] : Engine.TensorConcatenate(blocks.ToArray(), 0);
        return probabilities.ToMatrix();
    }

    /// <summary>
    /// Predicts class probabilities for a block of query rows.
    /// </summary>
    private Tensor<T> PredictBlockProbabilities(Tensor<T> queries)
    {
        var xTrain = _xTrain ?? throw new InvalidOperationException("_xTrain has not been initialized.");
        var yTrain = _yTrain ?? throw new InvalidOperationException("_yTrain has not been initialized.");
        var engine = Engine;

        int rows = queries.Shape[0];
        int n = xTrain.Rows;

        // Distances to all training samples, then the k nearest (stable: equal distances keep the lower index)
        var distances = ComputeDistances(queries, Tensor<T>.FromMatrix(xTrain));
        int k = Math.Min(Options.NNeighbors, n);
        var (sortedDistances, sortedIndices) = engine.TensorSort(distances, 1);
        var nearestDistances = sortedDistances.Slice(1, 0, k);
        var nearest = sortedIndices.Slice(1, 0, k).Reshape(rows * k);

        // Compute weighted votes: sum of weights of the neighbors of each class
        var neighborClasses = engine.TensorGather(GetClassIndexTensor(yTrain), nearest, 0);
        var oneHot = engine.TensorOneHot<T>(neighborClasses, NumClasses).Reshape(rows, k, NumClasses);
        var weights = GetWeights(nearestDistances).Reshape(rows, k, 1);
        var classVotes = engine.ReduceSum(engine.TensorMultiply(oneHot, weights), new[] { 1 }, false);

        // Normalize to probabilities (uniform when the total weight is not positive)
        var totalWeight = engine.ReduceSum(classVotes, new[] { 1 }, true);
        var hasWeight = engine.TensorGreaterThan(totalWeight, NumOps.Zero);
        var uniform = new Tensor<T>(new[] { rows, NumClasses });
        engine.TensorFill(uniform, NumOps.FromDouble(1.0 / NumClasses));
        var normalized = engine.TensorDivide(classVotes, totalWeight);
        var condition = engine.TensorAdd(new Tensor<T>(new[] { rows, NumClasses }), hasWeight);
        return engine.TensorWhere(condition, normalized, uniform);
    }

    /// <summary>
    /// Computes the [queries, train] distances between every query and training sample for the configured metric.
    /// </summary>
    private Tensor<T> ComputeDistances(Tensor<T> queries, Tensor<T> train)
    {
        var engine = Engine;
        int rows = queries.Shape[0];
        int n = train.Shape[0];
        int d = train.Shape[1];

        if (Options.Metric == DistanceMetric.Cosine)
        {
            return ComputeCosineDistances(queries, train);
        }

        var diff = engine.TensorSubtract(queries.Reshape(rows, 1, d), train.Reshape(1, n, d));
        int[] featureAxis = { 2 };
        return Options.Metric switch
        {
            DistanceMetric.Manhattan => engine.ReduceSum(engine.TensorAbs(diff), featureAxis, false),
            DistanceMetric.Minkowski => engine.TensorPower(
                engine.ReduceSum(engine.TensorPower(engine.TensorAbs(diff), NumOps.FromDouble(Options.P)), featureAxis, false),
                NumOps.FromDouble(1.0 / Options.P)),
            DistanceMetric.Chebyshev => engine.TensorClampMin(engine.ReduceMax(engine.TensorAbs(diff), featureAxis, false), NumOps.Zero),
            _ => engine.TensorSqrt(engine.ReduceSum(engine.TensorMultiply(diff, diff), featureAxis, false))
        };
    }

    /// <summary>
    /// 1 - cosine similarity; the similarity is clamped to [-1, 1] and is 0 when either norm is ~0.
    /// </summary>
    private Tensor<T> ComputeCosineDistances(Tensor<T> queries, Tensor<T> train)
    {
        var engine = Engine;
        int rows = queries.Shape[0];
        int n = train.Shape[0];

        var dot = engine.TensorMatMul(queries, engine.TensorTranspose(train));
        var queryNorms = engine.ReduceSum(engine.TensorMultiply(queries, queries), new[] { 1 }, true);
        var trainNorms = engine.ReduceSum(engine.TensorMultiply(train, train), new[] { 1 }, false).Reshape(1, n);
        var denominator = engine.TensorSqrt(engine.TensorMultiply(queryNorms, trainNorms));
        var similarity = engine.TensorClamp(engine.TensorDivide(dot, denominator), NumOps.FromDouble(-1.0), NumOps.One);
        var valid = engine.TensorGreaterThan(denominator, NumOps.FromDouble(1e-10));
        var zeros = new Tensor<T>(new[] { rows, n });
        return engine.ScalarMinusTensor(NumOps.One, engine.TensorWhere(valid, similarity, zeros));
    }

    /// <summary>
    /// Gets the weight of every neighbor based on the weighting scheme.
    /// </summary>
    private Tensor<T> GetWeights(Tensor<T> distances)
    {
        var engine = Engine;
        var ones = new Tensor<T>(distances.Shape.ToArray());
        engine.TensorFill(ones, NumOps.One);
        if (Options.Weights != WeightingScheme.Distance)
        {
            return ones;
        }

        // Very close to a training point (distance < 1e-10) gets a large weight, otherwise 1 / distance
        var close = engine.TensorLessThan(distances, NumOps.FromDouble(1e-10));
        var large = new Tensor<T>(distances.Shape.ToArray());
        engine.TensorFill(large, NumOps.FromDouble(1e10));
        return engine.TensorWhere(close, large, engine.TensorDivide(ones, distances));
    }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newModel = (KNeighborsClassifier<T>)Clone();
        newModel.SetParameters(parameters);
        return newModel;
    }

    /// <inheritdoc/>
    public Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        // KNN doesn't use gradient-based optimization
        // Return zero gradients for compatibility
        return new Vector<T>(NumFeatures);
    }

    /// <inheritdoc/>
    public void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // KNN doesn't use gradient-based optimization
        // This is a no-op for compatibility
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["NNeighbors"] = Options.NNeighbors;
        metadata.AdditionalInfo["Metric"] = Options.Metric.ToString();
        metadata.AdditionalInfo["Weights"] = Options.Weights.ToString();
        metadata.AdditionalInfo["TrainingSamples"] = _xTrain?.Rows ?? 0;
        return metadata;
    }
}

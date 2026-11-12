using AiDotNet.Attributes;
using AiDotNet.Classification;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Classification.Trees;

#pragma warning disable CS8601, CS8618 // Generic T defaults use default(T) - always used with value types

/// <summary>
/// A decision tree classifier that learns a hierarchy of decision rules from training data.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Decision trees are non-parametric supervised learning algorithms that learn decision rules
/// inferred from data features. They partition the feature space into regions and assign
/// class labels to each region.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Imagine playing a game of "20 Questions" to classify things. The decision tree learns
/// which questions (based on features) best separate the different classes.
///
/// Example: Classifying whether to play tennis
/// 1. Is it raining? -> No: Go to step 2, Yes: Don't play
/// 2. Is humidity > 75%? -> No: Play!, Yes: Don't play
///
/// Each question splits the data based on a feature value, and leaves contain the final decisions.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var newSample = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// var labels = new Vector&lt;double&gt;(new double[] { 0.0, 1.0, 0.0, 1.0 });
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// // Use AiModelBuilder facade for decisiontreeclassifier classification
/// var builder = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new DecisionTreeClassifier&lt;double&gt;(
///         new DecisionTreeClassifierOptions&lt;double&gt;()));
///
/// var result = builder.Build(features, labels);
/// var prediction = result.Predict(newSample);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.DecisionTree)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("Classification and Regression Trees", "https://doi.org/10.1201/9781315139470")]
public partial class DecisionTreeClassifier<T> : ProbabilisticClassifierBase<T>, ITreeBasedClassifier<T>
{

    // Its own comment: "Decision trees do not have traditional numeric parameters". The base fold
    // states that directly.
    /// <summary>
    /// Gets the decision tree specific options.
    /// </summary>
    protected new DecisionTreeClassifierOptions<T> Options => (DecisionTreeClassifierOptions<T>)base.Options;

    /// <summary>
    /// The root node of the decision tree.
    /// </summary>
    private DecisionNode<T>? _root;

    /// <summary>
    /// Random number generator for feature selection when MaxFeatures is set.
    /// </summary>
    private Random? _random;

    /// <inheritdoc/>
    public int MaxDepth => Options.MaxDepth ?? CalculateActualDepth(_root);

    /// <inheritdoc/>
    public Vector<T>? FeatureImportances { get; private set; }

    /// <inheritdoc/>
    public int LeafCount => CountLeaves(_root);

    /// <inheritdoc/>
    public int NodeCount => CountNodes(_root);

    /// <summary>
    /// Initializes a new instance of the DecisionTreeClassifier class.
    /// </summary>
    /// <param name="options">Configuration options for the decision tree.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public DecisionTreeClassifier(DecisionTreeClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options ?? new DecisionTreeClassifierOptions<T>(), regularization, new CrossEntropyWithLogitsLoss<T>())
    {
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>

    /// <summary>
    /// Trains the decision tree on the provided data.
    /// </summary>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("Number of samples in X must match length of y.");
        }

        var rows = Enumerable.Range(0, x.Rows).ToArray();
        TrainDense(CreateDenseData(x), y, rows, y);
    }

    /// <summary>
    /// Trains on the data rows <paramref name="rows"/> of already-ranked <paramref name="data"/> (a row listed twice
    /// counts twice), exactly as <see cref="Train"/> would on a matrix holding those rows in that order. A forest
    /// ranks its data once and trains every tree on its bootstrap rows this way, without copying them out.
    /// </summary>
    /// <param name="y">Labels indexed by data row.</param>
    internal void TrainOnRows(DenseTreeData<T> data, Vector<T> y, int[] rows)
    {
        var sampleLabels = AiDotNetEngine.Current.TensorTake(y.AsTensor(), new Tensor<int>(rows, new[] { rows.Length })).ToVector();
        TrainDense(data, y, rows, sampleLabels);
    }

    private void TrainDense(DenseTreeData<T> data, Vector<T> y, int[] rows, Vector<T> sampleLabels)
    {
        NumFeatures = data.Features;
        ClassLabels = ExtractClassLabels(sampleLabels);
        NumClasses = ClassLabels.Length;
        TaskType = InferTaskType(sampleLabels);

        // Initialize random number generator
        _random = Options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(Options.Seed.Value)
            : RandomHelper.CreateSeededRandom(42);

        // Initialize feature importances
        FeatureImportances = new Vector<T>(NumFeatures);

        // Grow the tree on the ranked data (DecisionTreeClassifier.DenseBuilder.cs).
        _root = BuildDense(data, y, rows);

        // Normalize feature importances
        NormalizeFeatureImportances();
    }

    /// <summary>
    /// Adds a split's impurity decrease weighted by its share of the samples (scikit-learn's
    /// N_t / N * impurity decrease; the normalization makes the scale irrelevant).
    /// </summary>
    private void UpdateFeatureImportances(int feature, T weightedGain)
    {
        var importances = FeatureImportances ?? throw new InvalidOperationException("FeatureImportances has not been initialized.");
        importances[feature] = NumOps.Add(importances[feature], weightedGain);
    }

    /// <summary>
    /// Normalizes feature importances to sum to 1.
    /// </summary>
    private void NormalizeFeatureImportances()
    {
        if (FeatureImportances == null)
            return;

        var engine = AiDotNetEngine.Current;
        T sum = engine.TensorSum(FeatureImportances.AsTensor());
        if (NumOps.Compare(sum, NumOps.Zero) > 0)
        {
            FeatureImportances = engine.TensorDivideScalar(FeatureImportances.AsTensor(), sum).ToVector();
        }
    }

    /// <summary>
    /// Gets the class index for a label.
    /// </summary>
    private int GetClassIndex(T label)
    {
        if (ClassLabels == null)
        {
            throw new InvalidOperationException("Model must be trained before getting class index.");
        }

        var engine = AiDotNetEngine.Current;
        var matches = engine.TensorEquals(ClassLabels.AsTensor(), label);
        if (NumOps.Compare(engine.TensorSum(matches), NumOps.Zero) == 0)
        {
            throw new ArgumentException($"Label {label} not found in class labels.");
        }

        return engine.TensorArgMax(matches, 0).GetFlat(0);
    }

    /// <summary>The trained tree as flat arrays for batch evaluation, or null before training.</summary>
    internal FlatDecisionTree<T>? Flatten() => _root is null ? null : FlatDecisionTree<T>.From(_root);

    /// <inheritdoc/>
    public override Matrix<T> PredictProbabilities(Matrix<T> input)
    {
        if (_root == null)
        {
            throw new InvalidOperationException("Model must be trained before prediction.");
        }

        var engine = AiDotNetEngine.Current;
        int n = input.Rows;
        if (n == 0)
        {
            return new Matrix<T>(0, NumClasses);
        }

        // Route all rows down the tree at once: each node splits its rows into value <= threshold (left) and the
        // rest, each leaf emits its class distribution for its rows, and the blocks are put back in row order.
        var x = new Tensor<T>(new[] { n, input.Columns }, input);
        var blocks = new List<Tensor<T>>();
        var positions = new List<Tensor<int>>();
        RouteRows(_root, x, new Tensor<int>(Enumerable.Range(0, n).ToArray(), new[] { n }), blocks, positions);

        var leafRows = blocks.Count == 1 ? blocks[0] : engine.TensorConcatenate(blocks.ToArray(), 0);
        var placed = positions.Count == 1 ? positions[0] : engine.TensorConcatenate(positions.ToArray(), 0);
        return engine.TensorGather(leafRows, engine.TensorArgsort(placed, 0, false), 0).ToMatrix();
    }

    private void RouteRows(DecisionNode<T> node, Tensor<T> x, Tensor<int> rows, List<Tensor<T>> blocks, List<Tensor<int>> positions)
    {
        if (rows.Length == 0) return;
        var engine = AiDotNetEngine.Current;
        if (node.IsLeaf)
        {
            var probabilities = node.ClassProbabilities ?? throw new InvalidOperationException("ClassProbabilities has not been initialized.");
            blocks.Add(engine.TensorBroadcastTo(
                engine.Reshape(probabilities.AsTensor(), new[] { 1, probabilities.Length }), new[] { rows.Length, probabilities.Length }));
            positions.Add(rows);
            return;
        }

        int n = rows.Length;
        var column = engine.Reshape(
            engine.TensorSlice(engine.TensorGather(x, rows, 0), new[] { 0, node.FeatureIndex }, new[] { n, 1 }), new[] { n });
        var goesRight = engine.TensorGreaterThan(column, node.Threshold);
        int rightCount = NumOps.ToInt32(engine.TensorSum(goesRight));
        var left = node.Left ?? throw new InvalidOperationException("node.Left is null where it is required.");
        var right = node.Right ?? throw new InvalidOperationException("node.Right is null where it is required.");
        if (rightCount == 0)
        {
            RouteRows(left, x, rows, blocks, positions);
            return;
        }

        if (rightCount == n)
        {
            RouteRows(right, x, rows, blocks, positions);
            return;
        }

        // key = goesRight * n + position puts the left rows first, then the right rows, each in order.
        var key = engine.TensorAdd(engine.TensorMultiplyScalar(goesRight, NumOps.FromDouble(n)), engine.TensorArange<T>(0, n, 1));
        var arranged = engine.TensorGather(rows, engine.TensorArgsort(key, 0, false), 0);
        int leftCount = n - rightCount;
        RouteRows(left, x, engine.TensorSlice(arranged, new[] { 0 }, new[] { leftCount }), blocks, positions);
        RouteRows(right, x, engine.TensorSlice(arranged, new[] { leftCount }, new[] { rightCount }), blocks, positions);
    }

    /// <summary>
    /// Calculates the actual depth of the tree.
    /// </summary>
    private int CalculateActualDepth(DecisionNode<T>? node)
    {
        if (node == null || node.IsLeaf)
            return 0;

        return 1 + Math.Max(CalculateActualDepth(node.Left), CalculateActualDepth(node.Right));
    }

    /// <summary>
    /// Counts the number of leaf nodes.
    /// </summary>
    private int CountLeaves(DecisionNode<T>? node)
    {
        if (node == null)
            return 0;
        if (node.IsLeaf)
            return 1;
        return CountLeaves(node.Left) + CountLeaves(node.Right);
    }

    /// <summary>
    /// Counts the total number of nodes.
    /// </summary>
    private int CountNodes(DecisionNode<T>? node)
    {
        if (node == null)
            return 0;
        return 1 + CountNodes(node.Left) + CountNodes(node.Right);
    }

    /// <summary>
    /// Deep clones a decision tree node.
    /// </summary>
    private DecisionNode<T> CloneNode(DecisionNode<T> node)
    {
        var cloned = new DecisionNode<T>
        {
            IsLeaf = node.IsLeaf,
            FeatureIndex = node.FeatureIndex,
            Threshold = node.Threshold,
            PredictedClass = node.PredictedClass,
            NumSamples = node.NumSamples
        };

        if (node.ClassProbabilities != null)
        {
            cloned.ClassProbabilities = node.ClassProbabilities.Clone();
        }

        if (node.Left != null)
        {
            cloned.Left = CloneNode(node.Left);
        }

        if (node.Right != null)
        {
            cloned.Right = CloneNode(node.Right);
        }

        return cloned;
    }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newModel = (DecisionTreeClassifier<T>)Clone();
        newModel.SetParameters(parameters);
        return newModel;
    }

    /// <inheritdoc/>
    public Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        // Decision trees don't typically use gradient-based optimization
        // Return zero gradients for compatibility
        return new Vector<T>(NumFeatures);
    }

    /// <inheritdoc/>
    public void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // Decision trees don't typically use gradient-based optimization
        // This is a no-op for compatibility
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["MaxDepth"] = Options.MaxDepth?.ToString() ?? "unlimited";
        metadata.AdditionalInfo["Criterion"] = Options.Criterion.ToString();
        metadata.AdditionalInfo["LeafCount"] = LeafCount;
        metadata.AdditionalInfo["NodeCount"] = NodeCount;
        return metadata;
    }

    /// <summary>
    /// Serializes a decision node to a dictionary for JSON serialization.
    /// </summary>
    private Dictionary<string, object> SerializeNode(DecisionNode<T> node)
    {
        var nodeData = new Dictionary<string, object>
        {
            { "IsLeaf", node.IsLeaf },
            { "FeatureIndex", node.FeatureIndex },
            { "Threshold", NumOps.ToDouble(node.Threshold) },
            { "PredictedClass", node.PredictedClass },
            { "NumSamples", node.NumSamples }
        };

        if (node.ClassProbabilities is not null)
        {
            var probsArray = node.ClassProbabilities.AsTensor().Cast<double>().ToArray();
            nodeData["ClassProbabilities"] = probsArray;
        }

        if (node.Left is not null)
        {
            nodeData["Left"] = SerializeNode(node.Left);
        }

        if (node.Right is not null)
        {
            nodeData["Right"] = SerializeNode(node.Right);
        }

        return nodeData;
    }

    /// <summary>
    /// Deserializes a decision node from a JSON object.
    /// </summary>
    private DecisionNode<T>? DeserializeNode(Newtonsoft.Json.Linq.JObject? nodeData)
    {
        if (nodeData == null)
        {
            return null;
        }

        var node = new DecisionNode<T>
        {
            IsLeaf = nodeData["IsLeaf"]?.ToObject<bool>() ?? false,
            FeatureIndex = nodeData["FeatureIndex"]?.ToObject<int>() ?? 0,
            Threshold = NumOps.FromDouble(nodeData["Threshold"]?.ToObject<double>() ?? 0.0),
            PredictedClass = nodeData["PredictedClass"]?.ToObject<int>() ?? 0,
            NumSamples = nodeData["NumSamples"]?.ToObject<int>() ?? 0
        };

        var probsToken = nodeData["ClassProbabilities"];
        if (probsToken is not null)
        {
            var probsArray = probsToken.ToObject<double[]>() ?? Array.Empty<double>();
            if (probsArray.Length > 0)
            {
                node.ClassProbabilities = new Tensor<double>(probsArray, new[] { probsArray.Length }).Cast<T>().ToVector();
            }
        }

        var leftToken = nodeData["Left"];
        if (leftToken is not null && leftToken.Type == Newtonsoft.Json.Linq.JTokenType.Object)
        {
            node.Left = DeserializeNode(leftToken as Newtonsoft.Json.Linq.JObject);
        }

        var rightToken = nodeData["Right"];
        if (rightToken is not null && rightToken.Type == Newtonsoft.Json.Linq.JTokenType.Object)
        {
            node.Right = DeserializeNode(rightToken as Newtonsoft.Json.Linq.JObject);
        }

        return node;
    }
}

/// <summary>
/// Represents a node in the decision tree.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
internal class DecisionNode<T>
{
    /// <summary>
    /// Whether this is a leaf node.
    /// </summary>
    public bool IsLeaf { get; set; }

    /// <summary>
    /// The feature index used for splitting (internal nodes only).
    /// </summary>
    public int FeatureIndex { get; set; }

    /// <summary>
    /// The threshold value for splitting (internal nodes only).
    /// </summary>
    public T Threshold { get; set; } = default;

    /// <summary>
    /// The left child node (values &lt;= threshold).
    /// </summary>
    public DecisionNode<T>? Left { get; set; }

    /// <summary>
    /// The right child node (values > threshold).
    /// </summary>
    public DecisionNode<T>? Right { get; set; }

    /// <summary>
    /// Class probabilities (leaf nodes only).
    /// </summary>
    public Vector<T>? ClassProbabilities { get; set; }

    /// <summary>
    /// Predicted class (leaf nodes only).
    /// </summary>
    public int PredictedClass { get; set; }

    /// <summary>
    /// Number of samples that reached this node during training.
    /// </summary>
    public int NumSamples { get; set; }
}

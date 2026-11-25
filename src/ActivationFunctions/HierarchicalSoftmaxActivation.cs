using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the Hierarchical Softmax activation function for neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// The classes are the leaves of a balanced binary tree (Morin and Bengio, 2005). Each of the
/// <c>numClasses - 1</c> internal nodes owns a weight vector <c>w_n</c> and decides between its two subtrees with
/// <c>sigmoid(w_n . x)</c> (right) and <c>1 - sigmoid(w_n . x)</c> (left). The probability of a class is the product
/// of the decisions on the path from the root to its leaf. Because every internal node splits its probability mass
/// between its two children, the class probabilities sum to 1 for any number of classes.
/// </para>
/// <para>
/// The tree is built by halving: a node that covers classes <c>[low, high)</c> sends the first
/// <c>ceil((high - low) / 2)</c> of them left and the rest right, so no path is longer than
/// <c>ceil(log2(numClasses))</c>. Internal nodes are numbered in pre-order (a node, then its left subtree, then its
/// right subtree); node 0 is the root.
/// </para>
/// <para>
/// <b>For Beginners:</b> Hierarchical Softmax is an efficient alternative to the standard Softmax function,
/// especially when dealing with a large number of output classes (like thousands of words in language models).
///
/// While regular Softmax calculates probabilities for all possible classes at once (which can be slow
/// with many classes), Hierarchical Softmax organizes classes in a tree structure:
///
/// - Think of it like a "20 Questions" game where each question narrows down the possibilities
/// - Each node in the tree asks its own yes/no question (left or right)
/// - The final probability is calculated by multiplying probabilities along the path to a class
///
/// Finding the probability of one class takes O(log N) questions instead of O(N), where N is the number of classes.
///
/// Common uses include:
/// - Natural language processing (predicting words from vocabularies)
/// - Classification problems with many categories
/// - Any task where computing standard Softmax would be too slow
/// </para>
/// </remarks>
[ActivationCategory(ActivationCategory.Normalization)]
[ActivationCategory(ActivationCategory.Output)]
[ActivationTask(ActivationTask.OutputLayer)]
[ActivationProperty(IsMonotonic = false, ZeroPreserving = false, IsBounded = true, IsVectorActivation = true, Cost = ComputeCost.High)]
public class HierarchicalSoftmaxActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// The total number of output classes (the leaves of the tree).
    /// </summary>
    private readonly int _numClasses;

    /// <summary>
    /// The number of internal nodes of the tree: <c>numClasses - 1</c>.
    /// </summary>
    private readonly int _nodeCount;

    /// <summary>
    /// <c>[numClasses, nodes]</c>: 1 where the path to the class turns right at the node, 0 elsewhere.
    /// </summary>
    private readonly Tensor<T> _rightTurns;

    /// <summary>
    /// <c>[numClasses, nodes]</c>: 1 where the path to the class turns left at the node, 0 elsewhere.
    /// </summary>
    private readonly Tensor<T> _leftTurns;

    /// <summary>
    /// The weight vector of every internal node, <c>[nodes, inputDim]</c>.
    /// </summary>
    private Tensor<T> _nodeWeights;

    /// <summary>
    /// The input dimension that the node weights were initialized for.
    /// </summary>
    private int _weightInputDim;

    /// <summary>
    /// Gets the node weights as a tensor.
    /// </summary>
    /// <value>
    /// A tensor with shape [numClasses - 1, inputDim]: one weight vector per internal node of the tree, in pre-order
    /// (row 0 is the root).
    /// </value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> every decision point in the tree has its own set of weights. This property exposes
    /// all of them so the activation can be traced, exported, or inspected. The tensor is a copy: changing it does
    /// not change the activation.
    /// </para>
    /// </remarks>
    public Tensor<T> NodeWeightsTensor => new Tensor<T>(_nodeWeights.ToArray(), new[] { _nodeCount, _weightInputDim });

    /// <summary>
    /// Initializes a new instance of the Hierarchical Softmax activation function.
    /// </summary>
    /// <param name="numClasses">The number of output classes; at least 1.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="numClasses"/> is less than 1.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the Hierarchical Softmax with the specified number of classes.
    ///
    /// When you create a Hierarchical Softmax:
    /// - It builds a balanced binary tree whose leaves are the classes
    /// - It gives every decision point in the tree (there are numClasses - 1 of them) its own random weights
    ///
    /// For example, with 5 classes the root separates classes 0-2 from classes 3-4, the next decision on the left
    /// separates classes 0-1 from class 2, and so on until every class sits alone at a leaf. A single class needs
    /// no decision at all and always has probability 1.
    /// </para>
    /// </remarks>
    public HierarchicalSoftmaxActivation(int numClasses)
    {
        if (numClasses < 1)
            throw new ArgumentOutOfRangeException(nameof(numClasses), numClasses, "At least one class is required.");

        _numClasses = numClasses;
        _nodeCount = numClasses - 1;
        (_rightTurns, _leftTurns) = BuildPaths(numClasses);
        _weightInputDim = numClasses;
        _nodeWeights = CreateWeights(_nodeCount, numClasses);
    }

    /// <summary>
    /// Indicates whether this activation function supports operations on individual scalar values.
    /// </summary>
    /// <returns>Always returns false as Hierarchical Softmax requires vector operations.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method tells the neural network that Hierarchical Softmax
    /// cannot work with single numbers - it needs to process entire vectors (arrays of numbers) at once.
    /// </para>
    /// </remarks>
    protected override bool SupportsScalarOperations() => false;

    /// <summary>
    /// Applies the Hierarchical Softmax activation function to an input vector.
    /// </summary>
    /// <param name="input">The input vector to transform.</param>
    /// <returns>A vector of probabilities for each class; they sum to 1.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes a vector of numbers (like features from previous layers)
    /// and converts them into probabilities for each possible class. The output values will be between
    /// 0 and 1, representing how likely each class is, and together they add up to 1.
    /// </para>
    /// <para>
    /// This is the tensor overload applied to the vector, so the two always agree.
    /// </para>
    /// </remarks>
    public override Vector<T> Activate(Vector<T> input)
        => Activate(Tensor<T>.FromVector(input)).ToVector();

    /// <summary>
    /// Applies Hierarchical Softmax to a tensor using only engine primitives so the gradient tape records every
    /// step. Works on any tensor where the trailing dimension is the feature dimension that feeds the tree
    /// classifier; the output has the same leading dimensions and a trailing dimension of <c>numClasses</c>.
    /// </summary>
    /// <param name="input">Input tensor. Trailing axis must match the feature dimension.</param>
    /// <returns>A tensor of class probabilities with trailing axis of size <c>numClasses</c>.</returns>
    /// <remarks>
    /// <para>
    /// With <c>s = x W^T</c> the node scores, the log-probability of class c is
    /// <c>sum_n right[c, n] * log sigmoid(s_n) + left[c, n] * log sigmoid(-s_n)</c>: two matrix products with the
    /// constant path matrices. <c>log sigmoid(-s) = log sigmoid(s) - s</c>, and the log-sigmoid is finite for every
    /// finite score, so a saturated node cannot turn an untaken branch into 0 * infinity.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> this is the batched version of the vector method: every row of the input gets its own
    /// set of class probabilities.
    /// </para>
    /// </remarks>
    public override Tensor<T> Activate(Tensor<T> input)
    {
        if (input is null)
            throw new ArgumentNullException(nameof(input));
        int rank = input.Shape.Length;
        if (rank == 0)
            throw new ArgumentException("HierarchicalSoftmax requires at least a 1D tensor.", nameof(input));

        int inputDim = input.Shape[rank - 1];
        EnsureWeights(inputDim);

        // Flatten leading dims into a single batch dim so we can matmul.
        int batchSize = 1;
        for (int d = 0; d < rank - 1; d++)
            batchSize *= input.Shape[d];

        var probabilities = PathProbabilities(Engine.Reshape(input, new[] { batchSize, inputDim }));

        // Restore original batch shape with numClasses as the trailing axis.
        int[] outShape = new int[rank];
        for (int d = 0; d < rank - 1; d++)
            outShape[d] = input.Shape[d];
        outShape[rank - 1] = _numClasses;
        return Engine.Reshape(probabilities, outShape);
    }

    /// <summary>
    /// Calculates the derivative (Jacobian matrix) of the Hierarchical Softmax function for an input vector.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <returns>A Jacobian matrix [numClasses, inputDim] of partial derivatives.</returns>
    /// <remarks>
    /// <para>
    /// For class c, <c>dP_c/dx = P_c * sum_n (right[c, n] * (1 - sigma_n) - left[c, n] * sigma_n) * w_n</c>: the
    /// derivative of the log of each decision on the path, times the class probability. Every column of the matrix
    /// sums to zero, because the probabilities always sum to 1.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The derivative tells us how much each output probability changes when we slightly
    /// change each input value. Each row represents an output class and each column an input feature.
    /// This information is crucial during neural network training to determine how to adjust the weights.
    /// </para>
    /// </remarks>
    public override Matrix<T> Derivative(Vector<T> input)
    {
        if (input is null)
            throw new ArgumentNullException(nameof(input));

        int inputDim = input.Length;
        EnsureWeights(inputDim);
        if (_nodeCount == 0 || inputDim == 0)
            return new Matrix<T>(_numClasses, inputDim);

        var row = Engine.Reshape(Tensor<T>.FromVector(input), new[] { 1, inputDim });
        var scores = NodeScores(row);
        var takeRight = Engine.Sigmoid(scores);
        var takeLeft = Engine.Sigmoid(Engine.TensorNegate(scores));

        // d log P_c / d s_n: (1 - sigma_n) on a right turn, -sigma_n on a left turn, 0 off the path.
        var logSlope = Engine.TensorSubtract(
            Engine.TensorMultiply(_rightTurns, takeLeft),
            Engine.TensorMultiply(_leftTurns, takeRight));
        var probabilities = Engine.Reshape(PathProbabilities(row), new[] { _numClasses, 1 });
        return Engine.TensorMultiply(Engine.TensorMatMul(logSlope, _nodeWeights), probabilities).ToMatrix();
    }

    /// <summary>
    /// Class probabilities <c>[batch, numClasses]</c> for input rows <c>[batch, inputDim]</c>.
    /// </summary>
    private Tensor<T> PathProbabilities(Tensor<T> rows)
    {
        // A single class has no decision to make: probability 1.
        if (_nodeCount == 0)
            return Tensor<T>.CreateDefault(new[] { rows.Shape[0], 1 }, NumOps.One);

        var scores = NodeScores(rows);
        var logRight = Engine.TensorLogSigmoid(scores);
        var logLeft = Engine.TensorSubtract(logRight, scores);
        var logProbability = Engine.TensorAdd(
            Engine.TensorMatMul(logRight, Engine.TensorPermute(_rightTurns, new[] { 1, 0 })),
            Engine.TensorMatMul(logLeft, Engine.TensorPermute(_leftTurns, new[] { 1, 0 })));
        return Engine.TensorExp(logProbability);
    }

    /// <summary>
    /// Node scores <c>x . w_n</c>, <c>[batch, nodes]</c>, for input rows <c>[batch, inputDim]</c>.
    /// </summary>
    private Tensor<T> NodeScores(Tensor<T> rows)
        // With no features every score is the empty dot product, 0.
        => rows.Shape[1] == 0
            ? new Tensor<T>(new[] { rows.Shape[0], _nodeCount })
            : Engine.TensorMatMul(rows, Engine.TensorPermute(_nodeWeights, new[] { 1, 0 }));

    /// <summary>
    /// Draws new node weights when the input dimension differs from the one the weights were drawn for.
    /// </summary>
    private void EnsureWeights(int inputDim)
    {
        if (inputDim == _weightInputDim)
            return;
        _weightInputDim = inputDim;
        _nodeWeights = CreateWeights(_nodeCount, inputDim);
    }

    /// <summary>
    /// Node weights <c>[nodes, inputDim]</c> drawn uniformly from [-0.5, 0.5), node by node.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> before training, every decision point in the tree starts with small random weights.
    /// Random starting values are important because they break symmetry and allow the network to learn
    /// different patterns at different nodes.
    /// </para>
    /// </remarks>
    private static Tensor<T> CreateWeights(int nodes, int inputDim)
    {
        var random = RandomHelper.CreateSecureRandom();
        var values = new double[nodes * inputDim];
        for (int i = 0; i < values.Length; i++)
            values[i] = random.NextDouble() - 0.5;
        return new Tensor<double>(values, new[] { nodes, inputDim }).Cast<T>();
    }

    /// <summary>
    /// Builds the balanced tree over the classes and returns, per class and internal node, whether the path to the
    /// class turns right or left there.
    /// </summary>
    private static (Tensor<T> RightTurns, Tensor<T> LeftTurns) BuildPaths(int numClasses)
    {
        int nodes = numClasses - 1;
        var right = new double[numClasses * nodes];
        var left = new double[numClasses * nodes];

        // Pre-order: a node covering [low, high) is numbered before its left subtree, which comes before its right.
        var pending = new Stack<(int Low, int High)>();
        pending.Push((0, numClasses));
        int next = 0;
        while (pending.Count > 0)
        {
            var (low, high) = pending.Pop();
            if (high - low < 2)
                continue;

            int node = next++;
            int middle = low + (high - low + 1) / 2;
            for (int c = low; c < middle; c++)
                left[c * nodes + node] = 1.0;
            for (int c = middle; c < high; c++)
                right[c * nodes + node] = 1.0;

            pending.Push((middle, high));
            pending.Push((low, middle));
        }

        var shape = new[] { numClasses, nodes };
        return (new Tensor<double>(right, shape).Cast<T>(), new Tensor<double>(left, shape).Cast<T>());
    }
}

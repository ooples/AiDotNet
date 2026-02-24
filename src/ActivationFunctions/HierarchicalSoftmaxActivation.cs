using AiDotNet.Autodiff;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the Hierarchical Softmax activation function for neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Hierarchical Softmax is an efficient alternative to the standard Softmax function,
/// especially when dealing with a large number of output classes (like thousands of words in language models).
/// 
/// While regular Softmax calculates probabilities for all possible classes at once (which can be slow
/// with many classes), Hierarchical Softmax organizes classes in a tree structure:
/// 
/// - Think of it like a "20 Questions" game where each question narrows down the possibilities
/// - Each node in the tree represents a binary decision (left or right)
/// - The final probability is calculated by multiplying probabilities along the path to a class
/// 
/// This approach reduces computation from O(N) to O(log N), where N is the number of classes,
/// making it much faster for problems with many output classes.
/// 
/// Common uses include:
/// - Natural language processing (predicting words from vocabularies)
/// - Classification problems with many categories
/// - Any task where computing standard Softmax would be too slow
/// </para>
/// </remarks>
public class HierarchicalSoftmaxActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// The total number of output classes.
    /// </summary>
    private readonly int _numClasses;

    /// <summary>
    /// The depth of the binary tree used to represent the hierarchical structure.
    /// </summary>
    private readonly int _treeDepth;

    /// <summary>
    /// The weights for each node in the binary tree.
    /// </summary>
    private Matrix<T> _nodeWeights;

    /// <summary>
    /// The input dimension that the weight matrix was initialized for.
    /// </summary>
    private int _weightInputDim;

    /// <summary>
    /// Gets the node weights as a tensor for use in computation graphs.
    /// </summary>
    /// <value>A tensor containing the node weights with shape [treeDepth, numClasses].</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This property provides access to the internal weights used by the hierarchical
    /// tree structure. When using JIT compilation, you can wrap these weights in a ComputationNode
    /// to enable gradient computation and weight updates during training.
    /// </para>
    /// </remarks>
    public Tensor<T> NodeWeightsTensor => Tensor<T>.FromRowMatrix(_nodeWeights);

    /// <summary>
    /// Initializes a new instance of the Hierarchical Softmax activation function.
    /// </summary>
    /// <param name="numClasses">The number of output classes to support.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the hierarchical structure needed for efficient probability calculation.
    /// 
    /// It creates a binary tree where:
    /// - The number of levels (tree depth) is calculated based on the number of classes
    /// - Each node in the tree gets its own set of weights
    /// - Weights are initialized randomly to start the learning process
    /// 
    /// For example, if you have 8 classes, it creates a 3-level tree (because 2³=8),
    /// allowing the model to make 3 binary decisions to reach any of the 8 classes.
    /// </para>
    /// </remarks>
    public HierarchicalSoftmaxActivation(int numClasses)
    {
        _numClasses = numClasses;
        _treeDepth = (int)Math.Ceiling(MathHelper.Log2(numClasses));
        _weightInputDim = numClasses;
        _nodeWeights = new Matrix<T>(_treeDepth, numClasses);
        InitializeWeights();
    }

    /// <summary>
    /// Indicates whether this activation function can operate on individual scalar values.
    /// </summary>
    /// <returns>Always returns false as Hierarchical Softmax requires vector operations.</returns>
    protected override bool SupportsScalarOperations() => false;

    /// <summary>
    /// Applies the Hierarchical Softmax activation function to transform input vectors into class probabilities.
    /// </summary>
    /// <param name="input">The input vector to transform.</param>
    /// <returns>A vector containing probabilities for each output class.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method converts your neural network's raw output (numbers) into probabilities
    /// that sum to approximately 1, making them suitable for classification tasks.
    /// 
    /// For each possible class:
    /// 1. It traces a unique path through the binary tree
    /// 2. At each node, it calculates the probability of going left or right
    /// 3. It multiplies these probabilities to get the final probability for that class
    /// 
    /// Unlike standard Softmax which computes all classes at once, this method calculates
    /// each class probability independently by following its path through the tree.
    /// </para>
    /// </remarks>
    public override Vector<T> Activate(Vector<T> input)
    {
        // Reinitialize weights if input dimension doesn't match
        if (input.Length != _weightInputDim)
        {
            _weightInputDim = input.Length;
            _nodeWeights = new Matrix<T>(_treeDepth, input.Length);
            InitializeWeights();
        }

        Vector<T> output = new Vector<T>(_numClasses);
        for (int i = 0; i < _numClasses; i++)
        {
            output[i] = ComputePathProbability(input, i);
        }

        return output;
    }

    /// <summary>
    /// Calculates the derivative (gradient) of the Hierarchical Softmax function.
    /// </summary>
    /// <param name="input">The input vector at which to calculate the derivative.</param>
    /// <returns>A Jacobian matrix containing the partial derivatives.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The derivative tells us how much the output probabilities change when we slightly
    /// change each input value. This information is essential during neural network training.
    /// 
    /// This method:
    /// 1. Creates a matrix where each row represents how one output class is affected by changes in each input
    /// 2. For each class, it calculates how changes to the input affect the probability of that class
    /// 3. These calculations help the neural network learn by adjusting weights in the right direction
    /// 
    /// The "Jacobian matrix" is simply a collection of all these derivatives organized in rows and columns,
    /// where each row corresponds to an output class and each column to an input dimension.
    /// </para>
    /// </remarks>
    public override Matrix<T> Derivative(Vector<T> input)
    {
        Matrix<T> jacobian = new Matrix<T>(_numClasses, input.Length);

        for (int i = 0; i < _numClasses; i++)
        {
            Vector<T> pathDerivative = ComputePathDerivative(input, i);
            jacobian.SetRow(i, pathDerivative);
        }

        return jacobian;
    }

    /// <summary>
    /// Calculates the derivative of the path probability for a specific class.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <param name="classIndex">The index of the class for which to calculate the derivative.</param>
    /// <returns>A vector containing the partial derivatives with respect to each input dimension.</returns>
    private Vector<T> ComputePathDerivative(Vector<T> input, int classIndex)
    {
        Vector<T> derivative = new Vector<T>(input.Length);
        int node = 1;

        for (int depth = 0; depth < _treeDepth; depth++)
        {
            Vector<T> nodeWeights = _nodeWeights.GetRow(depth);
            T nodeOutput = MathHelper.Sigmoid(input.DotProduct(nodeWeights));
            bool goRight = (classIndex & (1 << (_treeDepth - depth - 1))) != 0;

            T multiplier = goRight ? nodeOutput : NumOps.Subtract(NumOps.One, nodeOutput);
            T derivativeFactor = NumOps.Multiply(nodeOutput, NumOps.Subtract(NumOps.One, nodeOutput));

            if (goRight)
            {
                derivative = derivative.Add(nodeWeights.Multiply(derivativeFactor));
            }
            else
            {
                derivative = derivative.Subtract(nodeWeights.Multiply(derivativeFactor));
            }

            node = node * 2 + (goRight ? 1 : 0);
            if (node >= _numClasses) break;
        }

        return derivative;
    }

    /// <summary>
    /// Initializes the weights for all nodes in the binary tree with small random values.
    /// </summary>
    private void InitializeWeights()
    {
        var random = RandomHelper.CreateSecureRandom();
        int cols = _nodeWeights.Columns;
        for (int i = 0; i < _treeDepth; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                _nodeWeights[i, j] = NumOps.FromDouble(random.NextDouble() - 0.5);
            }
        }
    }

    /// <summary>
    /// Computes the probability of a specific class by traversing the binary tree.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <param name="classIndex">The index of the class for which to compute the probability.</param>
    /// <returns>The probability value for the specified class.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how likely a specific class is, given the input.
    /// 
    /// It works by:
    /// 1. Starting at the root (top) of the tree
    /// 2. At each node, deciding whether to go left or right based on the class's binary code
    /// 3. Calculating the probability of taking that path at each step
    /// 4. Multiplying all these probabilities together to get the final probability
    /// 
    /// The binary code of the class index determines the path through the tree:
    /// - Each bit in the binary representation tells us whether to go left (0) or right (1)
    /// - For example, class 5 (binary 101) would go right, then left, then right
    /// 
    /// This approach is much more efficient than calculating probabilities for all classes
    /// when you have thousands or millions of possible outputs.
    /// </para>
    /// </remarks>
    private T ComputePathProbability(Vector<T> input, int classIndex)
    {
        T probability = NumOps.One;
        int node = 1;
        for (int depth = 0; depth < _treeDepth; depth++)
        {
            T nodeOutput = MathHelper.Sigmoid(input.DotProduct(_nodeWeights.GetRow(depth)));
            bool goRight = (classIndex & (1 << (_treeDepth - depth - 1))) != 0;
            probability = NumOps.Multiply(probability, goRight ? nodeOutput : NumOps.Subtract(NumOps.One, nodeOutput));
            node = node * 2 + (goRight ? 1 : 0);
            if (node >= _numClasses) break;
        }

        return probability;
    }


    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>True because TensorOperations.HierarchicalSoftmax provides full forward and backward pass support.</value>
    /// <remarks>
    /// <para>
    /// HierarchicalSoftmax supports JIT compilation with gradient computation through the binary tree structure.
    /// The backward pass computes gradients for both the input and the node weights, enabling end-to-end training.
    /// </para>
    /// <para>
    /// The node weights are exposed via <see cref="NodeWeightsTensor"/> for use in computation graphs.
    /// For training, wrap the weights in a ComputationNode to track gradients.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Applies this activation function to a computation graph node.
    /// </summary>
    /// <param name="input">The computation node to apply the activation to.</param>
    /// <returns>A new computation node with HierarchicalSoftmax activation applied.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <remarks>
    /// <para>
    /// This method maps to TensorOperations&lt;T&gt;.HierarchicalSoftmax which handles both
    /// forward and backward passes for JIT compilation.
    /// </para>
    /// <para>
    /// The internal node weights are wrapped in a ComputationNode to enable gradient tracking.
    /// For full training support with weight updates, use <see cref="ApplyToGraph(ComputationNode{T}, ComputationNode{T})"/>
    /// with externally managed weights.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        // Wrap internal weights in a ComputationNode for JIT compilation
        var weightsNode = new ComputationNode<T>(NodeWeightsTensor, requiresGradient: true);
        return TensorOperations<T>.HierarchicalSoftmax(input, weightsNode, _numClasses);
    }

    /// <summary>
    /// Applies Hierarchical Softmax with externally provided weights for full training support.
    /// </summary>
    /// <param name="input">The computation node containing the input features.</param>
    /// <param name="nodeWeights">The computation node containing the tree node weights.</param>
    /// <returns>A new computation node with HierarchicalSoftmax activation applied.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input or nodeWeights is null.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this overload when you want to train the hierarchical softmax weights
    /// as part of your model. By providing the weights as a ComputationNode, gradients will flow
    /// through them during backpropagation, allowing the optimizer to update them.
    /// </para>
    /// <para>
    /// Example usage:
    /// <code>
    /// var weightsNode = new ComputationNode&lt;float&gt;(activation.NodeWeightsTensor, requiresGrad: true);
    /// var output = activation.ApplyToGraph(input, weightsNode);
    /// </code>
    /// </para>
    /// </remarks>
    public ComputationNode<T> ApplyToGraph(ComputationNode<T> input, ComputationNode<T> nodeWeights)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (nodeWeights == null)
            throw new ArgumentNullException(nameof(nodeWeights));

        return TensorOperations<T>.HierarchicalSoftmax(input, nodeWeights, _numClasses);
    }
}

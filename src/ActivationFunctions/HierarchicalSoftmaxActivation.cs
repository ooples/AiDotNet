using AiDotNet.Attributes;
using AiDotNet.Enums;

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
[ActivationCategory(ActivationCategory.Normalization)]
[ActivationCategory(ActivationCategory.Output)]
[ActivationTask(ActivationTask.OutputLayer)]
[ActivationProperty(IsMonotonic = false, ZeroPreserving = false, IsBounded = true, IsVectorActivation = true, Cost = ComputeCost.High)]
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
    /// Gets the node weights as a tensor.
    /// </summary>
    /// <value>A tensor containing the node weights with shape [treeDepth, numClasses].</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This property provides access to the internal weights used by the hierarchical
    /// tree structure, for inspection or export.
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

        if (_treeDepth == 0)
        {
            // No decision nodes: every class keeps the empty product
            return Engine.Fill(_numClasses, NumOps.One);
        }

        var engine = Engine;
        var (right, active) = PathMasks();
        var sigmoids = NodeSigmoids(input);
        var oneMinusSigmoids = engine.TensorAddScalar(engine.TensorNegate(sigmoids), NumOps.One);

        // step[c, d] is the factor class c picks up at depth d: the node output when its path goes
        // right, one minus it when the path goes left, and 1 for depths past the end of its path
        var step = engine.TensorWhere(
            active, engine.TensorWhere(right, sigmoids, oneMinusSigmoids), engine.TensorOnesLike(sigmoids));

        // The path probability multiplies the factors in depth order: the last cumulative product
        return engine.TensorSliceAxis(engine.TensorCumProd(step, 1), 1, _treeDepth - 1).ToVector();
    }

    /// <summary>
    /// The sigmoid output of every tree node for <paramref name="input"/>, repeated for each class as a
    /// [numClasses, treeDepth] tensor.
    /// </summary>
    private Tensor<T> NodeSigmoids(Vector<T> input)
    {
        var engine = Engine;
        var nodeOutputs = Tensor<T>.FromVector(engine.Sigmoid(engine.MatrixVectorMultiply(_nodeWeights, input)));
        return engine.TensorTile(engine.Reshape(nodeOutputs, new[] { 1, _treeDepth }), new[] { _numClasses, 1 });
    }

    /// <summary>
    /// Builds the constant [numClasses, treeDepth] path masks: <c>Right[c, d]</c> is 1 when class c takes
    /// the right branch at depth d, and <c>Active[c, d]</c> is 1 while depth d is still on class c's path
    /// (the walk stops once the node index reaches the number of classes).
    /// </summary>
    private (Tensor<T> Right, Tensor<T> Active) PathMasks()
    {
        var right = new Tensor<T>(new[] { _numClasses, _treeDepth });
        var active = new Tensor<T>(new[] { _numClasses, _treeDepth });
        var rightSpan = right.Data.Span;
        var activeSpan = active.Data.Span;
        for (int classIndex = 0; classIndex < _numClasses; classIndex++)
        {
            int node = 1;
            for (int depth = 0; depth < _treeDepth; depth++)
            {
                bool goRight = (classIndex & (1 << (_treeDepth - depth - 1))) != 0;
                int cell = classIndex * _treeDepth + depth;
                rightSpan[cell] = goRight ? NumOps.One : NumOps.Zero;
                activeSpan[cell] = NumOps.One;

                node = node * 2 + (goRight ? 1 : 0);
                if (node >= _numClasses) break;
            }
        }

        return (right, active);
    }

    /// <summary>
    /// Applies HierarchicalSoftmax to a tensor via engine primitives so the gradient
    /// tape records every step. Computes per-depth sigmoid scores via matmul, then
    /// builds class path probabilities using a constant "path mask" that selects
    /// <c>sig</c> or <c>1 - sig</c> at each depth, combined via log-sum-exp instead
    /// of a product reduction.
    /// </summary>
    public override Tensor<T> Activate(Tensor<T> input)
    {
        int rank = input.Shape.Length;
        if (rank == 0)
            throw new ArgumentException("HierarchicalSoftmax requires at least a 1D tensor.", nameof(input));

        int inputDim = input.Shape[rank - 1];
        if (inputDim != _weightInputDim)
        {
            _weightInputDim = inputDim;
            _nodeWeights = new Matrix<T>(_treeDepth, inputDim);
            InitializeWeights();
        }

        // Flatten leading dims into a single batch dim so we can matmul.
        int batchSize = 1;
        for (int d = 0; d < rank - 1; d++)
            batchSize *= input.Shape[d];

        var input2D = rank == 1
            ? Engine.Reshape(input, new[] { 1, inputDim })
            : Engine.Reshape(input, new[] { batchSize, inputDim });

        // Weights as [treeDepth, inputDim] → transpose to [inputDim, treeDepth].
        var weightsTensor = Tensor<T>.FromRowMatrix(_nodeWeights);
        var weightsT = Engine.TensorPermute(weightsTensor, new[] { 1, 0 });

        // scores = input @ W^T → [batch, treeDepth]
        var scores = Engine.TensorMatMul(input2D, weightsT);
        var sigs = Engine.Sigmoid(scores);

        // Build the constant path mask [1, numClasses, treeDepth]:
        // mask[c, d] = 1 if class c goes right at depth d, else 0.
        var mask = new Tensor<T>(new[] { 1, _numClasses, _treeDepth });
        var maskSpan = mask.Data.Span;
        for (int c = 0; c < _numClasses; c++)
        {
            for (int d = 0; d < _treeDepth; d++)
            {
                bool goRight = (c & (1 << (_treeDepth - d - 1))) != 0;
                maskSpan[c * _treeDepth + d] = goRight ? NumOps.One : NumOps.Zero;
            }
        }

        // Expand sigs to [batch, 1, treeDepth] and broadcast-combine with mask.
        var sigs3D = Engine.Reshape(sigs, new[] { batchSize, 1, _treeDepth });
        var oneMinusSigs3D = Engine.TensorSubtractScalar(
            Engine.TensorNegate(sigs3D), NumOps.Negate(NumOps.One));
        // perPath[b, c, d] = mask[c, d] * sigs[b, d] + (1 - mask[c, d]) * (1 - sigs[b, d])
        var maskedSigs = Engine.TensorMultiply(sigs3D, mask);
        var oneMinusMask = Engine.TensorSubtractScalar(
            Engine.TensorNegate(mask), NumOps.Negate(NumOps.One));
        var maskedOneMinusSigs = Engine.TensorMultiply(oneMinusSigs3D, oneMinusMask);
        var perPath = Engine.TensorAdd(maskedSigs, maskedOneMinusSigs);

        // Reduce path product via log-sum-exp on the depth axis.
        var logPerPath = Engine.TensorLog(perPath);
        var sumLog = Engine.ReduceSum(logPerPath, new[] { 2 }, keepDims: false);
        var output2D = Engine.TensorExp(sumLog);

        // Restore original batch shape with numClasses as the trailing axis.
        int[] outShape = new int[rank];
        for (int d = 0; d < rank - 1; d++)
            outShape[d] = input.Shape[d];
        outShape[rank - 1] = _numClasses;
        if (rank == 1)
            outShape = new[] { _numClasses };
        return Engine.Reshape(output2D, outShape);
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
        if (_treeDepth == 0)
        {
            return new Matrix<T>(_numClasses, input.Length);
        }

        var engine = Engine;
        var (right, active) = PathMasks();
        var sigmoids = NodeSigmoids(input);

        // Each visited node contributes its weights scaled by sigmoid * (1 - sigmoid), added when the
        // path goes right and subtracted when it goes left
        var factors = engine.TensorMultiply(
            sigmoids, engine.TensorAddScalar(engine.TensorNegate(sigmoids), NumOps.One));
        var signed = engine.TensorWhere(right, factors, engine.TensorNegate(factors));
        var coefficients = engine.TensorWhere(active, signed, engine.TensorZerosLike(signed));

        // Row c of the Jacobian is the coefficient-weighted sum of the node weight rows
        return engine.TensorMatMul(coefficients, Tensor<T>.FromRowMatrix(_nodeWeights)).ToMatrix();
    }

    /// <summary>
    /// Initializes the weights for all nodes in the binary tree with small random values.
    /// </summary>
    private void InitializeWeights()
    {
        int cols = _nodeWeights.Columns;
        if (_treeDepth == 0 || cols == 0)
        {
            return;
        }

        // One uniform draw per weight in row-major order, shifted to [-0.5, 0.5)
        var random = RandomHelper.CreateSecureRandom();
        var uniform = Tensor<T>.CreateRandom(random, _treeDepth, cols);
        _nodeWeights = Engine.TensorSubtractScalar(uniform, NumOps.FromDouble(0.5)).ToMatrix();
    }

}

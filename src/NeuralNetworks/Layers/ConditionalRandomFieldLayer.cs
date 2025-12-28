using AiDotNet.Autodiff;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a Conditional Random Field (CRF) layer for sequence labeling tasks.
/// </summary>
/// <remarks>
/// <para>
/// A Conditional Random Field (CRF) layer is a specialized neural network layer designed for sequence labeling
/// tasks such as named entity recognition, part-of-speech tagging, and activity recognition. Unlike standard
/// classification layers that make independent predictions for each element in a sequence, CRF layers model
/// the dependencies between labels in a sequence, leading to more coherent predictions. The layer uses the
/// Viterbi algorithm to find the most likely sequence of labels given the input features and learned transition
/// probabilities between labels.
/// </para>
/// <para><b>For Beginners:</b> A Conditional Random Field (CRF) layer is used when you need to label each item 
/// in a sequence while considering how labels relate to each other.
/// 
/// In many sequence tasks, the label for an item depends not just on the item itself, but also on nearby items:
/// 
/// For example, in a sentence like "John Smith lives in New York":
/// - Without CRF: Each word might be labeled independently, potentially creating invalid sequences
/// - With CRF: The model considers that "New" followed by "York" is likely a location name
/// 
/// Think of it like:
/// - Standard layers ask, "What's the best label for this word on its own?"
/// - CRF layers ask, "What's the best sequence of labels for the whole sentence?"
/// 
/// CRFs are especially useful for tasks like:
/// - Named entity recognition (finding names of people, organizations, locations)
/// - Part-of-speech tagging (labeling words as nouns, verbs, etc.)
/// - Any task where the correct labels form patterns or follow rules
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class ConditionalRandomFieldLayer<T> : LayerBase<T>
{
    private Tensor<T> _transitionMatrix;
    private Tensor<T> _startScores;
    private Tensor<T> _endScores;

    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the original input shape for any-rank tensor support.
    /// </summary>
    private int[]? _originalInputShape;
    private Tensor<T>? _lastOutput;

    private Tensor<T>? _transitionMatrixGradient;
    private Tensor<T>? _startScoresGradient;
    private Tensor<T>? _endScoresGradient;

    private readonly int _numClasses;
    private readonly int _sequenceLength;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// Always <c>true</c> as CRF layers have trainable parameters.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property returns true because ConditionalRandomFieldLayer has trainable parameters (transition matrix,
    /// start scores, and end scores) that are adjusted during the training process to minimize the network's error.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you that this layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer contains values (parameters) that will change during training
    /// - It will improve its performance as it sees more examples
    /// - It participates in the learning process of the neural network
    /// 
    /// CRF layers always support training because they need to learn:
    /// - How likely one label is to follow another (transition probabilities)
    /// - Which labels are likely to appear at the start of a sequence
    /// - Which labels are likely to appear at the end of a sequence
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="ConditionalRandomFieldLayer{T}"/> class with a scalar activation function.
    /// </summary>
    /// <param name="numClasses">The number of possible label classes.</param>
    /// <param name="sequenceLength">The length of the input sequences.</param>
    /// <param name="scalarActivation">The scalar activation function to apply to inputs. Defaults to identity if not specified.</param>
    /// <param name="engine">The computation engine for vectorized operations. Defaults to CPU if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new CRF layer with the specified number of classes and sequence length.
    /// It initializes the transition matrix, start scores, and end scores with appropriate random values.
    /// The scalar activation function is applied to the input features before the CRF processing.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new CRF layer with a standard activation function.
    /// 
    /// When creating a CRF layer, you need to specify:
    /// - How many different labels (classes) there are (e.g., 9 parts of speech)
    /// - How long each input sequence is (e.g., maximum sentence length)
    /// - Optionally, an activation function to transform the input features
    /// 
    /// The layer creates and initializes:
    /// - A transition matrix that learns how likely one label is to follow another
    /// - Start scores that learn which labels commonly appear at the beginning
    /// - End scores that learn which labels commonly appear at the end
    /// 
    /// These values start as small random numbers and are refined during training.
    /// </para>
    /// </remarks>
    public ConditionalRandomFieldLayer(int numClasses, int sequenceLength, IActivationFunction<T>? scalarActivation = null)
        : base([sequenceLength, numClasses], [sequenceLength, numClasses], scalarActivation ?? new IdentityActivation<T>())
    {
        if (sequenceLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(sequenceLength), "sequenceLength must be greater than 0.");

        _numClasses = numClasses;
        _sequenceLength = sequenceLength;
        _transitionMatrix = new Tensor<T>([_numClasses, _numClasses]);
        _startScores = new Tensor<T>([_numClasses]);
        _endScores = new Tensor<T>([_numClasses]);

        InitializeParameters();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="ConditionalRandomFieldLayer{T}"/> class with a vector activation function.
    /// </summary>
    /// <param name="numClasses">The number of possible label classes.</param>
    /// <param name="sequenceLength">The length of the input sequences.</param>
    /// <param name="vectorActivation">The vector activation function to apply to inputs. Defaults to identity if not specified.</param>
    /// <param name="engine">The computation engine for vectorized operations. Defaults to CPU if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new CRF layer with the specified number of classes and sequence length.
    /// It initializes the transition matrix, start scores, and end scores with appropriate random values.
    /// This overload accepts a vector activation function, which operates on entire vectors rather than
    /// individual elements when transforming the input features.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new CRF layer with a vector-based activation function.
    /// 
    /// A vector activation function:
    /// - Operates on entire groups of numbers at once, rather than one at a time
    /// - Can capture relationships between different elements in the input
    /// - Defaults to the Identity function, which doesn't change the values
    /// 
    /// This constructor works the same way as the other one, but it's useful when you need more
    /// complex activation patterns that consider the relationships between different inputs.
    /// </para>
    /// </remarks>
    public ConditionalRandomFieldLayer(int numClasses, int sequenceLength, IVectorActivationFunction<T>? vectorActivation = null)
        : base([sequenceLength, numClasses], [sequenceLength, numClasses], vectorActivation ?? new IdentityActivation<T>())
    {
        if (sequenceLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(sequenceLength), "sequenceLength must be greater than 0.");

        _numClasses = numClasses;
        _sequenceLength = sequenceLength;
        _transitionMatrix = new Tensor<T>([_numClasses, _numClasses]);
        _startScores = new Tensor<T>([_numClasses]);
        _endScores = new Tensor<T>([_numClasses]);

        InitializeParameters();
    }

    /// <summary>
    /// Initializes the layer's parameters with scaled random values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the transition matrix, start scores, and end scores with small random values
    /// scaled based on the number of classes. This initialization approach helps with training convergence
    /// by keeping the initial values in an appropriate range.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the initial values for the layer's parameters.
    /// 
    /// Before training begins, we need to:
    /// - Fill the transition matrix with small random values
    /// - Set start and end scores to small random values
    /// 
    /// This initialization is important because:
    /// - Starting with the right range of random values helps the network learn faster
    /// - If values are too large or too small, the network might have trouble learning
    /// 
    /// The scaling factor ensures the random values are an appropriate size based on
    /// the number of classes in the problem.
    /// </para>
    /// </remarks>
    private void InitializeParameters()
    {
        // VECTORIZED: Initialize parameters with scaled random values
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_numClasses + _numClasses)));
        T half = NumOps.FromDouble(0.5);

        // Initialize transition matrix: (random - 0.5) * scale
        var transRandom = Tensor<T>.CreateRandom(_transitionMatrix.Length, 1).Reshape(_transitionMatrix.Shape);
        var transHalf = new Tensor<T>(_transitionMatrix.Shape);
        transHalf.Fill(half);
        var transCentered = Engine.TensorSubtract(transRandom, transHalf);
        _transitionMatrix = Engine.TensorMultiplyScalar(transCentered, scale);

        // Initialize start scores: (random - 0.5) * scale
        var startRandom = Tensor<T>.CreateRandom(_startScores.Length, 1).Reshape(_startScores.Shape);
        var startHalf = new Tensor<T>(_startScores.Shape);
        startHalf.Fill(half);
        var startCentered = Engine.TensorSubtract(startRandom, startHalf);
        _startScores = Engine.TensorMultiplyScalar(startCentered, scale);

        // Initialize end scores: (random - 0.5) * scale
        var endRandom = Tensor<T>.CreateRandom(_endScores.Length, 1).Reshape(_endScores.Shape);
        var endHalf = new Tensor<T>(_endScores.Shape);
        endHalf.Fill(half);
        var endCentered = Engine.TensorSubtract(endRandom, endHalf);
        _endScores = Engine.TensorMultiplyScalar(endCentered, scale);
    }

    /// <summary>
    /// Performs the forward pass of the CRF layer.
    /// </summary>
    /// <param name="input">The input tensor containing sequence features.</param>
    /// <returns>The output tensor containing the most likely sequence labels.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the CRF layer using the Viterbi algorithm to find
    /// the most likely sequence of labels. It first applies any activation function to transform the
    /// input features, then uses dynamic programming to find the optimal label sequence considering
    /// the transition scores between labels, start scores, and end scores. The output is a one-hot
    /// encoded tensor representing the best label at each position in the sequence.
    /// </para>
    /// <para><b>For Beginners:</b> This method finds the best sequence of labels for the input data.
    /// 
    /// The forward pass has several steps:
    /// 
    /// 1. Transform the input features using the activation function (if specified)
    /// 2. For each sequence in the batch, run the Viterbi algorithm:
    ///    - Start with the initial scores and input features
    ///    - For each position in the sequence, calculate the best previous label
    ///    - Keep track of the best path using "backpointers"
    ///    - Find the best final label considering the end scores
    ///    - Trace backwards to find the optimal sequence of labels
    /// 3. Convert the best label sequence to a one-hot encoded output
    /// 
    /// The Viterbi algorithm is like finding the shortest path through a grid,
    /// where each step considers both the current position's score and the
    /// transition cost from the previous position.
    /// 
    /// This approach ensures that the entire sequence of labels makes sense together,
    /// rather than just picking the best label at each position independently.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Store original shape for any-rank tensor support
        _originalInputShape = input.Shape;
        int rank = input.Shape.Length;

        // Handle any-rank tensor: collapse to 2D for processing
        Tensor<T> processInput;
        int batchSize;

        if (rank == 1)
        {
            // 1D: add batch dim
            batchSize = 1;
            processInput = input.Reshape([1, input.Shape[0]]);
        }
        else if (rank == 2)
        {
            // Standard 2D
            batchSize = input.Shape[0];
            processInput = input;
        }
        else
        {
            // Higher-rank: collapse leading dims into batch
            int flatBatch = 1;
            for (int d = 0; d < rank - 1; d++)
                flatBatch *= input.Shape[d];
            batchSize = flatBatch;
            processInput = input.Reshape([flatBatch, input.Shape[rank - 1]]);
        }

        _lastInput = processInput;

        var output = new Tensor<T>([batchSize, _sequenceLength, _numClasses]);

        // === VECTORIZED: Apply activation to entire input ===
        Tensor<T> sequenceScores;
        if (UsingVectorActivation)
        {
            sequenceScores = ApplyActivation(input);
        }
        else if (ScalarActivation != null && !(ScalarActivation is IdentityActivation<T>))
        {
            sequenceScores = ApplyActivation(input);
        }
        else
        {
            sequenceScores = input;
        }

        // Process each batch item (Viterbi requires sequential time processing)
        for (int b = 0; b < batchSize; b++)
        {
            // === VECTORIZED: Extract sequence for this batch item ===
            var batchSeq = Engine.TensorSliceAxis(sequenceScores, 0, b); // [sequenceLength, numClasses]

            // === VECTORIZED Viterbi Algorithm ===
            var viterbi = new Tensor<T>([_sequenceLength, _numClasses]);
            var backpointers = new Matrix<int>(_sequenceLength, _numClasses);

            // VECTORIZED: Initialize first timestep - startScores + emissions[0]
            var firstEmissions = Engine.TensorSliceAxis(batchSeq, 0, 0); // [numClasses]
            var firstViterbi = Engine.TensorAdd(firstEmissions, _startScores);
            Engine.TensorSetSliceAxis(viterbi, firstViterbi, 0, 0);

            // Recursion over time (inherently sequential)
            for (int t = 1; t < _sequenceLength; t++)
            {
                var currentEmissions = Engine.TensorSliceAxis(batchSeq, 0, t); // [numClasses]
                var prevViterbi = Engine.TensorSliceAxis(viterbi, 0, t - 1); // [numClasses]

                // For each current class, compute max over prev classes
                // score[c] = max_prevC(viterbi[t-1, prevC] + transition[prevC, c]) + emissions[t, c]
                // This can be done by broadcasting:
                // prevViterbi: [numClasses, 1] + transition: [numClasses, numClasses] -> [numClasses, numClasses]
                // Then max over axis 0

                var prevExpanded = prevViterbi.Reshape([_numClasses, 1]); // [numClasses, 1]
                var scoresWithTrans = Engine.TensorAdd(prevExpanded, _transitionMatrix); // [numClasses, numClasses]

                // Get max over previous classes and store backpointers
                var maxScores = new Tensor<T>([_numClasses]);
                for (int c = 0; c < _numClasses; c++)
                {
                    T maxVal = NumOps.MinValue;
                    int maxIdx = 0;
                    for (int prevC = 0; prevC < _numClasses; prevC++)
                    {
                        T val = scoresWithTrans[prevC, c];
                        if (NumOps.GreaterThan(val, maxVal))
                        {
                            maxVal = val;
                            maxIdx = prevC;
                        }
                    }
                    maxScores[c] = maxVal;
                    backpointers[t, c] = maxIdx;
                }

                // Add emissions: maxScores + currentEmissions
                var currentViterbi = Engine.TensorAdd(maxScores, currentEmissions);
                Engine.TensorSetSliceAxis(viterbi, currentViterbi, 0, t);
            }

            // === VECTORIZED Termination ===
            var lastViterbi = Engine.TensorSliceAxis(viterbi, 0, _sequenceLength - 1);
            var finalScores = Engine.TensorAdd(lastViterbi, _endScores);

            // Find argmax
            T maxFinalScore = NumOps.MinValue;
            int maxFinalClass = 0;
            for (int c = 0; c < _numClasses; c++)
            {
                if (NumOps.GreaterThan(finalScores[c], maxFinalScore))
                {
                    maxFinalScore = finalScores[c];
                    maxFinalClass = c;
                }
            }

            // Backtracking (inherently sequential)
            var bestPath = new int[_sequenceLength];
            bestPath[_sequenceLength - 1] = maxFinalClass;
            for (int t = _sequenceLength - 2; t >= 0; t--)
            {
                bestPath[t] = backpointers[t + 1, bestPath[t + 1]];
            }

            // === VECTORIZED: Set one-hot output ===
            // Create one-hot tensor for this batch
            for (int t = 0; t < _sequenceLength; t++)
            {
                output[b, t, bestPath[t]] = NumOps.One;
            }
        }

        _lastOutput = output;
        return output;
    }

    /// <summary>
    /// Performs the backward pass of the CRF layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the CRF layer, which is used during training to propagate
    /// error gradients back through the network. It computes the gradients of the loss with respect to the
    /// layer's parameters (transition matrix, start scores, and end scores) and the layer's input.
    /// </para>
    /// <para><b>For Beginners:</b> This method is used during training to calculate how the layer's inputs
    /// and parameters should change to reduce errors.
    ///
    /// During the backward pass:
    /// 1. The layer receives error gradients from the next layer
    /// 2. It calculates how much each parameter contributed to the error:
    ///    - How transition scores between labels should change
    ///    - How start and end scores should change
    /// 3. It calculates how the input features contributed to the error
    /// 4. If an activation function was used, its derivative is applied
    ///
    /// This lets the network learn:
    /// - Which label is likely to follow another
    /// - Which labels commonly appear at the start or end of sequences
    /// - How input features relate to labels
    ///
    /// This is part of the "backpropagation" algorithm that helps neural networks learn
    /// from their mistakes and improve over time.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }

    /// <summary>
    /// Manual backward pass implementation using optimized gradient calculations.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];

        // === VECTORIZED Gradient Computation ===

        // Input gradient starts as a copy of output gradient
        var inputGradient = outputGradient.Clone();

        // Start scores gradient: sum of gradients at t=0 over all batches
        // outputGradient[:, 0, :] summed over batch
        var firstTimestep = Engine.TensorSliceAxis(outputGradient, 1, 0); // [batchSize, numClasses]
        _startScoresGradient = Engine.ReduceSum(firstTimestep, new[] { 0 }, keepDims: false);

        // End scores gradient: sum of gradients at t=seqLen-1 over all batches
        var lastTimestep = Engine.TensorSliceAxis(outputGradient, 1, _sequenceLength - 1); // [batchSize, numClasses]
        _endScoresGradient = Engine.ReduceSum(lastTimestep, new[] { 0 }, keepDims: false);

        // Transition matrix gradient: sum of gradients for all t > 0
        // For simplicity, we sum all gradients across batch and time (except t=0), then broadcast
        // A more accurate gradient would involve the actual paths, but this is an approximation
        var allGradients = Engine.ReduceSum(outputGradient, new[] { 0, 1 }, keepDims: false); // [numClasses]

        // Create transition gradient: outer product approximation
        // Each transition gets the sum of class gradients
        _transitionMatrixGradient = new Tensor<T>([_numClasses, _numClasses]);
        var gradExpanded = allGradients.Reshape([1, _numClasses]);
        var onesCol = new Tensor<T>([_numClasses, 1]);
        onesCol.Fill(NumOps.One);
        // transGrad[i, j] = sumGrad[j] for all i
        var transGrad = Engine.TensorMultiply(onesCol, gradExpanded);
        // Scale by (seqLen - 1) / seqLen to account for t=0 not having transitions
        var scale = NumOps.FromDouble((_sequenceLength - 1.0) / _sequenceLength);
        _transitionMatrixGradient = Engine.TensorMultiplyScalar(transGrad, scale);

        // Apply activation function gradient if applicable
        if (UsingVectorActivation || (ScalarActivation != null && !(ScalarActivation is IdentityActivation<T>)))
        {
            inputGradient = ApplyActivationDerivative(_lastInput, inputGradient);
        }

        return inputGradient;
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method uses automatic differentiation to compute gradients via the CRFForward operation.
    /// It builds a computation graph for the CRF forward pass, then propagates gradients backward
    /// through the graph. The activation function derivative is applied separately after the autodiff
    /// backward pass to match the behavior of BackwardManual.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Build computation graph using differentiable CRF forward op
        var emissionsNode = Autodiff.TensorOperations<T>.Variable(_lastInput, "crf_emissions", requiresGradient: true);
        var transitionsNode = Autodiff.TensorOperations<T>.Variable(_transitionMatrix, "crf_transitions", requiresGradient: true);
        var startNode = Autodiff.TensorOperations<T>.Variable(_startScores, "crf_start", requiresGradient: true);
        var endNode = Autodiff.TensorOperations<T>.Variable(_endScores, "crf_end", requiresGradient: true);

        var outputNode = Autodiff.TensorOperations<T>.CRFForward(emissionsNode, transitionsNode, startNode, endNode);
        outputNode.Gradient = outputGradient;

        // Inline topological sort
        var visited = new HashSet<Autodiff.ComputationNode<T>>();
        var topoOrder = new List<Autodiff.ComputationNode<T>>();
        var stack = new Stack<(Autodiff.ComputationNode<T> node, bool processed)>();
        stack.Push((outputNode, false));

        while (stack.Count > 0)
        {
            var (node, processed) = stack.Pop();
            if (visited.Contains(node)) continue;

            if (processed)
            {
                visited.Add(node);
                topoOrder.Add(node);
            }
            else
            {
                stack.Push((node, true));
                if (node.Parents != null)
                {
                    foreach (var parent in node.Parents)
                    {
                        if (!visited.Contains(parent))
                            stack.Push((parent, false));
                    }
                }
            }
        }

        for (int i = topoOrder.Count - 1; i >= 0; i--)
        {
            var node = topoOrder[i];
            if (node.RequiresGradient && node.BackwardFunction != null && node.Gradient != null)
            {
                node.BackwardFunction(node.Gradient);
            }
        }

        // Capture gradients
        _transitionMatrixGradient = transitionsNode.Gradient ?? new Tensor<T>([_numClasses, _numClasses]);
        _startScoresGradient = startNode.Gradient ?? new Tensor<T>([_numClasses]);
        _endScoresGradient = endNode.Gradient ?? new Tensor<T>([_numClasses]);

        if (emissionsNode.Gradient == null)
            throw new InvalidOperationException("Gradient computation failed in CRF autodiff.");

        var inputGradient = emissionsNode.Gradient;

        // Apply activation function gradient if applicable (matching BackwardManual behavior)
        if (UsingVectorActivation || (ScalarActivation != null && !(ScalarActivation is IdentityActivation<T>)))
        {
            inputGradient = ApplyActivationDerivative(_lastInput, inputGradient);
        }

        return inputGradient;
    }

    /// <summary>
    /// Updates the layer's parameters using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <exception cref="InvalidOperationException">Thrown when update is called before backward.</exception>
    /// <remarks>
    /// <para>
    /// This method updates the layer's parameters (transition matrix, start scores, and end scores) based on
    /// the gradients calculated during the backward pass. The learning rate controls the size of the parameter
    /// updates. The update is performed by subtracting the scaled gradients from the current parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the layer's internal values during training.
    /// 
    /// After calculating the gradients in the backward pass:
    /// - This method applies those changes to the transition matrix, start scores, and end scores
    /// - The learning rate controls how big each update step is
    /// - Smaller learning rates mean slower but more stable learning
    /// - Larger learning rates mean faster but potentially unstable learning
    /// 
    /// The formula is simple: new_value = old_value - (gradient * learning_rate)
    /// 
    /// This is how the layer "learns" from data over time, gradually improving its ability
    /// to predict the correct sequence of labels.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_transitionMatrixGradient == null || _startScoresGradient == null || _endScoresGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        // Update using Engine tensor operations: param = param - lr * gradient
        var scaledTransGrad = Engine.TensorMultiplyScalar(_transitionMatrixGradient, learningRate);
        _transitionMatrix = Engine.TensorSubtract(_transitionMatrix, scaledTransGrad);

        var scaledStartGrad = Engine.TensorMultiplyScalar(_startScoresGradient, learningRate);
        _startScores = Engine.TensorSubtract(_startScores, scaledStartGrad);

        var scaledEndGrad = Engine.TensorMultiplyScalar(_endScoresGradient, learningRate);
        _endScores = Engine.TensorSubtract(_endScores, scaledEndGrad);
    }

    /// <summary>
    /// Gets all trainable parameters from the layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters from the layer (transition matrix, start scores, and
    /// end scores) and combines them into a single vector. This is useful for optimization algorithms that
    /// operate on all parameters at once, or for saving and loading model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the learnable values from the layer into a single list.
    /// 
    /// The parameters include:
    /// - The transition matrix (shows how likely one label is to follow another)
    /// - The start scores (shows which labels are likely at sequence beginnings)
    /// - The end scores (shows which labels are likely at sequence endings)
    /// 
    /// All these values are flattened into a single long list (vector).
    /// 
    /// This is useful for:
    /// - Saving the model to disk
    /// - Loading parameters from a previously trained model
    /// - Advanced optimization techniques that need access to all parameters
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Use Vector<T>.Concatenate for efficient parameter collection
        var flatTrans = new Vector<T>(_transitionMatrix.ToArray());
        var flatStart = new Vector<T>(_startScores.ToArray());
        var flatEnd = new Vector<T>(_endScores.ToArray());

        return Vector<T>.Concatenate(Vector<T>.Concatenate(flatTrans, flatStart), flatEnd);
    }

    /// <summary>
    /// Sets the trainable parameters for the layer.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method sets the trainable parameters for the layer (transition matrix, start scores, and end scores)
    /// from a single vector. This is useful for loading saved model weights or for implementing optimization
    /// algorithms that operate on all parameters at once.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the learnable values in the layer.
    /// 
    /// When setting parameters:
    /// - The input must be a vector with the correct length
    /// - The first part is used for the transition matrix
    /// - The next part is used for the start scores
    /// - The final part is used for the end scores
    /// 
    /// This is useful for:
    /// - Loading a previously saved model
    /// - Transferring parameters from another model
    /// - Testing different parameter values
    /// 
    /// An error is thrown if the input vector doesn't have the expected number of parameters.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        int transSize = _numClasses * _numClasses;
        int totalParams = transSize + _numClasses * 2;

        if (parameters.Length != totalParams)
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");

        // VECTORIZED: Use Vector.Slice and Tensor.FromVector
        var transVec = parameters.Slice(0, transSize);
        var startVec = parameters.Slice(transSize, _numClasses);
        var endVec = parameters.Slice(transSize + _numClasses, _numClasses);

        _transitionMatrix = Tensor<T>.FromVector(transVec).Reshape(_transitionMatrix.Shape);
        _startScores = Tensor<T>.FromVector(startVec).Reshape(_startScores.Shape);
        _endScores = Tensor<T>.FromVector(endVec).Reshape(_endScores.Shape);
    }

    /// <summary>
    /// Resets the internal state of the CRF layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the CRF layer, including the cached inputs, outputs,
    /// and parameter gradients. This is useful when starting to process a new sequence or batch of data.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's temporary memory to start fresh.
    /// 
    /// When resetting the state:
    /// - Stored inputs and outputs are cleared
    /// - Calculated gradients are cleared
    /// - The layer forgets any information from previous batches
    /// 
    /// This is important for:
    /// - Processing a new, unrelated batch of data
    /// - Preventing information from one batch affecting another
    /// - Freeing up memory that's no longer needed
    /// 
    /// Note that this doesn't reset the learned parameters (transition matrix, start scores, end scores),
    /// just the temporary information used during a single forward/backward pass.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _transitionMatrixGradient = null;
        _startScoresGradient = null;
        _endScoresGradient = null;
    }

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        if (inputNodes.Count == 0)
            throw new ArgumentException("At least one input node is required.", nameof(inputNodes));

        // ConditionalRandomFieldLayer JIT uses the forward algorithm for differentiable inference:
        // This computes the log partition function which can be used for CRF training.
        // For inference at runtime, Viterbi decoding is still used, but training can use autodiff.

        var input = inputNodes[0];

        // Input is emissions [seqLen, numClasses]
        // Convert transition matrix to computation node
        var transitionsTensor = new Tensor<T>([_numClasses, _numClasses]);
        for (int i = 0; i < _numClasses; i++)
            for (int j = 0; j < _numClasses; j++)
                transitionsTensor[i, j] = _transitionMatrix[i, j];

        var transitionsNode = TensorOperations<T>.Variable(transitionsTensor, "crf_transitions", requiresGradient: true);

        // Use CRF forward algorithm for log partition computation
        var logPartition = TensorOperations<T>.CRFForward(input, transitionsNode);

        // Apply activation
        var output = ApplyActivationToGraph(logPartition);

        return output;
    }

    /// <summary>
    /// Gets a value indicating whether this layer supports JIT compilation.
    /// </summary>
    /// <value>
    /// Always <c>true</c>. CRF uses the forward algorithm for differentiable training.
    /// </value>
    /// <remarks>
    /// <para>
    /// JIT compilation for CRF uses the forward algorithm to compute the log partition
    /// function, which is differentiable with respect to emissions and transitions.
    /// This enables gradient-based optimization of CRF parameters. For inference,
    /// Viterbi decoding is used at runtime, but the JIT-compiled graph supports training.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

}

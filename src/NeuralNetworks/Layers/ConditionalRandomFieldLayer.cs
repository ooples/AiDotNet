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
    private Matrix<T> _transitionMatrix = default!;
    private Vector<T> _startScores = default!;
    private Vector<T> _endScores = default!;

    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;

    private Matrix<T>? _transitionMatrixGradient;
    private Vector<T>? _startScoresGradient;
    private Vector<T>? _endScoresGradient;

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
        _numClasses = numClasses;
        _sequenceLength = sequenceLength;
        _transitionMatrix = new Matrix<T>(_numClasses, _numClasses);
        _startScores = new Vector<T>(_numClasses);
        _endScores = new Vector<T>(_numClasses);

        InitializeParameters();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="ConditionalRandomFieldLayer{T}"/> class with a vector activation function.
    /// </summary>
    /// <param name="numClasses">The number of possible label classes.</param>
    /// <param name="sequenceLength">The length of the input sequences.</param>
    /// <param name="vectorActivation">The vector activation function to apply to inputs. Defaults to identity if not specified.</param>
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
        _numClasses = numClasses;
        _sequenceLength = sequenceLength;
        _transitionMatrix = new Matrix<T>(_numClasses, _numClasses);
        _startScores = new Vector<T>(_numClasses);
        _endScores = new Vector<T>(_numClasses);

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
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_numClasses + _numClasses)));
        InitializeMatrix(_transitionMatrix, scale);
        InitializeVector(_startScores, scale);
        InitializeVector(_endScores, scale);
    }

    /// <summary>
    /// Initializes a matrix with scaled random values.
    /// </summary>
    /// <param name="matrix">The matrix to initialize.</param>
    /// <param name="scale">The scale factor for the random values.</param>
    /// <remarks>
    /// <para>
    /// This method fills the given matrix with random values drawn from a uniform distribution
    /// centered at zero (-0.5 to 0.5) and scaled by the provided scale factor. This type of
    /// initialization helps with the training process by preventing issues like vanishing or
    /// exploding gradients.
    /// </para>
    /// <para><b>For Beginners:</b> This method fills a matrix with random values of an appropriate size.
    /// 
    /// When filling the matrix:
    /// - We generate random numbers between -0.5 and 0.5
    /// - We multiply each by a scaling factor to control their magnitude
    /// - The result is a matrix filled with small random values
    /// 
    /// For the transition matrix, these values represent the initial "guesses" about
    /// how likely one label is to follow another. These values will be adjusted during
    /// training to reflect the actual patterns in the data.
    /// </para>
    /// </remarks>
    private void InitializeMatrix(Matrix<T> matrix, T scale)
    {
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                matrix[i, j] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
            }
        }
    }

    /// <summary>
    /// Initializes a vector with scaled random values.
    /// </summary>
    /// <param name="vector">The vector to initialize.</param>
    /// <param name="scale">The scale factor for the random values.</param>
    /// <remarks>
    /// <para>
    /// This method fills the given vector with random values drawn from a uniform distribution
    /// centered at zero (-0.5 to 0.5) and scaled by the provided scale factor. This helps with
    /// the training process by providing a good starting point for the parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This method fills a vector with random values of an appropriate size.
    /// 
    /// When filling the vector:
    /// - We generate random numbers between -0.5 and 0.5
    /// - We multiply each by a scaling factor to control their magnitude
    /// - The result is a vector filled with small random values
    /// 
    /// These random values serve as starting points for:
    /// - Start scores: initial "guesses" about which labels appear at sequence beginnings
    /// - End scores: initial "guesses" about which labels appear at sequence endings
    /// 
    /// These values will be adjusted during training to reflect patterns in the actual data.
    /// </para>
    /// </remarks>
    private void InitializeVector(Vector<T> vector, T scale)
    {
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
        }
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
        _lastInput = input;
        int batchSize = input.Shape[0];

        var output = new Tensor<T>([batchSize, _sequenceLength, _numClasses]);

        for (int b = 0; b < batchSize; b++)
        {
            var sequenceScores = new Matrix<T>(_sequenceLength, _numClasses);

            // Apply optional feature transformation
            for (int t = 0; t < _sequenceLength; t++)
            {
                var featureVector = new Vector<T>(_numClasses);
                for (int c = 0; c < _numClasses; c++)
                {
                    featureVector[c] = input[b, t, c];
                }

                if (UsingVectorActivation)
                {
                    featureVector = VectorActivation!.Activate(featureVector);
                }
                else if (ScalarActivation != null)
                {
                    for (int c = 0; c < _numClasses; c++)
                    {
                        featureVector[c] = ScalarActivation.Activate(featureVector[c]);
                    }
                }

                for (int c = 0; c < _numClasses; c++)
                {
                    sequenceScores[t, c] = featureVector[c];
                }
            }

            // Viterbi algorithm
            var viterbi = new Matrix<T>(_sequenceLength, _numClasses);
            var backpointers = new Matrix<int>(_sequenceLength, _numClasses);

            // Initialize first timestep
            for (int c = 0; c < _numClasses; c++)
            {
                viterbi[0, c] = NumOps.Add(_startScores[c], sequenceScores[0, c]);
            }

            // Recursion
            for (int t = 1; t < _sequenceLength; t++)
            {
                for (int c = 0; c < _numClasses; c++)
                {
                    T maxScore = NumOps.MinValue;
                    int maxPrevClass = -1;

                    for (int prevC = 0; prevC < _numClasses; prevC++)
                    {
                        T score = NumOps.Add(
                            NumOps.Add(
                                viterbi[t - 1, prevC],
                                _transitionMatrix[prevC, c]
                            ),
                            sequenceScores[t, c]
                        );

                        if (NumOps.GreaterThan(score, maxScore))
                        {
                            maxScore = score;
                            maxPrevClass = prevC;
                        }
                    }

                    viterbi[t, c] = maxScore;
                    backpointers[t, c] = maxPrevClass;
                }
            }

            // Termination
            T maxFinalScore = NumOps.MinValue;
            int maxFinalClass = -1;
            for (int c = 0; c < _numClasses; c++)
            {
                T finalScore = NumOps.Add(viterbi[_sequenceLength - 1, c], _endScores[c]);
                if (NumOps.GreaterThan(finalScore, maxFinalScore))
                {
                    maxFinalScore = finalScore;
                    maxFinalClass = c;
                }
            }

            // Backtracking
            var bestPath = new int[_sequenceLength];
            bestPath[_sequenceLength - 1] = maxFinalClass;
            for (int t = _sequenceLength - 2; t >= 0; t--)
            {
                bestPath[t] = backpointers[t + 1, bestPath[t + 1]];
            }

            // Set output
            for (int t = 0; t < _sequenceLength; t++)
            {
                for (int c = 0; c < _numClasses; c++)
                {
                    output[b, t, c] = c == bestPath[t] ? NumOps.One : NumOps.Zero;
                }
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
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];

        var inputGradient = new Tensor<T>(_lastInput.Shape);
        _transitionMatrixGradient = new Matrix<T>(_numClasses, _numClasses);
        _startScoresGradient = new Vector<T>(_numClasses);
        _endScoresGradient = new Vector<T>(_numClasses);

        for (int b = 0; b < batchSize; b++)
        {
            // Compute gradients for transition matrix, start scores, and end scores
            for (int t = 0; t < _sequenceLength; t++)
            {
                for (int c = 0; c < _numClasses; c++)
                {
                    T grad = outputGradient[b, t, c];

                    if (t == 0)
                    {
                        _startScoresGradient[c] = NumOps.Add(_startScoresGradient[c], grad);
                    }
                    else if (t == _sequenceLength - 1)
                    {
                        _endScoresGradient[c] = NumOps.Add(_endScoresGradient[c], grad);
                    }

                    if (t > 0)
                    {
                        for (int prevC = 0; c < _numClasses; prevC++)
                        {
                            _transitionMatrixGradient[prevC, c] = NumOps.Add(_transitionMatrixGradient[prevC, c], grad);
                        }
                    }

                    // Compute input gradient
                    inputGradient[b, t, c] = grad;
                }
            }
        }

        // Apply activation function gradient if applicable
        if (UsingVectorActivation)
        {
            for (int b = 0; b < batchSize; b++)
            {
                for (int t = 0; t < _sequenceLength; t++)
                {
                    var input = new Vector<T>(_numClasses);
                    var grad = new Vector<T>(_numClasses);
                    for (int c = 0; c < _numClasses; c++)
                    {
                        input[c] = _lastInput[b, t, c];
                        grad[c] = inputGradient[b, t, c];
                    }

                    var derivativeMatrix = VectorActivation!.Derivative(input);
                    var result = derivativeMatrix.Multiply(grad);

                    for (int c = 0; c < _numClasses; c++)
                    {
                        inputGradient[b, t, c] = result[c];
                    }
                }
            }
        }
        else if (ScalarActivation != null)
        {
            for (int b = 0; b < batchSize; b++)
            {
                for (int t = 0; t < _sequenceLength; t++)
                {
                    for (int c = 0; c < _numClasses; c++)
                    {
                        T derivative = ScalarActivation.Derivative(_lastInput[b, t, c]);
                        inputGradient[b, t, c] = NumOps.Multiply(derivative, inputGradient[b, t, c]);
                    }
                }
            }
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

        for (int i = 0; i < _numClasses; i++)
        {
            for (int j = 0; j < _numClasses; j++)
            {
                _transitionMatrix[i, j] = NumOps.Subtract(_transitionMatrix[i, j], 
                    NumOps.Multiply(learningRate, _transitionMatrixGradient[i, j]));
            }

            _startScores[i] = NumOps.Subtract(_startScores[i], 
                NumOps.Multiply(learningRate, _startScoresGradient[i]));
            _endScores[i] = NumOps.Subtract(_endScores[i], 
                NumOps.Multiply(learningRate, _endScoresGradient[i]));
        }
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
        // Flatten all parameters into a single vector
        int totalParams = _numClasses * _numClasses + _numClasses * 2;
        var parameters = new Vector<T>(totalParams);
        
        int index = 0;
        
        // Copy transition matrix parameters
        for (int i = 0; i < _numClasses; i++)
        {
            for (int j = 0; j < _numClasses; j++)
            {
                parameters[index++] = _transitionMatrix[i, j];
            }
        }
        
        // Copy start scores
        for (int i = 0; i < _numClasses; i++)
        {
            parameters[index++] = _startScores[i];
        }
        
        // Copy end scores
        for (int i = 0; i < _numClasses; i++)
        {
            parameters[index++] = _endScores[i];
        }
        
        return parameters;
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
        int totalParams = _numClasses * _numClasses + _numClasses * 2;
        
        if (parameters.Length != totalParams)
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        
        int index = 0;
        
        // Set transition matrix parameters
        for (int i = 0; i < _numClasses; i++)
        {
            for (int j = 0; j < _numClasses; j++)
            {
                _transitionMatrix[i, j] = parameters[index++];
            }
        }
        
        // Set start scores
        for (int i = 0; i < _numClasses; i++)
        {
            _startScores[i] = parameters[index++];
        }
        
        // Set end scores
        for (int i = 0; i < _numClasses; i++)
        {
            _endScores[i] = parameters[index++];
        }
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
}
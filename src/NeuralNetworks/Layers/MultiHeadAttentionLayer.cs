namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements a multi-head attention layer for neural networks, a key component in transformer architectures.
/// </summary>
/// <typeparam name="T">The numeric type used for computations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Multi-head attention is like having multiple "experts" look at the same information
/// from different perspectives. Each "head" focuses on different parts of the input, allowing the model
/// to capture various relationships in the data simultaneously. This is similar to how you might ask
/// several friends for advice on a decision - each person might notice different important factors.
/// </para>
/// </remarks>
public class MultiHeadAttentionLayer<T> : LayerBase<T>
{
    /// <summary>
    /// Weights used to transform input into query representations.
    /// </summary>
    private Matrix<T> _queryWeights = default!;
    
    /// <summary>
    /// Weights used to transform input into key representations.
    /// </summary>
    private Matrix<T> _keyWeights = default!;
    
    /// <summary>
    /// Weights used to transform input into value representations.
    /// </summary>
    private Matrix<T> _valueWeights = default!;
    
    /// <summary>
    /// Weights used in the final output projection.
    /// </summary>
    private Matrix<T> _outputWeights = default!;
    
    /// <summary>
    /// Bias terms added to the final output.
    /// </summary>
    private Vector<T> _outputBias = default!;

    /// <summary>
    /// Cached input from the forward pass for use in the backward pass.
    /// </summary>
    private Tensor<T>? _lastInput;
    
    /// <summary>
    /// Cached output from the forward pass for use in the backward pass.
    /// </summary>
    private Tensor<T>? _lastOutput;
    
    /// <summary>
    /// Cached attention scores from the forward pass for use in the backward pass.
    /// </summary>
    private Tensor<T>? _lastAttentionScores;

    /// <summary>
    /// Gradients for query weights calculated during backward pass.
    /// </summary>
    private Matrix<T>? _queryWeightsGradient;
    
    /// <summary>
    /// Gradients for key weights calculated during backward pass.
    /// </summary>
    private Matrix<T>? _keyWeightsGradient;
    
    /// <summary>
    /// Gradients for value weights calculated during backward pass.
    /// </summary>
    private Matrix<T>? _valueWeightsGradient;
    
    /// <summary>
    /// Gradients for output weights calculated during backward pass.
    /// </summary>
    private Matrix<T>? _outputWeightsGradient;
    
    /// <summary>
    /// Gradients for output bias calculated during backward pass.
    /// </summary>
    private Vector<T>? _outputBiasGradient;

    /// <summary>
    /// The number of attention heads in this layer.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Think of this as the number of "experts" or different perspectives
    /// that will analyze the same input data.
    /// </remarks>
    protected readonly int _headCount;
    
    /// <summary>
    /// The size of each attention head.
    /// </summary>
    private readonly int _headDimension;

    /// <summary>
    /// The dropout rate applied for regularization.
    /// </summary>
    private readonly double _dropoutRate;

    /// <summary>
    /// Indicates whether this layer supports training.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Creates a new multi-head attention layer with the specified dimensions and head count.
    /// </summary>
    /// <param name="sequenceLength">The length of the input sequence.</param>
    /// <param name="embeddingDimension">The dimension of each element in the sequence.</param>
    /// <param name="headCount">The number of attention heads to use.</param>
    /// <param name="activationFunction">The activation function to apply (defaults to identity function if null).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the attention mechanism with:
    /// - sequenceLength: How many items are in your sequence (like words in a sentence)
    /// - embeddingDimension: How much information is stored about each item
    /// - headCount: How many different "perspectives" or "experts" will analyze the data
    /// </para>
    /// </remarks>
    public MultiHeadAttentionLayer(int sequenceLength, int embeddingDimension, int headCount, IActivationFunction<T>? activationFunction = null)
        : base([sequenceLength, embeddingDimension], [sequenceLength, embeddingDimension], activationFunction ?? new IdentityActivation<T>())
    {
        _headCount = headCount;
        _headDimension = embeddingDimension / headCount;
        _dropoutRate = 0.0; // No dropout for this constructor

        _queryWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _keyWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _valueWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _outputWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _outputBias = new Vector<T>(embeddingDimension);

        InitializeParameters();
    }

    /// <summary>
    /// Creates a new multi-head attention layer with the specified dimensions and head count.
    /// </summary>
    /// <param name="sequenceLength">The length of the input sequence.</param>
    /// <param name="embeddingDimension">The dimension of each element in the sequence.</param>
    /// <param name="headCount">The number of attention heads to use.</param>
    /// <param name="vectorActivationFunction">The vector activation function to apply (defaults to identity function if null).</param>
    public MultiHeadAttentionLayer(int sequenceLength, int embeddingDimension, int headCount, IVectorActivationFunction<T>? vectorActivationFunction = null)
        : base([sequenceLength, embeddingDimension], [sequenceLength, embeddingDimension], vectorActivationFunction ?? new IdentityActivation<T>())
    {
        _headCount = headCount;
        _headDimension = embeddingDimension / headCount;
        _dropoutRate = 0.0; // No dropout for this constructor

        _queryWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _keyWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _valueWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _outputWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _outputBias = new Vector<T>(embeddingDimension);

        InitializeParameters();
    }

    /// <summary>
    /// Creates a new multi-head attention layer with the specified dimensions, head count, and dropout.
    /// </summary>
    /// <param name="sequenceLength">The length of the input sequence.</param>
    /// <param name="embeddingDimension">The dimension of each element in the sequence.</param>
    /// <param name="headCount">The number of attention heads to use.</param>
    /// <param name="activationFunction">The activation function to apply (defaults to identity function if null).</param>
    /// <param name="dropoutRate">The dropout rate to apply for regularization.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor includes dropout for regularization during training.
    /// Dropout randomly sets some values to zero during training, which helps prevent overfitting.
    /// </para>
    /// </remarks>
    public MultiHeadAttentionLayer(int sequenceLength, int embeddingDimension, int headCount, IActivationFunction<T>? activationFunction, double dropoutRate)
        : base([sequenceLength, embeddingDimension], [sequenceLength, embeddingDimension], activationFunction ?? new IdentityActivation<T>())
    {
        _headCount = headCount;
        _headDimension = embeddingDimension / headCount;
        _dropoutRate = dropoutRate;

        _queryWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _keyWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _valueWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _outputWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _outputBias = new Vector<T>(embeddingDimension);

        InitializeParameters();
    }

    /// <summary>
    /// Creates a new multi-head attention layer with the specified dimensions, head count, and dropout.
    /// </summary>
    /// <param name="sequenceLength">The length of the input sequence.</param>
    /// <param name="embeddingDimension">The dimension of each element in the sequence.</param>
    /// <param name="headCount">The number of attention heads to use.</param>
    /// <param name="vectorActivationFunction">The vector activation function to apply (defaults to identity function if null).</param>
    /// <param name="dropoutRate">The dropout rate to apply for regularization.</param>
    public MultiHeadAttentionLayer(int sequenceLength, int embeddingDimension, int headCount, IVectorActivationFunction<T>? vectorActivationFunction, double dropoutRate)
        : base([sequenceLength, embeddingDimension], [sequenceLength, embeddingDimension], vectorActivationFunction ?? new IdentityActivation<T>())
    {
        _headCount = headCount;
        _headDimension = embeddingDimension / headCount;
        _dropoutRate = dropoutRate;

        _queryWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _keyWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _valueWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _outputWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _outputBias = new Vector<T>(embeddingDimension);

        InitializeParameters();
    }

    /// <summary>
    /// Initializes the weights and biases of the layer.
    /// </summary>
    private void InitializeParameters()
    {
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_queryWeights.Rows + _queryWeights.Columns)));
        InitializeMatrix(_queryWeights, scale);
        InitializeMatrix(_keyWeights, scale);
        InitializeMatrix(_valueWeights, scale);
        InitializeMatrix(_outputWeights, scale);

        for (int i = 0; i < _outputBias.Length; i++)
        {
            _outputBias[i] = NumOps.Zero;
        }
    }

    /// <summary>
    /// Initializes a matrix with random values scaled appropriately.
    /// </summary>
    /// <param name="matrix">The matrix to initialize.</param>
    /// <param name="scale">The scaling factor for the random values.</param>
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
    /// Performs the forward pass of the multi-head attention layer.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The output tensor after applying multi-head attention.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The forward pass is where the layer processes the input data. 
    /// Here's what happens:
    /// 1. The input is transformed into three different representations: queries, keys, and values
    /// 2. These are split into multiple "heads" (different perspectives)
    /// 3. Each head calculates how much attention to pay to different parts of the input
    /// 4. The results from all heads are combined to create the final output
    /// 
    /// Think of it like this: If you're reading a book, you might pay attention to different aspects
    /// like characters, plot, and setting all at once. Each "head" is like focusing on one of these aspects.
    /// </para>
    /// </remarks>
    /// <summary>
    /// Computes attention scores between query and key tensors.
    /// </summary>
    /// <param name="queries">The query tensor.</param>
    /// <param name="keys">The key tensor.</param>
    /// <param name="mask">Optional attention mask tensor.</param>
    /// <returns>The attention scores tensor.</returns>
    protected virtual Tensor<T> ComputeAttentionScores(Tensor<T> queries, Tensor<T> keys, Tensor<T>? mask)
    {
        var attentionScores = queries.Multiply(keys.Transpose([0, 1, 3, 2]));
        attentionScores = attentionScores.Multiply(NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension)));
        
        // Apply mask if provided
        if (mask != null)
        {
            // Apply mask - typically mask values are 0 for positions to attend to and very negative for positions to ignore
            attentionScores = attentionScores.Add(mask);
        }
        
        return attentionScores;
    }
    
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];
        int sequenceLength = input.Shape[1];
        int embeddingDimension = input.Shape[2];

        var queries = input.Multiply(_queryWeights);
        var keys = input.Multiply(_keyWeights);
        var values = input.Multiply(_valueWeights);

        queries = queries.Reshape(batchSize, sequenceLength, _headCount, _headDimension).Transpose([0, 2, 1, 3]);
        keys = keys.Reshape(batchSize, sequenceLength, _headCount, _headDimension).Transpose([0, 2, 1, 3]);
        values = values.Reshape(batchSize, sequenceLength, _headCount, _headDimension).Transpose([0, 2, 1, 3]);

        var attentionScores = ComputeAttentionScores(queries, keys, null);

        var softmaxActivation = new SoftmaxActivation<T>();
        var attentionWeights = softmaxActivation.Activate(attentionScores);
        _lastAttentionScores = attentionWeights;

        var attentionOutput = attentionWeights.Multiply(values);
        attentionOutput = attentionOutput.Transpose([0, 2, 1, 3]).Reshape(batchSize, sequenceLength, embeddingDimension);

        var output = attentionOutput.Multiply(_outputWeights).Add(_outputBias);
        _lastOutput = ApplyActivation(output);

        return _lastOutput;
    }

    /// <summary>
    /// Performs the backward pass of the multi-head attention layer, calculating gradients for learning.
    /// </summary>
    /// <param name="outputGradient">The gradient flowing back from the next layer.</param>
    /// <returns>The gradient to be passed to the previous layer.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The backward pass is how neural networks learn. Think of it like figuring out 
    /// which parts of a recipe need adjustment after tasting the final dish:
    /// 
    /// 1. We first check how our output differs from what was expected (the gradient)
    /// 2. Then we trace backward through all the calculations we did in the forward pass
    /// 3. We determine how much each weight contributed to any errors
    /// 4. These contributions become our gradients, which we'll use to update the weights
    /// 
    /// The complex matrix operations are just a mathematical way of figuring out 
    /// "if I change this weight a little bit, how much would it improve the output?"
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _lastAttentionScores == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        var activationGradient = ApplyActivationDerivative(_lastOutput, outputGradient);

        int batchSize = _lastInput.Shape[0];
        int sequenceLength = _lastInput.Shape[1];
        int embeddingDimension = _lastInput.Shape[2];

        var attentionOutputGradient = activationGradient.Multiply(_outputWeights.Transpose());
        _outputWeightsGradient = activationGradient.Transpose([0, 2, 1]).Multiply(_lastOutput).Sum([0]).ToMatrix();
        _outputBiasGradient = activationGradient.Sum([0, 1]).ToVector();

        attentionOutputGradient = attentionOutputGradient.Reshape([batchSize, sequenceLength, _headCount, _headDimension]).Transpose([0, 2, 1, 3]);

        var softmaxActivation = new SoftmaxActivation<T>();
        var softmaxDerivative = softmaxActivation.Derivative(_lastAttentionScores);
        var attentionWeightsGradient = softmaxDerivative.ElementwiseMultiply(attentionOutputGradient.Multiply(_lastInput.Reshape([batchSize, sequenceLength, _headCount, _headDimension]).Transpose([0, 2, 3, 1])));

        var queriesGradient = attentionWeightsGradient.Multiply(_lastInput.Reshape([batchSize, sequenceLength, _headCount, _headDimension]).Transpose([0, 2, 3, 1]));
        var keysGradient = attentionWeightsGradient.Transpose([0, 1, 3, 2]).Multiply(_lastInput.Reshape([batchSize, sequenceLength, _headCount, _headDimension]).Transpose([0, 2, 1, 3]));
        var valuesGradient = _lastAttentionScores.Transpose([0, 1, 3, 2]).Multiply(attentionOutputGradient);

        queriesGradient = queriesGradient.Transpose([0, 2, 1, 3]).Reshape([batchSize, sequenceLength, embeddingDimension]);
        keysGradient = keysGradient.Transpose([0, 2, 1, 3]).Reshape([batchSize, sequenceLength, embeddingDimension]);
        valuesGradient = valuesGradient.Transpose([0, 2, 1, 3]).Reshape([batchSize, sequenceLength, embeddingDimension]);

        _queryWeightsGradient = _lastInput.Transpose([0, 2, 1]).Multiply(queriesGradient).Sum([0]).ToMatrix();
        _keyWeightsGradient = _lastInput.Transpose([0, 2, 1]).Multiply(keysGradient).Sum([0]).ToMatrix();
        _valueWeightsGradient = _lastInput.Transpose([0, 2, 1]).Multiply(valuesGradient).Sum([0]).ToMatrix();

        var inputGradient = queriesGradient.Multiply(_queryWeights.Transpose())
                            .Add(keysGradient.Multiply(_keyWeights.Transpose()))
                            .Add(valuesGradient.Multiply(_valueWeights.Transpose()));

        return inputGradient;
    }

    /// <summary>
    /// Updates the layer's parameters (weights and biases) using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate that controls how much to adjust the parameters.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method is like adjusting a recipe based on feedback. The learning rate 
    /// is how bold we are with our changes - a higher rate means bigger adjustments, while a lower 
    /// rate means more cautious, smaller adjustments. The gradients tell us which direction to adjust 
    /// each parameter to improve the network's performance.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_queryWeightsGradient == null || _keyWeightsGradient == null || _valueWeightsGradient == null || _outputWeightsGradient == null || _outputBiasGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        _queryWeights = _queryWeights.Subtract(_queryWeightsGradient.Multiply(learningRate));
        _keyWeights = _keyWeights.Subtract(_keyWeightsGradient.Multiply(learningRate));
        _valueWeights = _valueWeights.Subtract(_valueWeightsGradient.Multiply(learningRate));
        _outputWeights = _outputWeights.Subtract(_outputWeightsGradient.Multiply(learningRate));
        _outputBias = _outputBias.Subtract(_outputBiasGradient.Multiply(learningRate));
    }

    /// <summary>
    /// Extracts all parameters (weights and biases) from the layer into a single vector.
    /// </summary>
    /// <returns>A vector containing all parameters of the layer.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method collects all the layer's adjustable values (weights and biases) 
    /// into a single list. Think of it like taking inventory of all the ingredients in a recipe.
    /// This is useful for saving the model's state or for optimization algorithms that need to 
    /// work with all parameters at once.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Calculate total number of parameters
        int totalParams = _queryWeights.Rows * _queryWeights.Columns +
                          _keyWeights.Rows * _keyWeights.Columns +
                          _valueWeights.Rows * _valueWeights.Columns +
                          _outputWeights.Rows * _outputWeights.Columns +
                          _outputBias.Length;

        var parameters = new Vector<T>(totalParams);
        int index = 0;

        // Copy query weights
        for (int i = 0; i < _queryWeights.Rows; i++)
        {
            for (int j = 0; j < _queryWeights.Columns; j++)
            {
                parameters[index++] = _queryWeights[i, j];
            }
        }

        // Copy key weights
        for (int i = 0; i < _keyWeights.Rows; i++)
        {
            for (int j = 0; j < _keyWeights.Columns; j++)
            {
                parameters[index++] = _keyWeights[i, j];
            }
        }

        // Copy value weights
        for (int i = 0; i < _valueWeights.Rows; i++)
        {
            for (int j = 0; j < _valueWeights.Columns; j++)
            {
                parameters[index++] = _valueWeights[i, j];
            }
        }

        // Copy output weights
        for (int i = 0; i < _outputWeights.Rows; i++)
        {
            for (int j = 0; j < _outputWeights.Columns; j++)
            {
                parameters[index++] = _outputWeights[i, j];
            }
        }

        // Copy output bias
        for (int i = 0; i < _outputBias.Length; i++)
        {
            parameters[index++] = _outputBias[i];
        }

        return parameters;
    }

    /// <summary>
    /// Sets all parameters (weights and biases) of the layer from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set in the layer.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method does the opposite of GetParameters - it takes a list of values 
    /// and distributes them back into the layer's weights and biases. It's like restocking all the 
    /// ingredients in your kitchen from a single shopping bag, putting each item in its proper place.
    /// This is useful when loading a saved model or when optimization algorithms have computed 
    /// improved parameter values.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        int totalParams = _queryWeights.Rows * _queryWeights.Columns +
                          _keyWeights.Rows * _keyWeights.Columns +
                          _valueWeights.Rows * _valueWeights.Columns +
                          _outputWeights.Rows * _outputWeights.Columns +
                          _outputBias.Length;

        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }

        int index = 0;

        // Set query weights
        for (int i = 0; i < _queryWeights.Rows; i++)
        {
            for (int j = 0; j < _queryWeights.Columns; j++)
            {
                _queryWeights[i, j] = parameters[index++];
            }
        }

        // Set key weights
        for (int i = 0; i < _keyWeights.Rows; i++)
        {
            for (int j = 0; j < _keyWeights.Columns; j++)
            {
                _keyWeights[i, j] = parameters[index++];
            }
        }

        // Set value weights
        for (int i = 0; i < _valueWeights.Rows; i++)
        {
            for (int j = 0; j < _valueWeights.Columns; j++)
            {
                _valueWeights[i, j] = parameters[index++];
            }
        }

        // Set output weights
        for (int i = 0; i < _outputWeights.Rows; i++)
        {
            for (int j = 0; j < _outputWeights.Columns; j++)
            {
                _outputWeights[i, j] = parameters[index++];
            }
        }

        // Set output bias
        for (int i = 0; i < _outputBias.Length; i++)
        {
            _outputBias[i] = parameters[index++];
        }
    }

    /// <summary>
    /// Resets the internal state of the multi-head attention layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method clears all cached values from previous forward and backward passes,
    /// effectively resetting the layer to its initial state but keeping the learned weights.
    /// This is useful when starting a new training sequence or when you want to clear
    /// any temporary data without losing the layer's learned parameters.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Think of this like clearing your scratch paper after solving a math problem.
    /// You're keeping all the knowledge you've gained (the weights), but you're getting rid of
    /// all the intermediate calculations (cached values) to make room for new work.
    /// 
    /// This is particularly important in neural networks because:
    /// 1. It frees up memory by removing data we no longer need
    /// 2. It ensures that each new input is processed with a "clean slate"
    /// 3. It prevents old calculations from affecting new ones, which could lead to incorrect results
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastOutput = null;
        _lastAttentionScores = null;

        _queryWeightsGradient = null;
        _keyWeightsGradient = null;
        _valueWeightsGradient = null;
        _outputWeightsGradient = null;
        _outputBiasGradient = null;
    }
}
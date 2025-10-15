using AiDotNet.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers.PositionalEncoding;

/// <summary>
/// Implements learnable absolute position embeddings, as used in BERT and many other models.
/// </summary>
/// <remarks>
/// <para>
/// Unlike fixed sinusoidal encodings, learned position embeddings are trainable parameters
/// that are updated during the model training process. This allows the model to learn the
/// optimal positional representations for the specific task.
/// </para>
/// <para><b>For Beginners:</b> This method lets the model learn what each position means.
/// 
/// Instead of using a mathematical formula to create position encodings:
/// - We create a table of position embeddings (one for each possible position)
/// - These embeddings start with random values
/// - During training, they get updated just like other model weights
/// - The model learns what each position means based on the training data
/// 
/// This is like having the model learn its own "position vocabulary" rather than using
/// a pre-defined mathematical pattern. Models like BERT use this approach.
/// 
/// The main limitation is that the model can only recognize positions it has seen during training.
/// If you train on sequences up to length 512, it won't work well on longer sequences.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class LearnedPositionalEncoding<T> : PositionalEncodingBase<T>
{
    /// <summary>
    /// The learnable position embeddings.
    /// </summary>
    private Tensor<T> _positionEmbeddings = default!;
    
    /// <summary>
    /// Gradients for the position embeddings.
    /// </summary>
    private Tensor<T>? _gradients;
    
    
    /// <summary>
    /// Initializes a new instance of the <see cref="LearnedPositionalEncoding{T}"/> class.
    /// </summary>
    /// <param name="maxSequenceLength">The maximum sequence length that this layer can handle.</param>
    /// <param name="embeddingSize">The size of each embedding vector.</param>
    public LearnedPositionalEncoding(int maxSequenceLength, int embeddingSize)
        : base(maxSequenceLength, embeddingSize)
    {
        // Initialize position embeddings with small random values
        _positionEmbeddings = new Tensor<T>([maxSequenceLength, embeddingSize]);
        InitializeEmbeddings();
    }
    
    /// <summary>
    /// Initializes the position embeddings with small random values.
    /// </summary>
    private void InitializeEmbeddings()
    {
        // Initialize with values from a normal distribution with small variance
        Random random = new Random(42); // Fixed seed for reproducibility
        double stdDev = 0.02; // Small standard deviation
        
        for (int pos = 0; pos < _maxSequenceLength; pos++)
        {
            for (int i = 0; i < _embeddingSize; i++)
            {
                // Box-Muller transform to generate normal distribution
                double u1 = 1.0 - random.NextDouble(); // Uniform(0,1) random doubles
                double u2 = 1.0 - random.NextDouble();
                double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                double randNormal = randStdNormal * stdDev;
                
                _positionEmbeddings[pos, i] = NumOps.FromDouble(randNormal);
            }
        }
    }
    
    /// <summary>
    /// Applies learned positional encoding to the input tensor.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor with positional encodings added.</returns>
    protected override Tensor<T> ApplyPositionalEncoding(Tensor<T> input)
    {
        int sequenceLength = input.Shape[0];
        var relevantEmbeddings = _positionEmbeddings.Slice(0, 0, sequenceLength, _embeddingSize);
        // Use tensor addition method instead of direct operator to follow INumericOperations pattern
        return input.Add(relevantEmbeddings);
    }
    
    /// <summary>
    /// Performs the backward pass for learned positional encoding.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    protected override Tensor<T> BackwardPositionalEncoding(Tensor<T> outputGradient)
    {
        // Store gradients for the position embeddings
        int sequenceLength = outputGradient.Shape[0];
        
        // Initialize gradients tensor if it's null
        if (_gradients == null)
        {
            _gradients = new Tensor<T>([_maxSequenceLength, _embeddingSize]);
        }
        
        // Update gradients for positions used in this forward pass
        for (int pos = 0; pos < sequenceLength; pos++)
        {
            for (int i = 0; i < _embeddingSize; i++)
            {
                _gradients[pos, i] = outputGradient[pos, i];
            }
        }
        
        // The gradient with respect to the input is the same as the output gradient
        return outputGradient;
    }
    
    /// <summary>
    /// Updates the position embeddings using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    public override void UpdateParameters(T learningRate)
    {
        if (_gradients != null)
        {
            // Only update embeddings for positions that were used
            for (int pos = 0; pos < _maxSequenceLength; pos++)
            {
                for (int i = 0; i < _embeddingSize; i++)
                {
                    _positionEmbeddings[pos, i] = NumOps.Subtract(
                        _positionEmbeddings[pos, i], 
                        NumOps.Multiply(learningRate, _gradients[pos, i]));
                }
            }
            
            // Reset gradients for next backward pass
            _gradients = new Tensor<T>([_maxSequenceLength, _embeddingSize]);
        }
    }
    
    /// <summary>
    /// Gets all trainable parameters from the positional encoding layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable position embedding parameters.</returns>
    public override Vector<T> GetParameters()
    {
        // Convert the position embeddings tensor to a vector
        return _positionEmbeddings.ToVector();
    }
    
    /// <summary>
    /// Updates the parameters of the layer using the provided parameter vector.
    /// </summary>
    /// <param name="parameters">A vector containing new values for all trainable parameters.</param>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (parameters.Length != _maxSequenceLength * _embeddingSize)
        {
            throw new ArgumentException($"Expected parameter vector of length {_maxSequenceLength * _embeddingSize}, but got {parameters.Length}");
        }
        
        _positionEmbeddings = Tensor<T>.FromVector(parameters, [_maxSequenceLength, _embeddingSize]);
    }
    
    /// <summary>
    /// Gets the number of trainable parameters in this layer.
    /// </summary>
    public override int ParameterCount => _maxSequenceLength * _embeddingSize;
    
    /// <summary>
    /// Encodes positional information for a sequence.
    /// </summary>
    /// <param name="sequenceLength">The length of the sequence to encode.</param>
    /// <returns>A tensor containing positional encodings for each position in the sequence.</returns>
    public override Tensor<T> Encode(int sequenceLength)
    {
        if (sequenceLength > _maxSequenceLength)
        {
            throw new ArgumentException($"Sequence length {sequenceLength} exceeds maximum length {_maxSequenceLength}");
        }
        
        // Return the learned position embeddings for the requested sequence length
        return _positionEmbeddings.Slice(0, 0, sequenceLength, _embeddingSize);
    }
}
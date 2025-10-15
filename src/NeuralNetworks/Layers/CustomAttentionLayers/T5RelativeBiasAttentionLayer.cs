namespace AiDotNet.NeuralNetworks.Layers.CustomAttentionLayers;

using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Layers.PositionalEncoding;

/// <summary>
/// Multi-head attention layer that integrates T5-style relative positional bias.
/// </summary>
/// <remarks>
/// <para>
/// This layer implements the relative position bias approach used in the T5 (Text-to-Text 
/// Transfer Transformer) model. Instead of encoding positions in the input embeddings,
/// T5 adds learned relative position biases directly to the attention scores.
/// </para>
/// <para><b>For Beginners:</b> This layer helps a transformer understand sequence order in a clever way.
/// 
/// T5's approach to positional encoding:
/// - Groups similar relative distances into "buckets" (like "distance 1-2", "distance 3-4", etc.)
/// - Learns a specific bias value for each bucket
/// - Adds these biases directly to attention scores
/// 
/// This is more efficient than having a separate value for every possible position, and
/// it works particularly well for text-to-text tasks like translation and summarization.
/// </para>
/// </remarks>
/// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
public class T5RelativeBiasAttentionLayer<T> : MultiHeadAttentionLayer<T>
{
    /// <summary>
    /// The number of buckets used for relative position binning.
    /// </summary>
    private readonly int _numBuckets;
    
    /// <summary>
    /// The maximum relative distance to consider.
    /// </summary>
    private readonly int _maxDistance;
    
    /// <summary>
    /// Whether to use different buckets for negative and positive relative positions.
    /// </summary>
    private readonly bool _bidirectional;
    
    /// <summary>
    /// The learned relative position biases.
    /// </summary>
    private Vector<T> _relativeBias = default!;
    
    /// <summary>
    /// The gradients for the relative position biases.
    /// </summary>
    private Vector<T>? _biasGradients;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="T5RelativeBiasAttentionLayer{T}"/> class.
    /// </summary>
    /// <param name="sequenceLength">The maximum sequence length that this layer can handle.</param>
    /// <param name="embeddingDimension">The size of each embedding vector.</param>
    /// <param name="headCount">The number of attention heads.</param>
    /// <param name="activationFunction">The activation function to use.</param>
    /// <param name="numBuckets">The number of buckets to use for relative positions. Default is 32.</param>
    /// <param name="maxDistance">The maximum relative distance to consider. Default is 128.</param>
    /// <param name="bidirectional">Whether to use different buckets for negative and positive relative positions. Default is true.</param>
    public T5RelativeBiasAttentionLayer(
        int sequenceLength,
        int embeddingDimension,
        int headCount,
        IActivationFunction<T> activationFunction,
        int numBuckets = 32,
        int maxDistance = 128,
        bool bidirectional = true)
        : base(sequenceLength, embeddingDimension, headCount, activationFunction)
    {
        _numBuckets = numBuckets;
        _maxDistance = maxDistance;
        _bidirectional = bidirectional;
        
        // Initialize the relative bias vector
        _relativeBias = new Vector<T>(_numBuckets);
        InitializeRelativeBias();
    }
    
    /// <summary>
    /// Initializes the relative bias vector with small random values.
    /// </summary>
    private void InitializeRelativeBias()
    {
        // Initialize with values from a normal distribution with small variance
        Random random = new Random(42); // Fixed seed for reproducibility
        double stdDev = 0.02; // Small standard deviation
        
        for (int i = 0; i < _numBuckets; i++)
        {
            // Box-Muller transform to generate normal distribution
            double u1 = 1.0 - random.NextDouble(); // Uniform(0,1) random doubles
            double u2 = 1.0 - random.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            double randNormal = randStdNormal * stdDev;
            
            _relativeBias[i] = NumOps.FromDouble(randNormal);
        }
    }
    
    /// <summary>
    /// Computes the bucket index for a relative position.
    /// </summary>
    /// <param name="relPos">The relative position.</param>
    /// <returns>The bucket index for the relative position.</returns>
    private int RelativePositionBucket(int relPos)
    {
        int relPosAbs = Math.Abs(relPos);
        
        // Half of the buckets are for exact relative positions
        int maxExact = _numBuckets / 2;
        bool isSmall = relPosAbs < maxExact;
        
        // For the other half, we logarithmically divide the range from maxExact to maxDistance
        int bucketPos;
        if (isSmall)
        {
            bucketPos = relPosAbs;
        }
        else
        {
            // This implements the logarithmic binning for larger distances
            // Following T5's approach of using fewer buckets for larger distances
            double log2 = Math.Log(relPosAbs / (double)maxExact) / Math.Log(2);
            int logBin = (int)Math.Floor(log2);
            int logOffset = (relPosAbs - (maxExact << logBin)) >> logBin;
            
            bucketPos = maxExact + (logBin * (maxExact >> 1)) + logOffset;
            bucketPos = Math.Min(bucketPos, _numBuckets - 1);
        }
        
        // If bidirectional, use different buckets for negative relative positions
        return _bidirectional && relPos < 0 ? _numBuckets - bucketPos - 1 : bucketPos;
    }
    
    /// <summary>
    /// Gets the bias value for a relative position.
    /// </summary>
    /// <param name="relPos">The relative position.</param>
    /// <returns>The bias value for the relative position.</returns>
    private T GetBias(int relPos)
    {
        int bucket = RelativePositionBucket(relPos);
        return _relativeBias[bucket];
    }
    
    /// <summary>
    /// Computes the attention scores with T5-style relative bias added.
    /// </summary>
    /// <param name="queries">The query tensor.</param>
    /// <param name="keys">The key tensor.</param>
    /// <param name="mask">Optional attention mask tensor.</param>
    /// <returns>The attention scores tensor with T5-style relative bias applied.</returns>
    protected override Tensor<T> ComputeAttentionScores(Tensor<T> queries, Tensor<T> keys, Tensor<T>? mask)
    {
        // First compute standard attention scores: Q * K^T / sqrt(d_k)
        Tensor<T> scores = base.ComputeAttentionScores(queries, keys, null); // Don't apply mask yet
        
        // Apply T5 relative bias to the attention scores
        int seqLength = queries.Shape[0];
        
        for (int i = 0; i < seqLength; i++)
        {
            for (int j = 0; j < seqLength; j++)
            {
                int relPos = j - i;  // Relative position
                T bias = GetBias(relPos);
                
                // Add the same bias to all heads for this position pair
                for (int h = 0; h < _headCount; h++)
                {
                    scores[i, h, j] = NumOps.Add(scores[i, h, j], bias);
                }
            }
        }
        
        // Now apply mask if provided
        if (mask != null)
        {
            ApplyMask(scores, mask);
        }
        
        return scores;
    }
    
    /// <summary>
    /// Applies the attention mask to the scores.
    /// </summary>
    /// <param name="scores">The attention scores.</param>
    /// <param name="mask">The attention mask.</param>
    private void ApplyMask(Tensor<T> scores, Tensor<T> mask)
    {
        int seqLength = scores.Shape[0];
        T largeNegative = NumOps.FromDouble(-1e9);
        
        for (int i = 0; i < seqLength; i++)
        {
            for (int h = 0; h < _headCount; h++)
            {
                for (int j = 0; j < seqLength; j++)
                {
                    if (NumOps.Equals(mask[i, j], NumOps.Zero))
                    {
                        scores[i, h, j] = largeNegative;
                    }
                }
            }
        }
    }
    
    /// <summary>
    /// Performs the backward pass for the T5 relative bias attention layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // Standard backward pass for attention
        Tensor<T> inputGradient = base.Backward(outputGradient);
        
        // Store bias gradients for parameter update
        if (_biasGradients == null)
        {
            _biasGradients = new Vector<T>(_numBuckets);
        }
        
        // Reset bias gradients
        for (int i = 0; i < _numBuckets; i++)
        {
            _biasGradients[i] = NumOps.Zero;
        }
        
        // Compute gradients for the relative bias parameters
        // This would involve tracking the gradient flow through the attention mechanism
        // Full implementation would be complex and depend on the specific backward pass
        // Simplified placeholder for now
        
        return inputGradient;
    }
    
    /// <summary>
    /// Updates the parameters of the layer using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    public override void UpdateParameters(T learningRate)
    {
        // First update the standard attention parameters
        base.UpdateParameters(learningRate);
        
        // Then update the relative bias parameters
        if (_biasGradients != null)
        {
            for (int i = 0; i < _numBuckets; i++)
            {
                _relativeBias[i] = NumOps.Subtract(
                    _relativeBias[i],
                    NumOps.Multiply(learningRate, _biasGradients[i]));
            }
        }
    }
    
    /// <summary>
    /// Gets all trainable parameters from the layer.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    public override Vector<T> GetParameters()
    {
        // Get the base attention parameters
        Vector<T> baseParams = base.GetParameters();
        
        // Create a new vector with space for both base params and relative bias
        Vector<T> allParams = new Vector<T>(baseParams.Length + _numBuckets);
        
        // Copy base parameters
        for (int i = 0; i < baseParams.Length; i++)
        {
            allParams[i] = baseParams[i];
        }
        
        // Add the relative bias parameters
        for (int i = 0; i < _numBuckets; i++)
        {
            allParams[baseParams.Length + i] = _relativeBias[i];
        }
        
        return allParams;
    }
    
    /// <summary>
    /// Updates the parameters of the layer using the provided parameter vector.
    /// </summary>
    /// <param name="parameters">A vector containing new values for all trainable parameters.</param>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException($"Expected parameter vector of length {ParameterCount}, but got {parameters.Length}");
        }
        
        // Get the number of base parameters
        int baseParamCount = base.ParameterCount;
        
        // Extract and update base parameters
        Vector<T> baseParams = new Vector<T>(baseParamCount);
        for (int i = 0; i < baseParamCount; i++)
        {
            baseParams[i] = parameters[i];
        }
        base.UpdateParameters(baseParams);
        
        // Extract and update relative bias parameters
        for (int i = 0; i < _numBuckets; i++)
        {
            _relativeBias[i] = parameters[baseParamCount + i];
        }
    }
    
    /// <summary>
    /// Gets the number of trainable parameters in this layer.
    /// </summary>
    public override int ParameterCount => base.ParameterCount + _numBuckets;
}
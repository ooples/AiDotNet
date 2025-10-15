namespace AiDotNet.NeuralNetworks.Layers.CustomAttentionLayers;

using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Layers.PositionalEncoding;

/// <summary>
/// Multi-head attention layer that integrates ALiBi (Attention with Linear Biases) positional encoding.
/// </summary>
/// <remarks>
/// <para>
/// This layer implements the ALiBi positional encoding directly in the attention mechanism.
/// ALiBi adds a linear bias to attention scores based on the distance between tokens,
/// which has shown excellent ability to extrapolate to longer sequences.
/// </para>
/// <para><b>For Beginners:</b> This layer helps a transformer understand sequence order while paying attention.
/// 
/// Unlike traditional positional encoding that modifies token embeddings, ALiBi:
/// - Directly changes how attention works
/// - Adds a distance-based penalty between tokens
/// - Makes tokens that are far apart pay less attention to each other
/// - Helps models generalize to longer sequences than seen during training
/// 
/// This is particularly useful for large language models that need to handle variable-length inputs.
/// </para>
/// </remarks>
/// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
public class ALiBiAttentionLayer<T> : MultiHeadAttentionLayer<T>
{
    /// <summary>
    /// The slope parameter that controls how quickly the penalty increases with distance.
    /// </summary>
    private readonly T _slope = default!;
    
    /// <summary>
    /// The pre-computed bias matrix.
    /// </summary>
    private readonly Tensor<T> _biasMatrix = default!;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="ALiBiAttentionLayer{T}"/> class.
    /// </summary>
    /// <param name="sequenceLength">The maximum sequence length that this layer can handle.</param>
    /// <param name="embeddingDimension">The size of each embedding vector.</param>
    /// <param name="headCount">The number of attention heads.</param>
    /// <param name="activationFunction">The activation function to use.</param>
    /// <param name="slope">The slope parameter that controls how quickly the penalty increases with distance. Default is -0.1.</param>
    public ALiBiAttentionLayer(
        int sequenceLength,
        int embeddingDimension,
        int headCount,
        IActivationFunction<T> activationFunction,
        double slope = -0.1)
        : base(sequenceLength, embeddingDimension, headCount, activationFunction)
    {
        _slope = NumOps.FromDouble(slope);
        _biasMatrix = new Tensor<T>([sequenceLength, sequenceLength]);
        InitializeBiasMatrix();
    }
    
    /// <summary>
    /// Initializes the bias matrix used to modify attention scores.
    /// </summary>
    private void InitializeBiasMatrix()
    {
        // ALiBi uses a linear bias matrix where the bias increases
        // with the distance between tokens
        int maxSequenceLength = InputShape[0];
        
        for (int i = 0; i < maxSequenceLength; i++)
        {
            for (int j = 0; j < maxSequenceLength; j++)
            {
                // Calculate the absolute distance between positions
                int distance = Math.Abs(i - j);
                
                // The bias is the negative distance multiplied by the slope
                // This creates a bias that decreases as distance increases
                _biasMatrix[i, j] = NumOps.Multiply(_slope, NumOps.FromDouble(distance));
            }
        }
    }
    
    /// <summary>
    /// Computes the attention scores with ALiBi bias added.
    /// </summary>
    /// <param name="queries">The query tensor.</param>
    /// <param name="keys">The key tensor.</param>
    /// <param name="mask">Optional attention mask tensor.</param>
    /// <returns>The attention scores tensor with ALiBi bias applied.</returns>
    protected override Tensor<T> ComputeAttentionScores(Tensor<T> queries, Tensor<T> keys, Tensor<T>? mask)
    {
        // First compute standard attention scores: Q * K^T / sqrt(d_k)
        Tensor<T> scores = base.ComputeAttentionScores(queries, keys, null); // Don't apply mask yet
        
        // Slice the bias matrix to match the actual sequence length
        int actualSeqLength = queries.Shape[0];
        Tensor<T> slicedBias = _biasMatrix.Slice(0, 0, actualSeqLength, actualSeqLength);
        
        // Apply ALiBi bias to the attention scores
        // For each attention head, add the bias matrix
        int headDim = scores.Shape[1];
        
        for (int i = 0; i < actualSeqLength; i++)
        {
            for (int j = 0; j < actualSeqLength; j++)
            {
                for (int h = 0; h < _headCount; h++)
                {
                    // Apply scaled bias based on head index
                    // Each head gets progressively stronger bias (2^(-i) for head i)
                    double headScale = Math.Pow(2, -h); // Different scale per head
                    T scaledBias = NumOps.Multiply(slicedBias[i, j], NumOps.FromDouble(headScale));
                    
                    scores[i, h, j] = NumOps.Add(scores[i, h, j], scaledBias);
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
}
namespace AiDotNet.NeuralNetworks.Layers;

public class TransformerEncoderLayer<T> : LayerBase<T>
{
    private readonly int _embeddingSize;
    private readonly int _numHeads;
    private readonly int _feedForwardDim;

    private MultiHeadAttentionLayer<T> _selfAttention;
    private LayerNormalizationLayer<T> _norm1;
    private FeedForwardLayer<T> _feedForward;
    private LayerNormalizationLayer<T> _norm2;

    public override bool SupportsTraining => true;

    public TransformerEncoderLayer(int embeddingSize, int numHeads, int feedForwardDim)
        : base([embeddingSize], [embeddingSize])
    {
        _embeddingSize = embeddingSize;
        _numHeads = numHeads;
        _feedForwardDim = feedForwardDim;

        int sequenceLength = 1; // Default to 1
        _selfAttention = new MultiHeadAttentionLayer<T>(
            sequenceLength, 
            _embeddingSize, 
            _numHeads, 
            new GELUActivation<T>() as IActivationFunction<T>);
            
        _norm1 = new LayerNormalizationLayer<T>(_embeddingSize);
        
        _feedForward = new FeedForwardLayer<T>(
            _embeddingSize, 
            _feedForwardDim, 
            new GELUActivation<T>() as IActivationFunction<T>);
            
        _norm2 = new LayerNormalizationLayer<T>(_embeddingSize);
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        var attention = _selfAttention.Forward(input);
        var normalized1 = _norm1.Forward(input + attention);
        var feedForward = _feedForward.Forward(normalized1);
        var output = _norm2.Forward(normalized1 + feedForward);

        return output;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // Backward pass through the second normalization layer
        var dNorm2 = _norm2.Backward(outputGradient);
    
        // Split the gradient for the residual connection
        var dFeedForward = dNorm2;
        var dNormalized1 = dNorm2;

        // Backward pass through the feed-forward layer
        var dFeedForwardInput = _feedForward.Backward(dFeedForward);
        dNormalized1 += dFeedForwardInput;

        // Backward pass through the first normalization layer
        var dNorm1 = _norm1.Backward(dNormalized1);

        // Split the gradient for the residual connection
        var dAttention = dNorm1;
        var dInput = dNorm1;

        // Backward pass through the self-attention layer
        var dSelfAttentionInput = _selfAttention.Backward(dAttention);
        dInput += dSelfAttentionInput;

        return dInput;
    }

    public override void UpdateParameters(T learningRate)
    {
        // Update parameters for each sub-layer
        _selfAttention.UpdateParameters(learningRate);
        _norm1.UpdateParameters(learningRate);
        _feedForward.UpdateParameters(learningRate);
        _norm2.UpdateParameters(learningRate);
    }

    public override Vector<T> GetParameters()
    {
        // Collect parameters from all sublayers
        var selfAttentionParams = _selfAttention.GetParameters();
        var norm1Params = _norm1.GetParameters();
        var feedForwardParams = _feedForward.GetParameters();
        var norm2Params = _norm2.GetParameters();
    
        // Calculate total parameter count
        int totalParamCount = selfAttentionParams.Length + 
                              norm1Params.Length + 
                              feedForwardParams.Length + 
                              norm2Params.Length;
    
        // Create a vector to hold all parameters
        var parameters = new Vector<T>(totalParamCount);
    
        // Copy all parameters into the combined vector
        int currentIndex = 0;
    
        // Copy self-attention parameters
        for (int i = 0; i < selfAttentionParams.Length; i++)
            parameters[currentIndex++] = selfAttentionParams[i];
    
        // Copy norm1 parameters
        for (int i = 0; i < norm1Params.Length; i++)
            parameters[currentIndex++] = norm1Params[i];
    
        // Copy feed-forward parameters
        for (int i = 0; i < feedForwardParams.Length; i++)
            parameters[currentIndex++] = feedForwardParams[i];
    
        // Copy norm2 parameters
        for (int i = 0; i < norm2Params.Length; i++)
            parameters[currentIndex++] = norm2Params[i];
    
        return parameters;
    }

    public override void ResetState()
    {
        // Reset all sublayers
        _selfAttention.ResetState();
        _norm1.ResetState();
        _feedForward.ResetState();
        _norm2.ResetState();
    }
}
namespace AiDotNet.NeuralNetworks.Layers;

public class TransformerDecoderLayer<T> : LayerBase<T>
{
    private readonly int _embeddingSize;
    private readonly int _numHeads;
    private readonly int _feedForwardDim;
    private readonly int _sequenceLength;

    private MultiHeadAttentionLayer<T> _selfAttention;
    private LayerNormalization<T> _norm1;
    private MultiHeadAttentionLayer<T> _crossAttention;
    private LayerNormalization<T> _norm2;
    private FeedForwardLayer<T> _feedForward;
    private LayerNormalization<T> _norm3;

    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastEncoderOutput;
    private Tensor<T>? _lastSelfAttentionOutput;
    private Tensor<T>? _lastNormalized1;
    private Tensor<T>? _lastCrossAttentionOutput;
    private Tensor<T>? _lastNormalized2;
    private Tensor<T>? _lastFeedForwardOutput;

    public override bool SupportsTraining => true;

    public TransformerDecoderLayer(int embeddingSize = 512, 
        int numHeads = 8, 
        int feedForwardDim = 2048, 
        int sequenceLength = 512,
        IActivationFunction<T>? ffnActivation = null)
        : base([embeddingSize], [embeddingSize])
    {
        _embeddingSize = embeddingSize;
        _numHeads = numHeads;
        _feedForwardDim = feedForwardDim;
        _sequenceLength = sequenceLength;

        var activation = ffnActivation ?? new GELUActivation<T>();
         
        // Self-attention layer (no activation)
        _selfAttention = new MultiHeadAttentionLayer<T>(_sequenceLength, _embeddingSize, _numHeads, activation);
        _norm1 = new LayerNormalization<T>([_embeddingSize]);

        // Cross-attention layer (no activation)
        _crossAttention = new MultiHeadAttentionLayer<T>(_sequenceLength, _embeddingSize, _numHeads, activation);
        _norm2 = new LayerNormalization<T>([_embeddingSize]);

        // Feed-forward layer (with activation)
        _feedForward = new FeedForwardLayer<T>(_embeddingSize, _feedForwardDim, activation);
        _norm3 = new LayerNormalization<T>([_embeddingSize]);
    }

    public TransformerDecoderLayer(int embeddingSize = 512, 
        int numHeads = 8, 
        int feedForwardDim = 2048, 
        int sequenceLength = 512,
        IVectorActivationFunction<T>? ffnVectorActivation = null)
        : base([embeddingSize], [embeddingSize])
    {
        _embeddingSize = embeddingSize;
        _numHeads = numHeads;
        _feedForwardDim = feedForwardDim;
        _sequenceLength = sequenceLength;

        var activation = ffnVectorActivation ?? new GELUActivation<T>();

        // Self-attention layer (no activation)
        _selfAttention = new MultiHeadAttentionLayer<T>(_sequenceLength, _embeddingSize, _numHeads, activation);
        _norm1 = new LayerNormalization<T>([_embeddingSize]);

        // Cross-attention layer (no activation)
        _crossAttention = new MultiHeadAttentionLayer<T>(_sequenceLength, _embeddingSize, _numHeads, activation);
        _norm2 = new LayerNormalization<T>([_embeddingSize]);

        // Feed-forward layer (with vector activation)
        _feedForward = new FeedForwardLayer<T>(_embeddingSize, _feedForwardDim, activation);
        _norm3 = new LayerNormalization<T>([_embeddingSize]);
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        throw new InvalidOperationException("Use Forward(Tensor<T> input, Tensor<T> encoderOutput) for TransformerDecoderLayer.");
    }

    public Tensor<T> Forward(Tensor<T> input, Tensor<T> encoderOutput)
    {
        _lastInput = input;
        _lastEncoderOutput = encoderOutput;

        _lastSelfAttentionOutput = _selfAttention.Forward(input);
        _lastNormalized1 = _norm1.Forward(input + _lastSelfAttentionOutput);
        _lastCrossAttentionOutput = _crossAttention.Forward(_lastNormalized1, encoderOutput);
        _lastNormalized2 = _norm2.Forward(_lastNormalized1 + _lastCrossAttentionOutput);
        _lastFeedForwardOutput = _feedForward.Forward(_lastNormalized2);
        var output = _norm3.Forward(_lastNormalized2 + _lastFeedForwardOutput);

        return output;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        var gradNorm3 = _norm3.Backward(outputGradient);
        var gradFeedForward = _feedForward.Backward(gradNorm3);
        var gradNorm2 = _norm2.Backward(gradFeedForward + gradNorm3);
        var gradCrossAttention = _crossAttention.Backward(gradNorm2);
        var gradNorm1 = _norm1.Backward(gradCrossAttention + gradNorm2);
        var gradSelfAttention = _selfAttention.Backward(gradNorm1);

        return gradSelfAttention + gradNorm1;
    }

    public override void UpdateParameters(T learningRate)
    {
        _selfAttention.UpdateParameters(learningRate);
        _norm1.UpdateParameters(learningRate);
        _crossAttention.UpdateParameters(learningRate);
        _norm2.UpdateParameters(learningRate);
        _feedForward.UpdateParameters(learningRate);
        _norm3.UpdateParameters(learningRate);
    }

    public override Vector<T> GetParameters()
    {
        // Collect parameters from all sublayers
        var selfAttentionParams = _selfAttention.GetParameters();
        var norm1Params = _norm1.GetParameters();
        var crossAttentionParams = _crossAttention.GetParameters();
        var norm2Params = _norm2.GetParameters();
        var feedForwardParams = _feedForward.GetParameters();
        var norm3Params = _norm3.GetParameters();
    
        // Calculate total parameter count
        int totalParamCount = selfAttentionParams.Length + 
                              norm1Params.Length + 
                              crossAttentionParams.Length + 
                              norm2Params.Length + 
                              feedForwardParams.Length + 
                              norm3Params.Length;
    
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
    
        // Copy cross-attention parameters
        for (int i = 0; i < crossAttentionParams.Length; i++)
            parameters[currentIndex++] = crossAttentionParams[i];
    
        // Copy norm2 parameters
        for (int i = 0; i < norm2Params.Length; i++)
            parameters[currentIndex++] = norm2Params[i];
    
        // Copy feed-forward parameters
        for (int i = 0; i < feedForwardParams.Length; i++)
            parameters[currentIndex++] = feedForwardParams[i];
    
        // Copy norm3 parameters
        for (int i = 0; i < norm3Params.Length; i++)
            parameters[currentIndex++] = norm3Params[i];
    
        return parameters;
    }

    public override void ResetState()
    {
        // Reset all sublayers
        _selfAttention.ResetState();
        _norm1.ResetState();
        _crossAttention.ResetState();
        _norm2.ResetState();
        _feedForward.ResetState();
        _norm3.ResetState();
    
        // Clear cached tensors
        _lastInput = null;
        _lastEncoderOutput = null;
        _lastSelfAttentionOutput = null;
        _lastNormalized1 = null;
        _lastCrossAttentionOutput = null;
        _lastNormalized2 = null;
        _lastFeedForwardOutput = null;
    }
}
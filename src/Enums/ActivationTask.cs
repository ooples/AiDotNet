namespace AiDotNet.Enums;

/// <summary>
/// Tasks or architectural positions where an activation function is commonly used.
/// </summary>
public enum ActivationTask
{
    /// <summary>Standard hidden layer activation in feedforward/convolutional networks.</summary>
    HiddenLayer,
    /// <summary>Output layer activation for final predictions (classification, regression).</summary>
    OutputLayer,
    /// <summary>Attention mechanism gating (Softmax over attention scores).</summary>
    AttentionGating,
    /// <summary>Recurrent network gating (LSTM/GRU forget/input/output gates).</summary>
    RecurrentGating,
    /// <summary>Producing normalized probability distributions.</summary>
    NormalizationOutput,
    /// <summary>Transformer feed-forward sublayers (GELU, SwiGLU).</summary>
    TransformerFFN,
    /// <summary>Generative model outputs (Tanh for image generation [-1,1]).</summary>
    GenerativeOutput,
    /// <summary>Capsule network squashing functions.</summary>
    CapsuleSquash,
    /// <summary>Spiking neural network activations.</summary>
    SpikingNeuron
}

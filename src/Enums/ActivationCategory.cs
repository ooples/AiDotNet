namespace AiDotNet.Enums;

/// <summary>
/// Categories of activation functions based on their architectural role and behavior.
/// </summary>
public enum ActivationCategory
{
    /// <summary>General-purpose activations suitable for hidden layers (ReLU, GELU, Swish).</summary>
    General,
    /// <summary>Gate activations that control information flow (Sigmoid for LSTM gates, Tanh for cell state).</summary>
    Gate,
    /// <summary>Output activations that produce final predictions (Softmax for classification, Sigmoid for binary).</summary>
    Output,
    /// <summary>Normalization activations that produce probability distributions (Softmax, Sparsemax, LogSoftmax).</summary>
    Normalization,
    /// <summary>Stochastic activations with random components (GumbelSoftmax, BinarySpiking, RReLU).</summary>
    Stochastic,
    /// <summary>Parametric activations with learnable parameters (PReLU, Maxout).</summary>
    Parametric
}

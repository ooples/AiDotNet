namespace AiDotNet.Enums;

/// <summary>
/// Defines the functional roles of activation functions in neural networks.
/// </summary>
/// <remarks>
/// <para>
/// Different parts of neural networks typically require different activation behaviors.
/// This enum categorizes activation functions by their role rather than by their mathematical form.
/// </para>
/// <para><b>For Beginners:</b> This enum helps organize the different "jobs" that activation functions perform
/// in neural networks, similar to how different workers in a factory have different specialized roles.
/// </para>
/// </remarks>
public enum ActivationFunctionRole
{
    /// <summary>
    /// Used for standard hidden layer activations.
    /// </summary>
    /// <remarks>Commonly ReLU, Tanh, etc.</remarks>
    Hidden,

    /// <summary>
    /// Used for output layer activations.
    /// </summary>
    /// <remarks>Commonly Softmax (classification), Linear/Identity (regression), Sigmoid (binary)</remarks>
    Output,

    /// <summary>
    /// Used for gate mechanisms that control information flow.
    /// </summary>
    /// <remarks>Commonly Sigmoid in LSTM, GRU, and NTM gates</remarks>
    Gate,

    /// <summary>
    /// Used for attention mechanisms.
    /// </summary>
    /// <remarks>Commonly Softmax in Transformers, NTM, etc.</remarks>
    Attention,

    /// <summary>
    /// Used for memory cell state updates.
    /// </summary>
    /// <remarks>Commonly Tanh in LSTM and memory networks</remarks>
    Cell,

    /// <summary>
    /// Used for normalization functions.
    /// </summary>
    /// <remarks>Used in Layer Normalization, Batch Normalization, etc.</remarks>
    Normalization,

    /// <summary>
    /// Used for probability distributions.
    /// </summary>
    /// <remarks>Commonly Softmax or Sigmoid</remarks>
    Probability
}

namespace AiDotNet.Data.Structures;

/// <summary>
/// Types of controllers for Neural Turing Machines.
/// </summary>
/// <remarks>
/// <para>
/// Controllers determine how the Neural Turing Machine reads from and writes to
/// its external memory. Different controller architectures offer different
/// trade-offs between complexity and performance.
/// </para>
/// </remarks>
public enum ControllerType
{
    /// <summary>
    /// Feedforward neural network controller.
    /// </summary>
    Feedforward,

    /// <summary>
    /// Long Short-Term Memory controller.
    /// </summary>
    LSTM,

    /// <summary>
    /// Gated Recurrent Unit controller.
    /// </summary>
    GRU
}
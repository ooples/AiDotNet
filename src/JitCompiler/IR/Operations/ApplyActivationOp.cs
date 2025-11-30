namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents a generic activation function application in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Corresponds to TensorOperations<T>.ApplyActivation().
/// Applies a named activation function to the input.
/// </para>
/// <para><b>For Beginners:</b> Applies any activation function by name.
///
/// This is a more generic operation that can apply various activations
/// (ReLU, Sigmoid, Tanh, etc.) based on a parameter rather than being
/// hard-coded to one specific activation.
/// </para>
/// </remarks>
public class ApplyActivationOp : IROp
{
    /// <summary>
    /// The name of the activation function to apply.
    /// </summary>
    public string ActivationName { get; set; } = string.Empty;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        if (string.IsNullOrWhiteSpace(ActivationName)) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = ApplyActivation(t{InputIds[0]}, \"{ActivationName}\") : {OutputType} {OutputShape.ShapeToString()}";
    }
}

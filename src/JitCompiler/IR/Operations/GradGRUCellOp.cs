namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for GRUCellOp.
/// </summary>
/// <remarks>
/// <para>
/// GRU backward pass computes gradients through:
/// - Update gate (z)
/// - Reset gate (r)
/// - Candidate hidden state (h_tilde)
/// </para>
/// <para><b>For Beginners:</b> GRU is simpler than LSTM with just 2 gates instead of 4.
/// The gradient computation is:
/// 1. Gradient through output combination: h = (1-z)*h_prev + z*h_tilde
/// 2. Gradient through candidate: h_tilde = tanh(W_h @ x + U_h @ (r * h_prev))
/// 3. Gradient through gates: z = sigmoid(...), r = sigmoid(...)
/// </para>
/// </remarks>
public class GradGRUCellOp : BackwardOp
{
    /// <summary>Hidden state size.</summary>
    public int HiddenSize { get; set; }

    /// <summary>Which gradient: 0 = input, 1 = hidden, 2 = W_ih, 3 = W_hh, 4 = bias.</summary>
    public int InputIndex { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length < 2) return false;
        if (HiddenSize <= 0) return false;
        return true;
    }

    public override string ToString()
    {
        var inputName = InputIndex switch
        {
            0 => "input",
            1 => "h_prev",
            2 => "W_ih",
            3 => "W_hh",
            4 => "bias",
            _ => $"input[{InputIndex}]"
        };
        return $"t{OutputId} = GradGRUCell[{inputName}, hidden={HiddenSize}](...) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

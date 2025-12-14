namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents a GRU (Gated Recurrent Unit) cell operation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// GRU cell computes:
/// - z = sigmoid(Wz @ x + Uz @ h + bz)  // Update gate
/// - r = sigmoid(Wr @ x + Ur @ h + br)  // Reset gate
/// - h_tilde = tanh(Wh @ x + Uh @ (r * h) + bh)  // Candidate hidden state
/// - h_new = (1 - z) * h + z * h_tilde  // New hidden state
/// </para>
/// </remarks>
public class GRUCellOp : IROp
{
    /// <summary>
    /// Size of the hidden state.
    /// </summary>
    public int HiddenSize { get; set; }

    /// <summary>
    /// Whether to include bias terms.
    /// </summary>
    public bool HasBias { get; set; } = true;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Inputs: input (x), hidden state (h), weights (W_ih, W_hh), optionally biases (b_ih, b_hh)
        if (InputIds.Length < 4) return false;
        if (HiddenSize <= 0) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GRUCell(t{InputIds[0]}, t{InputIds[1]}, hidden={HiddenSize}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

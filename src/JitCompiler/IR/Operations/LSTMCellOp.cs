namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents an LSTM (Long Short-Term Memory) cell operation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// LSTM cell computes:
/// - i = sigmoid(Wi @ x + Ui @ h + bi)  // Input gate
/// - f = sigmoid(Wf @ x + Uf @ h + bf)  // Forget gate
/// - g = tanh(Wg @ x + Ug @ h + bg)     // Cell candidate
/// - o = sigmoid(Wo @ x + Uo @ h + bo)  // Output gate
/// - c_new = f * c + i * g              // New cell state
/// - h_new = o * tanh(c_new)            // New hidden state
/// </para>
/// </remarks>
public class LSTMCellOp : IROp
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
        // Inputs: input (x), hidden state (h), cell state (c), weights (W_ih, W_hh), optionally biases (b_ih, b_hh)
        if (InputIds.Length < 5) return false;
        if (HiddenSize <= 0) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = LSTMCell(t{InputIds[0]}, h=t{InputIds[1]}, c=t{InputIds[2]}, hidden={HiddenSize}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

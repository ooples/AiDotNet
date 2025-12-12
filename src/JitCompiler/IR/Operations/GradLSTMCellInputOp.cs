namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for LSTMCellOp - computes gradient for input.
/// </summary>
/// <remarks>
/// <para>
/// LSTM backward pass uses the chain rule through the gate computations:
/// - grad flows back through output gate, cell state, forget/input gates
/// - Requires saved forward activations for correct gradient computation
/// </para>
/// <para><b>For Beginners:</b> LSTM has multiple paths for gradients to flow:
///
/// The LSTM has 4 gates (input, forget, cell candidate, output) and 2 states (hidden, cell).
/// During backpropagation, we need to compute how the loss changes when we change:
/// 1. The input at this timestep
/// 2. The hidden state from previous timestep
/// 3. The cell state from previous timestep
/// 4. All the weights (W_ih, W_hh) and biases
///
/// This complexity is what makes LSTM training work well for sequences!
/// </para>
/// </remarks>
public class GradLSTMCellInputOp : BackwardOp
{
    /// <summary>Hidden state size.</summary>
    public int HiddenSize { get; set; }

    /// <summary>Which gradient: 0 = input, 1 = hidden, 2 = cell, 3 = W_ih, 4 = W_hh, 5 = bias.</summary>
    public int InputIndex { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Needs: grad_h_out, grad_c_out, plus saved forward tensors
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
            2 => "c_prev",
            3 => "W_ih",
            4 => "W_hh",
            5 => "bias",
            _ => $"input[{InputIndex}]"
        };
        return $"t{OutputId} = GradLSTMCell[{inputName}, hidden={HiddenSize}](...) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

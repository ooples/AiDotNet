namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for full LSTM sequence.
/// </summary>
/// <remarks>
/// <para>
/// Computes gradients for all timesteps of an LSTM sequence.
/// Uses truncated backpropagation through time (TBPTT) if specified.
/// </para>
/// </remarks>
public class GradLSTMSequenceOp : BackwardOp
{
    /// <summary>Hidden state size.</summary>
    public int HiddenSize { get; set; }

    /// <summary>Sequence length.</summary>
    public int SequenceLength { get; set; }

    /// <summary>Number of layers (for stacked LSTM).</summary>
    public int NumLayers { get; set; } = 1;

    /// <summary>Whether LSTM is bidirectional.</summary>
    public bool Bidirectional { get; set; } = false;

    /// <summary>Truncation length for TBPTT (0 = no truncation).</summary>
    public int TruncationLength { get; set; } = 0;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length < 1) return false;
        if (HiddenSize <= 0) return false;
        return true;
    }

    public override string ToString()
    {
        var bidirStr = Bidirectional ? ", bidirectional" : "";
        return $"t{OutputId} = GradLSTMSeq[hidden={HiddenSize}, len={SequenceLength}, layers={NumLayers}{bidirStr}](...) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

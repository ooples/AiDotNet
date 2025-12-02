namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for full GRU sequence.
/// </summary>
public class GradGRUSequenceOp : BackwardOp
{
    /// <summary>Hidden state size.</summary>
    public int HiddenSize { get; set; }

    /// <summary>Sequence length.</summary>
    public int SequenceLength { get; set; }

    /// <summary>Number of layers.</summary>
    public int NumLayers { get; set; } = 1;

    /// <summary>Whether GRU is bidirectional.</summary>
    public bool Bidirectional { get; set; } = false;

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
        return $"t{OutputId} = GradGRUSeq[hidden={HiddenSize}, len={SequenceLength}, layers={NumLayers}{bidirStr}](...) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents 2D convolution in the IR.
/// </summary>
public class Conv2DOp : IROp
{
    /// <summary>Kernel size [height, width].</summary>
    public int[] KernelSize { get; set; } = new int[] { 3, 3 };

    /// <summary>Stride [height, width].</summary>
    public int[] Stride { get; set; } = new int[] { 1, 1 };

    /// <summary>Padding [height, width].</summary>
    public int[] Padding { get; set; } = new int[] { 0, 0 };

    /// <summary>Whether this convolution has a bias term.</summary>
    public bool HasBias { get; set; }

    /// <summary>Input shape [batch, channels, height, width] for kernel generation.</summary>
    public int[] InputShape { get; set; } = new int[] { 1, 1, 1, 1 };

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Input + kernel, optionally + bias
        if (InputIds.Length < 2 || InputIds.Length > 3) return false;
        if (InputIds.Length == 3 && !HasBias) return false;
        return true;
    }

    public override string ToString()
    {
        var inputs = HasBias ? $"t{InputIds[0]}, t{InputIds[1]}, t{InputIds[2]}" : $"t{InputIds[0]}, t{InputIds[1]}";
        return $"t{OutputId} = Conv2D({inputs}, kernel=[{string.Join(",", KernelSize)}], stride=[{string.Join(",", Stride)}], pad=[{string.Join(",", Padding)}]) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

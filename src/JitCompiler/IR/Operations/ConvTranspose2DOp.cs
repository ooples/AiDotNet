namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents transposed 2D convolution in the IR.
/// </summary>
public class ConvTranspose2DOp : IROp
{
    /// <summary>Kernel size [height, width].</summary>
    public int[] KernelSize { get; set; } = new int[] { 3, 3 };

    /// <summary>Stride [height, width].</summary>
    public int[] Stride { get; set; } = new int[] { 1, 1 };

    /// <summary>Padding [height, width].</summary>
    public int[] Padding { get; set; } = new int[] { 0, 0 };

    /// <summary>Output padding [height, width].</summary>
    public int[] OutputPadding { get; set; } = new int[] { 0, 0 };

    /// <summary>Input shape [batch, channels, height, width] for kernel generation.</summary>
    public int[] InputShape { get; set; } = new int[] { 1, 1, 1, 1 };

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length < 2) return false;
        return true;
    }
}

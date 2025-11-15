namespace AiDotNet.JitCompiler.IR.Operations;

// ============================================================================
// REDUCTION OPERATIONS
// ============================================================================

/// <summary>
/// Represents sum reduction in the IR.
/// </summary>
public class SumOp : IROp
{
    public int[]? Axes { get; set; }
    public bool KeepDims { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        var axesStr = Axes != null ? $"[{string.Join(",", Axes)}]" : "all";
        return $"t{OutputId} = Sum(t{InputIds[0]}, axes={axesStr}, keepDims={KeepDims}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents mean reduction in the IR.
/// </summary>
public class MeanOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

/// <summary>
/// Represents max reduction in the IR.
/// </summary>
public class ReduceMaxOp : IROp
{
    public int[]? Axes { get; set; }
    public bool KeepDims { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

/// <summary>
/// Represents mean reduction in the IR.
/// </summary>
public class ReduceMeanOp : IROp
{
    public int[]? Axes { get; set; }
    public bool KeepDims { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

/// <summary>
/// Represents log variance reduction in the IR.
/// </summary>
public class ReduceLogVarianceOp : IROp
{
    public int[]? Axes { get; set; }
    public bool KeepDims { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

// ============================================================================
// SHAPE OPERATIONS
// ============================================================================

/// <summary>
/// Represents reshape operation in the IR.
/// </summary>
public class ReshapeOp : IROp
{
    public int[] NewShape { get; set; } = Array.Empty<int>();

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        if (NewShape.Length == 0) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = Reshape(t{InputIds[0]}, {NewShape.ShapeToString()}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents concatenation along an axis in the IR.
/// </summary>
public class ConcatOp : IROp
{
    public int Axis { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length < 2) return false;  // Need at least 2 inputs to concat
        return true;
    }

    public override string ToString()
    {
        var inputs = string.Join(", ", InputIds.Select(id => $"t{id}"));
        return $"t{OutputId} = Concat([{inputs}], axis={Axis}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents padding operation in the IR.
/// </summary>
public class PadOp : IROp
{
    public int[,]? PadWidth { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

/// <summary>
/// Represents cropping operation in the IR.
/// </summary>
public class CropOp : IROp
{
    public int[] Cropping { get; set; } = Array.Empty<int>();

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

/// <summary>
/// Represents upsampling operation in the IR.
/// </summary>
public class UpsampleOp : IROp
{
    public int Scale { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        if (Scale <= 0) return false;
        return true;
    }
}

/// <summary>
/// Represents pixel shuffle (depth-to-space) operation in the IR.
/// </summary>
public class PixelShuffleOp : IROp
{
    public int UpscaleFactor { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        if (UpscaleFactor <= 0) return false;
        return true;
    }
}

// ============================================================================
// CONVOLUTION OPERATIONS
// ============================================================================

/// <summary>
/// Represents 2D convolution in the IR.
/// </summary>
public class Conv2DOp : IROp
{
    public int[] Stride { get; set; } = new int[] { 1, 1 };
    public int[] Padding { get; set; } = new int[] { 0, 0 };
    public bool HasBias { get; set; }

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
        return $"t{OutputId} = Conv2D({inputs}, stride=[{string.Join(",", Stride)}], pad=[{string.Join(",", Padding)}]) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents transposed 2D convolution in the IR.
/// </summary>
public class ConvTranspose2DOp : IROp
{
    public int[] Stride { get; set; } = new int[] { 1, 1 };
    public int[] Padding { get; set; } = new int[] { 0, 0 };
    public int[] OutputPadding { get; set; } = new int[] { 0, 0 };

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length < 2) return false;
        return true;
    }
}

/// <summary>
/// Represents depthwise 2D convolution in the IR.
/// </summary>
public class DepthwiseConv2DOp : IROp
{
    public int[] Stride { get; set; } = new int[] { 1, 1 };
    public int[] Padding { get; set; } = new int[] { 0, 0 };

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length < 2) return false;
        return true;
    }
}

/// <summary>
/// Represents dilated 2D convolution in the IR.
/// </summary>
public class DilatedConv2DOp : IROp
{
    public int[] Stride { get; set; } = new int[] { 1, 1 };
    public int[] Padding { get; set; } = new int[] { 0, 0 };
    public int[] Dilation { get; set; } = new int[] { 1, 1 };

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length < 2) return false;
        return true;
    }
}

/// <summary>
/// Represents locally connected 2D convolution in the IR.
/// </summary>
public class LocallyConnectedConv2DOp : IROp
{
    public int[] Stride { get; set; } = new int[] { 1, 1 };
    public int[] Padding { get; set; } = new int[] { 0, 0 };

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length < 2) return false;
        return true;
    }
}

// ============================================================================
// POOLING OPERATIONS
// ============================================================================

/// <summary>
/// Represents 2D max pooling in the IR.
/// </summary>
public class MaxPool2DOp : IROp
{
    public int[] PoolSize { get; set; } = new int[] { 2, 2 };
    public int[] Stride { get; set; } = new int[] { 2, 2 };
    public int[] Padding { get; set; } = new int[] { 0, 0 };

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

/// <summary>
/// Represents 2D average pooling in the IR.
/// </summary>
public class AvgPool2DOp : IROp
{
    public int[] PoolSize { get; set; } = new int[] { 2, 2 };
    public int[] Stride { get; set; } = new int[] { 2, 2 };
    public int[] Padding { get; set; } = new int[] { 0, 0 };

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}

// ============================================================================
// NORMALIZATION OPERATIONS
// ============================================================================

/// <summary>
/// Represents layer normalization in the IR.
/// </summary>
public class LayerNormOp : IROp
{
    public int[] NormalizedShape { get; set; } = Array.Empty<int>();
    public double Epsilon { get; set; } = 1e-5;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Input, gamma, beta
        if (InputIds.Length != 3) return false;
        return true;
    }
}

/// <summary>
/// Represents batch normalization in the IR.
/// </summary>
public class BatchNormOp : IROp
{
    public double Epsilon { get; set; } = 1e-5;
    public double Momentum { get; set; } = 0.1;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Input, gamma, beta, running_mean, running_var
        if (InputIds.Length != 5) return false;
        return true;
    }
}

// ============================================================================
// ADVANCED OPERATIONS
// ============================================================================

/// <summary>
/// Represents graph convolution in the IR.
/// </summary>
public class GraphConvOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // features, adjacency_matrix, weights
        if (InputIds.Length != 3) return false;
        return true;
    }
}

/// <summary>
/// Represents affine grid generation for spatial transformer in the IR.
/// </summary>
public class AffineGridOp : IROp
{
    public int[] OutputSize { get; set; } = Array.Empty<int>();

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;  // theta (affine transformation matrix)
        return true;
    }
}

/// <summary>
/// Represents grid sampling for spatial transformer in the IR.
/// </summary>
public class GridSampleOp : IROp
{
    public string InterpolationMode { get; set; } = "bilinear";
    public string PaddingMode { get; set; } = "zeros";

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false;  // input, grid
        return true;
    }
}

/// <summary>
/// Represents RBF (Radial Basis Function) kernel computation in the IR.
/// </summary>
public class RBFKernelOp : IROp
{
    public double Gamma { get; set; } = 1.0;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false;  // x, centers
        return true;
    }
}

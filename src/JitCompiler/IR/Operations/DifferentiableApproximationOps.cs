namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents a soft split operation for differentiable decision trees in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Implements differentiable decision tree nodes using sigmoid gating instead of hard branching.
/// This enables gradient-based learning and JIT compilation of tree-based models.
/// </para>
/// <para>
/// The soft split computes:
/// <code>
/// p_left = σ((threshold - x[featureIndex]) / temperature)
/// output = p_left * leftValue + (1 - p_left) * rightValue
/// </code>
/// </para>
/// <para><b>For Beginners:</b> Normal decision trees make hard yes/no decisions at each node.
/// A soft split makes a "probabilistic" decision - instead of choosing left OR right,
/// it takes a weighted average of both paths based on how close the input is to the threshold.
///
/// Example with temperature=1:
/// - If x[feature] is much less than threshold: p_left ≈ 1 (mostly goes left)
/// - If x[feature] equals threshold: p_left = 0.5 (50/50 split)
/// - If x[feature] is much greater than threshold: p_left ≈ 0 (mostly goes right)
///
/// This makes the tree differentiable (can compute gradients for training) while still
/// approximating hard decision behavior when temperature is low.
/// </para>
/// </remarks>
public class SoftSplitOp : IROp
{
    /// <summary>
    /// Gets or sets the index of the feature to split on.
    /// </summary>
    public int FeatureIndex { get; set; }

    /// <summary>
    /// Gets or sets the threshold value for the split.
    /// </summary>
    public double Threshold { get; set; }

    /// <summary>
    /// Gets or sets the temperature parameter controlling split sharpness.
    /// Lower temperature = sharper (more like hard split), higher = softer.
    /// </summary>
    public double Temperature { get; set; } = 1.0;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Inputs: [input_features, left_value, right_value]
        if (InputIds.Length != 3) return false;
        if (Temperature <= 0) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = SoftSplit(t{InputIds[0]}, t{InputIds[1]}, t{InputIds[2]}, " +
               $"feature={FeatureIndex}, threshold={Threshold}, temp={Temperature}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents a soft K-Nearest Neighbors operation for differentiable instance-based learning.
/// </summary>
/// <remarks>
/// <para>
/// Implements differentiable KNN using attention-weighted contributions from all support vectors
/// instead of hard k-selection. This enables gradient-based optimization and JIT compilation.
/// </para>
/// <para>
/// The soft KNN computes:
/// <code>
/// distances[i] = ||input - supportVectors[i]||²
/// weights = softmax(-distances / temperature)
/// output = Σ weights[i] * labels[i]
/// </code>
/// </para>
/// <para><b>For Beginners:</b> Normal KNN finds the k closest neighbors and averages their labels.
/// Soft KNN considers ALL neighbors but weights them by how close they are:
/// - Very close neighbors get high weights (contribute more to prediction)
/// - Far neighbors get very low weights (contribute almost nothing)
///
/// This is like "all neighbors vote, but closer neighbors have louder voices."
/// The temperature controls how much we favor close neighbors over far ones.
/// </para>
/// </remarks>
public class SoftKNNOp : IROp
{
    /// <summary>
    /// Gets or sets the temperature parameter controlling attention sharpness.
    /// Lower temperature = more focused on nearest neighbors.
    /// </summary>
    public double Temperature { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the distance metric type (0=L2/Euclidean, 1=L1/Manhattan).
    /// </summary>
    public int DistanceType { get; set; } = 0;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Inputs: [input, supportVectors, labels]
        if (InputIds.Length != 3) return false;
        if (Temperature <= 0) return false;
        return true;
    }

    public override string ToString()
    {
        var distName = DistanceType == 0 ? "L2" : "L1";
        return $"t{OutputId} = SoftKNN(t{InputIds[0]}, t{InputIds[1]}, t{InputIds[2]}, " +
               $"temp={Temperature}, dist={distName}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents a soft locally-weighted regression operation for differentiable instance-based learning.
/// </summary>
/// <remarks>
/// <para>
/// Implements differentiable locally-weighted regression using attention-based weighting.
/// This enables gradient-based optimization and JIT compilation of LOESS/LOWESS-style models.
/// </para>
/// <para>
/// The operation computes:
/// <code>
/// distances[i] = ||input - X_train[i]||²
/// weights = softmax(-distances / bandwidth)
/// output = Σ weights[i] * y_train[i]
/// </code>
/// </para>
/// <para><b>For Beginners:</b> Locally-weighted regression makes predictions by computing
/// a weighted average of nearby training examples, where "nearby" is determined by distance.
///
/// This soft version uses attention (softmax) to compute weights, making it fully differentiable:
/// - Points close to the query get high attention weights
/// - Points far from the query get low attention weights
/// - The bandwidth controls how quickly attention drops off with distance
/// </para>
/// </remarks>
public class SoftLocallyWeightedOp : IROp
{
    /// <summary>
    /// Gets or sets the bandwidth parameter controlling the locality of weighting.
    /// Smaller bandwidth = more local (only nearby points matter).
    /// </summary>
    public double Bandwidth { get; set; } = 1.0;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Inputs: [input, X_train, y_train]
        if (InputIds.Length != 3) return false;
        if (Bandwidth <= 0) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = SoftLocallyWeighted(t{InputIds[0]}, t{InputIds[1]}, t{InputIds[2]}, " +
               $"bandwidth={Bandwidth}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

/// <summary>
/// Represents a fake quantization operation with Straight-Through Estimator (STE).
/// </summary>
/// <remarks>
/// <para>
/// Implements differentiable quantization using the Straight-Through Estimator (STE) approach.
/// The forward pass applies quantization, while the backward pass passes gradients through unchanged.
/// This enables training quantization-aware models and JIT compilation of quantized inference.
/// </para>
/// <para>
/// The operation computes:
/// <code>
/// Forward: output = round(input / scale) * scale
/// Backward: ∂L/∂input = ∂L/∂output (gradient passes through)
/// </code>
/// </para>
/// <para><b>For Beginners:</b> Quantization reduces precision (e.g., from 32-bit to 8-bit)
/// to make models smaller and faster. The challenge is that rounding isn't differentiable.
///
/// Fake quantization solves this by:
/// - Forward pass: Actually quantize the values (round to discrete levels)
/// - Backward pass: Pretend quantization didn't happen (let gradients flow through)
///
/// This trick (Straight-Through Estimator) lets us train models that will be quantized later.
/// </para>
/// </remarks>
public class FakeQuantizationOp : IROp
{
    /// <summary>
    /// Gets or sets the number of quantization bits.
    /// </summary>
    public int NumBits { get; set; } = 8;

    /// <summary>
    /// Gets or sets the scale factor for quantization.
    /// If not specified, it will be computed from min/max values.
    /// </summary>
    public double? Scale { get; set; }

    /// <summary>
    /// Gets or sets the zero point for asymmetric quantization.
    /// </summary>
    public double ZeroPoint { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets whether to use symmetric quantization.
    /// </summary>
    public bool Symmetric { get; set; } = true;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        if (NumBits < 1 || NumBits > 32) return false;
        return true;
    }

    public override string ToString()
    {
        var scaleStr = Scale.HasValue ? Scale.Value.ToString("F4") : "auto";
        return $"t{OutputId} = FakeQuantize(t{InputIds[0]}, bits={NumBits}, scale={scaleStr}, " +
               $"zeroPoint={ZeroPoint}, symmetric={Symmetric}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

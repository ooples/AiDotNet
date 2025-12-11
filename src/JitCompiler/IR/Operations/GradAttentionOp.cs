namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for attention (Q*K^T + softmax + matmul V).
/// </summary>
/// <remarks>
/// <para>
/// Attention backward computes gradients for Q, K, V through:
/// 1. grad_V = attention_weights^T @ grad_output
/// 2. grad_attention_weights = grad_output @ V^T
/// 3. grad_scores = softmax_backward(grad_attention_weights)
/// 4. grad_Q = grad_scores @ K
/// 5. grad_K = grad_scores^T @ Q
/// </para>
/// </remarks>
public class GradAttentionOp : BackwardOp
{
    /// <summary>Which input: 0 = Q, 1 = K, 2 = V.</summary>
    public int InputIndex { get; set; }

    /// <summary>Scaling factor used in forward.</summary>
    public double Scale { get; set; } = 1.0;

    /// <summary>Whether causal masking was used.</summary>
    public bool CausalMask { get; set; } = false;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Needs grad_output and saved attention weights
        if (InputIds.Length < 2) return false;
        return true;
    }

    public override string ToString()
    {
        var inputName = InputIndex switch { 0 => "Q", 1 => "K", 2 => "V", _ => $"input[{InputIndex}]" };
        return $"t{OutputId} = GradAttention[{inputName}, scale={Scale}](...) : {OutputType} {OutputShape.ShapeToString()}";
    }
}

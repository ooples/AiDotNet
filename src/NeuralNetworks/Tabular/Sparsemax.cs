using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// Implements the Sparsemax activation function, which projects input onto the probability simplex
/// with sparse outputs (many exact zeros).
/// </summary>
/// <remarks>
/// <para>
/// Sparsemax is an alternative to softmax that produces sparse probability distributions.
/// Unlike softmax, which always produces non-zero probabilities for all inputs, sparsemax
/// can produce exact zeros, making it ideal for feature selection in attention mechanisms.
/// </para>
/// <para>
/// <b>For Beginners:</b> Sparsemax is like softmax but produces cleaner, more focused attention.
///
/// Imagine you're selecting which features to pay attention to:
/// - Softmax says "pay a little attention to everything"
/// - Sparsemax says "focus only on the important features, ignore the rest completely"
///
/// This makes neural networks more interpretable because you can clearly see
/// which features the model considers important (non-zero values) versus
/// which ones it ignores (exact zeros).
///
/// In TabNet, sparsemax is used to select which features to use at each decision step,
/// providing built-in interpretability about feature importance.
/// </para>
/// <para>
/// <b>Mathematical Background:</b>
/// Sparsemax solves: argmin_{p ∈ Δ^K} ||p - z||²
/// where Δ^K is the (K-1)-dimensional probability simplex.
///
/// Reference: "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification"
/// by André F. T. Martins and Ramón Fernandez Astudillo (ICML 2016)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ComponentType(ComponentType.Optimizer)]
[PipelineStage(PipelineStage.Training)]
public class Sparsemax<T>
{
    /// <summary>
    /// Initializes a new instance of the Sparsemax class.
    /// </summary>
    public Sparsemax()
    {
    }

    /// <summary>
    /// Applies the sparsemax function to the input tensor along the specified axis.
    /// </summary>
    /// <param name="input">The input tensor of logits.</param>
    /// <param name="axis">The axis along which to apply sparsemax. Default is -1 (last axis).</param>
    /// <returns>A tensor with sparsemax-normalized values (sparse probability distribution).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This transforms raw scores into a sparse probability distribution.
    /// The output values sum to 1 (like probabilities), but many values will be exactly 0.
    /// </para>
    /// <para>
    /// Runs the engine's sparsemax kernel (Martins &amp; Astudillo, 2016): sort each slice descending, take the
    /// largest k with z_(k) &gt; (sum_{j&lt;=k} z_(j) - 1) / k, set tau = (sum_{j&lt;=k} z_(j) - 1) / k and output
    /// max(z - tau, 0). The op is recorded on the gradient tape, so layers that use it (the TabNet attentive
    /// transformer) receive the sparsemax Jacobian-vector product in backward.
    /// </para>
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> input, int axis = -1)
    {
        // Handle negative axis
        if (axis < 0)
        {
            axis = input.Rank + axis;
        }

        return AiDotNetEngine.Current.Sparsemax(input, axis);
    }
}

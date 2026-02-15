using AiDotNet.Enums;

namespace AiDotNet.CausalDiscovery.DeepLearning;

/// <summary>
/// Base class for deep learning-based causal discovery algorithms.
/// </summary>
/// <remarks>
/// <para>
/// Deep learning methods learn causal structure by training neural networks that
/// parameterize the structural equation model. The DAG constraint is typically
/// enforced through continuous relaxation (e.g., NOTEARS-style) during training.
/// </para>
/// <para>
/// <b>For Beginners:</b> These methods use neural networks to discover causal relationships.
/// They can capture complex nonlinear effects but require more data and computation than
/// traditional methods. Think of them as "letting the neural network figure out which
/// variables cause which" by training it on the data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class DeepCausalBase<T> : CausalDiscoveryBase<T>
{
    /// <inheritdoc/>
    public override CausalDiscoveryCategory Category => CausalDiscoveryCategory.DeepLearning;

    /// <summary>
    /// Number of hidden units in neural network layers.
    /// </summary>
    protected int HiddenUnits { get; set; } = 64;

    /// <summary>
    /// Learning rate for gradient-based optimization.
    /// </summary>
    protected double LearningRate { get; set; } = 1e-3;

    /// <summary>
    /// Maximum training epochs.
    /// </summary>
    protected int MaxEpochs { get; set; } = 100;

    /// <summary>
    /// Applies deep learning options.
    /// </summary>
    protected void ApplyDeepOptions(Models.Options.CausalDiscoveryOptions? options)
    {
        if (options == null) return;
        if (options.MaxIterations.HasValue) MaxEpochs = options.MaxIterations.Value;
        if (options.SparsityPenalty.HasValue) LearningRate = options.SparsityPenalty.Value;
        if (options.MaxParents.HasValue) HiddenUnits = options.MaxParents.Value;
    }

    /// <summary>
    /// Converts double array to Matrix&lt;T&gt;.
    /// </summary>
    protected Matrix<T> DoubleArrayToMatrix(double[,] data)
    {
        int rows = data.GetLength(0), cols = data.GetLength(1);
        var result = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[i, j] = NumOps.FromDouble(data[i, j]);
        return result;
    }
}

using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.Bootstrap;

/// <summary>
/// .632 Bootstrap that provides bias-corrected error estimation.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Regular bootstrap error estimates can be biased (too optimistic).
/// The .632 bootstrap corrects this by combining training error and OOB error.
/// </para>
/// <para>
/// <b>The Formula:</b>
/// Error_632 = 0.368 × TrainingError + 0.632 × OOB_Error
/// </para>
/// <para>
/// <b>Why .632?</b>
/// This number comes from the probability that a sample is NOT in a bootstrap sample:
/// (1 - 1/n)^n ≈ 1/e ≈ 0.368
/// So approximately 63.2% of unique samples end up in training.
/// </para>
/// <para>
/// <b>Note:</b> This splitter generates the same splits as regular bootstrap.
/// The .632 error calculation should be done during model evaluation.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class Bootstrap632Splitter<T> : BootstrapSplitter<T>
{
    /// <summary>
    /// Creates a new .632 bootstrap splitter.
    /// </summary>
    /// <param name="nIterations">Number of bootstrap iterations. Default is 100.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public Bootstrap632Splitter(int nIterations = 100, int randomSeed = 42)
        : base(nIterations, randomSeed)
    {
    }

    /// <inheritdoc/>
    public override string Description => $".632 Bootstrap ({NumSplits} iterations)";
}

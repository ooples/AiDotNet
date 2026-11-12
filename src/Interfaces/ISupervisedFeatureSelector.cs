namespace AiDotNet.Interfaces;

/// <summary>
/// Defines a feature selector that needs the target values to decide which columns to keep.
/// </summary>
/// <remarks>
/// <para>
/// Filter methods that score each column against the target (mutual information, chi-squared, ANOVA F, ReliefF and
/// similar) and wrapper methods that train a model cannot choose columns from the inputs alone, so they are fitted
/// on the inputs together with the target.
/// </para>
/// <para><b>For Beginners:</b> These selectors ask "which columns help predict the answer?", so they need the
/// answers (the target) while fitting. Afterwards, transforming new data only needs the inputs.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public interface ISupervisedFeatureSelector<T> : IFeatureSelector<T>
{
    /// <summary>
    /// Learns which columns to keep from the inputs and their target values.
    /// </summary>
    /// <param name="data">The input matrix, one row per sample and one column per feature.</param>
    /// <param name="target">The target value of each row of <paramref name="data"/>.</param>
    void Fit(AiDotNet.Tensors.LinearAlgebra.Matrix<T> data, AiDotNet.Tensors.LinearAlgebra.Vector<T> target);

    /// <summary>
    /// Learns which columns to keep from the inputs and their target values, then returns the inputs with only
    /// those columns.
    /// </summary>
    /// <param name="data">The input matrix, one row per sample and one column per feature.</param>
    /// <param name="target">The target value of each row of <paramref name="data"/>.</param>
    /// <returns>The selected columns of <paramref name="data"/>, in their original order.</returns>
    AiDotNet.Tensors.LinearAlgebra.Matrix<T> FitTransform(
        AiDotNet.Tensors.LinearAlgebra.Matrix<T> data, AiDotNet.Tensors.LinearAlgebra.Vector<T> target);
}

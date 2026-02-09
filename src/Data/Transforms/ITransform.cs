namespace AiDotNet.Data.Transforms;

/// <summary>
/// Core interface for composable data transforms in the data pipeline.
/// </summary>
/// <typeparam name="TInput">The type of the input data.</typeparam>
/// <typeparam name="TOutput">The type of the output data after transformation.</typeparam>
/// <remarks>
/// <para>
/// Transforms are deterministic, composable operations applied during data loading.
/// They differ from augmentations (in <c>AiDotNet.Augmentation</c>), which are
/// stochastic and applied during training.
/// </para>
/// <para><b>For Beginners:</b> A transform converts data from one form to another.
/// For example, normalizing pixel values from [0, 255] to [0, 1], or converting
/// a label integer to a one-hot encoded vector.
/// </para>
/// </remarks>
public interface ITransform<in TInput, out TOutput>
{
    /// <summary>
    /// Applies the transform to the input data.
    /// </summary>
    /// <param name="input">The input data to transform.</param>
    /// <returns>The transformed output data.</returns>
    TOutput Apply(TInput input);
}

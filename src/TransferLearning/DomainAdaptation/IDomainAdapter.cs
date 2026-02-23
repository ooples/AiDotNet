namespace AiDotNet.TransferLearning.DomainAdaptation;

/// <summary>
/// Defines the interface for adapting models to reduce distribution shift between source and target domains.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Domain adaptation is like helping someone adjust to a new environment.
/// Even when features are mapped correctly, the source and target domains might have different
/// statistical properties (like different averages or variability). A domain adapter helps
/// reduce these differences so a model trained on source data works better on target data.
/// </para>
/// <para>
/// Think of it like adjusting your eyes when moving from a bright room to a dim room - the
/// objects are the same, but your perception needs to adapt to the new lighting conditions.
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("DomainAdapter")]
public interface IDomainAdapter<T>
{
    /// <summary>
    /// Adapts source domain data to better match the target domain distribution.
    /// </summary>
    /// <param name="sourceData">Data from the source domain.</param>
    /// <param name="targetData">Data from the target domain (used to learn the target distribution).</param>
    /// <returns>The adapted source data that better matches the target distribution.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method adjusts source data so it looks more like target data
    /// statistically. It's like color-correcting photos from different cameras to make them
    /// look consistent.
    /// </para>
    /// </remarks>
    Matrix<T> AdaptSource(Matrix<T> sourceData, Matrix<T> targetData);

    /// <summary>
    /// Adapts target domain data to better match the source domain distribution.
    /// </summary>
    /// <param name="targetData">Data from the target domain.</param>
    /// <param name="sourceData">Data from the source domain (used to learn the source distribution).</param>
    /// <returns>The adapted target data that better matches the source distribution.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the reverse operation - adjusting target data to look more
    /// like source data. Sometimes this is useful when you want to evaluate how well transfer
    /// learning might work.
    /// </para>
    /// </remarks>
    Matrix<T> AdaptTarget(Matrix<T> targetData, Matrix<T> sourceData);

    /// <summary>
    /// Computes the domain discrepancy (how different the domains are).
    /// </summary>
    /// <param name="sourceData">Data from the source domain.</param>
    /// <param name="targetData">Data from the target domain.</param>
    /// <returns>A non-negative value where higher values indicate greater domain shift.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This measures how different the two domains are. A small value means
    /// the domains are similar (transfer learning should work well). A large value means the
    /// domains are very different (transfer learning might be challenging).
    /// </para>
    /// </remarks>
    T ComputeDomainDiscrepancy(Matrix<T> sourceData, Matrix<T> targetData);

    /// <summary>
    /// Gets the name of the adaptation method.
    /// </summary>
    string AdaptationMethod { get; }

    /// <summary>
    /// Determines if the adapter requires training before use.
    /// </summary>
    bool RequiresTraining { get; }

    /// <summary>
    /// Trains the domain adapter if required.
    /// </summary>
    /// <param name="sourceData">Training data from the source domain.</param>
    /// <param name="targetData">Training data from the target domain.</param>
    void Train(Matrix<T> sourceData, Matrix<T> targetData);
}

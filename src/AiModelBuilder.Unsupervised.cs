using AiDotNet.Clustering.Base;
using AiDotNet.Helpers;
using AiDotNet.Models.Results;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet;

public partial class AiModelBuilder<T, TInput, TOutput>
{
    /// <summary>
    /// Builds an unsupervised model from data alone.
    /// </summary>
    /// <param name="features">The data to fit, one row per sample.</param>
    /// <returns>The trained result, the same one <see cref="Build(TInput, TOutput)"/> returns.</returns>
    /// <remarks>
    /// <para>
    /// Clustering has no labels to learn from — that is what makes it clustering. Until this existed the
    /// only way through the builder was <see cref="Build(TInput, TOutput)"/> with a label argument the
    /// model ignores, which every clustering example had to construct and explain. A parameter that
    /// exists only to be discarded is worth removing rather than documenting, so this overload
    /// constructs it and the caller stops seeing it.
    /// </para>
    /// <para>
    /// The assignments come off the result afterwards through <c>GetClusterLabels</c>, and a row the
    /// model has not seen is placed with <c>Predict</c>.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Supervised learning is shown the right answers and learns to reproduce
    /// them. Clustering is shown none: it groups rows by how similar they are, and the groups are the
    /// output rather than something checked against a key.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="features"/> is null.</exception>
    /// <exception cref="InvalidOperationException">
    /// No model has been configured, the configured model is not an unsupervised one, or this builder's
    /// output type is one the placeholder cannot be built for.
    /// </exception>
    /// <example>
    /// <code>
    /// var dataMatrix = new Matrix&lt;double&gt;(new double[,]
    /// {
    ///     { 1.0, 2.0 }, { 1.5, 1.8 }, { 5.0, 8.0 }, { 8.0, 8.0 }, { 1.0, 0.6 }, { 9.0, 11.0 }
    /// });
    ///
    /// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureModel(new KMeans&lt;double&gt;(new KMeansOptions&lt;double&gt;()))
    ///     .Build(dataMatrix);
    ///
    /// // one cluster index per row
    /// var assignments = result.GetClusterLabels();
    /// </code>
    /// </example>
    public AiModelResult<T, TInput, TOutput> Build(TInput features)
    {
        if (features is null) throw new ArgumentNullException(nameof(features));

        if (_model is null)
        {
            throw new InvalidOperationException(
                "Build(features) needs a model. Call ConfigureModel with an unsupervised model — a " +
                "clustering model such as KMeans, BIRCH, CURE, CLARANS, SpectralClustering, " +
                "GaussianMixtureModel or SelfOrganizingMap — before building.");
        }

        if (_model is not ClusteringBase<T>)
        {
            throw new InvalidOperationException(
                $"Build(features) is for unsupervised models, which have nothing to learn from labels; " +
                $"the configured model is {_model.GetType().Name} and does. Use Build(features, labels).");
        }

        // GetBatchSize, not GetInputSize: the latter is the feature width, and a placeholder sized to
        // the number of columns fails the data loader's row-count check on anything but a square input.
        int samples = InputHelper<T, TInput>.GetBatchSize(features);

        // The model ignores this, and constructing it here is the point: it is the argument every
        // clustering example used to build by hand and then explain away.
        if (CreateIgnoredLabels(samples) is not TOutput ignored)
        {
            throw new InvalidOperationException(
                $"Build(features) needs a builder whose output type is Vector<{typeof(T).Name}> or " +
                $"Tensor<{typeof(T).Name}>; this one is over {typeof(TOutput).Name}. Use " +
                $"Build(features, labels) and supply a placeholder of that type yourself.");
        }

        return Build(features, ignored);
    }

    /// <summary>
    /// Builds the placeholder an unsupervised model will not read. Shaped to the builder's output type
    /// rather than assumed, so a tensor-typed builder is served as readily as a vector-typed one.
    /// </summary>
    private static object? CreateIgnoredLabels(int samples)
    {
        if (typeof(TOutput) == typeof(Vector<T>))
        {
            return new Vector<T>(samples);
        }

        if (typeof(TOutput) == typeof(Tensor<T>))
        {
            return new Tensor<T>([samples]);
        }

        return null;
    }
}

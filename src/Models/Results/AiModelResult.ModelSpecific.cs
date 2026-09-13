using AiDotNet.AutoML;
using AiDotNet.Clustering.Base;
using AiDotNet.Data.Structures;
using AiDotNet.MetaLearning;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Models.Results;

/// <summary>
/// Outputs that are the whole point of a particular model family, but which prediction cannot express:
/// which cluster each row landed in, what a generator produces, how a meta-learner does on a new task,
/// and the architecture a search settled on.
/// </summary>
/// <remarks>
/// <para>
/// Each of these is the reason you would reach for that model at all. Without them on the result, using
/// any of these families meant holding the model directly and stepping around the builder — which skips
/// the validation, preprocessing and training pipeline the builder exists to run.
/// </para>
/// </remarks>
public partial class AiModelResult<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the cluster each training row was assigned to.
    /// </summary>
    /// <returns>One cluster index per row, or <c>null</c> if the model has not been fitted.</returns>
    /// <exception cref="NotSupportedException">The built model is not a clustering model.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Clustering has no right answers to predict — it groups rows by similarity.
    /// This tells you which group each of your training rows ended up in. To place a row the model has
    /// not seen, use <see cref="Predict(TInput)"/> instead.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// var dataMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 1.5, 1.8 }, { 5.0, 8.0 }, { 8.0, 8.0 } });
    /// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureModel(new SpectralClustering&lt;double&gt;(new SpectralOptions&lt;double&gt;()))
    ///     .Build(dataMatrix);
    ///
    /// var assignments = result.GetClusterLabels();
    /// </code>
    /// </example>
    public Vector<T>? GetClusterLabels()
    {
        if (EnsureModel is ClusteringBase<T> clustering)
        {
            return clustering.Labels;
        }

        throw new NotSupportedException(
            $"GetClusterLabels requires a clustering model (ClusteringBase<T>); the built model is " +
            $"{EnsureModel.GetType().Name}.");
    }

    /// <summary>
    /// Adapts the meta-learned model to one new task and reports how it did.
    /// </summary>
    /// <param name="task">The task to adapt to: its support set is learned from, its query set scored.</param>
    /// <returns>The adaptation result, including the post-adaptation score.</returns>
    /// <exception cref="NotSupportedException">The built model is not a meta-learner.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A meta-learner is trained to be good at *learning*, not at one job. This is
    /// how you use that: hand it a job it has never seen with a handful of examples, and it adapts on the
    /// spot and tells you how well it then does.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(inputFeatures: 4, outputSize: 2);
    /// var metaModel = new NeuralNetwork&lt;double&gt;(architecture);
    /// var trainX = Tensor&lt;double&gt;.CreateRandom(4, 4);
    /// var trainY = Tensor&lt;double&gt;.CreateRandom(4, 2);
    ///
    /// var result = new AiModelBuilder&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;()
    ///     .ConfigureModel(new MAMLAlgorithm&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(
    ///         new MAMLOptions&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(metaModel)))
    ///     .Build(trainX, trainY);
    ///
    /// var newTask = new MetaLearningTask&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;();
    /// var adaptResult = result.AdaptAndEvaluate(newTask);
    /// </code>
    /// </example>
    public MetaAdaptationResult<T> AdaptAndEvaluate(MetaLearningTask<T, TInput, TOutput> task)
    {
        if (EnsureModel is MetaLearnerBase<T, TInput, TOutput> metaLearner)
        {
            return metaLearner.AdaptAndEvaluate(task);
        }

        throw new NotSupportedException(
            $"AdaptAndEvaluate requires a meta-learner (MetaLearnerBase<T, TInput, TOutput>); the built " +
            $"model is {EnsureModel.GetType().Name}.");
    }

    /// <summary>
    /// Draws synthetic rows from a fitted generator.
    /// </summary>
    /// <param name="numSamples">How many rows to generate.</param>
    /// <param name="conditionColumn">Optional column to condition on, so the sample matches a chosen value.</param>
    /// <param name="conditionValue">The value that column should take.</param>
    /// <returns>The generated rows.</returns>
    /// <exception cref="NotSupportedException">The built model is not a synthetic-data generator.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A generator learns the shape of your data and then invents new rows that
    /// look like it — useful when the real data cannot leave the building. Conditioning lets you ask for
    /// a particular kind of row: "generate 100 patients who tested positive".
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// var trainX = Tensor&lt;double&gt;.CreateRandom(16, 8);
    /// var trainY = Tensor&lt;double&gt;.CreateRandom(16, 8);
    ///
    /// var result = new AiModelBuilder&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;()
    ///     .ConfigureModel(new MedGANGenerator&lt;double&gt;(
    ///         new NeuralNetworkArchitecture&lt;double&gt;(
    ///             InputType.OneDimensional, NeuralNetworkTaskType.Regression,
    ///             inputSize: 8, outputSize: 8)))
    ///     .Build(trainX, trainY);
    ///
    /// var synthetic = result.GenerateSamples(numSamples: 100);
    /// </code>
    /// </example>
    public Matrix<T> GenerateSamples(
        int numSamples,
        Vector<T>? conditionColumn = null,
        Vector<T>? conditionValue = null)
    {
        if (EnsureModel is MedGANGenerator<T> generator)
        {
            return generator.Generate(numSamples, conditionColumn, conditionValue);
        }

        throw new NotSupportedException(
            $"GenerateSamples requires a synthetic-data generator (MedGANGenerator<T>); the built model " +
            $"is {EnsureModel.GetType().Name}.");
    }

    /// <summary>
    /// Extracts the architecture a neural architecture search settled on.
    /// </summary>
    /// <returns>The derived architecture.</returns>
    /// <exception cref="NotSupportedException">The built model is not an architecture search.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A supernet trains many candidate designs at once. When the search is done,
    /// this is how you get the winning design out so you can train it properly on its own.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// var searchSpace = new SearchSpaceBase&lt;float&gt;();
    /// var trainX = Tensor&lt;float&gt;.CreateRandom(4, 8);
    /// var trainY = Tensor&lt;float&gt;.CreateRandom(4, 2);
    ///
    /// var result = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
    ///     .ConfigureModel(new SuperNet&lt;float&gt;(searchSpace, numNodes: 4))
    ///     .Build(trainX, trainY);
    ///
    /// var architecture = result.DeriveArchitecture();
    /// </code>
    /// </example>
    public Architecture<T> DeriveArchitecture()
    {
        if (EnsureModel is SuperNet<T> superNet)
        {
            return superNet.DeriveArchitecture();
        }

        throw new NotSupportedException(
            $"DeriveArchitecture requires an architecture search (SuperNet<T>); the built model is " +
            $"{EnsureModel.GetType().Name}.");
    }
}

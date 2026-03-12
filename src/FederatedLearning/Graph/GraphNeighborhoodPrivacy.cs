using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Graph;

/// <summary>
/// Applies local differential privacy (LDP) to neighborhood queries to prevent topology leakage.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> GNN embeddings encode information about a node's neighborhood structure.
/// If shared naively, an adversary could reconstruct the local graph topology. This class adds
/// calibrated noise to neighborhood aggregations before they leave a client, ensuring that
/// individual edges cannot be inferred from the shared embeddings.</para>
///
/// <para><b>Mechanism:</b></para>
/// <list type="bullet">
/// <item><description><b>Degree perturbation:</b> Add Laplace noise to node degree reports.</description></item>
/// <item><description><b>Embedding noise:</b> Add calibrated Gaussian noise to aggregated neighborhood
/// embeddings before sharing (sensitivity based on max neighbor influence).</description></item>
/// <item><description><b>Randomized response:</b> When reporting edge existence, flip the answer with
/// probability calibrated by epsilon.</description></item>
/// </list>
///
/// <para><b>Privacy guarantee:</b> (epsilon, delta)-differential privacy for individual edges.
/// Lower epsilon = stronger privacy but more noise.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class GraphNeighborhoodPrivacy<T> : FederatedLearningComponentBase<T>
{
    private readonly double _epsilon;
    private readonly double _delta;
    private readonly double _sensitivity;

    /// <summary>
    /// Initializes a new instance of <see cref="GraphNeighborhoodPrivacy{T}"/>.
    /// </summary>
    /// <param name="epsilon">Privacy budget epsilon. Default 2.0.</param>
    /// <param name="delta">Privacy failure probability. Default 1e-5.</param>
    /// <param name="sensitivity">Maximum influence of a single edge on output. Default 1.0.</param>
    public GraphNeighborhoodPrivacy(double epsilon = 2.0, double delta = 1e-5, double sensitivity = 1.0)
    {
        if (epsilon <= 0) throw new ArgumentOutOfRangeException(nameof(epsilon), "Epsilon must be positive.");
        if (delta <= 0 || delta >= 1) throw new ArgumentOutOfRangeException(nameof(delta), "Delta must be in (0, 1).");
        if (sensitivity <= 0) throw new ArgumentOutOfRangeException(nameof(sensitivity), "Sensitivity must be positive.");

        _epsilon = epsilon;
        _delta = delta;
        _sensitivity = sensitivity;
    }

    /// <summary>
    /// Adds calibrated Gaussian noise to node embeddings to protect neighborhood structure.
    /// </summary>
    /// <param name="embeddings">Node embeddings to privatize (flattened [numNodes * embDim]).</param>
    /// <param name="embeddingDim">Dimensionality of each embedding vector.</param>
    /// <returns>Privatized embeddings with Gaussian noise added.</returns>
    public Tensor<T> PrivatizeEmbeddings(Tensor<T> embeddings, int embeddingDim)
    {
        if (embeddings is null) throw new ArgumentNullException(nameof(embeddings));
        if (embeddingDim <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingDim), "Embedding dimension must be positive.");

        int totalElements = embeddings.Shape[0];

        if (totalElements % embeddingDim != 0)
        {
            throw new ArgumentException(
                $"Embedding tensor length ({totalElements}) is not divisible by embeddingDim ({embeddingDim}). " +
                "Expected a flattened [numNodes * embDim] tensor.",
                nameof(embeddings));
        }

        var privatized = new Tensor<T>(new[] { totalElements });

        // Gaussian mechanism: sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon
        double sigma = _sensitivity * Math.Sqrt(2 * Math.Log(1.25 / _delta)) / _epsilon;
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();

        for (int i = 0; i < totalElements; i++)
        {
            double value = NumOps.ToDouble(embeddings[i]);
            double noise = SampleGaussian(rng, 0, sigma);
            privatized[i] = NumOps.FromDouble(value + noise);
        }

        return privatized;
    }

    /// <summary>
    /// Perturbs node degree information with Laplace noise.
    /// </summary>
    /// <param name="degrees">Array of node degrees.</param>
    /// <returns>Perturbed degrees.</returns>
    public double[] PerturbDegrees(int[] degrees)
    {
        if (degrees is null) throw new ArgumentNullException(nameof(degrees));

        // Laplace mechanism: scale = sensitivity / epsilon
        double scale = _sensitivity / _epsilon;
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var perturbed = new double[degrees.Length];

        for (int i = 0; i < degrees.Length; i++)
        {
            double noise = SampleLaplace(rng, 0, scale);
            perturbed[i] = Math.Max(0, degrees[i] + noise);
        }

        return perturbed;
    }

    /// <summary>
    /// Applies randomized response to edge existence queries.
    /// </summary>
    /// <param name="edgeExists">True if the edge actually exists.</param>
    /// <returns>Reported value (may be flipped for privacy).</returns>
    public bool RandomizedResponseEdge(bool edgeExists)
    {
        double flipProbability = 1.0 / (Math.Exp(_epsilon) + 1.0);
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();

        if (rng.NextDouble() < flipProbability)
        {
            return !edgeExists; // Flip with calibrated probability
        }

        return edgeExists;
    }

    /// <summary>
    /// Clips and adds noise to an aggregated neighborhood feature vector.
    /// </summary>
    /// <param name="aggregatedFeatures">Mean-aggregated neighbor features.</param>
    /// <param name="clipNorm">Maximum L2 norm before noise addition.</param>
    /// <returns>Clipped and noised feature vector.</returns>
    public Tensor<T> ClipAndNoiseNeighborhood(Tensor<T> aggregatedFeatures, double clipNorm)
    {
        if (aggregatedFeatures is null) throw new ArgumentNullException(nameof(aggregatedFeatures));

        int dim = aggregatedFeatures.Shape[0];
        var result = new Tensor<T>(new[] { dim });

        // Compute L2 norm
        double normSquared = 0;
        for (int i = 0; i < dim; i++)
        {
            double v = NumOps.ToDouble(aggregatedFeatures[i]);
            normSquared += v * v;
        }

        double norm = Math.Sqrt(normSquared);
        double clipFactor = norm > clipNorm ? clipNorm / norm : 1.0;

        // Clip and add Gaussian noise
        double sigma = clipNorm * Math.Sqrt(2 * Math.Log(1.25 / _delta)) / _epsilon;
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();

        for (int i = 0; i < dim; i++)
        {
            double clipped = NumOps.ToDouble(aggregatedFeatures[i]) * clipFactor;
            double noise = SampleGaussian(rng, 0, sigma);
            result[i] = NumOps.FromDouble(clipped + noise);
        }

        return result;
    }

    private static double SampleGaussian(Random rng, double mean, double stdDev)
    {
        // Box-Muller transform
        double u1 = 1.0 - rng.NextDouble(); // Avoid log(0)
        double u2 = rng.NextDouble();
        double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        return mean + stdDev * z;
    }

    private static double SampleLaplace(Random rng, double mean, double scale)
    {
        double u = rng.NextDouble() - 0.5;
        return mean - scale * Math.Sign(u) * Math.Log(1 - 2 * Math.Abs(u));
    }
}

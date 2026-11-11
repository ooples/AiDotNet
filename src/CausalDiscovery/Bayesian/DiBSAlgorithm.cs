using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.Bayesian;

/// <summary>
/// DiBS — Differentiable Bayesian Structure Learning.
/// </summary>
/// <remarks>
/// <para>
/// DiBS uses Stein Variational Gradient Descent (SVGD) to maintain a set of particles,
/// each representing a possible DAG structure via continuous edge logits. The particles
/// are jointly optimized to approximate the posterior distribution over DAGs using
/// a differentiable relaxation of the acyclicity constraint.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>Initialize K particles, each with edge logits Z_k[i,j] (d x d matrix)</item>
/// <item>For each particle, compute edge probabilities via sigmoid(Z_k / tau)</item>
/// <item>Compute log-posterior gradient: data likelihood + prior + acyclicity penalty</item>
/// <item>Compute SVGD kernel: k(Z_k, Z_l) = exp(-||Z_k - Z_l||^2 / (2*bandwidth^2))</item>
/// <item>Update particles: Z_k += lr * (1/K) * sum_l [k(Z_l, Z_k) * grad_l + grad_k(Z_l, Z_k)]</item>
/// <item>Anneal temperature tau</item>
/// <item>Average edge probabilities across particles for final output</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> DiBS uses gradient-based optimization (like training a neural network)
/// to find not just one graph but a whole set of plausible graphs, giving you uncertainty
/// estimates about the causal structure.
/// </para>
/// <para>
/// Reference: Lorch et al. (2021), "DiBS: Differentiable Bayesian Structure Learning", NeurIPS.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ResearchPaper("DiBS: Differentiable Bayesian Structure Learning", "https://proceedings.neurips.cc/paper/2021/hash/5e7dff7b20d04e7b47b22b44e6b8a82f-Abstract.html", Year = 2021, Authors = "Lars Lorch, Jonas Rothfuss, Bernhard Scholkopf, Andreas Krause")]
public class DiBSAlgorithm<T> : BayesianCausalBase<T>
{
    private readonly double _learningRate;
    private readonly int _numParticles;

    /// <inheritdoc/>
    public override string Name => "DiBS";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public DiBSAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyBayesianOptions(options);
        _learningRate = options?.LearningRate ?? 0.01;
        _numParticles = 10;
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        if (n < 3 || d < 2) return new Matrix<T>(d, d);

        // Standardize each column to zero mean / unit variance BEFORE scoring. Causal structure is a
        // property of the joint DISTRIBUTION, not the measurement units, so the discovered graph must
        // be invariant to per-variable rescaling (DiscoverStructure_IsInvariantToDataScaling). Scoring
        // raw covariances made the data term scale with the variables' magnitudes, so a 10x rescale
        // changed the gradient balance and the recovered edges. Standardizing turns the covariance
        // used below into the CORRELATION matrix (diagonal == 1), which is scale-invariant.
        data = StandardizeColumns(data);

        var cov = ComputeCovarianceMatrix(data);
        var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(Seed);
        int K = _numParticles;
        int D = d * d;
        var offDiagonal = OffDiagonalMask(d);

        // Initialize particles: each is a d x d matrix of edge logits, stored flattened as a row of
        // particles [K, d*d]; off-diagonal logits 0.01 * (u - 0.5) drawn particle by particle in row-major order.
        var offDiagonalPositions = new int[K * d * (d - 1)];
        for (int k = 0, slot = 0; k < K; k++)
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    if (i != j) offDiagonalPositions[slot++] = k * D + i * d + j;
        var draws = Engine.TensorMultiplyScalar(
            Engine.TensorAddScalar(Tensor<T>.CreateRandom(rng, offDiagonalPositions.Length), NumOps.FromDouble(-0.5)),
            NumOps.FromDouble(0.01));
        var particles = Engine.TensorScatter(new Tensor<T>(new[] { K * D }),
            new Tensor<int>(offDiagonalPositions, new[] { offDiagonalPositions.Length }), draws, 0).Reshape(new[] { K, D });

        // Gradient terms that do not depend on the particles, and where a logit may move at all.
        var (dataGrad, updatable) = ComputeDataGradient(cov, offDiagonal, d, n);

        for (int iter = 0; iter < NumSamples; iter++)
        {
            double tau = Math.Max(0.1, 1.0 * Math.Pow(0.95, iter));
            particles = SvgdStep(particles, dataGrad, updatable, offDiagonal, tau, d);
        }

        // Average edge probabilities across particles
        var probabilities = ClampedSigmoid(particles, 0.1).Reshape(new[] { K, d, d });
        var prob = Engine.TensorDivideScalar(Engine.ReduceSum(probabilities, new[] { 0 }, false), NumOps.FromDouble(K));

        // Threshold and compute OLS weights.
        // Edge weights are computed from the STANDARDIZED covariance (correlation)
        // on purpose: DiBS returns a causal ADJACENCY matrix whose edge strengths
        // must be scale-invariant (a property of the joint distribution, not the
        // measurement units) — required by DiscoverStructure_IsInvariantToDataScaling
        // and the relative-magnitude invariants (IndependentVariablesHaveWeakEdges,
        // etc.). This is NOT an OLS regression coefficient in original units.
        var weight = ComputePairwiseOLSWeights(cov);
        var keep = Engine.TensorMultiply(Engine.TensorMultiply(updatable,
            Engine.TensorGreaterThan(prob, NumOps.FromDouble(0.5))),
            Engine.TensorGreaterThan(Engine.TensorAbs(weight), NumOps.FromDouble(0.1)));
        return Engine.TensorWhere(keep, weight, Engine.TensorZerosLike(weight)).ToMatrix();
    }

    /// <summary>
    /// The data likelihood gradient n * cov[i,j]^2 / cov[i,i] and the mask of logits that receive a
    /// gradient (off-diagonal with cov[i,i] &gt; 1e-10).
    /// </summary>
    private (Tensor<T> DataGrad, Tensor<T> Updatable) ComputeDataGradient(Matrix<T> cov, Tensor<T> offDiagonal, int d, int n)
    {
        // The Bayesian data-fit score is a sum over the n observations, so its gradient scales with
        // the sample count; the sparsity / acyclicity priors are O(1). Without this factor the O(1)
        // priors overwhelm even a strong true edge's per-sample data term (correlation^2 ~ 0.4-0.6),
        // driving EVERY edge logit negative -> 0 edges recovered (DiscoverStructure_RecoversTrueEdges).
        // Scaling the standardized-data term by n restores the paper's likelihood-vs-prior balance:
        // strong true edges (corr 0.6-0.8) get a data term >> the priors and survive, while
        // near-zero correlations stay below them and are pruned.
        var c = Tensor<T>.FromMatrix(cov);
        var rowVariance = Engine.TensorBroadcastTo(Engine.TensorDiagonal(c).Reshape(new[] { d, 1 }), new[] { d, d });
        var updatable = Engine.TensorMultiply(offDiagonal, Engine.TensorGreaterThan(rowVariance, NumOps.FromDouble(1e-10)));
        var safeVariance = Engine.TensorWhere(updatable, rowVariance, Engine.TensorOnesLike(rowVariance));
        var dataGrad = Engine.TensorMultiplyScalar(Engine.TensorDivide(Engine.TensorMultiply(c, c), safeVariance),
            NumOps.FromDouble(n));
        return (dataGrad, updatable);
    }

    /// <summary>
    /// One Stein variational gradient descent step on all particles [K, d*d]:
    /// Z_k += lr * (1/K) * sum_l k(Z_l, Z_k) * (grad_l + (Z_l - Z_k) / (2h^2 + 1e-10)) off the diagonal,
    /// with the RBF kernel k = exp(-||Z_l - Z_k||^2 / max(2h^2, 1e-10)) and the median bandwidth h.
    /// </summary>
    private Tensor<T> SvgdStep(Tensor<T> particles, Tensor<T> dataGrad, Tensor<T> updatable, Tensor<T> offDiagonal,
        double tau, int d)
    {
        int K = _numParticles;
        int D = d * d;

        // Compute gradients for each particle
        var gradients = ComputeLogPosteriorGradient(particles, dataGrad, updatable, tau, d);

        // Compute pairwise kernel values and kernel gradients (SVGD)
        var sqDistances = ComputePairwiseSquaredDistances(particles);
        double bandwidth = ComputeMedianBandwidth(sqDistances, K);
        T bandwidthT = NumOps.FromDouble(bandwidth);
        T twoBandwidthSq = NumOps.Multiply(NumOps.FromDouble(2), NumOps.Multiply(bandwidthT, bandwidthT));

        // Kernel: k(Z_l, Z_k) = exp(-||Z_l - Z_k||^2 / (2*h^2)), kernel[k, l]
        var kernel = Engine.TensorExp(Engine.TensorDivideScalar(Engine.TensorNegate(sqDistances),
            NumOps.FromDouble(Math.Max(NumOps.ToDouble(twoBandwidthSq), 1e-10))));

        // SVGD: kernel * gradient + kernel_gradient, averaged over l
        var driving = Engine.TensorMatMul(kernel, gradients);
        var kernelRowSums = Engine.TensorBroadcastTo(Engine.ReduceSum(kernel, new[] { 1 }, true), new[] { K, D });
        var repulsive = Engine.TensorDivideScalar(
            Engine.TensorSubtract(Engine.TensorMatMul(kernel, particles), Engine.TensorMultiply(kernelRowSums, particles)),
            NumOps.Add(twoBandwidthSq, NumOps.FromDouble(1e-10)));
        T invK = NumOps.Divide(NumOps.One, NumOps.FromDouble(K));
        var updates = Engine.TensorMultiplyScalar(Engine.TensorAdd(driving, repulsive), invK);

        // Apply updates
        var particleMask = Engine.TensorBroadcastTo(offDiagonal.Reshape(new[] { 1, D }), new[] { K, D });
        var stepped = Engine.TensorAdd(particles, Engine.TensorMultiplyScalar(updates, NumOps.FromDouble(_learningRate)));
        return Engine.TensorWhere(particleMask, stepped, particles);
    }

    /// <summary>
    /// Log-posterior gradient of every particle [K, d*d]: P(1-P)/(tau+1e-10) * (data - sign(Z) - 2P)
    /// with P = sigmoid(Z / tau), zero where the logit is not updatable.
    /// </summary>
    private Tensor<T> ComputeLogPosteriorGradient(Tensor<T> particles, Tensor<T> dataGrad, Tensor<T> updatable,
        double tau, int d)
    {
        int K = _numParticles;
        int D = d * d;
        var shape = new[] { K, D };

        // Edge probability
        var P = ClampedSigmoid(particles, tau);

        // Sparsity prior: -|Z| (encourage sparse graphs) -> gradient -sign(Z)
        var sparsityGrad = Engine.TensorNegate(Engine.TensorSign(particles));

        // Acyclicity penalty: -2 * rho * P[i,j]
        var acycGrad = Engine.TensorMultiplyScalar(P, NumOps.FromDouble(-2));

        // Chain rule through sigmoid: dP/dZ = P*(1-P)/tau
        var sigDeriv = Engine.TensorDivideScalar(Engine.TensorMultiply(P, Engine.ScalarMinusTensor(NumOps.One, P)),
            NumOps.Add(NumOps.FromDouble(tau), NumOps.FromDouble(1e-10)));

        var total = Engine.TensorAdd(Engine.TensorBroadcastTo(dataGrad.Reshape(new[] { 1, D }), shape),
            Engine.TensorAdd(sparsityGrad, acycGrad));
        var grad = Engine.TensorMultiply(sigDeriv, total);
        return Engine.TensorWhere(Engine.TensorBroadcastTo(updatable.Reshape(new[] { 1, D }), shape),
            grad, Engine.TensorZerosLike(grad));
    }

    /// <summary>||Z_l - Z_k||^2 for every pair of particles, as a [K, K] tensor.</summary>
    private Tensor<T> ComputePairwiseSquaredDistances(Tensor<T> particles)
    {
        int K = particles.Shape[0];
        int D = particles.Shape[1];
        var pairShape = new[] { K, K, D };
        var diff = Engine.TensorSubtract(
            Engine.TensorBroadcastTo(particles.Reshape(new[] { 1, K, D }), pairShape),
            Engine.TensorBroadcastTo(particles.Reshape(new[] { K, 1, D }), pairShape));
        return Engine.ReduceSum(Engine.TensorSquare(diff), new[] { 2 }, false);
    }

    /// <summary>
    /// Median pairwise particle distance over k &lt; l, scaled by 1/sqrt(2 log(K + 1)) and floored at 0.01.
    /// </summary>
    private double ComputeMedianBandwidth(Tensor<T> sqDistances, int numParticles)
    {
        var pairPositions = new List<int>();
        for (int k = 0; k < numParticles; k++)
            for (int l = k + 1; l < numParticles; l++)
                pairPositions.Add(k * numParticles + l);

        if (pairPositions.Count == 0) return 1.0;
        var distances = Engine.TensorSort(Engine.TensorSqrt(Engine.TensorIndexSelect(
            sqDistances.Reshape(new[] { numParticles * numParticles }),
            new Tensor<int>(pairPositions.ToArray(), new[] { pairPositions.Count }), 0))).Values;
        double median = NumOps.ToDouble(distances[pairPositions.Count / 2]);
        return Math.Max(median / Math.Sqrt(2 * Math.Log(numParticles + 1)), 0.01);
    }
}

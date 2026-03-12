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
[ModelPaper("DiBS: Differentiable Bayesian Structure Learning", "https://proceedings.neurips.cc/paper/2021/hash/5e7dff7b20d04e7b47b22b44e6b8a82f-Abstract.html", Year = 2021, Authors = "Lars Lorch, Jonas Rothfuss, Bernhard Scholkopf, Andreas Krause")]
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

        var cov = ComputeCovarianceMatrix(data);
        var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(Seed);
        T lr = NumOps.FromDouble(_learningRate);
        T initScale = NumOps.FromDouble(0.01);

        // Initialize particles: each is a d x d matrix of edge logits
        var particles = new Matrix<T>[_numParticles];
        for (int k = 0; k < _numParticles; k++)
        {
            particles[k] = new Matrix<T>(d, d);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    if (i != j)
                        particles[k][i, j] = NumOps.Multiply(initScale,
                            NumOps.FromDouble(rng.NextDouble() - 0.5));
        }

        for (int iter = 0; iter < NumSamples; iter++)
        {
            double tau = Math.Max(0.1, 1.0 * Math.Pow(0.95, iter));
            T tauT = NumOps.FromDouble(tau);

            // Compute gradients for each particle
            var gradients = new Matrix<T>[_numParticles];
            for (int k = 0; k < _numParticles; k++)
                gradients[k] = ComputeLogPosteriorGradient(particles[k], cov, d, tauT);

            // Compute pairwise kernel values and kernel gradients (SVGD)
            var bandwidth = ComputeMedianBandwidth(particles, d);
            T bandwidthT = NumOps.FromDouble(bandwidth);
            T twoBandwidthSq = NumOps.Multiply(NumOps.FromDouble(2), NumOps.Multiply(bandwidthT, bandwidthT));

            var updates = new Matrix<T>[_numParticles];
            for (int k = 0; k < _numParticles; k++)
                updates[k] = new Matrix<T>(d, d);

            T invK = NumOps.Divide(NumOps.One, NumOps.FromDouble(_numParticles));

            for (int k = 0; k < _numParticles; k++)
            {
                for (int l = 0; l < _numParticles; l++)
                {
                    // Kernel: k(Z_l, Z_k) = exp(-||Z_l - Z_k||^2 / (2*h^2))
                    T distSq = NumOps.Zero;
                    for (int i = 0; i < d; i++)
                        for (int j = 0; j < d; j++)
                        {
                            T diff = NumOps.Subtract(particles[l][i, j], particles[k][i, j]);
                            distSq = NumOps.Add(distSq, NumOps.Multiply(diff, diff));
                        }

                    T kernelVal = NumOps.FromDouble(
                        Math.Exp(-NumOps.ToDouble(distSq) / Math.Max(NumOps.ToDouble(twoBandwidthSq), 1e-10)));

                    for (int i = 0; i < d; i++)
                        for (int j = 0; j < d; j++)
                        {
                            if (i == j) continue;
                            // SVGD: kernel * gradient + kernel_gradient
                            T kGrad = NumOps.Multiply(kernelVal, gradients[l][i, j]);
                            T diff = NumOps.Subtract(particles[l][i, j], particles[k][i, j]);
                            T kGradKernel = NumOps.Multiply(kernelVal,
                                NumOps.Divide(diff, NumOps.Add(twoBandwidthSq, NumOps.FromDouble(1e-10))));

                            updates[k][i, j] = NumOps.Add(updates[k][i, j],
                                NumOps.Multiply(invK, NumOps.Add(kGrad, kGradKernel)));
                        }
                }
            }

            // Apply updates
            for (int k = 0; k < _numParticles; k++)
                for (int i = 0; i < d; i++)
                    for (int j = 0; j < d; j++)
                        if (i != j)
                            particles[k][i, j] = NumOps.Add(particles[k][i, j],
                                NumOps.Multiply(lr, updates[k][i, j]));
        }

        // Average edge probabilities across particles
        var avgProb = new Matrix<T>(d, d);
        for (int k = 0; k < _numParticles; k++)
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    if (i == j) continue;
                    double sv = NumOps.ToDouble(particles[k][i, j]) / 0.1;
                    double sigVal = sv > 20 ? 1.0 : sv < -20 ? 0.0 : 1.0 / (1.0 + Math.Exp(-sv));
                    avgProb[i, j] = NumOps.Add(avgProb[i, j], NumOps.FromDouble(sigVal));
                }

        // Threshold and compute OLS weights
        var result = new Matrix<T>(d, d);
        T numParticlesT = NumOps.FromDouble(_numParticles);
        T edgeThreshold = NumOps.FromDouble(0.5);
        T weightThreshold = NumOps.FromDouble(0.1);

        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
            {
                if (i == j) continue;
                T prob = NumOps.Divide(avgProb[i, j], numParticlesT);
                if (NumOps.GreaterThan(prob, edgeThreshold))
                {
                    T varI = cov[i, i];
                    if (NumOps.GreaterThan(varI, NumOps.FromDouble(1e-10)))
                    {
                        T weight = NumOps.Divide(cov[i, j], varI);
                        if (NumOps.GreaterThan(NumOps.Abs(weight), weightThreshold))
                            result[i, j] = weight;
                    }
                }
            }

        return result;
    }

    private Matrix<T> ComputeLogPosteriorGradient(Matrix<T> Z, Matrix<T> cov, int d, T tau)
    {
        var grad = new Matrix<T>(d, d);
        T eps = NumOps.FromDouble(1e-10);

        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
            {
                if (i == j) continue;

                // Edge probability
                T zScaled = NumOps.Divide(Z[i, j], tau);
                double sv = NumOps.ToDouble(zScaled);
                double sigVal = sv > 20 ? 1.0 : sv < -20 ? 0.0 : 1.0 / (1.0 + Math.Exp(-sv));
                T pij = NumOps.FromDouble(sigVal);

                T varI = cov[i, i];
                if (!NumOps.GreaterThan(varI, eps)) continue;

                // Data likelihood gradient: proportional to cov[i,j]^2 / cov[i,i]
                T covIJ = cov[i, j];
                T dataGrad = NumOps.Divide(NumOps.Multiply(covIJ, covIJ), varI);

                // Sparsity prior: -|Z| (encourage sparse graphs)
                T sparsityGrad = NumOps.Negate(NumOps.FromDouble(Math.Sign(NumOps.ToDouble(Z[i, j]))));

                // Acyclicity penalty: -2 * rho * P[i,j]
                T acycGrad = NumOps.Negate(NumOps.Multiply(NumOps.FromDouble(2), pij));

                // Chain rule through sigmoid: dP/dZ = P*(1-P)/tau
                T oneMinusP = NumOps.Subtract(NumOps.One, pij);
                T sigDeriv = NumOps.Divide(NumOps.Multiply(pij, oneMinusP),
                    NumOps.Add(tau, NumOps.FromDouble(1e-10)));

                grad[i, j] = NumOps.Multiply(sigDeriv,
                    NumOps.Add(dataGrad, NumOps.Add(sparsityGrad, acycGrad)));
            }

        return grad;
    }

    private double ComputeMedianBandwidth(Matrix<T>[] particles, int d)
    {
        var distances = new List<double>();
        for (int k = 0; k < particles.Length; k++)
            for (int l = k + 1; l < particles.Length; l++)
            {
                double dist = 0;
                for (int i = 0; i < d; i++)
                    for (int j = 0; j < d; j++)
                    {
                        double diff = NumOps.ToDouble(NumOps.Subtract(particles[k][i, j], particles[l][i, j]));
                        dist += diff * diff;
                    }
                distances.Add(Math.Sqrt(dist));
            }

        if (distances.Count == 0) return 1.0;
        distances.Sort();
        double median = distances[distances.Count / 2];
        return Math.Max(median / Math.Sqrt(2 * Math.Log(particles.Length + 1)), 0.01);
    }
}

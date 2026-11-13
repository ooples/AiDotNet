using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.DeepLearning;

/// <summary>
/// CGNN — Causal Generative Neural Networks.
/// </summary>
/// <remarks>
/// <para>
/// CGNN generates data according to a causal model parameterized by neural networks.
/// For each candidate edge (i→j), a generative model f_j(parents(j), noise) is trained.
/// The model quality is measured by Maximum Mean Discrepancy (MMD) between generated
/// and observed data. Edges are scored pairwise: the direction with lower MMD indicates
/// the causal direction.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>Initialize from correlation-based skeleton</item>
/// <item>For each pair (i,j) with non-zero correlation:</item>
/// <item>  Train MLP f: x_i + noise → x_j, compute MMD(generated_j, real_j)</item>
/// <item>  Train MLP g: x_j + noise → x_i, compute MMD(generated_i, real_i)</item>
/// <item>  If MMD(f) &lt; MMD(g), edge is i→j; otherwise j→i</item>
/// <item>Compute OLS weights for the oriented edges</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> CGNN tests different causal graph candidates by asking "If this
/// graph were correct, could a neural network generate data that looks like the real data?"
/// The graph that produces the most realistic synthetic data is chosen as the answer.
/// </para>
/// <para>
/// Reference: Goudet et al. (2018), "Learning Functional Causal Models with Generative
/// Neural Networks", Explainable and Interpretable Models in CV and ML.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ResearchPaper("Learning Functional Causal Models with Generative Neural Networks", "https://doi.org/10.1007/978-3-030-28954-6_3", Year = 2018, Authors = "Olivier Goudet, Diviyan Kalainathan, Philippe Caillou, Isabelle Guyon, David Lopez-Paz, Michele Sebag")]
public class CGNNAlgorithm<T> : DeepCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "CGNN";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;
    private readonly int _seed;

    /// <summary>
    /// Seed used when the caller does not supply one, so that a run is reproducible by default.
    /// </summary>
    /// <remarks>
    /// Previously this fell back to an unseeded secure RNG, which made the model nondeterministic
    /// and its generated model-family tests flaky: the identical test binary produced 13 failures
    /// in one run and 8 in the next over the same 42 tests, which makes regression detection on
    /// this path impossible. Callers still override via the options' Seed property. Matches the
    /// fixed-seed convention already used by DAGMANonlinear in this module.
    /// </remarks>
    private const int DefaultRandomSeed = 42;
    public CGNNAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyDeepOptions(options);
        _seed = options?.Seed ?? DefaultRandomSeed;
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        int h = HiddenUnits;
        if (n < 3 || d < 2) return new Matrix<T>(d, d);

        var cov = ComputeCovarianceMatrix(data);
        var corr = CovarianceToCorrelation(cov);
        var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(_seed);

        var candidates = new List<(int From, int To, T Weight, double Confidence)>();

        // For each pair with significant correlation, determine direction via MMD
        for (int i = 0; i < d; i++)
            for (int j = i + 1; j < d; j++)
            {
                var candidate = OrientPair(data, cov, corr, i, j, h, n, rng);
                if (candidate.HasValue)
                    candidates.Add(candidate.Value);
            }

        // Pairwise MMD preferences are local: three individually preferred directions can still form
        // A->B->C->A. CGNN's output contract is a DAG, so assemble the graph globally in descending
        // orientation confidence and omit a weaker edge when it would close a directed cycle. The
        // deterministic tie-break also makes equal-score projections reproducible.
        candidates.Sort((a, b) =>
        {
            int byConfidence = b.Confidence.CompareTo(a.Confidence);
            if (byConfidence != 0) return byConfidence;
            int byFrom = a.From.CompareTo(b.From);
            return byFrom != 0 ? byFrom : a.To.CompareTo(b.To);
        });

        // Every candidate weight exceeds 0.1 in magnitude, so an accepted edge is always a live edge
        // for the cycle test.
        var accepted = new bool[d, d];

        bool CreatesCycle(int from, int to)
        {
            var seen = new bool[d];
            var pending = new Stack<int>();
            pending.Push(to);

            while (pending.Count > 0)
            {
                int node = pending.Pop();
                if (node == from) return true;
                if (seen[node]) continue;

                seen[node] = true;
                for (int next = 0; next < d; next++)
                {
                    if (next != node && accepted[node, next])
                        pending.Push(next);
                }

            }

            return false;
        }

        var positions = new List<int>();
        var weights = new List<T>();
        foreach (var candidate in candidates)
        {
            if (!CreatesCycle(candidate.From, candidate.To))
            {
                accepted[candidate.From, candidate.To] = true;
                positions.Add(candidate.From * d + candidate.To);
                weights.Add(candidate.Weight);
            }
        }

        var result = new Tensor<T>(new[] { d * d });
        if (positions.Count > 0)
        {
            result = Engine.TensorScatter(result, new Tensor<int>(positions.ToArray(), new[] { positions.Count }),
                new Tensor<T>(weights.ToArray(), new[] { weights.Count }), 0);
        }

        return result.Reshape(new[] { d, d }).ToMatrix();
    }

    /// <summary>
    /// For a pair whose |correlation| exceeds EdgeThreshold, trains both directions and orients the edge
    /// toward the lower MMD; returns the edge with its OLS weight and |MMD difference| confidence, or
    /// null when the pair is uncorrelated, the source variance is not above 1e-10, or |weight| &lt;= 0.1.
    /// </summary>
    private (int From, int To, T Weight, double Confidence)? OrientPair(Matrix<T> data, Matrix<T> cov, Matrix<T> corr,
        int i, int j, int h, int n, Random rng)
    {
        T absCorr = NumOps.Abs(corr[i, j]);
        if (!NumOps.GreaterThan(absCorr, NumOps.FromDouble(EdgeThreshold))) return null;

        // Train MLP i→j and compute MMD
        T mmdIJ = TrainAndComputeMMD(data, i, j, h, n, rng);
        // Train MLP j→i and compute MMD
        T mmdJI = TrainAndComputeMMD(data, j, i, h, n, rng);

        // Direction with lower MMD wins
        int from, to;
        if (NumOps.GreaterThan(mmdJI, mmdIJ))
        {
            (from, to) = (i, j);
        }
        else
        {
            (from, to) = (j, i);
        }

        T varFrom = cov[from, from];
        if (!NumOps.GreaterThan(varFrom, NumOps.FromDouble(1e-10))) return null;

        T olsWeight = NumOps.Divide(cov[from, to], varFrom);
        if (!NumOps.GreaterThan(NumOps.Abs(olsWeight), NumOps.FromDouble(0.1))) return null;

        double confidence = Math.Abs(NumOps.ToDouble(mmdIJ) - NumOps.ToDouble(mmdJI));
        if (double.IsNaN(confidence) || double.IsInfinity(confidence)) confidence = 0.0;
        return (from, to, olsWeight, confidence);
    }

    /// <summary>
    /// Trains a small MLP to predict target from source+noise and returns the distribution
    /// distance (mean + variance discrepancy) between predicted and actual target values.
    /// </summary>
    private T TrainAndComputeMMD(Matrix<T> data, int source, int target, int h, int n, Random rng)
    {
        int inputDim = 2; // source + noise
        double scale = Math.Sqrt(2.0 / inputDim);

        // MLP: [source, noise] -> hidden(h) -> target. Draw order per hidden unit k: wh[0,k], wh[1,k], wo[k].
        var draws = Engine.TensorMultiplyScalar(
            Engine.TensorAddScalar(Tensor<T>.CreateRandom(rng, h, 3), NumOps.FromDouble(-0.5)),
            NumOps.FromDouble(scale));
        var wh = Engine.TensorTranspose(Engine.TensorSlice(draws, new[] { 0, 0 }, new[] { h, 2 }));   // [2,h]
        var wo = Engine.TensorSlice(draws, new[] { 0, 2 }, new[] { h, 1 });                             // [h,1]

        var sourceColumn = Engine.GetColumn(data, source).AsTensor();
        var targetColumn = Engine.GetColumn(data, target).AsTensor().Reshape(new[] { n, 1 });

        for (int step = 0; step < MaxEpochs; step++)
            (wh, wo) = MmdTrainingStep(sourceColumn, targetColumn, wh, wo, n, rng);

        // Compute distribution distance: ||mean(predicted) - mean(actual)||^2 + |var(predicted) - var(actual)|
        var (predictions, _) = ForwardMLP(WithNoise(sourceColumn, n, rng), wh, wo);
        T nT = NumOps.FromDouble(n);
        T meanPred = NumOps.Divide(Engine.TensorSum(predictions), nT);
        T meanActual = NumOps.Divide(Engine.TensorSum(targetColumn), nT);

        T varPred = Engine.TensorSum(Engine.TensorSquare(Engine.TensorAddScalar(predictions, NumOps.Negate(meanPred))));
        T varActual = Engine.TensorSum(Engine.TensorSquare(Engine.TensorAddScalar(targetColumn, NumOps.Negate(meanActual))));

        T meanDiff = NumOps.Subtract(meanPred, meanActual);
        T varDiff = NumOps.Subtract(NumOps.Divide(varPred, nT), NumOps.Divide(varActual, nT));
        return NumOps.Add(NumOps.Multiply(meanDiff, meanDiff), NumOps.Abs(varDiff));
    }

    /// <summary>[n, 2] inputs [source, noise] with one fresh noise draw u*2 - 1 per sample, in sample order.</summary>
    private Tensor<T> WithNoise(Tensor<T> sourceColumn, int n, Random rng)
    {
        var noise = Engine.TensorAddScalar(
            Engine.TensorMultiplyScalar(Tensor<T>.CreateRandom(rng, n), NumOps.FromDouble(2)), NumOps.FromDouble(-1));
        return Engine.TensorStack(new[] { sourceColumn, noise }, 1);
    }

    /// <summary>One full-batch gradient step on the MSE of the [source, noise] -> target MLP.</summary>
    private (Tensor<T> Wh, Tensor<T> Wo) MmdTrainingStep(Tensor<T> sourceColumn, Tensor<T> targetColumn,
        Tensor<T> wh, Tensor<T> wo, int n, Random rng)
    {
        var inputs = WithNoise(sourceColumn, n, rng);
        var (pred, hidden) = ForwardMLP(inputs, wh, wo);

        T invN = NumOps.FromDouble(1.0 / n);
        var residual = Engine.TensorMultiplyScalar(Engine.TensorSubtract(pred, targetColumn), invN); // [n,1]

        var gWo = Engine.TensorMatMul(Engine.TensorTranspose(hidden), residual);                      // [h,1]
        var sigD = Engine.TensorMultiply(hidden, Engine.ScalarMinusTensor(NumOps.One, hidden));
        var dH = Engine.TensorMultiply(Engine.TensorMatMul(residual, Engine.TensorTranspose(wo)), sigD); // [n,h]
        var gWh = Engine.TensorMatMul(Engine.TensorTranspose(inputs), dH);                           // [2,h]

        T lr = NumOps.FromDouble(LearningRate);
        return (Engine.TensorSubtract(wh, Engine.TensorMultiplyScalar(gWh, lr)),
            Engine.TensorSubtract(wo, Engine.TensorMultiplyScalar(gWo, lr)));
    }

    /// <summary>
    /// Forward pass through the 2-input MLP for every sample: sigmoid hidden layer (saturated beyond ±20),
    /// linear output.
    /// </summary>
    private (Tensor<T> Prediction, Tensor<T> Hidden) ForwardMLP(Tensor<T> inputs, Tensor<T> wh, Tensor<T> wo)
    {
        var hidden = ClampedSigmoid(Engine.TensorMatMul(inputs, wh), 1.0);
        return (Engine.TensorMatMul(hidden, wo), hidden);
    }
}

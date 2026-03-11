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
[ModelPaper("Learning Functional Causal Models with Generative Neural Networks", "https://doi.org/10.1007/978-3-030-28954-6_3", Year = 2018, Authors = "Olivier Goudet, Diviyan Kalainathan, Philippe Caillou, Isabelle Guyon, David Lopez-Paz, Michele Sebag")]
public class CGNNAlgorithm<T> : DeepCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "CGNN";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    public CGNNAlgorithm(CausalDiscoveryOptions? options = null) { ApplyDeepOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        int h = Math.Min(HiddenUnits, 10);
        if (n < 3 || d < 2) return new Matrix<T>(d, d);

        var cov = ComputeCovarianceMatrix(data);
        var corr = CovarianceToCorrelation(cov);
        var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(42);
        T corrThreshold = NumOps.FromDouble(0.1);
        T eps = NumOps.FromDouble(1e-10);

        var result = new Matrix<T>(d, d);

        // For each pair with significant correlation, determine direction via MMD
        for (int i = 0; i < d; i++)
            for (int j = i + 1; j < d; j++)
            {
                T absCorr = NumOps.Abs(corr[i, j]);
                if (!NumOps.GreaterThan(absCorr, corrThreshold)) continue;

                // Train MLP i→j and compute MMD
                T mmdIJ = TrainAndComputeMMD(data, i, j, h, n, rng);
                // Train MLP j→i and compute MMD
                T mmdJI = TrainAndComputeMMD(data, j, i, h, n, rng);

                // Direction with lower MMD wins
                T varFrom, olsWeight;
                if (NumOps.GreaterThan(mmdJI, mmdIJ))
                {
                    // i→j
                    varFrom = cov[i, i];
                    if (NumOps.GreaterThan(varFrom, eps))
                    {
                        olsWeight = NumOps.Divide(cov[i, j], varFrom);
                        if (NumOps.GreaterThan(NumOps.Abs(olsWeight), NumOps.FromDouble(0.1)))
                            result[i, j] = olsWeight;
                    }
                }
                else
                {
                    // j→i
                    varFrom = cov[j, j];
                    if (NumOps.GreaterThan(varFrom, eps))
                    {
                        olsWeight = NumOps.Divide(cov[j, i], varFrom);
                        if (NumOps.GreaterThan(NumOps.Abs(olsWeight), NumOps.FromDouble(0.1)))
                            result[j, i] = olsWeight;
                    }
                }
            }

        return result;
    }

    /// <summary>
    /// Trains a small MLP to predict target from source+noise and returns the MMD
    /// between predicted and actual target distributions.
    /// </summary>
    private T TrainAndComputeMMD(Matrix<T> data, int source, int target, int h, int n, Random rng)
    {
        T scale = NumOps.FromDouble(Math.Sqrt(2.0 / 2)); // 2 inputs: source + noise
        T lr = NumOps.FromDouble(LearningRate);

        // MLP: [source, noise] → hidden(h) → target
        var wh = new Matrix<T>(2, h);
        var wo = new Matrix<T>(h, 1);
        for (int k = 0; k < h; k++)
        {
            wh[0, k] = NumOps.Multiply(scale, NumOps.FromDouble(rng.NextDouble() - 0.5));
            wh[1, k] = NumOps.Multiply(scale, NumOps.FromDouble(rng.NextDouble() - 0.5));
            wo[k, 0] = NumOps.Multiply(scale, NumOps.FromDouble(rng.NextDouble() - 0.5));
        }

        int trainSteps = Math.Min(MaxEpochs, 50);
        T invN = NumOps.FromDouble(1.0 / n);

        for (int step = 0; step < trainSteps; step++)
        {
            var gWh = new Matrix<T>(2, h);
            var gWo = new Matrix<T>(h, 1);

            for (int s = 0; s < n; s++)
            {
                T noise = NumOps.FromDouble(rng.NextDouble() * 2 - 1);
                var hidden = new T[h];

                for (int k = 0; k < h; k++)
                {
                    T z = NumOps.Add(NumOps.Multiply(data[s, source], wh[0, k]),
                                     NumOps.Multiply(noise, wh[1, k]));
                    double sv = NumOps.ToDouble(z);
                    hidden[k] = NumOps.FromDouble(sv > 20 ? 1.0 : sv < -20 ? 0.0 : 1.0 / (1.0 + Math.Exp(-sv)));
                }

                T pred = NumOps.Zero;
                for (int k = 0; k < h; k++)
                    pred = NumOps.Add(pred, NumOps.Multiply(hidden[k], wo[k, 0]));

                T residual = NumOps.Multiply(NumOps.Subtract(pred, data[s, target]), invN);

                for (int k = 0; k < h; k++)
                {
                    gWo[k, 0] = NumOps.Add(gWo[k, 0], NumOps.Multiply(residual, hidden[k]));
                    T sigD = NumOps.Multiply(hidden[k], NumOps.Subtract(NumOps.One, hidden[k]));
                    T dH = NumOps.Multiply(residual, NumOps.Multiply(wo[k, 0], sigD));
                    gWh[0, k] = NumOps.Add(gWh[0, k], NumOps.Multiply(dH, data[s, source]));
                    gWh[1, k] = NumOps.Add(gWh[1, k], NumOps.Multiply(dH, noise));
                }
            }

            for (int k = 0; k < h; k++)
            {
                wh[0, k] = NumOps.Subtract(wh[0, k], NumOps.Multiply(lr, gWh[0, k]));
                wh[1, k] = NumOps.Subtract(wh[1, k], NumOps.Multiply(lr, gWh[1, k]));
                wo[k, 0] = NumOps.Subtract(wo[k, 0], NumOps.Multiply(lr, gWo[k, 0]));
            }
        }

        // Compute MMD: ||mean(predicted) - mean(actual)||^2 + |var(predicted) - var(actual)|
        T meanPred = NumOps.Zero, meanActual = NumOps.Zero;
        T varPred = NumOps.Zero, varActual = NumOps.Zero;
        T nT = NumOps.FromDouble(n);

        for (int s = 0; s < n; s++)
        {
            T noise = NumOps.FromDouble(rng.NextDouble() * 2 - 1);
            var hidden = new T[h];
            for (int k = 0; k < h; k++)
            {
                T z = NumOps.Add(NumOps.Multiply(data[s, source], wh[0, k]),
                                 NumOps.Multiply(noise, wh[1, k]));
                double sv = NumOps.ToDouble(z);
                hidden[k] = NumOps.FromDouble(sv > 20 ? 1.0 : sv < -20 ? 0.0 : 1.0 / (1.0 + Math.Exp(-sv)));
            }

            T pred = NumOps.Zero;
            for (int k = 0; k < h; k++)
                pred = NumOps.Add(pred, NumOps.Multiply(hidden[k], wo[k, 0]));

            meanPred = NumOps.Add(meanPred, pred);
            meanActual = NumOps.Add(meanActual, data[s, target]);
        }

        meanPred = NumOps.Divide(meanPred, nT);
        meanActual = NumOps.Divide(meanActual, nT);

        for (int s = 0; s < n; s++)
        {
            T noise = NumOps.FromDouble(rng.NextDouble() * 2 - 1);
            var hidden = new T[h];
            for (int k = 0; k < h; k++)
            {
                T z = NumOps.Add(NumOps.Multiply(data[s, source], wh[0, k]),
                                 NumOps.Multiply(noise, wh[1, k]));
                double sv = NumOps.ToDouble(z);
                hidden[k] = NumOps.FromDouble(sv > 20 ? 1.0 : sv < -20 ? 0.0 : 1.0 / (1.0 + Math.Exp(-sv)));
            }

            T pred = NumOps.Zero;
            for (int k = 0; k < h; k++)
                pred = NumOps.Add(pred, NumOps.Multiply(hidden[k], wo[k, 0]));

            T dpred = NumOps.Subtract(pred, meanPred);
            varPred = NumOps.Add(varPred, NumOps.Multiply(dpred, dpred));
            T dact = NumOps.Subtract(data[s, target], meanActual);
            varActual = NumOps.Add(varActual, NumOps.Multiply(dact, dact));
        }

        // MMD approximation
        T meanDiff = NumOps.Subtract(meanPred, meanActual);
        T varDiff = NumOps.Subtract(NumOps.Divide(varPred, nT), NumOps.Divide(varActual, nT));
        return NumOps.Add(NumOps.Multiply(meanDiff, meanDiff), NumOps.Abs(varDiff));
    }
}

using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.TimeSeries;

/// <summary>
/// PCMCI+ — extension of PCMCI that also discovers contemporaneous causal links.
/// </summary>
/// <remarks>
/// <para>
/// PCMCI+ extends PCMCI to handle both lagged AND contemporaneous (same time-step)
/// causal links by adding a skeleton discovery and orientation step for lag-0 effects.
/// It applies the same Fisher z-test based conditional independence testing at lag 0
/// conditioned on the lagged parents discovered by PCMCI.
/// </para>
/// <para>
/// <b>For Beginners:</b> PCMCI only finds "yesterday's X causes today's Y" relationships.
/// PCMCI+ also finds "today's X causes today's Y" relationships, which are important
/// when variables influence each other faster than the measurement interval.
/// </para>
/// <para>
/// Reference: Runge (2020), "Discovering Contemporaneous and Lagged Causal Relations
/// in Autocorrelated Nonlinear Time Series Datasets", UAI.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ResearchPaper("Discovering Contemporaneous and Lagged Causal Relations in Autocorrelated Nonlinear Time Series Datasets", "https://proceedings.mlr.press/v124/runge20a.html", Year = 2020, Authors = "Jakob Runge")]
public class PCMCIPlusAlgorithm<T> : TimeSeriesCausalBase<T>
{
    private double _alpha = 0.05;

    /// <inheritdoc/>
    public override string Name => "PCMCI+";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public PCMCIPlusAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyTimeSeriesOptions(options);
        if (options?.SignificanceLevel.HasValue == true) _alpha = options.SignificanceLevel.Value;
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;

        int effectiveN = n - MaxLag;
        if (effectiveN < 2 * d) return new Matrix<T>(d, d);

        // Step 1: Run PCMCI to get lagged causal structure
        var pcmci = new PCMCIAlgorithm<T>(new CausalDiscoveryOptions
        {
            SignificanceLevel = _alpha,
            MaxIterations = MaxLag
        });
        var laggedGraph = pcmci.DiscoverStructure(data);

        // Lagged edges start the result; parentsOf[v] lists (ascending) every k with a nonzero lagged edge k -> v
        var lagged = Tensor<T>.FromMatrix(laggedGraph.AdjacencyMatrix);
        var hasLaggedEdge = Engine.TensorGreaterThan(Engine.TensorAbs(lagged), NumOps.Zero);
        var parentsOf = new List<int>[d];
        for (int v = 0; v < d; v++) parentsOf[v] = new List<int>();
        foreach (int position in Engine.TensorNonzero(Engine.TensorTranspose(hasLaggedEdge).Reshape(new[] { d * d })).ToArray())
            parentsOf[position / d].Add(position % d);

        // Contemporaneous strengths, applied where they exceed the lagged weight already there
        var contemporaneous = new Dictionary<int, double>();

        // Step 2: Contemporaneous discovery via conditional independence testing
        // Condition on the lagged parents of both variables (discovered from PCMCI)
        int offset = n - effectiveN;
        for (int i = 0; i < d; i++)
        {
            for (int j = i + 1; j < d; j++)
            {
                // Build conditioning set from lagged parents of both i and j
                var condVarIndices = new SortedSet<int>(parentsOf[i].Concat(parentsOf[j])).ToList();

                double partCorr;
                if (condVarIndices.Count == 0)
                {
                    // No lagged parents: use simple correlation over effective sample window
                    partCorr = ComputeWindowCorrelation(data, effectiveN, offset, i, j);
                }
                else
                {
                    // Residualize both variables on their lagged parents
                    partCorr = ComputeContemporaneousPartialCorrelation(
                        data, i, j, condVarIndices, effectiveN, offset);
                }

                double pValue = FisherZTestPValue(partCorr, effectiveN, condVarIndices.Count);

                if (pValue <= _alpha)
                {
                    // Significant contemporaneous link — orient using causal ordering heuristic
                    int parentsOfI = parentsOf[i].Count, parentsOfJ = parentsOf[j].Count;

                    // Each pair is visited once, so the weight already at [i, j] / [j, i] is the lagged one.
                    if (parentsOfI <= parentsOfJ)
                        contemporaneous[i * d + j] = Math.Abs(partCorr);
                    else
                        contemporaneous[j * d + i] = Math.Abs(partCorr);
                }
            }
        }

        var strength = Tensor<T>.FromMatrix(CausalTensorHelpers.FromStatistics<T>(d, d, contemporaneous));
        var tested = Tensor<T>.FromMatrix(CausalTensorHelpers.FromStatistics<T>(d, d,
            contemporaneous.ToDictionary(link => link.Key, _ => 1.0)));
        var replace = Engine.TensorMultiply(tested, Engine.TensorGreaterThan(strength, lagged));
        return Engine.TensorWhere(replace, strength, lagged).ToMatrix();
    }

    /// <summary>
    /// Computes partial correlation between variables i(t) and j(t) at lag 0,
    /// conditioned on lagged parents via OLS residualization.
    /// </summary>
    private double ComputeContemporaneousPartialCorrelation(
        Matrix<T> data, int i, int j,
        List<int> condVarIndices, int effectiveN, int offset)
    {
        var x = Tensor<T>.FromMatrix(data);

        // Build conditioning matrix from lag-1 values of conditioning variables (rows outside the data are 0)
        var condMatrix = Engine.TensorConcatenate(condVarIndices
            .Select(cVar => LaggedSeries(x, cVar, 1, offset, effectiveN).Reshape(new[] { effectiveN, 1 }))
            .ToArray(), 1);

        // Build target vectors at lag 0
        var iVals = LaggedSeries(x, i, 0, offset, effectiveN);
        var jVals = LaggedSeries(x, j, 0, offset, effectiveN);

        // Residualize both on conditioning set (ridge 1e-10, LU)
        var residI = OlsResidualSeries(condMatrix, iVals, false, 1e-10, MatrixDecompositionType.Lu);
        var residJ = OlsResidualSeries(condMatrix, jVals, false, 1e-10, MatrixDecompositionType.Lu);

        return SeriesCorrelation(residI, residJ);
    }

    private double ComputeWindowCorrelation(Matrix<T> data, int count, int offset, int i, int j)
    {
        var x = Tensor<T>.FromMatrix(data);
        return SeriesCorrelation(LaggedSeries(x, i, 0, offset, count), LaggedSeries(x, j, 0, offset, count));
    }

    private static double FisherZTestPValue(double partialCorrelation, int n, int condSetSize)
    {
        int dof = n - condSetSize - 3;
        if (dof < 1) return 1.0;

        double r = Math.Max(-0.9999, Math.Min(0.9999, partialCorrelation));
        double z = 0.5 * Math.Log((1 + r) / (1 - r));
        double testStat = Math.Abs(z) * Math.Sqrt(dof);

        return 2.0 * NormalCdfComplement(testStat);
    }

    private static double NormalCdfComplement(double x)
    {
        if (x < 0) return 1.0 - NormalCdfComplement(-x);

        const double p = 0.2316419;
        const double b1 = 0.319381530;
        const double b2 = -0.356563782;
        const double b3 = 1.781477937;
        const double b4 = -1.821255978;
        const double b5 = 1.330274429;

        double t = 1.0 / (1.0 + p * x);
        double t2 = t * t;
        double t3 = t2 * t;
        double t4 = t3 * t;
        double t5 = t4 * t;

        double phi = Math.Exp(-0.5 * x * x) / Math.Sqrt(2.0 * Math.PI);
        return phi * (b1 * t + b2 * t2 + b3 * t3 + b4 * t4 + b5 * t5);
    }
}

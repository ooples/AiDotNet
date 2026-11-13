using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.TimeSeries;

/// <summary>
/// CCM — Convergent Cross-Mapping for detecting causation in nonlinear dynamical systems.
/// </summary>
/// <remarks>
/// <para>
/// CCM is based on Takens' theorem from dynamical systems theory. If X causes Y, then
/// the shadow manifold reconstructed from Y should contain information about X, and
/// cross-mapping accuracy should improve (converge) with longer time series.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>For each pair (i,j), construct delay embedding M_j from variable j with lag tau and dimension E</item>
/// <item>For each point in M_j, find E+1 nearest neighbors in M_j</item>
/// <item>Compute simplex weights from distances: w_k = exp(-d_k / d_1)</item>
/// <item>Cross-map: predict x_i(t) as weighted combination of x_i at neighbor times</item>
/// <item>Compute correlation rho between predicted and actual x_i</item>
/// <item>If rho converges (improves) with increasing library size L, j cross-maps i → i causes j</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> CCM tests causation by checking whether one variable's history
/// can "predict" another variable using nearest-neighbor reconstruction in delay-coordinate
/// space. Crucially, if X causes Y, then Y's history cross-maps to X (not the other way),
/// which is the opposite of Granger causality's logic.
/// </para>
/// <para>
/// Reference: Sugihara et al. (2012), "Detecting Causality in Complex Ecosystems", Science.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ResearchPaper("Detecting Causality in Complex Ecosystems", "https://doi.org/10.1126/science.1227079", Year = 2012, Authors = "George Sugihara, Robert May, Hao Ye, Chih-hao Hsieh, Ethan Deyle, Michael Fogarty, Stephan Munch")]
public class CCMAlgorithm<T> : TimeSeriesCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "CCM";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    private readonly double _convergenceThreshold;
    private readonly double _correlationThreshold;
    private readonly double _directionalityAsymmetryThreshold;

    public CCMAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyTimeSeriesOptions(options);
        _convergenceThreshold = options?.EdgeThreshold ?? 0.05;
        _correlationThreshold = options?.CorrelationThreshold ?? 0.1;
        _directionalityAsymmetryThreshold = options?.DirectionalityAsymmetryThreshold ?? 0.2;
        if (double.IsNaN(_convergenceThreshold) || _convergenceThreshold < 0 || _convergenceThreshold > 1)
            throw new ArgumentException("EdgeThreshold (convergence threshold) must be a finite number between 0 and 1.");
        if (double.IsNaN(_correlationThreshold) || _correlationThreshold < 0 || _correlationThreshold > 1)
            throw new ArgumentException("CorrelationThreshold must be a finite number between 0 and 1.");
        if (double.IsNaN(_directionalityAsymmetryThreshold) || _directionalityAsymmetryThreshold < 0 || _directionalityAsymmetryThreshold > 1)
            throw new ArgumentException("DirectionalityAsymmetryThreshold must be a finite number between 0 and 1.");
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        // Use MaxLag + 1 as embedding dimension (uncapped)
        int embDim = MaxLag + 1;
        int tau = 1;
        int minLib = embDim + 2;
        // The half-library convergence test requires validN/2 >= minLib,
        // so validN >= 2*minLib, meaning n >= 2*minLib + (embDim-1)*tau
        int minSamples = 2 * minLib + (embDim - 1) * tau;

        if (d < 2)
            throw new ArgumentException($"CCM requires at least 2 variables, got {d}.");
        if (n < minSamples)
            throw new ArgumentException($"CCM requires at least {minSamples} samples for embedding dimension {embDim}, got {n}.");

        int validN = n - (embDim - 1) * tau;
        var x = Tensor<T>.FromMatrix(data);
        // Targets aligned with the embedding rows: row t of the embedding is time t + (embDim - 1) * tau
        var targets = Engine.TensorSlice(x, new[] { (embDim - 1) * tau, 0 }, new[] { validN, d });
        T zero = NumOps.Zero;
        T convergenceThreshold = NumOps.FromDouble(_convergenceThreshold);
        T correlationThreshold = NumOps.FromDouble(_correlationThreshold);
        T saturation = NumOps.FromDouble(0.95);
        T degradationTolerance = NumOps.FromDouble(-1e-3);

        var columns = new Tensor<T>[d];
        for (int j = 0; j < d; j++)
        {
            // Build delay embedding from variable j: embeddings[t, e] = x_j(t + (embDim - 1 - e) * tau)
            var column = Engine.TensorSliceAxis(x, 1, j);
            var embeddings = Engine.TensorConcatenate(Enumerable.Range(0, embDim)
                .Select(e => Engine.TensorSlice(column, new[] { (embDim - 1 - e) * tau }, new[] { validN }).Reshape(new[] { validN, 1 }))
                .ToArray(), 1);

            // CCM convergence (Sugihara et al. 2012, Science): an edge is accepted
            // when cross-map skill ρ INCREASES as the library grows from a
            // small size toward the full series and plateaus at a high value. The
            // signature is measured from a SMALL fixed library (minLib) to the
            // full library — not half-vs-full, where both endpoints scale with n and
            // already sit on ρ's plateau, so the measured gain vanishes for large n.
            // The manifold of j cross-maps every target i at once (ρ for each column of targets).
            var rhoSmall = ComputeCrossMapCorrelation(targets, embeddings, minLib, embDim);
            var rhoFull = ComputeCrossMapCorrelation(targets, embeddings, validN, embDim);

            var convergence = Engine.TensorSubtract(rhoFull, rhoSmall);
            var convergent = Engine.TensorMultiply(Engine.TensorGreaterThan(convergence, convergenceThreshold),
                Engine.TensorGreaterThan(rhoFull, correlationThreshold));

            // Already-saturated regime: on deterministic or near-deterministic
            // systems, the cross-mapping reaches its ceiling at the minimum
            // library size already, so ρ_full - ρ_small ≈ 0 even though the
            // causal coupling is overwhelmingly strong. Sugihara et al. 2012
            // §SI.2 ("Convergence with library size") explicitly notes that
            // for highly-coupled systems ρ saturates near 1 well below the
            // full library; the convergence test is informative for noisy
            // partial couplings, not for already-perfect couplings. Accept
            // the edge when both ρ_small and ρ_full are above 0.95 AND
            // ρ_full ≥ ρ_small (no degradation from increased library size).
            var notDegraded = Engine.TensorAdd(Engine.TensorGreaterThan(convergence, degradationTolerance),
                Engine.TensorEquals(convergence, degradationTolerance));
            var saturatedConvergence = Engine.TensorMultiply(Engine.TensorMultiply(
                Engine.TensorGreaterThan(rhoSmall, saturation), Engine.TensorGreaterThan(rhoFull, saturation)), notDegraded);

            var accept = Engine.TensorGreaterThan(Engine.TensorAdd(convergent, saturatedConvergence), zero);
            columns[j] = Engine.TensorWhere(accept, rhoFull, Engine.TensorZerosLike(rhoFull));
        }

        // result[i, j] = ρ of target i cross-mapped from the manifold of j (never i == j)
        var raw = Engine.TensorStack(columns, 1);
        var offDiagonal = OffDiagonalMask(d);
        var result = Engine.TensorWhere(offDiagonal, raw, Engine.TensorZerosLike(raw));

        // CCM directionality (Sugihara et al. 2012, Science, "Detecting Causality in Complex
        // Ecosystems"): causation is inferred from the ASYMMETRY of convergent cross-map skill — X
        // drives Y when Y's reconstructed manifold cross-maps X better than the reverse. The loop
        // above fills BOTH directions with their raw cross-map skill; convert those raw skills into an
        // inferred causal graph by keeping only the DOMINANT direction of each clearly-asymmetric pair
        // (the weaker reverse mapping is a reconstruction artifact, not a second causal link). Genuine
        // bidirectional coupling — where the two skills are comparable — is left intact (the paper
        // reports mutual causation in exactly that near-symmetric regime).
        // For a pair i < j with forward skill |r_ij| and backward |r_ji|, both positive and asymmetry
        // |fwd − bwd| / (max + 1e-10) at least the threshold, the backward entry is cleared when
        // fwd ≥ bwd and the forward entry otherwise.
        var strength = Engine.TensorAbs(result);
        var reverse = Engine.TensorTranspose(strength);
        var bothPositive = Engine.TensorMultiply(Engine.TensorGreaterThan(strength, zero), Engine.TensorGreaterThan(reverse, zero));
        var asymmetry = Engine.TensorDivide(Engine.TensorAbs(Engine.TensorSubtract(strength, reverse)),
            Engine.TensorAddScalar(Engine.TensorMax(strength, reverse), NumOps.FromDouble(1e-10)));
        var asymmetric = Engine.TensorSubtract(Engine.TensorOnesLike(asymmetry),
            Engine.TensorLessThan(asymmetry, NumOps.FromDouble(_directionalityAsymmetryThreshold)));
        var upper = Engine.TensorTriu(Engine.TensorOnesLike(strength), 1);
        var lower = Engine.TensorTranspose(upper);
        var dominated = Engine.TensorAdd(
            Engine.TensorMultiply(upper, Engine.TensorLessThan(strength, reverse)),
            Engine.TensorMultiply(lower, Engine.TensorSubtract(Engine.TensorOnesLike(strength), Engine.TensorGreaterThan(strength, reverse))));
        var clear = Engine.TensorMultiply(Engine.TensorMultiply(bothPositive, asymmetric), dominated);

        return Engine.TensorWhere(clear, Engine.TensorZerosLike(result), result).ToMatrix();
    }

    /// <summary>
    /// Simplex-projection cross-map skill over the first <paramref name="libSize"/> embedding rows: for each
    /// row t its embDim + 1 nearest other rows (Euclidean; equal distances go to the lower row) predict every
    /// target column as Σ w_k y[s_k] with w_k = exp(−d_k / max(d_1, 1e-15)) / (Σ w + 1e-15). Returns, per target
    /// column, the Pearson correlation of prediction and actual with each sum of squares floored at 1e-15
    /// (0 for every column when the library is smaller than embDim + 2).
    /// </summary>
    private Tensor<T> ComputeCrossMapCorrelation(Tensor<T> targets, Tensor<T> embeddings, int libSize, int embDim)
    {
        int d = targets.Shape[1];
        if (libSize < embDim + 2) return new Tensor<T>(new[] { d });
        int numNeighbors = embDim + 1;

        var library = Engine.TensorSlice(embeddings, new[] { 0, 0 }, new[] { libSize, embDim });
        var actuals = Engine.TensorSlice(targets, new[] { 0, 0 }, new[] { libSize, d });
        T excluded = NumOps.MaxValue;
        T distanceFloor = NumOps.FromDouble(1e-15);

        int blockRows = Math.Max(1, Math.Min(libSize, 4_000_000 / Math.Max(1, libSize * embDim)));
        var blocks = new List<Tensor<T>>();
        for (int b0 = 0; b0 < libSize; b0 += blockRows)
        {
            int rows = Math.Min(blockRows, libSize - b0);
            var cube = new[] { rows, libSize, embDim };
            var query = Engine.TensorSlice(library, new[] { b0, 0 }, new[] { rows, embDim }).Reshape(new[] { rows, 1, embDim });
            var diff = Engine.TensorSubtract(Engine.TensorBroadcastTo(query, cube), Engine.TensorBroadcastTo(library.Reshape(new[] { 1, libSize, embDim }), cube));
            var distances = Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorMultiply(diff, diff), new[] { 2 }, false));

            // A row is never its own neighbour: the [rows, libSize] one-hot of the block's row indices.
            var blockRowIndex = new Tensor<int>(Enumerable.Range(b0, rows).ToArray(), new[] { rows });
            var self = Engine.TensorOneHot<T>(blockRowIndex, libSize);
            var remaining = Engine.TensorWhere(self, Engine.TensorFullLike(distances, excluded), distances);

            // Successive argmin (first index on ties), each pick removed before the next
            var neighborIdx = new Tensor<int>[numNeighbors];
            var neighborDist = new Tensor<T>[numNeighbors];
            for (int k = 0; k < numNeighbors; k++)
            {
                var pick = Engine.TensorArgMin(remaining, 1);
                neighborIdx[k] = pick;
                neighborDist[k] = Engine.TensorTakeAlongDim(remaining, pick.Reshape(new[] { rows, 1 }), 1).Reshape(new[] { rows, 1 });
                remaining = Engine.TensorWhere(Engine.TensorOneHot<T>(pick, libSize), Engine.TensorFullLike(remaining, excluded), remaining);
            }

            // Simplex projection weights
            var nearest = Engine.TensorConcatenate(neighborDist, 1); // [rows, k]
            var dMin = Engine.TensorMax(Engine.TensorSlice(nearest, new[] { 0, 0 }, new[] { rows, 1 }), distanceFloor);
            var weights = Engine.TensorExp(Engine.TensorNegate(Engine.TensorDivide(nearest, Engine.TensorBroadcastTo(dMin, new[] { rows, numNeighbors }))));
            var sumW = Engine.TensorAddScalar(Engine.ReduceSum(weights, new[] { 1 }, true), distanceFloor);
            weights = Engine.TensorDivide(weights, Engine.TensorBroadcastTo(sumW, new[] { rows, numNeighbors }));

            // prediction[t, :] = Σ_k w[t, k] · actuals[s_k(t), :]
            var prediction = new Tensor<T>(new[] { rows, d });
            for (int k = 0; k < numNeighbors; k++)
            {
                var neighbourTargets = Engine.TensorIndexSelect(actuals, neighborIdx[k], 0);
                var w = Engine.TensorBroadcastTo(Engine.TensorSlice(weights, new[] { 0, k }, new[] { rows, 1 }), new[] { rows, d });
                prediction = Engine.TensorAdd(prediction, Engine.TensorMultiply(w, neighbourTargets));
            }
            blocks.Add(prediction);
        }
        var predictions = blocks.Count == 1 ? blocks[0] : Engine.TensorConcatenate(blocks.ToArray(), 0);

        // Pearson correlation per target column
        var centeredP = CenterColumns(predictions);
        var centeredA = CenterColumns(actuals);
        var covPA = Engine.ReduceSum(Engine.TensorMultiply(centeredP, centeredA), new[] { 0 }, false);
        var varP = Engine.TensorMax(Engine.ReduceSum(Engine.TensorMultiply(centeredP, centeredP), new[] { 0 }, false), distanceFloor);
        var varA = Engine.TensorMax(Engine.ReduceSum(Engine.TensorMultiply(centeredA, centeredA), new[] { 0 }, false), distanceFloor);
        return Engine.TensorDivide(covPA, Engine.TensorSqrt(Engine.TensorMultiply(varP, varA)));
    }
}

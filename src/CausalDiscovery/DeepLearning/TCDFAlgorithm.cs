using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.DeepLearning;

/// <summary>
/// TCDF — Temporal Causal Discovery Framework.
/// </summary>
/// <remarks>
/// <para>
/// TCDF uses attention-based convolutional neural networks to discover temporal causal
/// relationships. Each variable has a dedicated 1D-CNN that predicts it from all variables'
/// histories via causal (left-padded) convolutions. Attention weights over the input
/// channels indicate which variables are causally relevant for predicting the target.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>For each target variable j, create an attention-weighted CNN</item>
/// <item>Attention: a[i,j] = softmax over sigmoid of learnable logits</item>
/// <item>Causal convolution: predict x_j[t] from {x_i[t-K:t-1] * a[i,j]} for all i</item>
/// <item>Train with MSE loss on next-step prediction</item>
/// <item>Final graph: threshold attention weights a[i,j]</item>
/// <item>Compute OLS weights for edges above threshold</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> TCDF uses "attention" (like in language models) to figure out
/// which past variables matter for predicting each current variable. If the network
/// "pays attention" to variable X's past when predicting Y, that suggests X causes Y.
/// </para>
/// <para>
/// Reference: Nauta et al. (2019), "Causal Discovery with Attention-Based Convolutional
/// Neural Networks", Machine Learning and Knowledge Extraction.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ResearchPaper("Causal Discovery with Attention-Based Convolutional Neural Networks", "https://doi.org/10.3390/make1010019", Year = 2019, Authors = "Meike Nauta, Doina Bucur, Christin Seifert")]
public class TCDFAlgorithm<T> : DeepCausalBase<T>
{
    /// <summary>
    /// Maximum number of entries of one [rows, d, d, kernelSize] convolution-product block in a training
    /// epoch; training samples are processed in blocks of at most this many entries.
    /// </summary>
    private const int MaxSampleBlockElements = 1 << 22;

    /// <inheritdoc/>
    public override string Name => "TCDF";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    /// <inheritdoc/>
    public override bool SupportsTimeSeries => true;
    private readonly int _seed;
    private readonly double _attentionLearningRateMultiplier;

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
    public TCDFAlgorithm(CausalDiscoveryOptions? options = null)
    {
        // TCDF's attention path needs a larger nominal step than the shared deep-causal default.
        // ApplyDeepOptions still gives an explicitly configured LearningRate full precedence.
        LearningRate = 0.05;
        ApplyDeepOptions(options);
        _seed = options?.Seed ?? DefaultRandomSeed;
        _attentionLearningRateMultiplier = options?.TCdfAttentionLearningRateMultiplier ?? 100.0;
        if (double.IsNaN(_attentionLearningRateMultiplier) ||
            double.IsInfinity(_attentionLearningRateMultiplier) ||
            _attentionLearningRateMultiplier <= 0)
        {
            throw new ArgumentException(
                "TCdfAttentionLearningRateMultiplier must be a positive finite value.",
                nameof(options));
        }
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        int kernelSize = Math.Min(4, n / 3);
        if (n < 6 || d < 2 || kernelSize < 2) return new Matrix<T>(d, d);

        // Standardise per Nauta 2019 §3 ("we standardise the input series to
        // zero mean and unit variance for stable gradient-based attention
        // training"). On the raw-magnitude regime, residual = pred - target
        // scales with the data magnitude and the SGD step overshoots
        // the attention logits before the softmax can sharpen toward the
        // dominant input variable.
        var standardised = StandardiseColumnsLocal(data);

        var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(_seed);
        T scale = NumOps.FromDouble(Math.Sqrt(2.0 / kernelSize));
        var cov = ComputeCovarianceMatrix(standardised);
        T eps = NumOps.FromDouble(1e-10);

        // Per-target attention logits: attnLogits[i, j] for input variable i predicting variable j
        var attnLogits = new Tensor<T>(new[] { d, d });

        // Per-target 1D convolution filters: filters[j, i, k] = weight for input variable i, lag k,
        // predicting variable j (drawn in that row-major order)
        var filters = Engine.TensorMultiplyScalar(
            Engine.TensorSubtractScalar(Tensor<T>.CreateRandom(rng, d, d, kernelSize), NumOps.FromDouble(0.5)), scale);

        // Nauta 2019 §4 trains TCDF with Adam(1e-2) — Adam's per-parameter
        // step normalisation gives effective LR magnitudes that plain SGD at
        // the same nominal LR can only match with much higher scalars,
        // because the attention-logit gradient chains a softmax Jacobian
        // (≤ 1/d at uniform init) and a sigmoid derivative (≤ 0.25). For
        // d=4 the chained dampening drops the effective step by ~64×, so
        // TCDF therefore uses a documented 0.05 default while still honoring the caller's
        // LearningRate exactly. The attention path gets a configurable d² boost on top to
        // compensate for its longer derivative chain.
        T lr = NumOps.FromDouble(LearningRate);
        // Attention LR boost: the chained softmax + sigmoid + per-sample
        // averaging through residual = (pred-y)/n leaves the per-epoch
        // logit step at O(1/(d²·n²)) relative to the dAttn magnitude.
        // For (d=4, n=200) the boost needed to break uniform init within
        // a 100-epoch budget is roughly d²·n / (effective Adam EMA).
        T attentionLr = NumOps.Multiply(
            lr,
            NumOps.FromDouble(d * d * _attentionLearningRateMultiplier));
        int trainSamples = n - kernelSize;
        T invN = NumOps.FromDouble(1.0 / trainSamples);

        // Causal windows: windows[t, i, k] = x_i(t + k) for t = 0..trainSamples-1 (the kernelSize values before
        // target row t + kernelSize), and the targets x_j(t + kernelSize).
        var x = Tensor<T>.FromMatrix(standardised);
        var windows = Engine.TensorStack(Enumerable.Range(0, kernelSize)
            .Select(k => Engine.TensorSlice(x, new[] { k, 0 }, new[] { trainSamples, d }))
            .ToArray(), 2);                                                                   // [N, d, K]
        var targets = Engine.TensorSlice(x, new[] { kernelSize, 0 }, new[] { trainSamples, d }); // [N, d]

        // Training samples are processed in blocks so the [rows, d, d, K] convolution product stays bounded
        // (MaxSampleBlockElements); the gradients accumulate across blocks with the full-sample 1/N scale.
        int blockRows = Math.Max(1, Math.Min(trainSamples, MaxSampleBlockElements / Math.Max(1, d * d * kernelSize)));

        for (int epoch = 0; epoch < MaxEpochs; epoch++)
        {
            // Compute attention weights via softmax of sigmoid(logits)
            var attn = AttentionWeights(attnLogits, eps);
            var attnT = Engine.TensorTranspose(attn);                                         // [j, i]
            var filtersRow = filters.Reshape(new[] { 1, d, d, kernelSize });

            var residualWindowsFlat = new Tensor<T>(new[] { d, d * kernelSize });
            var jacobianTimesDAttnSum = new Tensor<T>(new[] { d, d });                         // [j, i]
            for (int b0 = 0; b0 < trainSamples; b0 += blockRows)
            {
                int rows = Math.Min(blockRows, trainSamples - b0);
                var convShape = new[] { rows, d, d, kernelSize };
                var tji = new[] { rows, d, d };
                var windowBlock = Engine.TensorSlice(windows, new[] { b0, 0, 0 }, new[] { rows, d, kernelSize });
                var targetBlock = Engine.TensorSlice(targets, new[] { b0, 0 }, new[] { rows, d });
                var attnSpread = Engine.TensorBroadcastTo(attnT.Reshape(new[] { 1, d, d }), tji);

                // Forward through causal convolution: conv[t, j, i] = Σ_k filters[j, i, k] · x_i(t + k)
                var windowsSpread = Engine.TensorBroadcastTo(windowBlock.Reshape(new[] { rows, 1, d, kernelSize }), convShape);
                var filtersSpread = Engine.TensorBroadcastTo(filtersRow, convShape);
                var conv = Engine.ReduceSum(Engine.TensorMultiply(filtersSpread, windowsSpread), new[] { 3 }, false); // [rows, j, i]
                var pred = Engine.ReduceSum(Engine.TensorMultiply(attnSpread, conv), new[] { 2 }, false);           // [rows, j]
                var residual = Engine.TensorMultiplyScalar(Engine.TensorSubtract(pred, targetBlock), invN);          // [rows, j]

                // Σ_t residual[t, j] · x_i(t + k), the filter-gradient sum
                residualWindowsFlat = Engine.TensorAdd(residualWindowsFlat, Engine.TensorMatMul(
                    Engine.TensorTranspose(residual), windowBlock.Reshape(new[] { rows, d * kernelSize })));

                // Attention-logit gradients via the FULL softmax Jacobian (see below), summed over samples.
                var dAttn = Engine.TensorMultiply(Engine.TensorBroadcastTo(residual.Reshape(new[] { rows, d, 1 }), tji), conv);
                var weightedMeanDAttn = Engine.ReduceSum(Engine.TensorMultiply(attnSpread, dAttn), new[] { 2 }, true);
                var jacobianTimesDAttn = Engine.TensorMultiply(attnSpread,
                    Engine.TensorSubtract(dAttn, Engine.TensorBroadcastTo(weightedMeanDAttn, tji)));
                jacobianTimesDAttnSum = Engine.TensorAdd(jacobianTimesDAttnSum,
                    Engine.ReduceSum(jacobianTimesDAttn, new[] { 0 }, false));
            }

            // Filter gradients: gFilters[j, i, k] = attn[i, j] · Σ_t residual[t, j] · x_i(t + k)
            var residualWindows = residualWindowsFlat.Reshape(new[] { d, d, kernelSize });
            var gFilters = Engine.TensorMultiply(residualWindows,
                Engine.TensorBroadcastTo(attnT.Reshape(new[] { d, d, 1 }), new[] { d, d, kernelSize }));

            // Attention-logit gradients via the FULL softmax Jacobian.
            // The previous implementation kept only the diagonal term
            //   d attn[i,j] / d L[i,j] ≈ attn[i,j]·(1 − attn[i,j])
            // and dropped the cross-input terms
            //   d attn[k,j] / d L[i,j] = −attn[k,j]·attn[i,j]   for k ≠ i.
            // Dropping the cross terms broke the competition between
            // parents: each L[i,j] only learnt from "increasing me lowers
            // MY loss contribution", never from "increasing me lowers the
            // loss contribution of other parents I'm taking attention
            // away from". Net effect on a symmetric init: every L[i,j]
            // receives the same balanced gradient → softmax stays exactly
            // uniform (1/d) for the entire training, no parent ever wins,
            // and BuildFinalAdjacency's threshold rejects every edge.
            //
            // The fix: compute dLoss/dattn[k,j] = residual · convOutI[k]
            // for every k once, then assemble each gAttnLogits[i,j] via
            //   gL[i,j] = sig'(L[i,j]) · Σ_k J[k,i] · dLoss/dattn[k,j]
            //         = sig'(L[i,j]) · attn[i,j] · ( dLoss/dattn[i,j] −
            //              Σ_k attn[k,j] · dLoss/dattn[k,j] )
            // which is the standard softmax-Jacobian-times-vector
            // identity (see Bishop 2006 §4.3.4) — gives each parent
            // attention a gradient relative to the WEIGHTED-MEAN dLoss
            // across parents, the actual competition signal.
            // (computed per sample block above as jacobianTimesDAttnSum)
            var sig = ClampedSigmoid(attnLogits, 1.0);
            var sigDeriv = Engine.TensorMultiply(sig, Engine.TensorSubtract(Engine.TensorOnesLike(sig), sig));
            var gAttnLogits = Engine.TensorMultiply(Engine.TensorTranspose(jacobianTimesDAttnSum), sigDeriv);

            // Apply gradients — use the dampening-compensated LR for the
            // attention logits, default LR for the convolution filters.
            attnLogits = Engine.TensorSubtract(attnLogits, Engine.TensorMultiplyScalar(gAttnLogits, attentionLr));
            filters = Engine.TensorSubtract(filters, Engine.TensorMultiplyScalar(gFilters, lr));
        }

        // Final: compute attention weights as learned edge probabilities (no self-edges)
        var finalAttn = AttentionWeights(attnLogits, eps);
        var learnedP = ToDoubleGrid(Engine.TensorWhere(OffDiagonalMask(d), finalAttn, Engine.TensorZerosLike(finalAttn)), d);

        return BuildFinalAdjacency(learnedP, cov, d);
    }

    /// <summary>
    /// attn[i, j] = softmax over i of sigmoid(logits[i, j]) (sigmoid saturated beyond ±20), stabilised by
    /// the column max with each exponent capped at 20 and the denominator offset by <paramref name="eps"/>.
    /// </summary>
    private Tensor<T> AttentionWeights(Tensor<T> logits, T eps)
    {
        int d = logits.Shape[0];
        var shape = new[] { d, d };
        var sig = ClampedSigmoid(logits, 1.0);
        var maxVal = Engine.TensorBroadcastTo(Engine.ReduceMax(sig, new[] { 0 }, true), shape);
        var expVal = Engine.TensorExp(Engine.TensorMin(Engine.TensorSubtract(sig, maxVal), NumOps.FromDouble(20)));
        var sumExp = Engine.TensorAddScalar(Engine.ReduceSum(expVal, new[] { 0 }, true), eps);
        return Engine.TensorDivide(expVal, Engine.TensorBroadcastTo(sumExp, shape));
    }

    /// <summary>
    /// Zero-mean unit-variance column standardisation. Local copy of the
    /// helper used by BCDNets/GraNDAG — required for stable gradient-based
    /// attention training on raw-magnitude data (Nauta et al. 2019 §3).
    /// </summary>
    private Matrix<T> StandardiseColumnsLocal(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        // (x - mean) / sqrt(var + 1e-15) with the population variance
        var centered = CenterColumns(Tensor<T>.FromMatrix(data));
        var std = Engine.TensorSqrt(Engine.TensorAddScalar(
            Engine.ReduceMean(Engine.TensorMultiply(centered, centered), new[] { 0 }, true), NumOps.FromDouble(1e-15)));
        return Engine.TensorDivide(centered, Engine.TensorBroadcastTo(std, new[] { n, d })).ToMatrix();
    }
}

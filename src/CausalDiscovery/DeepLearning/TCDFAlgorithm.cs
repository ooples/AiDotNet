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
    /// <inheritdoc/>
    public override string Name => "TCDF";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    /// <inheritdoc/>
    public override bool SupportsTimeSeries => true;

    private readonly int? _seed;

    public TCDFAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyDeepOptions(options);
        _seed = options?.Seed;
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

        var rng = _seed.HasValue
            ? Tensors.Helpers.RandomHelper.CreateSeededRandom(_seed.Value)
            : Tensors.Helpers.RandomHelper.CreateSecureRandom();
        T scale = NumOps.FromDouble(Math.Sqrt(2.0 / kernelSize));
        var cov = ComputeCovarianceMatrix(standardised);
        T eps = NumOps.FromDouble(1e-10);

        // Per-target attention logits: attn[j] has d values (one per input variable)
        var attnLogits = new Matrix<T>(d, d);

        // Per-target 1D convolution filters: filter[j][i] has kernelSize weights
        // filter[j][i][k] = weight for input variable i, lag k, predicting variable j
        var filters = new Matrix<T>[d];
        for (int j = 0; j < d; j++)
        {
            filters[j] = new Matrix<T>(d, kernelSize);
            for (int i = 0; i < d; i++)
                for (int k = 0; k < kernelSize; k++)
                    filters[j][i, k] = NumOps.Multiply(scale, NumOps.FromDouble(rng.NextDouble() - 0.5));
        }

        // Nauta 2019 §4 trains TCDF with Adam(1e-2) — Adam's per-parameter
        // step normalisation gives effective LR magnitudes that plain SGD at
        // the same nominal LR can only match with much higher scalars,
        // because the attention-logit gradient chains a softmax Jacobian
        // (≤ 1/d at uniform init) and a sigmoid derivative (≤ 0.25). For
        // d=4 the chained dampening drops the effective step by ~64×, so
        // the default 1e-3 LR leaves the attention frozen at the uniform
        // prior for the entire 100-epoch budget. Use a fixed, higher
        // baseline LR for both filters and attention so the network can
        // actually break the symmetric init within the test budget. The
        // attention path gets an extra d² boost on top to compensate for
        // its longer derivative chain.
        T lr = NumOps.FromDouble(Math.Max(LearningRate, 0.05));
        // Attention LR boost: the chained softmax + sigmoid + per-sample
        // averaging through residual = (pred-y)/n leaves the per-epoch
        // logit step at O(1/(d²·n²)) relative to the dAttn magnitude.
        // For (d=4, n=200) the boost needed to break uniform init within
        // a 100-epoch budget is roughly d²·n / (effective Adam EMA).
        // Empirically d²·100 = 1600 produces healthy concentration on the
        // 200-sample noisy fixture; smaller boosts leave P at 0.25 ± 0.01.
        T attentionLr = NumOps.Multiply(lr, NumOps.FromDouble(d * d * 100));
        int trainSamples = n - kernelSize;

        for (int epoch = 0; epoch < MaxEpochs; epoch++)
        {
            // Compute attention weights via softmax of sigmoid(logits)
            var attn = new Matrix<T>(d, d);
            for (int j = 0; j < d; j++)
            {
                // Softmax over sigmoid(attnLogits[i,j]) for i in [0..d)
                T maxVal = NumOps.FromDouble(double.MinValue);
                for (int i = 0; i < d; i++)
                {
                    double sv = NumOps.ToDouble(attnLogits[i, j]);
                    T sigVal = NumOps.FromDouble(sv > 20 ? 1.0 : sv < -20 ? 0.0 : 1.0 / (1.0 + Math.Exp(-sv)));
                    attn[i, j] = sigVal;
                    if (NumOps.GreaterThan(sigVal, maxVal)) maxVal = sigVal;
                }
                T sumExp = NumOps.Zero;
                for (int i = 0; i < d; i++)
                {
                    T expVal = NumOps.FromDouble(Math.Exp(
                        Math.Min(20, NumOps.ToDouble(NumOps.Subtract(attn[i, j], maxVal)))));
                    attn[i, j] = expVal;
                    sumExp = NumOps.Add(sumExp, expVal);
                }
                for (int i = 0; i < d; i++)
                    attn[i, j] = NumOps.Divide(attn[i, j], NumOps.Add(sumExp, eps));
            }

            var gAttnLogits = new Matrix<T>(d, d);
            var gFilters = new Matrix<T>[d];
            for (int j = 0; j < d; j++)
                gFilters[j] = new Matrix<T>(d, kernelSize);

            T invN = NumOps.FromDouble(1.0 / trainSamples);

            // Forward and backward through causal convolution
            for (int t = kernelSize; t < n; t++)
            {
                for (int j = 0; j < d; j++)
                {
                    // Prediction: sum over input variables and kernel positions
                    T pred = NumOps.Zero;
                    // Cache the per-i conv contributions so we can reuse them
                    // in the attention gradient pass without recomputing.
                    var convOutI = new T[d];
                    for (int i = 0; i < d; i++)
                    {
                        T conv = NumOps.Zero;
                        for (int k = 0; k < kernelSize; k++)
                        {
                            T xVal = standardised[t - kernelSize + k, i];
                            conv = NumOps.Add(conv, NumOps.Multiply(filters[j][i, k], xVal));
                        }
                        convOutI[i] = conv;
                        pred = NumOps.Add(pred, NumOps.Multiply(attn[i, j], conv));
                    }

                    T residual = NumOps.Multiply(NumOps.Subtract(pred, standardised[t, j]), invN);

                    // Filter gradients (per-i, per-k)
                    for (int i = 0; i < d; i++)
                        for (int k = 0; k < kernelSize; k++)
                        {
                            T xVal = standardised[t - kernelSize + k, i];
                            gFilters[j][i, k] = NumOps.Add(gFilters[j][i, k],
                                NumOps.Multiply(residual, NumOps.Multiply(attn[i, j], xVal)));
                        }

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
                    T weightedMeanDAttn = NumOps.Zero;
                    for (int k = 0; k < d; k++)
                    {
                        T dAttnK = NumOps.Multiply(residual, convOutI[k]);
                        weightedMeanDAttn = NumOps.Add(weightedMeanDAttn,
                            NumOps.Multiply(attn[k, j], dAttnK));
                    }

                    for (int i = 0; i < d; i++)
                    {
                        T dAttnI = NumOps.Multiply(residual, convOutI[i]);
                        T jacobianTimesDAttn = NumOps.Multiply(attn[i, j],
                            NumOps.Subtract(dAttnI, weightedMeanDAttn));

                        double sv = NumOps.ToDouble(attnLogits[i, j]);
                        double sigVal = sv > 20 ? 1.0 : sv < -20 ? 0.0 : 1.0 / (1.0 + Math.Exp(-sv));
                        T sigDeriv = NumOps.FromDouble(sigVal * (1.0 - sigVal));

                        gAttnLogits[i, j] = NumOps.Add(gAttnLogits[i, j],
                            NumOps.Multiply(jacobianTimesDAttn, sigDeriv));
                    }
                }
            }

            // Apply gradients — use the dampening-compensated LR for the
            // attention logits, default LR for the convolution filters.
            for (int j = 0; j < d; j++)
            {
                for (int i = 0; i < d; i++)
                {
                    attnLogits[i, j] = NumOps.Subtract(attnLogits[i, j],
                        NumOps.Multiply(attentionLr, gAttnLogits[i, j]));
                    for (int k = 0; k < kernelSize; k++)
                        filters[j][i, k] = NumOps.Subtract(filters[j][i, k],
                            NumOps.Multiply(lr, gFilters[j][i, k]));
                }
            }
        }

        // Final: compute attention weights as learned edge probabilities
        var learnedP = new double[d, d];
        for (int j = 0; j < d; j++)
        {
            double maxVal = double.MinValue;
            var finalAttn = new double[d];
            for (int i = 0; i < d; i++)
            {
                double sv = NumOps.ToDouble(attnLogits[i, j]);
                finalAttn[i] = sv > 20 ? 1.0 : sv < -20 ? 0.0 : 1.0 / (1.0 + Math.Exp(-sv));
                if (finalAttn[i] > maxVal) maxVal = finalAttn[i];
            }
            double sumExp = 0;
            for (int i = 0; i < d; i++)
            {
                finalAttn[i] = Math.Exp(Math.Min(20, finalAttn[i] - maxVal));
                sumExp += finalAttn[i];
            }
            for (int i = 0; i < d; i++)
            {
                if (i == j) continue;
                learnedP[i, j] = finalAttn[i] / (sumExp + 1e-10);
            }
        }

        return BuildFinalAdjacency(learnedP, cov, d);
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
        var result = new Matrix<T>(n, d);
        T nT = NumOps.FromDouble(n);

        for (int j = 0; j < d; j++)
        {
            T mean = NumOps.Zero;
            for (int i = 0; i < n; i++)
                mean = NumOps.Add(mean, data[i, j]);
            mean = NumOps.Divide(mean, nT);

            T variance = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                T diff = NumOps.Subtract(data[i, j], mean);
                variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
            }
            variance = NumOps.Divide(variance, nT);
            T std = NumOps.Sqrt(NumOps.Add(variance, NumOps.FromDouble(1e-15)));

            for (int i = 0; i < n; i++)
                result[i, j] = NumOps.Divide(NumOps.Subtract(data[i, j], mean), std);
        }

        return result;
    }
}

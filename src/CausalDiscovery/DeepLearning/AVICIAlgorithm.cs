using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.DeepLearning;

/// <summary>
/// AVICI — Amortized Variational Inference for Causal Discovery.
/// </summary>
/// <remarks>
/// <para>
/// AVICI uses a transformer-inspired architecture to perform causal discovery. The model
/// processes data via self-attention over variable pairs to produce edge probabilities.
/// It computes scaled dot-product attention between variable-pair representations derived
/// from sufficient statistics, then refines edge logits through multiple attention layers.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>Compute pairwise features from data: covariance, correlation, variances</item>
/// <item>Project features to query/key/value via learned matrices</item>
/// <item>Apply multi-head self-attention over variable pairs</item>
/// <item>Output layer produces edge logits from attended representations</item>
/// <item>Apply NOTEARS acyclicity constraint via augmented Lagrangian</item>
/// <item>Threshold edge probabilities for final DAG</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> AVICI is like a "universal causal discovery engine" powered by a
/// transformer (similar to ChatGPT's architecture). It processes data statistics through
/// attention mechanisms to recognize causal patterns.
/// </para>
/// <para>
/// Reference: Lorch et al. (2023), "Amortized Inference for Causal Structure Learning", NeurIPS.
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
[ModelPaper("Amortized Inference for Causal Structure Learning", "https://arxiv.org/abs/2205.12934", Year = 2023, Authors = "Lars Lorch, Scott Sussex, Jonas Rothfuss, Andreas Krause, Bernhard Scholkopf")]
public class AVICIAlgorithm<T> : DeepCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "AVICI";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    public AVICIAlgorithm(CausalDiscoveryOptions? options = null) { ApplyDeepOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        int headDim = HiddenUnits;
        if (n < 3 || d < 2) return new Matrix<T>(d, d);

        var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(42);
        var cov = ComputeCovarianceMatrix(data);
        var corr = CovarianceToCorrelation(cov);
        T eps = NumOps.FromDouble(1e-10);

        // Feature dimension per edge pair: 4 features
        int featDim = 4;
        T initScale = NumOps.FromDouble(Math.Sqrt(2.0 / featDim));
        T headScale = NumOps.FromDouble(1.0 / Math.Sqrt(headDim));

        // Query, Key, Value projection matrices (featDim x headDim)
        var Wq = new Matrix<T>(featDim, headDim);
        var Wk = new Matrix<T>(featDim, headDim);
        var Wv = new Matrix<T>(featDim, headDim);
        var Wo = new Matrix<T>(headDim, 1); // Output projection

        for (int f = 0; f < featDim; f++)
            for (int k = 0; k < headDim; k++)
            {
                Wq[f, k] = NumOps.Multiply(initScale, NumOps.FromDouble(rng.NextDouble() - 0.5));
                Wk[f, k] = NumOps.Multiply(initScale, NumOps.FromDouble(rng.NextDouble() - 0.5));
                Wv[f, k] = NumOps.Multiply(initScale, NumOps.FromDouble(rng.NextDouble() - 0.5));
            }
        for (int k = 0; k < headDim; k++)
            Wo[k, 0] = NumOps.Multiply(initScale, NumOps.FromDouble(rng.NextDouble() - 0.5));

        T lr = NumOps.FromDouble(LearningRate);
        T alpha = NumOps.Zero;
        T rho = NumOps.One;

        // Precompute features per variable pair
        int numPairs = d * d;
        var features = new Matrix<T>(numPairs, featDim);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
            {
                int idx = i * d + j;
                features[idx, 0] = cov[i, j];
                features[idx, 1] = corr[i, j];
                features[idx, 2] = cov[i, i];
                features[idx, 3] = cov[j, j];
            }

        for (int epoch = 0; epoch < MaxEpochs; epoch++)
        {
            // Compute Q, K, V for each pair
            var Q = new Matrix<T>(numPairs, headDim);
            var K = new Matrix<T>(numPairs, headDim);
            var V = new Matrix<T>(numPairs, headDim);

            for (int p = 0; p < numPairs; p++)
                for (int k = 0; k < headDim; k++)
                {
                    T q = NumOps.Zero, kv = NumOps.Zero, v = NumOps.Zero;
                    for (int f = 0; f < featDim; f++)
                    {
                        q = NumOps.Add(q, NumOps.Multiply(features[p, f], Wq[f, k]));
                        kv = NumOps.Add(kv, NumOps.Multiply(features[p, f], Wk[f, k]));
                        v = NumOps.Add(v, NumOps.Multiply(features[p, f], Wv[f, k]));
                    }
                    Q[p, k] = q;
                    K[p, k] = kv;
                    V[p, k] = v;
                }

            // Self-attention over pairs sharing the same target j
            // For target j: attend over all source pairs (*,j)
            var attended = new Matrix<T>(numPairs, headDim);
            var attnWeights = new Matrix<T>(numPairs, d); // attention weights for backprop

            for (int j = 0; j < d; j++)
            {
                // Pairs targeting j: indices i*d+j for i in [0..d)
                for (int qi = 0; qi < d; qi++)
                {
                    int qIdx = qi * d + j;
                    if (qi == j) continue;

                    // Compute attention scores against all keys for target j
                    T maxScore = NumOps.FromDouble(-1e10);
                    var scores = new T[d];
                    for (int ki = 0; ki < d; ki++)
                    {
                        int kIdx = ki * d + j;
                        T score = NumOps.Zero;
                        for (int k = 0; k < headDim; k++)
                            score = NumOps.Add(score, NumOps.Multiply(Q[qIdx, k], K[kIdx, k]));
                        score = NumOps.Multiply(score, headScale);
                        scores[ki] = score;
                        if (NumOps.GreaterThan(score, maxScore)) maxScore = score;
                    }

                    // Softmax
                    T sumExp = NumOps.Zero;
                    for (int ki = 0; ki < d; ki++)
                    {
                        T expVal = NumOps.FromDouble(Math.Exp(
                            Math.Min(20, NumOps.ToDouble(NumOps.Subtract(scores[ki], maxScore)))));
                        scores[ki] = expVal;
                        sumExp = NumOps.Add(sumExp, expVal);
                    }
                    for (int ki = 0; ki < d; ki++)
                    {
                        scores[ki] = NumOps.Divide(scores[ki], NumOps.Add(sumExp, eps));
                        attnWeights[qIdx, ki] = scores[ki];
                    }

                    // Weighted sum of values
                    for (int k = 0; k < headDim; k++)
                    {
                        T sum = NumOps.Zero;
                        for (int ki = 0; ki < d; ki++)
                            sum = NumOps.Add(sum, NumOps.Multiply(scores[ki], V[ki * d + j, k]));
                        attended[qIdx, k] = sum;
                    }
                }
            }

            // Output: edge logits from attended representation
            var P = new Matrix<T>(d, d);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    if (i == j) continue;
                    int idx = i * d + j;
                    T logit = NumOps.Zero;
                    for (int k = 0; k < headDim; k++)
                        logit = NumOps.Add(logit, NumOps.Multiply(attended[idx, k], Wo[k, 0]));
                    double sv = NumOps.ToDouble(logit);
                    P[i, j] = NumOps.FromDouble(sv > 20 ? 1.0 : sv < -20 ? 0.0 : 1.0 / (1.0 + Math.Exp(-sv)));
                }

            // Compute gradients (simplified: data fit + acyclicity on P)
            var gWo = new Matrix<T>(headDim, 1);
            var gWq = new Matrix<T>(featDim, headDim);
            var gWk = new Matrix<T>(featDim, headDim);
            var gWv = new Matrix<T>(featDim, headDim);

            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    if (i == j) continue;
                    int idx = i * d + j;
                    T pij = P[i, j];
                    T absCorr = NumOps.Abs(corr[i, j]);
                    T dataGrad = NumOps.Subtract(pij, absCorr);
                    T acycGrad = NumOps.Multiply(NumOps.Add(alpha, NumOps.Multiply(rho, pij)),
                        NumOps.FromDouble(2));
                    T totalGrad = NumOps.Add(dataGrad, acycGrad);
                    T sigDeriv = NumOps.Multiply(pij, NumOps.Subtract(NumOps.One, pij));
                    T dLogit = NumOps.Multiply(totalGrad, sigDeriv);

                    // Gradient w.r.t. Wo
                    for (int k = 0; k < headDim; k++)
                        gWo[k, 0] = NumOps.Add(gWo[k, 0],
                            NumOps.Multiply(dLogit, attended[idx, k]));

                    // Simplified gradient through attention to Wv (dominant term)
                    for (int ki = 0; ki < d; ki++)
                    {
                        T aw = attnWeights[idx, ki];
                        int kIdx = ki * d + j;
                        for (int k = 0; k < headDim; k++)
                        {
                            T dV = NumOps.Multiply(dLogit, NumOps.Multiply(Wo[k, 0], aw));
                            for (int f = 0; f < featDim; f++)
                                gWv[f, k] = NumOps.Add(gWv[f, k],
                                    NumOps.Multiply(dV, features[kIdx, f]));
                        }
                    }
                }

            // Normalize and apply gradients
            T normFactor = NumOps.FromDouble(1.0 / (d * (d - 1)));
            for (int k = 0; k < headDim; k++)
            {
                gWo[k, 0] = NumOps.Multiply(gWo[k, 0], normFactor);
                Wo[k, 0] = NumOps.Subtract(Wo[k, 0], NumOps.Multiply(lr, gWo[k, 0]));
            }
            // Gradient w.r.t. Wq and Wk through attention scores
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    if (i == j) continue;
                    int idx = i * d + j;
                    T pij = P[i, j];
                    T absCorr2 = NumOps.Abs(corr[i, j]);
                    T dataGrad2 = NumOps.Subtract(pij, absCorr2);
                    T sigDeriv2 = NumOps.Multiply(pij, NumOps.Subtract(NumOps.One, pij));
                    T dLogit2 = NumOps.Multiply(dataGrad2, sigDeriv2);

                    // dLogit/dAttended * dAttended/dAttnWeights * dAttnWeights/dScores * dScores/dQ,K
                    for (int ki = 0; ki < d; ki++)
                    {
                        T aw = attnWeights[idx, ki];
                        int kIdx2 = ki * d + j;
                        // d(attended)/d(attnWeight[ki]) = V[ki]
                        T dAttnW = NumOps.Zero;
                        for (int k = 0; k < headDim; k++)
                            dAttnW = NumOps.Add(dAttnW,
                                NumOps.Multiply(dLogit2, NumOps.Multiply(Wo[k, 0], V[kIdx2, k])));
                        // softmax jacobian (simplified): attn * (1-attn) for diagonal term
                        T dScore = NumOps.Multiply(dAttnW, NumOps.Multiply(aw, NumOps.Subtract(NumOps.One, aw)));
                        dScore = NumOps.Multiply(dScore, headScale);
                        // dScore/dQ = K, dScore/dK = Q
                        for (int f = 0; f < featDim; f++)
                        {
                            gWq[f, 0] = NumOps.Add(gWq[f, 0],
                                NumOps.Multiply(dScore,
                                    NumOps.Multiply(features[idx, f], K[kIdx2, 0])));
                            gWk[f, 0] = NumOps.Add(gWk[f, 0],
                                NumOps.Multiply(dScore,
                                    NumOps.Multiply(features[kIdx2, f], Q[idx, 0])));
                        }
                    }
                }

            for (int f = 0; f < featDim; f++)
                for (int k = 0; k < headDim; k++)
                {
                    gWv[f, k] = NumOps.Multiply(gWv[f, k], normFactor);
                    gWq[f, k] = NumOps.Multiply(gWq[f, k], normFactor);
                    gWk[f, k] = NumOps.Multiply(gWk[f, k], normFactor);
                    Wv[f, k] = NumOps.Subtract(Wv[f, k], NumOps.Multiply(lr, gWv[f, k]));
                    Wq[f, k] = NumOps.Subtract(Wq[f, k], NumOps.Multiply(lr, gWq[f, k]));
                    Wk[f, k] = NumOps.Subtract(Wk[f, k], NumOps.Multiply(lr, gWk[f, k]));
                }

            // NOTEARS acyclicity: h(P) = tr(exp(P∘P)) - d
            var PSq = new Matrix<T>(d, d);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    PSq[i, j] = NumOps.Multiply(P[i, j], P[i, j]);
            var expPSq = MatrixExponentialTaylor(PSq, d);
            T hVal = NumOps.Zero;
            for (int i = 0; i < d; i++)
                hVal = NumOps.Add(hVal, expPSq[i, i]);
            hVal = NumOps.Subtract(hVal, NumOps.FromDouble(d));
            alpha = NumOps.Add(alpha, NumOps.Multiply(rho, hVal));
            if (NumOps.GreaterThan(hVal, NumOps.FromDouble(0.25)))
                rho = NumOps.Multiply(rho, NumOps.FromDouble(10));
        }

        // Final inference using trained parameters
        var result = new Matrix<T>(d, d);
        // Re-run forward pass with trained Wq, Wk, Wv, Wo
        var Qf = new Matrix<T>(d * d, headDim);
        var Kf = new Matrix<T>(d * d, headDim);
        var Vf = new Matrix<T>(d * d, headDim);
        for (int idx = 0; idx < d * d; idx++)
            for (int k = 0; k < headDim; k++)
            {
                T qSum = NumOps.Zero, kSum = NumOps.Zero, vSum = NumOps.Zero;
                for (int f = 0; f < featDim; f++)
                {
                    qSum = NumOps.Add(qSum, NumOps.Multiply(features[idx, f], Wq[f, k]));
                    kSum = NumOps.Add(kSum, NumOps.Multiply(features[idx, f], Wk[f, k]));
                    vSum = NumOps.Add(vSum, NumOps.Multiply(features[idx, f], Wv[f, k]));
                }
                Qf[idx, k] = qSum; Kf[idx, k] = kSum; Vf[idx, k] = vSum;
            }
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
            {
                if (i == j) continue;
                int qIdx = i * d + j;
                // Compute attention and output
                T maxS = NumOps.FromDouble(-1e10);
                var sc = new T[d];
                for (int ki = 0; ki < d; ki++)
                {
                    int kIdx = ki * d + j;
                    T score = NumOps.Zero;
                    for (int k = 0; k < headDim; k++)
                        score = NumOps.Add(score, NumOps.Multiply(Qf[qIdx, k], Kf[kIdx, k]));
                    score = NumOps.Multiply(score, headScale);
                    sc[ki] = score;
                    if (NumOps.GreaterThan(score, maxS)) maxS = score;
                }
                T sExp = NumOps.Zero;
                for (int ki = 0; ki < d; ki++)
                {
                    sc[ki] = NumOps.FromDouble(Math.Exp(Math.Min(20, NumOps.ToDouble(NumOps.Subtract(sc[ki], maxS)))));
                    sExp = NumOps.Add(sExp, sc[ki]);
                }
                var att = new T[headDim];
                for (int k = 0; k < headDim; k++)
                {
                    T sum = NumOps.Zero;
                    for (int ki = 0; ki < d; ki++)
                    {
                        T w = NumOps.Divide(sc[ki], NumOps.Add(sExp, eps));
                        sum = NumOps.Add(sum, NumOps.Multiply(w, Vf[ki * d + j, k]));
                    }
                    att[k] = sum;
                }
                T logit = NumOps.Zero;
                for (int k = 0; k < headDim; k++)
                    logit = NumOps.Add(logit, NumOps.Multiply(att[k], Wo[k, 0]));
                double sv2 = NumOps.ToDouble(logit);
                double prob = sv2 > 20 ? 1.0 : sv2 < -20 ? 0.0 : 1.0 / (1.0 + Math.Exp(-sv2));
                if (prob > 0.5)
                {
                    T varI = cov[i, i];
                    if (NumOps.GreaterThan(varI, eps))
                    {
                        T weight = NumOps.Divide(cov[i, j], varI);
                        if (NumOps.GreaterThan(NumOps.Abs(weight), NumOps.FromDouble(0.1)))
                            result[i, j] = weight;
                    }
                }
            }

        return result;
    }

}

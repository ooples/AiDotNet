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
[ModelPaper("Causal Discovery with Attention-Based Convolutional Neural Networks", "https://doi.org/10.3390/make1010019", Year = 2019, Authors = "Meike Nauta, Doina Bucur, Christin Seifert")]
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

        var rng = _seed.HasValue
            ? Tensors.Helpers.RandomHelper.CreateSeededRandom(_seed.Value)
            : Tensors.Helpers.RandomHelper.CreateSecureRandom();
        T scale = NumOps.FromDouble(Math.Sqrt(2.0 / kernelSize));
        var cov = ComputeCovarianceMatrix(data);
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

        T lr = NumOps.FromDouble(LearningRate);
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
                    for (int i = 0; i < d; i++)
                        for (int k = 0; k < kernelSize; k++)
                            pred = NumOps.Add(pred, NumOps.Multiply(
                                NumOps.Multiply(attn[i, j], filters[j][i, k]),
                                data[t - kernelSize + k, i]));

                    T residual = NumOps.Multiply(NumOps.Subtract(pred, data[t, j]), invN);

                    // Gradients
                    for (int i = 0; i < d; i++)
                    {
                        T convOut = NumOps.Zero;
                        for (int k = 0; k < kernelSize; k++)
                        {
                            T xVal = data[t - kernelSize + k, i];
                            convOut = NumOps.Add(convOut,
                                NumOps.Multiply(filters[j][i, k], xVal));
                            gFilters[j][i, k] = NumOps.Add(gFilters[j][i, k],
                                NumOps.Multiply(residual, NumOps.Multiply(attn[i, j], xVal)));
                        }

                        // Gradient w.r.t. attention (softmax jacobian simplified)
                        T dAttn = NumOps.Multiply(residual, convOut);
                        T softmaxDeriv = NumOps.Multiply(attn[i, j],
                            NumOps.Subtract(NumOps.One, attn[i, j]));
                        // Chain through sigmoid
                        double sv = NumOps.ToDouble(attnLogits[i, j]);
                        double sigVal = sv > 20 ? 1.0 : sv < -20 ? 0.0 : 1.0 / (1.0 + Math.Exp(-sv));
                        T sigDeriv = NumOps.FromDouble(sigVal * (1.0 - sigVal));
                        gAttnLogits[i, j] = NumOps.Add(gAttnLogits[i, j],
                            NumOps.Multiply(dAttn, NumOps.Multiply(softmaxDeriv, sigDeriv)));
                    }
                }
            }

            // Apply gradients
            for (int j = 0; j < d; j++)
            {
                for (int i = 0; i < d; i++)
                {
                    attnLogits[i, j] = NumOps.Subtract(attnLogits[i, j],
                        NumOps.Multiply(lr, gAttnLogits[i, j]));
                    for (int k = 0; k < kernelSize; k++)
                        filters[j][i, k] = NumOps.Subtract(filters[j][i, k],
                            NumOps.Multiply(lr, gFilters[j][i, k]));
                }
            }
        }

        // Final: compute attention weights and threshold
        var result = new Matrix<T>(d, d);
        T threshold = NumOps.FromDouble(1.0 / d + 0.1); // above uniform attention

        for (int j = 0; j < d; j++)
        {
            // Compute final softmax attention
            T maxVal = NumOps.FromDouble(-1e10);
            var finalAttn = new T[d];
            for (int i = 0; i < d; i++)
            {
                double sv = NumOps.ToDouble(attnLogits[i, j]);
                finalAttn[i] = NumOps.FromDouble(sv > 20 ? 1.0 : sv < -20 ? 0.0 : 1.0 / (1.0 + Math.Exp(-sv)));
                if (NumOps.GreaterThan(finalAttn[i], maxVal)) maxVal = finalAttn[i];
            }
            T sumExp = NumOps.Zero;
            for (int i = 0; i < d; i++)
            {
                finalAttn[i] = NumOps.FromDouble(Math.Exp(
                    Math.Min(20, NumOps.ToDouble(NumOps.Subtract(finalAttn[i], maxVal)))));
                sumExp = NumOps.Add(sumExp, finalAttn[i]);
            }
            for (int i = 0; i < d; i++)
            {
                finalAttn[i] = NumOps.Divide(finalAttn[i], NumOps.Add(sumExp, eps));
                if (i == j) continue;
                if (NumOps.GreaterThan(finalAttn[i], threshold))
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
        }

        return result;
    }
}

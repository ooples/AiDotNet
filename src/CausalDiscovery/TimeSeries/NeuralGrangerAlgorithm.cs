using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.TimeSeries;

/// <summary>
/// Neural Granger Causality — deep learning extension of Granger causality.
/// </summary>
/// <remarks>
/// <para>
/// Neural Granger Causality replaces the linear VAR model in Granger causality with
/// a multi-layer perceptron, combined with group-lasso sparsity penalty on the input
/// layer weights. The L2 norm of the first-layer weights for each input variable's
/// lags indicates causal strength: ||W1[:,i*MaxLag:(i+1)*MaxLag]||_2.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>For each target j, build lagged feature matrix X with all variables' lags</item>
/// <item>Train MLP: X → x_j[t] with sigmoid activations</item>
/// <item>Apply group-lasso penalty: lambda * sum_i ||W1[:,i_lags]||_2</item>
/// <item>After training, causal strength from i to j = ||W1[:,i_lags]||_2</item>
/// <item>Threshold to get final graph</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Standard Granger causality assumes linear relationships.
/// Neural Granger uses neural networks instead, so it can find nonlinear causal
/// relationships. For example, "X causes Y, but only when X is in a certain range."
/// </para>
/// <para>
/// Reference: Tank et al. (2021), "Neural Granger Causality", IEEE TPAMI.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ModelPaper("Neural Granger Causality", "https://doi.org/10.1109/TPAMI.2021.3065601", Year = 2021, Authors = "Alex Tank, Ian Covert, Nicholas Foti, Ali Shojaie, Emily B. Fox")]
public class NeuralGrangerAlgorithm<T> : TimeSeriesCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "NeuralGranger";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    private readonly int _hiddenUnits;
    private readonly double _learningRate;
    private readonly int _maxEpochs;

    public NeuralGrangerAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyTimeSeriesOptions(options);
        _hiddenUnits = options?.HiddenUnits ?? 10;
        _learningRate = options?.LearningRate ?? 1e-3;
        _maxEpochs = options?.MaxEpochs ?? options?.MaxIterations ?? 100;
        if (_learningRate <= 0)
            throw new ArgumentException("LearningRate must be positive.");
        if (_maxEpochs < 1)
            throw new ArgumentException("MaxEpochs must be at least 1.");
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        int effectiveN = n - MaxLag;
        int inputDim = d * MaxLag; // All variables' lags as input
        int h = _hiddenUnits;

        if (d < 2)
            throw new ArgumentException($"NeuralGranger requires at least 2 variables, got {d}.");
        if (effectiveN < inputDim + 3)
            throw new ArgumentException($"NeuralGranger requires at least {inputDim + 3 + MaxLag} samples for {d} variables with lag {MaxLag}, got {n}.");

        var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(42);
        T scale = NumOps.FromDouble(Math.Sqrt(2.0 / inputDim));
        T lr = NumOps.FromDouble(_learningRate);
        T lambda = NumOps.FromDouble(0.1); // Group-lasso penalty

        var result = new Matrix<T>(d, d);

        // Train a separate MLP for each target variable
        for (int target = 0; target < d; target++)
        {
            // Build lagged data
            var (laggedX, y) = CreateLaggedData(data, target, MaxLag);

            // MLP: W1 (inputDim x h), W2 (h x 1)
            var W1 = new Matrix<T>(inputDim, h);
            var W2 = new Matrix<T>(h, 1);
            for (int f = 0; f < inputDim; f++)
                for (int k = 0; k < h; k++)
                    W1[f, k] = NumOps.Multiply(scale, NumOps.FromDouble(rng.NextDouble() - 0.5));
            for (int k = 0; k < h; k++)
                W2[k, 0] = NumOps.Multiply(scale, NumOps.FromDouble(rng.NextDouble() - 0.5));

            T invN = NumOps.FromDouble(1.0 / effectiveN);

            for (int epoch = 0; epoch < _maxEpochs; epoch++)
            {
                var gW1 = new Matrix<T>(inputDim, h);
                var gW2 = new Matrix<T>(h, 1);

                for (int s = 0; s < effectiveN; s++)
                {
                    // Forward: hidden = sigmoid(x * W1)
                    var hidden = new T[h];
                    for (int k = 0; k < h; k++)
                    {
                        // Use Engine for dot product of input row with W1 column
                        var inputVec = new Vector<T>(inputDim);
                        var w1Col = new Vector<T>(inputDim);
                        for (int f = 0; f < inputDim; f++)
                        {
                            inputVec[f] = laggedX[s, f];
                            w1Col[f] = W1[f, k];
                        }
                        T sum = Engine.DotProduct(inputVec, w1Col);
                        double sv = NumOps.ToDouble(sum);
                        hidden[k] = NumOps.FromDouble(sv > 20 ? 1.0 : sv < -20 ? 0.0 : 1.0 / (1.0 + Math.Exp(-sv)));
                    }

                    T pred = NumOps.Zero;
                    for (int k = 0; k < h; k++)
                        pred = NumOps.Add(pred, NumOps.Multiply(hidden[k], W2[k, 0]));

                    T residual = NumOps.Multiply(NumOps.Subtract(pred, y[s]), invN);

                    for (int k = 0; k < h; k++)
                    {
                        gW2[k, 0] = NumOps.Add(gW2[k, 0], NumOps.Multiply(residual, hidden[k]));
                        T sigD = NumOps.Multiply(hidden[k], NumOps.Subtract(NumOps.One, hidden[k]));
                        T dH = NumOps.Multiply(residual, NumOps.Multiply(W2[k, 0], sigD));
                        for (int f = 0; f < inputDim; f++)
                            gW1[f, k] = NumOps.Add(gW1[f, k], NumOps.Multiply(dH, laggedX[s, f]));
                    }
                }

                // Group-lasso gradient: for each input variable i, penalize ||W1[i_lags, :]||_2
                for (int i = 0; i < d; i++)
                {
                    // Compute group norm for variable i's lag block
                    T groupNorm = NumOps.Zero;
                    for (int l = 0; l < MaxLag; l++)
                    {
                        int f = l * d + i;
                        if (f >= inputDim) continue;
                        for (int k = 0; k < h; k++)
                            groupNorm = NumOps.Add(groupNorm, NumOps.Multiply(W1[f, k], W1[f, k]));
                    }
                    double gnorm = Math.Sqrt(Math.Max(NumOps.ToDouble(groupNorm), 1e-12));

                    for (int l = 0; l < MaxLag; l++)
                    {
                        int f = l * d + i;
                        if (f >= inputDim) continue;
                        for (int k = 0; k < h; k++)
                        {
                            T w1grad = NumOps.Divide(NumOps.Multiply(lambda, W1[f, k]),
                                NumOps.FromDouble(gnorm));
                            gW1[f, k] = NumOps.Add(gW1[f, k], w1grad);
                        }
                    }
                }

                // Update weights
                for (int f = 0; f < inputDim; f++)
                    for (int k = 0; k < h; k++)
                        W1[f, k] = NumOps.Subtract(W1[f, k], NumOps.Multiply(lr, gW1[f, k]));
                for (int k = 0; k < h; k++)
                    W2[k, 0] = NumOps.Subtract(W2[k, 0], NumOps.Multiply(lr, gW2[k, 0]));
            }

            // Extract causal strengths: ||W1[i_lags, :]||_2 for each input variable i
            for (int i = 0; i < d; i++)
            {
                if (i == target) continue;
                T groupNorm = NumOps.Zero;
                for (int l = 0; l < MaxLag; l++)
                {
                    int f = l * d + i;
                    if (f >= inputDim) continue;
                    for (int k = 0; k < h; k++)
                        groupNorm = NumOps.Add(groupNorm, NumOps.Multiply(W1[f, k], W1[f, k]));
                }
                T norm = NumOps.FromDouble(Math.Sqrt(NumOps.ToDouble(groupNorm)));
                if (NumOps.GreaterThan(norm, NumOps.FromDouble(0.1)))
                    result[i, target] = norm;
            }
        }

        return result;
    }
}

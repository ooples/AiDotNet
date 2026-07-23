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
[ResearchPaper("Neural Granger Causality", "https://doi.org/10.1109/TPAMI.2021.3065601", Year = 2021, Authors = "Alex Tank, Ian Covert, Nicholas Foti, Ali Shojaie, Emily B. Fox")]
public class NeuralGrangerAlgorithm<T> : TimeSeriesCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "NeuralGranger";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    private readonly int _hiddenUnits;
    private readonly double _learningRate;
    private readonly int _maxEpochs;
    private readonly double _sparsityPenalty;
    private readonly double _edgeThreshold;

    public NeuralGrangerAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyTimeSeriesOptions(options);
        _hiddenUnits = options?.HiddenUnits ?? 10;
        _learningRate = options?.LearningRate ?? 1e-3;
        _maxEpochs = options?.MaxEpochs ?? options?.MaxIterations ?? 100;
        _sparsityPenalty = options?.SparsityPenalty ?? 0.1;
        _edgeThreshold = options?.EdgeThreshold ?? 0.1;
        if (_hiddenUnits < 1)
            throw new ArgumentException("HiddenUnits must be at least 1.");
        if (double.IsNaN(_learningRate) || double.IsInfinity(_learningRate) || _learningRate <= 0)
            throw new ArgumentException("LearningRate must be a positive finite value.");
        if (_maxEpochs < 1)
            throw new ArgumentException("MaxEpochs must be at least 1.");
        if (double.IsNaN(_sparsityPenalty) || double.IsInfinity(_sparsityPenalty) || _sparsityPenalty < 0)
            throw new ArgumentException("SparsityPenalty must be a non-negative finite value.");
        if (double.IsNaN(_edgeThreshold) || double.IsInfinity(_edgeThreshold) || _edgeThreshold < 0)
            throw new ArgumentException("EdgeThreshold must be a non-negative finite value.");
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

        // Put every series on a common scale before applying a single group-sparsity
        // coefficient. Otherwise a change of physical units changes the balance between
        // prediction loss and lambda and therefore changes the selected causal graph.
        var standardizedData = StandardizeSeries(data, n, d);

        var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(42);
        T scale = NumOps.FromDouble(Math.Sqrt(2.0 / inputDim));
        T lr = NumOps.FromDouble(_learningRate);

        var result = new Matrix<T>(d, d);

        // Train a separate MLP for each target variable
        for (int target = 0; target < d; target++)
        {
            // Build lagged data
            var (laggedX, y) = CreateLaggedData(standardizedData, target, MaxLag);

            // Component-wise MLP: W1 (inputDim x h), b1 (h), W2 (h x 1), b2 (1).
            // The paper's equations include a bias at every layer; omitting them forces the
            // sparsity-bearing input weights to also model each series' mean level.
            var W1 = new Matrix<T>(inputDim, h);
            var W2 = new Matrix<T>(h, 1);
            var b1 = new Vector<T>(h);
            T b2 = NumOps.Zero;
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
                var gb1 = new Vector<T>(h);
                T gb2 = NumOps.Zero;

                for (int s = 0; s < effectiveN; s++)
                {
                    // Forward: hidden = sigmoid(x * W1 + b1). Use the raw matrix storage
                    // directly instead of allocating two temporary vectors per hidden unit.
                    var hidden = new T[h];
                    for (int k = 0; k < h; k++)
                    {
                        T sum = b1[k];
                        for (int f = 0; f < inputDim; f++)
                            sum = NumOps.Add(sum, NumOps.Multiply(laggedX[s, f], W1[f, k]));
                        double sv = NumOps.ToDouble(sum);
                        hidden[k] = NumOps.FromDouble(sv > 20 ? 1.0 : sv < -20 ? 0.0 : 1.0 / (1.0 + Math.Exp(-sv)));
                    }

                    T pred = b2;
                    for (int k = 0; k < h; k++)
                        pred = NumOps.Add(pred, NumOps.Multiply(hidden[k], W2[k, 0]));

                    T residual = NumOps.Multiply(NumOps.Subtract(pred, y[s]), invN);
                    gb2 = NumOps.Add(gb2, residual);

                    for (int k = 0; k < h; k++)
                    {
                        gW2[k, 0] = NumOps.Add(gW2[k, 0], NumOps.Multiply(residual, hidden[k]));
                        T sigD = NumOps.Multiply(hidden[k], NumOps.Subtract(NumOps.One, hidden[k]));
                        T dH = NumOps.Multiply(residual, NumOps.Multiply(W2[k, 0], sigD));
                        gb1[k] = NumOps.Add(gb1[k], dH);
                        for (int f = 0; f < inputDim; f++)
                            gW1[f, k] = NumOps.Add(gW1[f, k], NumOps.Multiply(dH, laggedX[s, f]));
                    }
                }

                // ISTA smooth-loss step. The group penalty is deliberately NOT added to this
                // gradient: the paper applies its exact proximal map after the smooth update so
                // complete input groups can become exactly zero (the interpretable criterion for
                // Granger non-causality).
                for (int f = 0; f < inputDim; f++)
                    for (int k = 0; k < h; k++)
                        W1[f, k] = NumOps.Subtract(W1[f, k], NumOps.Multiply(lr, gW1[f, k]));
                for (int k = 0; k < h; k++)
                {
                    W2[k, 0] = NumOps.Subtract(W2[k, 0], NumOps.Multiply(lr, gW2[k, 0]));
                    b1[k] = NumOps.Subtract(b1[k], NumOps.Multiply(lr, gb1[k]));
                }
                b2 = NumOps.Subtract(b2, NumOps.Multiply(lr, gb2));

                ApplyGroupLassoProximal(W1, d, h, _learningRate * _sparsityPenalty);
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
                if (NumOps.GreaterThan(norm, NumOps.FromDouble(_edgeThreshold)))
                    result[i, target] = norm;
            }
        }

        return result;
    }

    private Matrix<T> StandardizeSeries(Matrix<T> data, int sampleCount, int variableCount)
    {
        var standardized = new Matrix<T>(sampleCount, variableCount);
        for (int variable = 0; variable < variableCount; variable++)
        {
            double mean = 0.0;
            for (int sample = 0; sample < sampleCount; sample++)
                mean += NumOps.ToDouble(data[sample, variable]);
            mean /= sampleCount;

            double squaredDeviation = 0.0;
            for (int sample = 0; sample < sampleCount; sample++)
            {
                double deviation = NumOps.ToDouble(data[sample, variable]) - mean;
                squaredDeviation += deviation * deviation;
            }

            double standardDeviation = Math.Sqrt(squaredDeviation / sampleCount);
            if (standardDeviation < 1e-12)
                continue;

            for (int sample = 0; sample < sampleCount; sample++)
            {
                double value = (NumOps.ToDouble(data[sample, variable]) - mean) / standardDeviation;
                standardized[sample, variable] = NumOps.FromDouble(value);
            }
        }

        return standardized;
    }

    /// <summary>
    /// Applies the group-lasso proximal map to all lagged first-layer weights for each input series.
    /// </summary>
    private void ApplyGroupLassoProximal(Matrix<T> weights, int variableCount, int hiddenUnits, double threshold)
    {
        if (threshold <= 0) return;

        for (int variable = 0; variable < variableCount; variable++)
        {
            double squaredNorm = 0.0;
            for (int lag = 0; lag < MaxLag; lag++)
            {
                int feature = lag * variableCount + variable;
                for (int hidden = 0; hidden < hiddenUnits; hidden++)
                {
                    double value = NumOps.ToDouble(weights[feature, hidden]);
                    squaredNorm += value * value;
                }
            }

            double norm = Math.Sqrt(squaredNorm);
            double shrinkage = norm <= threshold ? 0.0 : 1.0 - threshold / norm;
            T shrinkageT = NumOps.FromDouble(shrinkage);
            for (int lag = 0; lag < MaxLag; lag++)
            {
                int feature = lag * variableCount + variable;
                for (int hidden = 0; hidden < hiddenUnits; hidden++)
                    weights[feature, hidden] = NumOps.Multiply(weights[feature, hidden], shrinkageT);
            }
        }
    }
}

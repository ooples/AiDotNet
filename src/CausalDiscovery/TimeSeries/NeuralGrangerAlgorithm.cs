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
        var standardizedData = StandardizeColumns(data);

        var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(42);
        T scale = NumOps.FromDouble(Math.Sqrt(2.0 / inputDim));
        T lr = NumOps.FromDouble(_learningRate);

        var columns = new Tensor<T>[d];
        T one = NumOps.One;
        T saturateHigh = NumOps.FromDouble(20);
        T saturateLow = NumOps.FromDouble(-20);
        T edgeThreshold = NumOps.FromDouble(_edgeThreshold);
        T half = NumOps.FromDouble(0.5);
        T invN = NumOps.FromDouble(1.0 / effectiveN);
        var offDiagonal = OffDiagonalMask(d);

        // Train a separate MLP for each target variable
        for (int target = 0; target < d; target++)
        {
            // Build lagged data
            var (laggedX, y) = CreateLaggedData(standardizedData, target, MaxLag);
            var x = Tensor<T>.FromMatrix(laggedX);
            var xT = Engine.TensorTranspose(x);
            var response = y.AsTensor().Reshape(new[] { effectiveN, 1 });

            // Component-wise MLP: W1 (inputDim x h), b1 (h), W2 (h x 1), b2 (1).
            // The paper's equations include a bias at every layer; omitting them forces the
            // sparsity-bearing input weights to also model each series' mean level.
            // Weights are scale * (U[0,1) - 0.5), drawn row-major: all of W1, then W2.
            var W1 = Engine.TensorMultiplyScalar(Engine.TensorSubtractScalar(Tensor<T>.CreateRandom(rng, inputDim, h), half), scale);
            var W2 = Engine.TensorMultiplyScalar(Engine.TensorSubtractScalar(Tensor<T>.CreateRandom(rng, h, 1), half), scale);
            var b1 = new Tensor<T>(new[] { 1, h });
            var b2 = new Tensor<T>(new[] { 1, 1 });

            for (int epoch = 0; epoch < _maxEpochs; epoch++)
            {
                // Forward over all samples: hidden = sigmoid(x * W1 + b1), saturated to exactly 1 above 20
                // and 0 below -20.
                var preActivation = Engine.TensorAdd(Engine.TensorMatMul(x, W1), Engine.TensorBroadcastTo(b1, new[] { effectiveN, h }));
                var logistic = Engine.TensorReciprocal(Engine.TensorAddScalar(Engine.TensorExp(Engine.TensorNegate(preActivation)), one));
                var hidden = Engine.TensorWhere(Engine.TensorGreaterThan(preActivation, saturateHigh), Engine.TensorOnesLike(preActivation),
                    Engine.TensorWhere(Engine.TensorLessThan(preActivation, saturateLow), Engine.TensorZerosLike(preActivation), logistic));

                var pred = Engine.TensorAdd(Engine.TensorMatMul(hidden, W2), Engine.TensorBroadcastTo(b2, new[] { effectiveN, 1 }));
                var residual = Engine.TensorMultiplyScalar(Engine.TensorSubtract(pred, response), invN); // [N, 1]

                // Backward
                var gb2 = Engine.ReduceSum(residual, new[] { 0 }, true);
                var gW2 = Engine.TensorMatMul(Engine.TensorTranspose(hidden), residual);
                var sigD = Engine.TensorMultiply(hidden, Engine.TensorSubtract(Engine.TensorOnesLike(hidden), hidden));
                var dH = Engine.TensorMultiply(
                    Engine.TensorMultiply(Engine.TensorBroadcastTo(residual, new[] { effectiveN, h }),
                        Engine.TensorBroadcastTo(Engine.TensorTranspose(W2), new[] { effectiveN, h })),
                    sigD);
                var gb1 = Engine.ReduceSum(dH, new[] { 0 }, true);
                var gW1 = Engine.TensorMatMul(xT, dH);

                // ISTA smooth-loss step. The group penalty is deliberately NOT added to this
                // gradient: the paper applies its exact proximal map after the smooth update so
                // complete input groups can become exactly zero (the interpretable criterion for
                // Granger non-causality).
                W1 = Engine.TensorSubtract(W1, Engine.TensorMultiplyScalar(gW1, lr));
                W2 = Engine.TensorSubtract(W2, Engine.TensorMultiplyScalar(gW2, lr));
                b1 = Engine.TensorSubtract(b1, Engine.TensorMultiplyScalar(gb1, lr));
                b2 = Engine.TensorSubtract(b2, Engine.TensorMultiplyScalar(gb2, lr));

                W1 = ApplyGroupLassoProximal(W1, d, h, _learningRate * _sparsityPenalty);
            }

            // Extract causal strengths: ||W1[i_lags, :]||_2 for each input variable i
            var norm = Engine.TensorSqrt(GroupSquaredNorms(W1, d, h));
            var keep = Engine.TensorGreaterThan(norm, edgeThreshold);
            columns[target] = Engine.TensorWhere(keep, norm, Engine.TensorZerosLike(norm));
        }

        // result[i, target] = strength of input series i for that target (never i == target)
        var strengths = Engine.TensorStack(columns, 1);
        return Engine.TensorWhere(offDiagonal, strengths, Engine.TensorZerosLike(strengths)).ToMatrix();
    }

    /// <summary>
    /// Squared L2 norm of every input series' group of first-layer weights: the rows lag * variableCount + variable
    /// of all lags, across all hidden units. Returns [variableCount].
    /// </summary>
    private Tensor<T> GroupSquaredNorms(Tensor<T> weights, int variableCount, int hiddenUnits)
    {
        var grouped = weights.Reshape(new[] { MaxLag, variableCount, hiddenUnits });
        return Engine.ReduceSum(Engine.TensorMultiply(grouped, grouped), new[] { 0, 2 }, false);
    }

    /// <summary>
    /// Applies the group-lasso proximal map to all lagged first-layer weights for each input series.
    /// </summary>
private Tensor<T> ApplyGroupLassoProximal(Tensor<T> weights, int variableCount, int hiddenUnits, double threshold)
    {
        if (threshold <= 0) return weights;

        // Each series' group is scaled by max(0, 1 - threshold / ||group||); a NaN norm keeps NaN.
        var norm = Engine.TensorSqrt(GroupSquaredNorms(weights, variableCount, hiddenUnits));
        T thresholdT = NumOps.FromDouble(threshold);
        var withinThreshold = Engine.TensorAdd(Engine.TensorLessThan(norm, thresholdT), Engine.TensorEquals(norm, thresholdT));
        var shrinkage = Engine.TensorWhere(withinThreshold, Engine.TensorZerosLike(norm),
            Engine.TensorSubtract(Engine.TensorOnesLike(norm), Engine.TensorDivide(Engine.TensorFullLike(norm, thresholdT), norm)));

        var grouped = weights.Reshape(new[] { MaxLag, variableCount, hiddenUnits });
        var factor = Engine.TensorBroadcastTo(shrinkage.Reshape(new[] { 1, variableCount, 1 }), new[] { MaxLag, variableCount, hiddenUnits });
        return Engine.TensorMultiply(grouped, factor).Reshape(new[] { MaxLag * variableCount, hiddenUnits });
    }
}

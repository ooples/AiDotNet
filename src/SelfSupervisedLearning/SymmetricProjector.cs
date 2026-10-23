using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.SelfSupervisedLearning;

/// <summary>
/// Symmetric Projector Head for BYOL and SimSiam-style methods.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> The symmetric projector is used in BYOL and SimSiam.
/// It consists of a projector MLP followed by a predictor MLP. The predictor
/// creates asymmetry between online and target branches, which is key to avoiding collapse.</para>
///
/// <para><b>Architecture:</b></para>
/// <list type="bullet">
/// <item><b>Projector:</b> Linear -> BN -> ReLU -> Linear -> BN</item>
/// <item><b>Predictor:</b> Linear -> BN -> ReLU -> Linear</item>
/// </list>
///
/// <para><b>Key insight:</b> The predictor is only applied to the online branch,
/// creating asymmetry. The target branch only uses the projector.</para>
///
/// <para><b>Gradients:</b> every operation is an engine op on tensor parameters, so a training step records the
/// projector and predictor on the gradient tape. The earlier implementation kept its weights in plain arrays and
/// computed with hand-written loops, so no gradient could reach it, or the encoder behind it, through the tape.</para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Embedding)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Bootstrap Your Own Latent - A New Approach to Self-Supervised Learning", "https://arxiv.org/abs/2006.07733", Year = 2020, Authors = "Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Rémi Munos, Michal Valko")]
public class SymmetricProjector<T> : IProjectorHead<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private static IEngine Engine => AiDotNetEngine.Current;
    private const double BatchNormEpsilon = 1e-5;
    // torch.nn.BatchNorm1d's default momentum for the running statistics.
    private const double BatchNormMomentum = 0.1;

    private readonly int _inputDim;
    private readonly int _hiddenDim;
    private readonly int _projectionDim;
    private readonly int _predictorHiddenDim;
    private readonly bool _hasPredictor;

    // Projector parameters. Weights are [in, out], row-major, the layout GetParameters has always used.
    private readonly Tensor<T> _projWeight1;
    private readonly Tensor<T> _projBias1;
    private readonly Tensor<T> _projBn1Gamma;
    private readonly Tensor<T> _projBn1Beta;
    private readonly Tensor<T> _projWeight2;
    private readonly Tensor<T> _projBias2;
    private readonly Tensor<T> _projBn2Gamma;
    private readonly Tensor<T> _projBn2Beta;

    // Predictor parameters (optional)
    private readonly Tensor<T>? _predWeight1;
    private readonly Tensor<T>? _predBias1;
    private readonly Tensor<T>? _predBn1Gamma;
    private readonly Tensor<T>? _predBn1Beta;
    private readonly Tensor<T>? _predWeight2;
    private readonly Tensor<T>? _predBias2;

    // Running statistics of each BatchNorm (buffers, not parameters): updated in training mode, used in evaluation
    // mode, as torch.nn.BatchNorm1d does.
    private readonly Tensor<T> _projBn1Mean;
    private readonly Tensor<T> _projBn1Var;
    private readonly Tensor<T> _projBn2Mean;
    private readonly Tensor<T> _projBn2Var;
    private readonly Tensor<T>? _predBn1Mean;
    private readonly Tensor<T>? _predBn1Var;

    private Tensor<T> PredWeight1 => _predWeight1 ?? throw new InvalidOperationException("Predictor weight1 not initialized. Ensure _hasPredictor is true.");
    private Tensor<T> PredBias1 => _predBias1 ?? throw new InvalidOperationException("Predictor bias1 not initialized.");
    private Tensor<T> PredBn1Gamma => _predBn1Gamma ?? throw new InvalidOperationException("Predictor BN1 gamma not initialized.");
    private Tensor<T> PredBn1Beta => _predBn1Beta ?? throw new InvalidOperationException("Predictor BN1 beta not initialized.");
    private Tensor<T> PredWeight2 => _predWeight2 ?? throw new InvalidOperationException("Predictor weight2 not initialized.");
    private Tensor<T> PredBias2 => _predBias2 ?? throw new InvalidOperationException("Predictor bias2 not initialized.");

    /// <inheritdoc />
    public int InputDimension => _inputDim;

    /// <inheritdoc />
    public int OutputDimension => _projectionDim;

    /// <inheritdoc />
    public int? HiddenDimension => _hiddenDim;

    /// <inheritdoc />
    /// <remarks>
    /// Sums the lengths of the same tensors <see cref="GetParameters"/> concatenates, so the two cannot disagree.
    /// </remarks>
    public long ParameterCount
    {
        get
        {
            long total = 0;
            foreach (var block in GetTrainableTensors()) total += block.Length;
            return total;
        }
    }

    private bool _isTraining = true;

    /// <summary>
    /// Gets whether this projector has a predictor head.
    /// </summary>
    public bool HasPredictor => _hasPredictor;

    /// <summary>
    /// Initializes a new instance of the SymmetricProjector class.
    /// </summary>
    /// <param name="inputDim">Input dimension from encoder.</param>
    /// <param name="hiddenDim">Hidden dimension of the projector (default: 4096).</param>
    /// <param name="projectionDim">Output dimension (default: 256).</param>
    /// <param name="predictorHiddenDim">Hidden dimension of predictor (default: 4096). Set to 0 to disable predictor.</param>
    /// <param name="seed">Random seed for initialization.</param>
    public SymmetricProjector(
        int inputDim,
        int hiddenDim = 4096,
        int projectionDim = 256,
        int predictorHiddenDim = 4096,
        int? seed = null)
    {
        _inputDim = inputDim;
        _hiddenDim = hiddenDim;
        _projectionDim = projectionDim;
        _predictorHiddenDim = predictorHiddenDim;
        _hasPredictor = predictorHiddenDim > 0;

        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.Shared;

        // Same draw order as before, so a seeded projector starts from the same values.
        _projWeight1 = InitializeWeight(inputDim, hiddenDim, rng);
        _projBias1 = new Tensor<T>(new[] { hiddenDim });
        _projBn1Gamma = InitializeOnes(hiddenDim);
        _projBn1Beta = new Tensor<T>(new[] { hiddenDim });
        _projWeight2 = InitializeWeight(hiddenDim, projectionDim, rng);
        _projBias2 = new Tensor<T>(new[] { projectionDim });
        _projBn2Gamma = InitializeOnes(projectionDim);
        _projBn2Beta = new Tensor<T>(new[] { projectionDim });
        _projBn1Mean = new Tensor<T>(new[] { hiddenDim });
        _projBn1Var = InitializeOnes(hiddenDim);
        _projBn2Mean = new Tensor<T>(new[] { projectionDim });
        _projBn2Var = InitializeOnes(projectionDim);

        if (_hasPredictor)
        {
            _predWeight1 = InitializeWeight(projectionDim, predictorHiddenDim, rng);
            _predBias1 = new Tensor<T>(new[] { predictorHiddenDim });
            _predBn1Gamma = InitializeOnes(predictorHiddenDim);
            _predBn1Beta = new Tensor<T>(new[] { predictorHiddenDim });
            _predWeight2 = InitializeWeight(predictorHiddenDim, projectionDim, rng);
            _predBias2 = new Tensor<T>(new[] { projectionDim });
            _predBn1Mean = new Tensor<T>(new[] { predictorHiddenDim });
            _predBn1Var = InitializeOnes(predictorHiddenDim);
        }
    }

    /// <inheritdoc />
    public Tensor<T> Project(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));

        // Projector: Linear -> BN -> ReLU -> Linear -> BN
        var h = Engine.ReLU(BatchNorm(Linear(input, _projWeight1, _projBias1), _projBn1Gamma, _projBn1Beta, _projBn1Mean, _projBn1Var));
        return BatchNorm(Linear(h, _projWeight2, _projBias2), _projBn2Gamma, _projBn2Beta, _projBn2Mean, _projBn2Var);
    }

    /// <summary>
    /// Applies the predictor head to a projection (for the online branch only); a projector without one returns the
    /// projection.
    /// </summary>
    /// <param name="projection">Output from the projector.</param>
    public Tensor<T> Predict(Tensor<T> projection)
    {
        if (projection is null) throw new ArgumentNullException(nameof(projection));
        if (!_hasPredictor || _predBn1Mean is null || _predBn1Var is null)
            return projection;

        // Predictor: Linear -> BN -> ReLU -> Linear
        var h = Engine.ReLU(BatchNorm(Linear(projection, PredWeight1, PredBias1), PredBn1Gamma, PredBn1Beta, _predBn1Mean, _predBn1Var));
        return Linear(h, PredWeight2, PredBias2);
    }

    /// <summary>
    /// Applies the predictor head. The branch index is ignored: the predictor holds no per-branch state, so every
    /// branch computes the same function.
    /// </summary>
    /// <param name="projection">Output from the projector.</param>
    /// <param name="branchIndex">0, 1 or -1; ignored.</param>
    [Obsolete("The predictor computes the same function for every branch; call Predict(projection).")]
    public Tensor<T> Predict(Tensor<T> projection, int branchIndex)
    {
        if (branchIndex < -1 || branchIndex > 1)
            throw new ArgumentOutOfRangeException(nameof(branchIndex), "Branch index must be -1, 0, or 1.");
        return Predict(projection);
    }

    /// <summary>
    /// Projects and predicts in one call (convenience method).
    /// </summary>
    public Tensor<T> ProjectAndPredict(Tensor<T> input) => Predict(Project(input));

    /// <inheritdoc />
    public Vector<T> GetParameters()
    {
        var all = new List<T>();
        foreach (var block in GetTrainableTensors())
            for (int i = 0; i < block.Length; i++) all.Add(block[i]);
        return new Vector<T>([.. all]);
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        int expected = ParameterCountHelper.ToFlatVectorSize(ParameterCount);
        if (parameters.Length != expected)
        {
            throw new ArgumentException(
                $"Parameter vector length {parameters.Length} does not match expected {expected}.",
                nameof(parameters));
        }

        // Written into the existing tensors, so an optimizer keyed by tensor reference keeps its state.
        int offset = 0;
        foreach (var block in GetTrainableTensors())
        {
            for (int i = 0; i < block.Length; i++) block[i] = parameters[offset + i];
            offset += block.Length;
        }
    }

    /// <inheritdoc />
    public IReadOnlyList<Tensor<T>> GetTrainableTensors()
    {
        var tensors = new List<Tensor<T>>
        {
            _projWeight1, _projBias1, _projBn1Gamma, _projBn1Beta,
            _projWeight2, _projBias2, _projBn2Gamma, _projBn2Beta,
        };
        if (_hasPredictor)
        {
            tensors.Add(PredWeight1);
            tensors.Add(PredBias1);
            tensors.Add(PredBn1Gamma);
            tensors.Add(PredBn1Beta);
            tensors.Add(PredWeight2);
            tensors.Add(PredBias2);
        }

        return tensors;
    }

    /// <inheritdoc />
    /// <exception cref="NotSupportedException">Always: the gradient tape differentiates this head.</exception>
    public Vector<T> GetParameterGradients() => throw new NotSupportedException(
        "SymmetricProjector keeps no gradients: record Project and Predict on a GradientTape and differentiate GetTrainableTensors().");

    /// <inheritdoc />
    /// <remarks>This head keeps no gradients, so there is nothing to clear.</remarks>
    public void ClearGradients()
    {
    }

    /// <inheritdoc />
    /// <remarks>Training mode normalizes with each batch's statistics and updates the running ones; evaluation mode
    /// normalizes with the running statistics.</remarks>
    public void SetTrainingMode(bool isTraining)
    {
        _isTraining = isTraining;
    }

    /// <inheritdoc />
    /// <remarks>The head keeps no per-call state; Reset leaves the weights and running statistics as they are.</remarks>
    public void Reset()
    {
    }

    private static Tensor<T> InitializeWeight(int fanIn, int fanOut, Random rng)
    {
        var weights = new Tensor<T>(new[] { fanIn, fanOut });
        var scale = Math.Sqrt(2.0 / fanIn); // He initialization

        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = NumOps.FromDouble((rng.NextDouble() * 2 - 1) * scale);
        }

        return weights;
    }

    private static Tensor<T> InitializeOnes(int size)
    {
        var ones = new Tensor<T>(new[] { size });
        for (int i = 0; i < size; i++)
        {
            ones[i] = NumOps.One;
        }
        return ones;
    }

    // input [batch, in] @ weight [in, out] + bias [out].
    private static Tensor<T> Linear(Tensor<T> input, Tensor<T> weight, Tensor<T> bias)
    {
        var output = Engine.TensorMatMul(input, weight);
        return Engine.TensorAdd(output, Engine.Reshape(bias, new[] { 1, bias.Length }));
    }

    // torch.nn.BatchNorm1d: in training, the batch's statistics normalize and update the running ones (the variance
    // unbiased, as PyTorch keeps it); in evaluation, the running statistics normalize.
    private Tensor<T> BatchNorm(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta, Tensor<T> runningMean, Tensor<T> runningVar)
    {
        int batch = input.Shape[0], dim = input.Shape[1];
        if (_isTraining)
        {
            var output = Engine.BatchNorm(input, gamma, beta, BatchNormEpsilon, out var mean, out var variance);
            double unbias = batch > 1 ? batch / (batch - 1.0) : 1.0;
            for (int d = 0; d < dim; d++)
            {
                runningMean[d] = NumOps.FromDouble((1 - BatchNormMomentum) * NumOps.ToDouble(runningMean[d])
                    + BatchNormMomentum * NumOps.ToDouble(mean[d]));
                runningVar[d] = NumOps.FromDouble((1 - BatchNormMomentum) * NumOps.ToDouble(runningVar[d])
                    + BatchNormMomentum * NumOps.ToDouble(variance[d]) * unbias);
            }
            return output;
        }

        // y = x * scale + shift, scale = gamma / sqrt(var + eps), shift = beta - mean * scale.
        var scale = new Tensor<T>(new[] { 1, dim });
        var shift = new Tensor<T>(new[] { 1, dim });
        for (int d = 0; d < dim; d++)
        {
            double s = NumOps.ToDouble(gamma[d]) / Math.Sqrt(NumOps.ToDouble(runningVar[d]) + BatchNormEpsilon);
            scale[0, d] = NumOps.FromDouble(s);
            shift[0, d] = NumOps.FromDouble(NumOps.ToDouble(beta[d]) - NumOps.ToDouble(runningMean[d]) * s);
        }
        var shape = new[] { batch, dim };
        return Engine.TensorAdd(Engine.TensorMultiply(input, Engine.TensorBroadcastTo(scale, shape)), Engine.TensorBroadcastTo(shift, shape));
    }
}
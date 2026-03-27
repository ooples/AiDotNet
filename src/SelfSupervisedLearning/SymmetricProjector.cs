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
/// <para><b>Dual-branch caching:</b> This projector supports two concurrent forward contexts
/// (branch 1 and branch 2) so that symmetric multi-view training (BYOL, SimSiam, BarlowTwins)
/// can call Project() twice and then Backward() twice without the second forward overwriting
/// the first branch's cached activations.</para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Embedding)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ModelPaper("Bootstrap Your Own Latent - A New Approach to Self-Supervised Learning", "https://arxiv.org/abs/2006.07733", Year = 2020, Authors = "Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Rémi Munos, Michal Valko")]
public class SymmetricProjector<T> : IProjectorHead<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private static IEngine Engine => AiDotNetEngine.Current;

    private readonly int _inputDim;
    private readonly int _hiddenDim;
    private readonly int _projectionDim;
    private readonly int _predictorHiddenDim;
    private readonly bool _hasPredictor;

    // Projector parameters
    private T[] _projWeight1;
    private T[] _projBias1;
    private T[] _projBn1Gamma;
    private T[] _projBn1Beta;
    private T[] _projWeight2;
    private T[] _projBias2;
    private T[] _projBn2Gamma;
    private T[] _projBn2Beta;

    // Predictor parameters (optional)
    private T[]? _predWeight1;
    private T[]? _predBias1;
    private T[]? _predBn1Gamma;
    private T[]? _predBn1Beta;
    private T[]? _predWeight2;
    private T[]? _predBias2;

    /// <summary>
    /// Per-forward-pass cached activations and BN statistics for one branch.
    /// </summary>
    private sealed class ForwardContext
    {
        public Tensor<T>? CachedInput;
        public Tensor<T>? CachedH1;
        public Tensor<T>? CachedH1Bn;
        public Tensor<T>? CachedH1Relu;
        public Tensor<T>? CachedProjection;
        public Tensor<T>? CachedPredictorInput; // The actual input passed to Predict (post-BN2)
        public Tensor<T>? CachedPredH1;
        public Tensor<T>? CachedPredH1Bn;
        public Tensor<T>? CachedPredH1Relu;

        // BatchNorm cached statistics
        public T[]? ProjBn1Mean;
        public T[]? ProjBn1Var;
        public Tensor<T>? ProjBn1Normalized;
        public T[]? ProjBn2Mean;
        public T[]? ProjBn2Var;
        public Tensor<T>? ProjBn2Normalized;
        public T[]? PredBn1Mean;
        public T[]? PredBn1Var;
        public Tensor<T>? PredBn1Normalized;

        // Cached intermediate backward results (to avoid recomputation in ComputeParameterGradients)
        public Tensor<T>? GradBeforeBn2;
        public Tensor<T>? GradAtH1;

        // BatchNorm gradients
        public T[]? ProjBn1GammaGrad;
        public T[]? ProjBn1BetaGrad;
        public T[]? ProjBn2GammaGrad;
        public T[]? ProjBn2BetaGrad;
        public T[]? PredBn1GammaGrad;
        public T[]? PredBn1BetaGrad;

        public void Clear()
        {
            CachedInput = null;
            CachedH1 = null;
            CachedH1Bn = null;
            CachedH1Relu = null;
            CachedProjection = null;
            CachedPredictorInput = null;
            CachedPredH1 = null;
            CachedPredH1Bn = null;
            CachedPredH1Relu = null;

            ProjBn1Mean = null;
            ProjBn1Var = null;
            ProjBn1Normalized = null;
            ProjBn2Mean = null;
            ProjBn2Var = null;
            ProjBn2Normalized = null;
            PredBn1Mean = null;
            PredBn1Var = null;
            PredBn1Normalized = null;

            GradBeforeBn2 = null;
            GradAtH1 = null;

            ProjBn1GammaGrad = null;
            ProjBn1BetaGrad = null;
            ProjBn2GammaGrad = null;
            ProjBn2BetaGrad = null;
            PredBn1GammaGrad = null;
            PredBn1BetaGrad = null;
        }
    }

    // Dual-branch forward contexts for symmetric multi-view training
    private readonly ForwardContext _branch1 = new();
    private readonly ForwardContext _branch2 = new();
    private int _nextBranch;           // 0 = branch1, 1 = branch2
    private int _nextBackwardBranch;   // 0 = branch1, 1 = branch2

    private Vector<T>? _gradients;

    // Guard properties for predictor parameters
    private T[] PredWeight1 => _predWeight1 ?? throw new InvalidOperationException("Predictor weight1 not initialized. Ensure _hasPredictor is true.");
    private T[] PredBias1 => _predBias1 ?? throw new InvalidOperationException("Predictor bias1 not initialized.");
    private T[] PredBn1Gamma => _predBn1Gamma ?? throw new InvalidOperationException("Predictor BN1 gamma not initialized.");
    private T[] PredBn1Beta => _predBn1Beta ?? throw new InvalidOperationException("Predictor BN1 beta not initialized.");
    private T[] PredWeight2 => _predWeight2 ?? throw new InvalidOperationException("Predictor weight2 not initialized.");
    private T[] PredBias2 => _predBias2 ?? throw new InvalidOperationException("Predictor bias2 not initialized.");

    /// <inheritdoc />
    public int InputDimension => _inputDim;

    /// <inheritdoc />
    public int OutputDimension => _projectionDim;

    /// <inheritdoc />
    public int? HiddenDimension => _hiddenDim;

    /// <inheritdoc />
    public int ParameterCount => ComputeParameterCount();

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

        // Initialize projector
        _projWeight1 = InitializeWeight(inputDim, hiddenDim, rng);
        _projBias1 = new T[hiddenDim];
        _projBn1Gamma = InitializeOnes(hiddenDim);
        _projBn1Beta = new T[hiddenDim];
        _projWeight2 = InitializeWeight(hiddenDim, projectionDim, rng);
        _projBias2 = new T[projectionDim];
        _projBn2Gamma = InitializeOnes(projectionDim);
        _projBn2Beta = new T[projectionDim];

        // Initialize predictor if needed
        if (_hasPredictor)
        {
            _predWeight1 = InitializeWeight(projectionDim, predictorHiddenDim, rng);
            _predBias1 = new T[predictorHiddenDim];
            _predBn1Gamma = InitializeOnes(predictorHiddenDim);
            _predBn1Beta = new T[predictorHiddenDim];
            _predWeight2 = InitializeWeight(predictorHiddenDim, projectionDim, rng);
            _predBias2 = new T[projectionDim];
        }
    }

    private ForwardContext GetNextForwardContext()
    {
        var ctx = _nextBranch == 0 ? _branch1 : _branch2;
        _nextBranch = (_nextBranch + 1) % 2;
        return ctx;
    }

    private ForwardContext GetNextBackwardContext()
    {
        var ctx = _nextBackwardBranch == 0 ? _branch1 : _branch2;
        _nextBackwardBranch = (_nextBackwardBranch + 1) % 2;
        return ctx;
    }

    /// <inheritdoc />
    public Tensor<T> Project(Tensor<T> input)
    {
        var ctx = GetNextForwardContext();
        ctx.Clear();
        ctx.CachedInput = input;

        // Projector: Linear -> BN -> ReLU -> Linear -> BN
        ctx.CachedH1 = Linear(input, _projWeight1, _projBias1, _inputDim, _hiddenDim);
        ctx.CachedH1Bn = BatchNorm(ctx.CachedH1, _projBn1Gamma, _projBn1Beta,
            out ctx.ProjBn1Mean, out ctx.ProjBn1Var, out ctx.ProjBn1Normalized);
        ctx.CachedH1Relu = ReLU(ctx.CachedH1Bn);
        ctx.CachedProjection = Linear(ctx.CachedH1Relu, _projWeight2, _projBias2, _hiddenDim, _projectionDim);
        var projNorm = BatchNorm(ctx.CachedProjection, _projBn2Gamma, _projBn2Beta,
            out ctx.ProjBn2Mean, out ctx.ProjBn2Var, out ctx.ProjBn2Normalized);

        return projNorm;
    }

    /// <summary>
    /// Applies the predictor head using the most recently used branch from <see cref="Project"/>.
    /// </summary>
    public Tensor<T> Predict(Tensor<T> projection) => Predict(projection, -1);

    /// <summary>
    /// Applies the predictor head (for online branch only).
    /// </summary>
    /// <param name="projection">Output from the projector.</param>
    /// <param name="branchIndex">0 for branch1, 1 for branch2, or -1 for most recent (legacy).</param>
    public Tensor<T> Predict(Tensor<T> projection, int branchIndex)
    {
        if (!_hasPredictor)
            return projection;

        ForwardContext ctx;
        if (branchIndex >= 0)
        {
            if (branchIndex > 1)
            {
                throw new ArgumentOutOfRangeException(nameof(branchIndex),
                    "Branch index must be -1, 0, or 1.");
            }
            ctx = branchIndex == 0 ? _branch1 : _branch2;
        }
        else if (branchIndex == -1)
        {
            ctx = _nextBranch == 0 ? _branch2 : _branch1;
        }
        else
        {
            throw new ArgumentOutOfRangeException(nameof(branchIndex),
                "Branch index must be -1, 0, or 1.");
        }

        ctx.CachedPredictorInput = projection;

        // Predictor: Linear -> BN -> ReLU -> Linear
        ctx.CachedPredH1 = Linear(projection, PredWeight1, PredBias1, _projectionDim, _predictorHiddenDim);
        ctx.CachedPredH1Bn = BatchNorm(ctx.CachedPredH1, PredBn1Gamma, PredBn1Beta,
            out ctx.PredBn1Mean, out ctx.PredBn1Var, out ctx.PredBn1Normalized);
        ctx.CachedPredH1Relu = ReLU(ctx.CachedPredH1Bn);
        var output = Linear(ctx.CachedPredH1Relu, PredWeight2, PredBias2, _predictorHiddenDim, _projectionDim);

        return output;
    }

    /// <summary>
    /// Projects and predicts in one call (convenience method).
    /// </summary>
    public Tensor<T> ProjectAndPredict(Tensor<T> input)
    {
        var projection = Project(input);
        return Predict(projection);
    }

    /// <inheritdoc />
    public Tensor<T> Backward(Tensor<T> gradOutput) => Backward(gradOutput, -1);

    /// <summary>
    /// Backward pass with explicit branch selection.
    /// </summary>
    /// <param name="gradOutput">The gradient from downstream.</param>
    /// <param name="branchIndex">0 for branch1, 1 for branch2, or -1 for round-robin (legacy).</param>
    public Tensor<T> Backward(Tensor<T> gradOutput, int branchIndex)
    {
        ForwardContext ctx;
        if (branchIndex >= 0)
        {
            if (branchIndex > 1)
                throw new ArgumentOutOfRangeException(nameof(branchIndex), "Branch index must be 0 or 1.");
            ctx = branchIndex == 0 ? _branch1 : _branch2;
        }
        else
        {
            ctx = GetNextBackwardContext();
        }

        var grad = gradOutput;

        // Backward through predictor if present
        if (_hasPredictor && ctx.CachedPredH1Relu is not null)
        {
            grad = LinearBackward(grad, PredWeight2, _predictorHiddenDim, _projectionDim);
            var cachedPredH1Bn = ctx.CachedPredH1Bn ?? throw new InvalidOperationException(
                "Cached predictor H1 BN not available. Call Project() before Backward().");
            grad = ReLUBackward(grad, cachedPredH1Bn);
            grad = BatchNormBackward(grad, PredBn1Gamma, ctx.PredBn1Var, ctx.PredBn1Normalized,
                out ctx.PredBn1GammaGrad, out ctx.PredBn1BetaGrad);
            grad = LinearBackward(grad, PredWeight1, _projectionDim, _predictorHiddenDim);
        }

        var gradAtProjectorOutput = grad;

        var bn2Var = ctx.ProjBn2Var ?? throw new InvalidOperationException(
            "ProjBn2Var not available. Call Project() before Backward().");
        var bn2Norm = ctx.ProjBn2Normalized ?? throw new InvalidOperationException(
            "ProjBn2Normalized not available. Call Project() before Backward().");
        grad = BatchNormBackward(grad, _projBn2Gamma, bn2Var, bn2Norm,
            out ctx.ProjBn2GammaGrad, out ctx.ProjBn2BetaGrad);
        ctx.GradBeforeBn2 = grad; // Cache for reuse in ComputeParameterGradients
        grad = LinearBackward(grad, _projWeight2, _hiddenDim, _projectionDim);
        var cachedH1Bn = ctx.CachedH1Bn ?? throw new InvalidOperationException(
            "Cached H1 BN not available. Call Project() before Backward().");
        grad = ReLUBackward(grad, cachedH1Bn);
        var bn1Var = ctx.ProjBn1Var ?? throw new InvalidOperationException(
            "ProjBn1Var not available. Call Project() before Backward().");
        var bn1Norm = ctx.ProjBn1Normalized ?? throw new InvalidOperationException(
            "ProjBn1Normalized not available. Call Project() before Backward().");
        grad = BatchNormBackward(grad, _projBn1Gamma, bn1Var, bn1Norm,
            out ctx.ProjBn1GammaGrad, out ctx.ProjBn1BetaGrad);
        ctx.GradAtH1 = grad; // Cache for reuse in ComputeParameterGradients
        grad = LinearBackward(grad, _projWeight1, _inputDim, _hiddenDim);

        // Compute parameter gradients and accumulate across backward passes
        var branchGradients = ComputeParameterGradients(gradAtProjectorOutput, gradOutput, ctx);
        if (_gradients is null)
        {
            _gradients = branchGradients;
        }
        else
        {
            var accumulated = new T[_gradients.Length];
            for (int i = 0; i < accumulated.Length; i++)
            {
                accumulated[i] = NumOps.Add(_gradients[i], branchGradients[i]);
            }
            _gradients = new Vector<T>(accumulated);
        }

        return grad;
    }

    private Tensor<T> LinearBackward(Tensor<T> gradOutput, T[] weight, int inDim, int outDim)
    {
        var batchSize = gradOutput.Shape[0];
        var gradInput = new T[batchSize * inDim];

        // gradInput = gradOutput @ weight.T
        for (int b = 0; b < batchSize; b++)
        {
            // Extract gradOutput row for this batch
            var gradRow = new Vector<T>(outDim);
            for (int j = 0; j < outDim; j++)
            {
                gradRow[j] = gradOutput[b, j];
            }

            for (int i = 0; i < inDim; i++)
            {
                // Extract weight row (contiguous in flat array: weight[i*outDim .. i*outDim+outDim])
                var weightRow = new Vector<T>(outDim);
                for (int j = 0; j < outDim; j++)
                {
                    weightRow[j] = weight[i * outDim + j];
                }
                gradInput[b * inDim + i] = Engine.DotProduct(gradRow, weightRow);
            }
        }

        return new Tensor<T>(gradInput, [batchSize, inDim]);
    }

    private Tensor<T> ReLUBackward(Tensor<T> gradOutput, Tensor<T> preActivation)
    {
        var batchSize = gradOutput.Shape[0];
        var dim = gradOutput.Shape[1];
        var gradInput = new T[batchSize * dim];

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < dim; i++)
            {
                // Gradient is passed through only where input was positive
                var wasPositive = NumOps.GreaterThan(preActivation[b, i], NumOps.Zero);
                gradInput[b * dim + i] = wasPositive ? gradOutput[b, i] : NumOps.Zero;
            }
        }

        return new Tensor<T>(gradInput, [batchSize, dim]);
    }

    /// <summary>
    /// Full BatchNorm backward pass computing gradients for input, gamma, and beta.
    /// </summary>
    /// <remarks>
    /// The full BatchNorm backward follows these equations:
    /// dx = (gamma / std) * (dout - mean(dout) - xhat * mean(dout * xhat))
    /// dgamma = sum(dout * xhat, axis=batch)
    /// dbeta = sum(dout, axis=batch)
    /// where xhat is the normalized input and std = sqrt(var + eps)
    /// </remarks>
    private Tensor<T> BatchNormBackward(
        Tensor<T> gradOutput,
        T[] gamma,
        T[]? variance,
        Tensor<T>? normalizedInput,
        out T[] gammaGrad,
        out T[] betaGrad)
    {
        var batchSize = gradOutput.Shape[0];
        var dim = gradOutput.Shape[1];
        var gradInput = new T[batchSize * dim];
        gammaGrad = new T[dim];
        betaGrad = new T[dim];

        var eps = NumOps.FromDouble(1e-5);
        var invN = NumOps.FromDouble(1.0 / batchSize);

        for (int j = 0; j < dim; j++)
        {
            // Compute std = sqrt(variance + eps)
            var std = variance != null
                ? NumOps.Sqrt(NumOps.Add(variance[j], eps))
                : NumOps.One;
            var invStd = NumOps.Divide(NumOps.One, std);

            // Compute dgamma = sum(dout * xhat, axis=batch)
            // Compute dbeta = sum(dout, axis=batch)
            T dgamma = NumOps.Zero;
            T dbeta = NumOps.Zero;
            for (int b = 0; b < batchSize; b++)
            {
                var dout = gradOutput[b, j];
                var xhat = normalizedInput != null ? normalizedInput[b, j] : NumOps.Zero;
                dgamma = NumOps.Add(dgamma, NumOps.Multiply(dout, xhat));
                dbeta = NumOps.Add(dbeta, dout);
            }
            gammaGrad[j] = dgamma;
            betaGrad[j] = dbeta;

            // Compute mean(dout) and mean(dout * xhat)
            T meanDout = NumOps.Multiply(dbeta, invN);
            T meanDoutXhat = NumOps.Multiply(dgamma, invN);

            // Compute gradInput for each sample:
            // dx = (gamma / std) * (dout - mean(dout) - xhat * mean(dout * xhat))
            var gammaOverStd = NumOps.Multiply(gamma[j], invStd);
            for (int b = 0; b < batchSize; b++)
            {
                var dout = gradOutput[b, j];
                var xhat = normalizedInput != null ? normalizedInput[b, j] : NumOps.Zero;

                // dout - mean(dout) - xhat * mean(dout * xhat)
                var term = NumOps.Subtract(dout, meanDout);
                term = NumOps.Subtract(term, NumOps.Multiply(xhat, meanDoutXhat));

                gradInput[b * dim + j] = NumOps.Multiply(gammaOverStd, term);
            }
        }

        return new Tensor<T>(gradInput, [batchSize, dim]);
    }

    private Vector<T> ComputeParameterGradients(Tensor<T> gradAtProjectorOutput, Tensor<T> originalGradOutput, ForwardContext ctx)
    {
        var grads = new T[ParameterCount];
        var batchSize = gradAtProjectorOutput.Shape[0];
        var invBatchSize = NumOps.FromDouble(1.0 / batchSize);
        int offset = 0;

        // If we don't have cached activations, we can't compute proper gradients
        if (ctx.CachedInput is null || ctx.CachedH1Relu is null || ctx.CachedProjection is null)
        {
            return new Vector<T>(grads);
        }

        var cachedInput = ctx.CachedInput;
        var cachedH1Relu = ctx.CachedH1Relu;

        // Reuse BN2 backward result cached during Backward() to avoid redundant recomputation
        var gradBeforeBn2 = ctx.GradBeforeBn2 ?? throw new InvalidOperationException(
            "GradBeforeBn2 not cached. Call Backward() before ComputeParameterGradients().");

        // Compute gradients for projWeight2: cachedH1Relu.T @ gradBeforeBn2
        for (int i = 0; i < _hiddenDim; i++)
        {
            // Extract column i from cachedH1Relu (all batches)
            var h1Col = new Vector<T>(batchSize);
            for (int b = 0; b < batchSize; b++)
            {
                h1Col[b] = cachedH1Relu[b, i];
            }

            for (int j = 0; j < _projectionDim; j++)
            {
                // Extract column j from gradBeforeBn2 (all batches)
                var gradCol = new Vector<T>(batchSize);
                for (int b = 0; b < batchSize; b++)
                {
                    gradCol[b] = gradBeforeBn2[b, j];
                }
                grads[offset + _projWeight1.Length + _projBias1.Length + _projBn1Gamma.Length + _projBn1Beta.Length + i * _projectionDim + j] =
                    NumOps.Multiply(Engine.DotProduct(h1Col, gradCol), invBatchSize);
            }
        }

        // Compute gradients for projBias2: sum of gradBeforeBn2 across batch
        int bias2Offset = offset + _projWeight1.Length + _projBias1.Length + _projBn1Gamma.Length + _projBn1Beta.Length + _projWeight2.Length;
        for (int j = 0; j < _projectionDim; j++)
        {
            T sum = NumOps.Zero;
            for (int b = 0; b < batchSize; b++)
            {
                sum = NumOps.Add(sum, gradBeforeBn2[b, j]);
            }
            grads[bias2Offset + j] = NumOps.Multiply(sum, invBatchSize);
        }

        // Reuse BN1 backward result cached during Backward() to avoid redundant recomputation
        var gradAtH1 = ctx.GradAtH1 ?? throw new InvalidOperationException(
            "GradAtH1 not cached. Call Backward() before ComputeParameterGradients().");

        // Compute gradients for projWeight1: cachedInput.T @ gradAtH1
        for (int i = 0; i < _inputDim; i++)
        {
            // Extract column i from cachedInput (all batches)
            var inputCol = new Vector<T>(batchSize);
            for (int b = 0; b < batchSize; b++)
            {
                inputCol[b] = cachedInput[b, i];
            }

            for (int j = 0; j < _hiddenDim; j++)
            {
                // Extract column j from gradAtH1 (all batches)
                var gradCol = new Vector<T>(batchSize);
                for (int b = 0; b < batchSize; b++)
                {
                    gradCol[b] = gradAtH1[b, j];
                }
                grads[offset + i * _hiddenDim + j] = NumOps.Multiply(Engine.DotProduct(inputCol, gradCol), invBatchSize);
            }
        }

        // Compute gradients for projBias1
        int bias1Offset = offset + _projWeight1.Length;
        for (int j = 0; j < _hiddenDim; j++)
        {
            T sum = NumOps.Zero;
            for (int b = 0; b < batchSize; b++)
            {
                sum = NumOps.Add(sum, gradAtH1[b, j]);
            }
            grads[bias1Offset + j] = NumOps.Multiply(sum, invBatchSize);
        }

        // Use the properly computed BN gradients from BatchNormBackward
        int bn1GammaOffset = bias1Offset + _projBias1.Length;
        int bn1BetaOffset = bn1GammaOffset + _projBn1Gamma.Length;
        if (ctx.ProjBn1GammaGrad != null && ctx.ProjBn1BetaGrad != null)
        {
            for (int j = 0; j < _hiddenDim; j++)
            {
                grads[bn1GammaOffset + j] = ctx.ProjBn1GammaGrad[j];
                grads[bn1BetaOffset + j] = ctx.ProjBn1BetaGrad[j];
            }
        }

        int bn2GammaOffset = bias2Offset + _projBias2.Length;
        int bn2BetaOffset = bn2GammaOffset + _projBn2Gamma.Length;
        if (ctx.ProjBn2GammaGrad != null && ctx.ProjBn2BetaGrad != null)
        {
            for (int j = 0; j < _projectionDim; j++)
            {
                grads[bn2GammaOffset + j] = ctx.ProjBn2GammaGrad[j];
                grads[bn2BetaOffset + j] = ctx.ProjBn2BetaGrad[j];
            }
        }

        // Predictor gradients (weight, bias, and BN)
        if (_hasPredictor)
        {
            int predOffset = bn2BetaOffset + _projBn2Beta.Length;
            int predWeight1Offset = predOffset;
            int predBias1Offset = predWeight1Offset + PredWeight1.Length;
            int predBn1GammaOffset = predBias1Offset + PredBias1.Length;
            int predBn1BetaOffset = predBn1GammaOffset + PredBn1Gamma.Length;
            int predWeight2Offset = predBn1BetaOffset + PredBn1Beta.Length;
            int predBias2Offset = predWeight2Offset + PredWeight2.Length;

            // PredWeight2/PredBias2 gradients
            if (ctx.CachedPredH1Relu is not null)
            {
                for (int i = 0; i < _predictorHiddenDim; i++)
                {
                    // Extract column i from CachedPredH1Relu (all batches)
                    var predH1Col = new Vector<T>(batchSize);
                    for (int b = 0; b < batchSize; b++)
                    {
                        predH1Col[b] = ctx.CachedPredH1Relu[b, i];
                    }

                    for (int j = 0; j < _projectionDim; j++)
                    {
                        // Extract column j from originalGradOutput (all batches)
                        var origGradCol = new Vector<T>(batchSize);
                        for (int b = 0; b < batchSize; b++)
                        {
                            origGradCol[b] = originalGradOutput[b, j];
                        }
                        grads[predWeight2Offset + i * _projectionDim + j] = NumOps.Multiply(Engine.DotProduct(predH1Col, origGradCol), invBatchSize);
                    }
                }

                for (int j = 0; j < _projectionDim; j++)
                {
                    T sum = NumOps.Zero;
                    for (int b = 0; b < batchSize; b++)
                    {
                        sum = NumOps.Add(sum, originalGradOutput[b, j]);
                    }
                    grads[predBias2Offset + j] = NumOps.Multiply(sum, invBatchSize);
                }
            }

            // PredWeight1/PredBias1 gradients
            if (ctx.CachedPredictorInput is not null)
            {
                var gradBeforePredBn1 = LinearBackward(originalGradOutput, PredWeight2, _predictorHiddenDim, _projectionDim);
                if (ctx.CachedPredH1Bn is not null)
                {
                    gradBeforePredBn1 = ReLUBackward(gradBeforePredBn1, ctx.CachedPredH1Bn);
                }
                var gradAtPredH1 = BatchNormBackward(gradBeforePredBn1, PredBn1Gamma, ctx.PredBn1Var, ctx.PredBn1Normalized,
                    out _, out _);

                for (int i = 0; i < _projectionDim; i++)
                {
                    // Extract column i from CachedPredictorInput (all batches)
                    var predInputCol = new Vector<T>(batchSize);
                    for (int b = 0; b < batchSize; b++)
                    {
                        predInputCol[b] = ctx.CachedPredictorInput[b, i];
                    }

                    for (int j = 0; j < _predictorHiddenDim; j++)
                    {
                        // Extract column j from gradAtPredH1 (all batches)
                        var gradPredCol = new Vector<T>(batchSize);
                        for (int b = 0; b < batchSize; b++)
                        {
                            gradPredCol[b] = gradAtPredH1[b, j];
                        }
                        grads[predWeight1Offset + i * _predictorHiddenDim + j] = NumOps.Multiply(Engine.DotProduct(predInputCol, gradPredCol), invBatchSize);
                    }
                }

                for (int j = 0; j < _predictorHiddenDim; j++)
                {
                    T sum = NumOps.Zero;
                    for (int b = 0; b < batchSize; b++)
                    {
                        sum = NumOps.Add(sum, gradAtPredH1[b, j]);
                    }
                    grads[predBias1Offset + j] = NumOps.Multiply(sum, invBatchSize);
                }
            }

            // Predictor BN1 gamma/beta gradients
            if (ctx.PredBn1GammaGrad != null && ctx.PredBn1BetaGrad != null)
            {
                for (int j = 0; j < _predictorHiddenDim; j++)
                {
                    grads[predBn1GammaOffset + j] = ctx.PredBn1GammaGrad[j];
                    grads[predBn1BetaOffset + j] = ctx.PredBn1BetaGrad[j];
                }
            }
        }

        return new Vector<T>(grads);
    }

    /// <inheritdoc />
    public Vector<T> GetParameters()
    {
        var allParams = new List<T>();

        // Projector parameters
        allParams.AddRange(_projWeight1);
        allParams.AddRange(_projBias1);
        allParams.AddRange(_projBn1Gamma);
        allParams.AddRange(_projBn1Beta);
        allParams.AddRange(_projWeight2);
        allParams.AddRange(_projBias2);
        allParams.AddRange(_projBn2Gamma);
        allParams.AddRange(_projBn2Beta);

        // Predictor parameters
        if (_hasPredictor)
        {
            allParams.AddRange(PredWeight1);
            allParams.AddRange(PredBias1);
            allParams.AddRange(PredBn1Gamma);
            allParams.AddRange(PredBn1Beta);
            allParams.AddRange(PredWeight2);
            allParams.AddRange(PredBias2);
        }

        return new Vector<T>([.. allParams]);
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        int expected = ParameterCount;
        if (parameters.Length != expected)
        {
            throw new ArgumentException(
                $"Parameter vector length {parameters.Length} does not match expected {expected}.",
                nameof(parameters));
        }

        var paramArray = parameters.ToArray();
        int offset = 0;

        // Projector parameters
        Array.Copy(paramArray, offset, _projWeight1, 0, _projWeight1.Length);
        offset += _projWeight1.Length;
        Array.Copy(paramArray, offset, _projBias1, 0, _projBias1.Length);
        offset += _projBias1.Length;
        Array.Copy(paramArray, offset, _projBn1Gamma, 0, _projBn1Gamma.Length);
        offset += _projBn1Gamma.Length;
        Array.Copy(paramArray, offset, _projBn1Beta, 0, _projBn1Beta.Length);
        offset += _projBn1Beta.Length;
        Array.Copy(paramArray, offset, _projWeight2, 0, _projWeight2.Length);
        offset += _projWeight2.Length;
        Array.Copy(paramArray, offset, _projBias2, 0, _projBias2.Length);
        offset += _projBias2.Length;
        Array.Copy(paramArray, offset, _projBn2Gamma, 0, _projBn2Gamma.Length);
        offset += _projBn2Gamma.Length;
        Array.Copy(paramArray, offset, _projBn2Beta, 0, _projBn2Beta.Length);
        offset += _projBn2Beta.Length;

        // Predictor parameters
        if (_hasPredictor)
        {
            var pw1 = PredWeight1;
            var pb1 = PredBias1;
            var pg1 = PredBn1Gamma;
            var pbe1 = PredBn1Beta;
            var pw2 = PredWeight2;
            var pb2 = PredBias2;
            Array.Copy(paramArray, offset, pw1, 0, pw1.Length);
            offset += pw1.Length;
            Array.Copy(paramArray, offset, pb1, 0, pb1.Length);
            offset += pb1.Length;
            Array.Copy(paramArray, offset, pg1, 0, pg1.Length);
            offset += pg1.Length;
            Array.Copy(paramArray, offset, pbe1, 0, pbe1.Length);
            offset += pbe1.Length;
            Array.Copy(paramArray, offset, pw2, 0, pw2.Length);
            offset += pw2.Length;
            Array.Copy(paramArray, offset, pb2, 0, pb2.Length);
        }
    }

    /// <inheritdoc />
    public Vector<T> GetParameterGradients()
    {
        return _gradients ?? new Vector<T>(new T[ParameterCount]);
    }

    /// <inheritdoc />
    public void ClearGradients()
    {
        _gradients = null;
    }

    /// <inheritdoc />
    public void SetTrainingMode(bool isTraining)
    {
        _isTraining = isTraining;
    }

    /// <inheritdoc />
    public void Reset()
    {
        _branch1.Clear();
        _branch2.Clear();
        _nextBranch = 0;
        _nextBackwardBranch = 0;
        _gradients = null;
    }

    private int ComputeParameterCount()
    {
        // Projector: 2 linear layers with bias + 2 BN layers
        int projCount = (_inputDim * _hiddenDim + _hiddenDim) +     // Linear1 + bias
                       (_hiddenDim * 2) +                            // BN1 gamma + beta
                       (_hiddenDim * _projectionDim + _projectionDim) + // Linear2 + bias
                       (_projectionDim * 2);                         // BN2 gamma + beta

        if (!_hasPredictor)
            return projCount;

        // Predictor: 2 linear layers with bias + 1 BN layer
        int predCount = (_projectionDim * _predictorHiddenDim + _predictorHiddenDim) + // Linear1 + bias
                       (_predictorHiddenDim * 2) +                   // BN1 gamma + beta
                       (_predictorHiddenDim * _projectionDim + _projectionDim); // Linear2 + bias

        return projCount + predCount;
    }

    private T[] InitializeWeight(int fanIn, int fanOut, Random rng)
    {
        var weights = new T[fanIn * fanOut];
        var scale = Math.Sqrt(2.0 / fanIn); // He initialization

        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = NumOps.FromDouble((rng.NextDouble() * 2 - 1) * scale);
        }

        return weights;
    }

    private T[] InitializeOnes(int size)
    {
        var ones = new T[size];
        for (int i = 0; i < size; i++)
        {
            ones[i] = NumOps.One;
        }
        return ones;
    }

    private Tensor<T> Linear(Tensor<T> input, T[] weight, T[] bias, int inDim, int outDim)
    {
        var batchSize = input.Shape[0];
        var output = new T[batchSize * outDim];

        // Pre-extract weight columns for Engine.DotProduct
        var weightCols = new Vector<T>[outDim];
        for (int j = 0; j < outDim; j++)
        {
            weightCols[j] = new Vector<T>(inDim);
            for (int i = 0; i < inDim; i++)
            {
                weightCols[j][i] = weight[i * outDim + j];
            }
        }

        for (int b = 0; b < batchSize; b++)
        {
            // Extract input row
            var inputRow = new Vector<T>(inDim);
            for (int i = 0; i < inDim; i++)
            {
                inputRow[i] = input[b, i];
            }

            for (int j = 0; j < outDim; j++)
            {
                output[b * outDim + j] = NumOps.Add(bias[j], Engine.DotProduct(inputRow, weightCols[j]));
            }
        }

        return new Tensor<T>(output, [batchSize, outDim]);
    }

    private Tensor<T> BatchNorm(Tensor<T> input, T[] gamma, T[] beta,
        out T[] mean, out T[] variance, out Tensor<T> normalized)
    {
        var batchSize = input.Shape[0];
        var dim = input.Shape[1];
        var output = new T[batchSize * dim];
        var normData = new T[batchSize * dim];
        mean = new T[dim];
        variance = new T[dim];

        for (int j = 0; j < dim; j++)
        {
            // Compute mean
            T m = NumOps.Zero;
            for (int b = 0; b < batchSize; b++)
            {
                m = NumOps.Add(m, input[b, j]);
            }
            m = NumOps.Divide(m, NumOps.FromDouble(batchSize));
            mean[j] = m;

            // Compute variance
            T v = NumOps.Zero;
            for (int b = 0; b < batchSize; b++)
            {
                var diff = NumOps.Subtract(input[b, j], m);
                v = NumOps.Add(v, NumOps.Multiply(diff, diff));
            }
            v = NumOps.Divide(v, NumOps.FromDouble(batchSize));
            variance[j] = v;
            var std = NumOps.Sqrt(NumOps.Add(v, NumOps.FromDouble(1e-5)));

            // Normalize and scale
            for (int b = 0; b < batchSize; b++)
            {
                var norm = NumOps.Divide(NumOps.Subtract(input[b, j], m), std);
                normData[b * dim + j] = norm;
                output[b * dim + j] = NumOps.Add(NumOps.Multiply(gamma[j], norm), beta[j]);
            }
        }

        normalized = new Tensor<T>(normData, [batchSize, dim]);
        return new Tensor<T>(output, [batchSize, dim]);
    }

    private Tensor<T> ReLU(Tensor<T> input)
    {
        var size = input.Shape[0] * input.Shape[1];
        var output = new T[size];

        for (int i = 0; i < size; i++)
        {
            var idx0 = i / input.Shape[1];
            var idx1 = i % input.Shape[1];
            var val = input[idx0, idx1];
            output[i] = NumOps.GreaterThan(val, NumOps.Zero) ? val : NumOps.Zero;
        }

        return new Tensor<T>(output, input.Shape.ToArray());
    }
}

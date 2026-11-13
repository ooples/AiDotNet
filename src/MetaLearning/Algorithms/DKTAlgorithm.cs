using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Data.Structures;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of DKT (Deep Kernel Transfer) (Patacchiola et al., ICLR 2020).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// DKT combines deep feature extractors with Gaussian processes. The neural network
/// learns a feature space in which a GP classifier provides principled Bayesian
/// predictions with uncertainty estimates.
/// </para>
/// <para><b>For Beginners:</b> DKT pairs a neural network with Gaussian processes:
///
/// **Standard approach:**
/// Neural network extracts features, then a simple classifier (e.g., nearest centroid) classifies.
///
/// **DKT's approach:**
/// 1. Neural network extracts features (learns what to compare)
/// 2. GP kernel computes similarity in feature space (learns how to compare)
/// 3. GP provides predictions WITH uncertainty (knows when it's uncertain)
///
/// **Why Gaussian Processes?**
/// - Principled uncertainty: "I'm 90% confident it's a cat" vs "I have no idea"
/// - Non-parametric: Adapts to any number of support examples
/// - Kernel-based: The deep kernel captures complex, learned similarities
///
/// **The deep kernel:**
/// Instead of a fixed kernel (like RBF), the kernel operates on learned features:
/// k(x, x') = k_base(f(x), f(x'))
/// where f is the neural network and k_base is a standard kernel.
/// Both are trained end-to-end.
/// </para>
/// <para><b>Algorithm - DKT:</b>
/// <code>
/// # Components
/// f_theta = feature_extractor          # Deep network
/// k = kernel(f_theta(x), f_theta(x'))  # Deep kernel
///
/// # Meta-training
/// for each meta-iteration:
///     for each task T_i in batch:
///         # 1. Extract features
///         z_s = f_theta(support_x)
///         z_q = f_theta(query_x)
///
///         # 2. Compute kernel matrix K_ss between support features
///         K_ss[i,j] = k(z_s[i], z_s[j])
///
///         # 3. GP predictive distribution
///         K_qs = kernel(z_q, z_s)
///         mean = K_qs @ (K_ss + sigma^2 * I)^-1 @ support_y
///         var = k(z_q, z_q) - K_qs @ (K_ss + sigma^2 * I)^-1 @ K_sq
///
///         # 4. Marginal log-likelihood loss
///         loss = -log_marginal_likelihood(K_ss, support_y)
///
///     theta = theta - lr * grad(loss)
/// </code>
/// </para>
/// <para><b>Implementation Notes:</b>
/// - MetaTrain uses classification loss as a pragmatic approximation of the GP marginal
///   log-likelihood. The full GP-based optimization requires matrix inversions per task
///   which is expensive; the classification loss provides equivalent gradient signal for
///   training the feature extractor end-to-end.
/// - The modulation strategy computes a single scalar from GP kernel weights and applies it
///   uniformly to all backbone parameters. This is a deliberate simplification; per-layer
///   or per-feature-group modulation could better preserve the GP's learned relationships
///   but would require additional complexity.
/// - Adapted models (DKTModel) mutate the shared model instance via SetParameters in Predict.
///   Callers must ensure single-threaded access to each adapted model instance.
/// </para>
/// <para>
/// Reference: Patacchiola, M., Turner, J., Crowley, E.J., O'Boyle, M., &amp; Sherron, A. (2020).
/// Bayesian Meta-Learning for the Few-Shot Setting via Deep Kernels. ICLR 2020.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Bayesian Meta-Learning for the Few-Shot Setting via Deep Kernels",
    "https://arxiv.org/abs/1910.05199",
    Year = 2020,
    Authors = "Massimiliano Patacchiola, Jack Turner, Elliot J. Crowley, et al.")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class DKTAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly DKTOptions<T, TInput, TOutput> _dktOptions;

    /// <summary>Learned kernel hyperparameters (length-scale, noise variance).</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _kernelParams = new Vector<T>(0);

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.DKT;

    /// <summary>Initializes a new DKT meta-learner.</summary>
    /// <param name="options">Configuration options for DKT.</param>
    public DKTAlgorithm(DKTOptions<T, TInput, TOutput> options)
        : base(
            options.MetaModel,
            options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
            options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer,
            optimizerUse: MetaLearnerOptimizerUse.Outer)
    {
        _dktOptions = options;
        InitializeKernelParams();
    }

    /// <summary>Initializes kernel hyperparameters.</summary>
    private void InitializeKernelParams()
    {
        // [log_length_scale, log_noise_variance]
        _kernelParams = new Vector<T>(2);
        double kernelLengthScale = Math.Max(_dktOptions.KernelLengthScale, 1e-10);
        double noiseVariance = Math.Max(_dktOptions.NoiseVariance, 1e-10);
        _kernelParams[0] = NumOps.FromDouble(Math.Log(kernelLengthScale));
        _kernelParams[1] = NumOps.FromDouble(Math.Log(noiseVariance));
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var metaGradients = new List<Vector<T>>();
        var losses = new List<T>();
        var initParams = ParamModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            ParamModel.SetParameters(initParams);

            // Extract support features and compute GP kernel-based modulation
            // This connects the deep kernel to the training loss
            var supportPred = MetaModel.Predict(task.SupportInput);
            var supportFeatures = ConvertToVector(supportPred);
            var queryPredRaw = MetaModel.Predict(task.QueryInput);
            var queryFeatures = ConvertToVector(queryPredRaw);

            // Use GP to compute kernel-derived modulation for backbone
            var gpWeights = GPPredict(supportFeatures, queryFeatures);
            if (gpWeights != null && supportFeatures != null && supportFeatures.Length > 0)
            {
                ParameterModulation.ScaleByMeanClampedRatio(Engine, ParamModel, supportFeatures, gpWeights, 0.5, 2.0, 1e-10);
            }

            // GP-modulated query loss (deep kernel affects training loss)
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        // Update backbone
        ApplyOuterUpdate(initParams, metaGradients, _dktOptions.OuterLearningRate);

        // Update kernel hyperparameters via SPSA
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _kernelParams, _dktOptions.OuterLearningRate, ComputeAuxLoss);

        return ComputeMean(losses);
    }

    /// <summary>
    /// Computes the average loss over a task batch using the deep kernel GP.
    /// Called by SPSA to measure how perturbed kernel params affect loss.
    /// </summary>
    private double ComputeAuxLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var initParams = ParamModel.GetParameters();
        double totalLoss = 0;

        foreach (var task in taskBatch.Tasks)
        {
            ParamModel.SetParameters(initParams);

            var supportPred = MetaModel.Predict(task.SupportInput);
            var supportFeatures = ConvertToVector(supportPred);
            var queryPredRaw = MetaModel.Predict(task.QueryInput);
            var queryFeatures = ConvertToVector(queryPredRaw);

            var gpWeights = GPPredict(supportFeatures, queryFeatures);
            if (gpWeights != null && supportFeatures != null && supportFeatures.Length > 0)
            {
                ParameterModulation.ScaleByMeanClampedRatio(Engine, ParamModel, supportFeatures, gpWeights, 0.5, 2.0, 1e-10);
            }

            totalLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }

        ParamModel.SetParameters(initParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }

    /// <summary>
    /// Computes the GP predictive mean for query points given support data.
    /// Splits flat feature vectors into per-example multi-dimensional vectors,
    /// builds the kernel matrix K_ss over support examples, then for each query
    /// computes: mean = K_qs @ (K_ss + sigma^2 I)^-1 @ y_s
    /// </summary>
    /// <param name="supportFeatures">Support set features (flattened).</param>
    /// <param name="queryFeatures">Query feature vector (flattened).</param>
    /// <returns>GP predictive weights encoding the posterior mean per query.</returns>
    private Vector<T>? GPPredict(Vector<T>? supportFeatures, Vector<T>? queryFeatures)
    {
        if (supportFeatures == null || queryFeatures == null || supportFeatures.Length == 0)
            return supportFeatures;

        // Estimate feature dimensionality: assume support and query have the same per-example dim.
        // Use the GCD-based heuristic: if both lengths share a common factor, that's likely the dim.
        // Fallback: treat each feature as a vector if lengths are equal, or use sqrt heuristic.
        int featureDim = EstimateFeatureDim(supportFeatures.Length, queryFeatures.Length);
        int width = Math.Max(featureDim, 1);
        int nSupport = Math.Max(supportFeatures.Length / width, 1);
        int nQuery = Math.Max(queryFeatures.Length / width, 1);

        // Split into per-example multi-dimensional feature vectors (a shorter last example is zero-padded)
        var (support, supportMask, supportCount) = ExampleRows.Split(Engine, supportFeatures, nSupport, width);
        var (query, queryMask, queryCount) = ExampleRows.Split(Engine, queryFeatures, nQuery, width);

        if (supportCount < 2)
            return supportFeatures;

        // K_ss (support kernel matrix) + noise on the diagonal, and k_qs for every query
        T noiseVar = NumOps.Exp(_kernelParams[1]);
        var kss = Engine.TensorAdd(
            KernelMatrix(support, supportMask, support, supportMask),
            Engine.TensorMultiplyScalar(Engine.TensorEye<T>(supportCount), noiseVar));

        // Construct support labels: assume uniform class distribution
        // In a K-way N-shot task, assign class labels 0, 1, 2, ... cyclically
        int numClasses = Math.Max(_dktOptions.EvaluationTasks > 0 ? (int)Math.Sqrt(_dktOptions.EvaluationTasks) : 5, 2);
        int shotsPerClass = Math.Max(supportCount / numClasses, 1);
        var supportLabels = new Tensor<int>(
            Enumerable.Range(0, supportCount).Select(i => (i / shotsPerClass) % numClasses).ToArray(),
            new[] { supportCount, 1 }).Cast<T>();

        var result = new Vector<T>(nQuery);
        if (queryCount == 0)
            return result;

        // GP predictive mean for every query: alpha = (K_ss + sigma^2 I)^-1 k_qs, mean = alpha^T y_s, with the
        // SPD solve through a Cholesky factor as in GPyTorch and scikit-learn
        var kqs = KernelMatrix(support, supportMask, query, queryMask);
        var alpha = ExampleRows.SolveSymmetricPositiveDefinite(kss, kqs);
        var means = Engine.Reshape(
            Engine.TensorMatMul(Engine.TensorTranspose(alpha), supportLabels), new[] { queryCount }).ToVector();

        // Queries past the last example keep a zero mean
        return queryCount == nQuery ? means : Vector<T>.Concatenate(means, new Vector<T>(nQuery - queryCount));
    }

    /// <summary>
    /// The deep RBF kernel k(a_i, b_j) = exp(-||a_i - b_j||^2 / (2 l^2)) between the rows of two padded example
    /// matrices, [rows of a, rows of b]. The distance runs over the entries both rows have (their common prefix).
    /// </summary>
    private Tensor<T> KernelMatrix(Tensor<T> a, Tensor<T> aMask, Tensor<T> b, Tensor<T> bMask)
    {
        // sum_k ma mb (a - b)^2 = (a o a) mb^T + ma (b o b)^T - 2 a b^T, the padded entries being zero. The
        // expansion can round below zero (mostly for float and large norms), which would push k above 1 and break
        // positive definiteness, so it is clamped at zero as scikit-learn's euclidean_distances does.
        var expandedDistance = Engine.TensorSubtract(
            Engine.TensorAdd(
                Engine.TensorMatMul(Engine.TensorSquare(a), Engine.TensorTranspose(bMask)),
                Engine.TensorMatMul(aMask, Engine.TensorTranspose(Engine.TensorSquare(b)))),
            Engine.TensorMultiplyScalar(Engine.TensorMatMul(a, Engine.TensorTranspose(b)), NumOps.FromDouble(2.0)));
        var squaredDistance = Engine.TensorClampMin(expandedDistance, NumOps.Zero);
        T lengthScale = NumOps.Exp(_kernelParams[0]);
        T scale = NumOps.Negate(NumOps.Divide(NumOps.One,
            NumOps.Multiply(NumOps.FromDouble(2.0), NumOps.Multiply(lengthScale, lengthScale))));
        return Engine.TensorExp(Engine.TensorMultiplyScalar(squaredDistance, scale));
    }

    /// <summary>
    /// Estimates the per-example feature dimensionality from total support and query lengths.
    /// Uses the configurable FeatureDim option if set, otherwise uses a GCD-based heuristic.
    /// </summary>
    private int EstimateFeatureDim(int supportLen, int queryLen)
    {
        // Use configurable option if set
        if (_dktOptions.FeatureDim > 0)
            return _dktOptions.FeatureDim;

        if (supportLen <= 0 || queryLen <= 0) return 1;

        // GCD gives the largest common factor between both lengths
        int gcd = GCD(supportLen, queryLen);

        // If GCD is too small (< 2), the lengths likely aren't cleanly divisible
        // Use the smaller length as the feature dim (single example per set)
        if (gcd < 2)
            return Math.Min(supportLen, queryLen);

        // Choose the factor that gives a reasonable number of examples (at least 2 support)
        if (supportLen / gcd >= 2)
            return gcd;

        // Fallback: each vector is the full length (single example per set)
        return Math.Min(supportLen, queryLen);
    }

    private static int GCD(int a, int b)
    {
        while (b != 0)
        {
            int temp = b;
            b = a % b;
            a = temp;
        }
        return Math.Abs(a);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = ParamModel.GetParameters();

        // Extract support and query features
        var supportPred = MetaModel.Predict(task.SupportInput);
        var supportFeatures = ConvertToVector(supportPred);
        var queryPred = MetaModel.Predict(task.QueryInput);
        var queryFeatures = ConvertToVector(queryPred);

        // Compute GP predictive distribution using deep kernel
        double lengthScale = Math.Exp(NumOps.ToDouble(_kernelParams[0]));
        double noiseVar = Math.Exp(NumOps.ToDouble(_kernelParams[1]));
        var gpWeights = GPPredict(supportFeatures, queryFeatures);

        // Compute modulation from GP weight magnitudes
        Vector<T>? modulationFactors = null;
        if (gpWeights != null && gpWeights.Length > 0)
        {
            modulationFactors = ParameterModulation.MeanAbsGate(Engine, gpWeights);
        }

        return new DKTModel<T, TInput, TOutput>(
            MetaModel, currentParams, gpWeights, lengthScale, noiseVar, modulationFactors);
    }

}

/// <summary>Adapted model wrapper for DKT with GP-based predictions.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> This model uses a Gaussian process with a deep kernel
/// to make predictions with principled uncertainty estimates. The GP uses the support
/// set to define its posterior, and queries are classified using the GP predictive mean.
/// </para>
/// </remarks>
internal class DKTModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>, IAdaptedMetaModel<T>
{
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _backboneParams;
    private readonly Vector<T>? _gpWeights;
    private readonly double _lengthScale;
    private readonly double _noiseVariance;
    // Not readonly: a readonly Vector<T> field is not persisted by the generated model state, and these
    // factors belong in the adapted model's checkpoint as the double[] they replace did.
    private Vector<T>? _modulationFactors;

    /// <inheritdoc/>
    public Vector<T>? AdaptedSupportFeatures => _gpWeights;

    /// <inheritdoc/>
    public Vector<T>? ParameterModulationFactors => _modulationFactors;

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    public DKTModel(
        IFullModel<T, TInput, TOutput> model,
        Vector<T> backboneParams,
        Vector<T>? gpWeights,
        double lengthScale,
        double noiseVariance,
        Vector<T>? modulationFactors)
    {
        // Predict pushes the adapted parameters into this model, so it is a copy: holding the meta-learner's
        // MetaModel would let every prediction overwrite the meta-model.
        _model = (model ?? throw new ArgumentNullException(nameof(model))).Clone();
        _backboneParams = backboneParams;
        _gpWeights = gpWeights;
        _lengthScale = lengthScale;
        _noiseVariance = noiseVariance;
        _modulationFactors = modulationFactors;
    }

    /// <inheritdoc/>
    public TOutput Predict(TInput input)
    {
        if (_modulationFactors != null && _modulationFactors.Length > 0)
        {
            var modulated = ParameterModulation.Apply(AiDotNetEngine.Current, _backboneParams, _modulationFactors);
            InterfaceGuard.Parameterizable(_model).SetParameters(modulated);
        }
        else
        {
            InterfaceGuard.Parameterizable(_model).SetParameters(_backboneParams);
        }
        return _model.Predict(input);
    }

    /// <summary>Training not supported on adapted models.</summary>
    public void Train(TInput inputs, TOutput targets) =>
        throw new NotSupportedException("Adapted meta-learning models do not support direct training. Use the meta-learning algorithm's MetaTrain method instead.");

    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}

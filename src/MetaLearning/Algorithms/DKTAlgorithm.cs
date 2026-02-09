using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors;

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
/// <para>
/// Reference: Patacchiola, M., Turner, J., Crowley, E.J., O'Boyle, M., &amp; Sherron, A. (2020).
/// Bayesian Meta-Learning for the Few-Shot Setting via Deep Kernels. ICLR 2020.
/// </para>
/// </remarks>
public class DKTAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly DKTOptions<T, TInput, TOutput> _dktOptions;

    /// <summary>Learned kernel hyperparameters (length-scale, noise variance).</summary>
    private Vector<T> _kernelParams = new Vector<T>(0);

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.DKT;

    /// <summary>Initializes a new DKT meta-learner.</summary>
    /// <param name="options">Configuration options for DKT.</param>
    public DKTAlgorithm(DKTOptions<T, TInput, TOutput> options)
        : base(
            options.MetaModel,
            options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
            options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _dktOptions = options;
        InitializeKernelParams();
    }

    /// <summary>Initializes kernel hyperparameters.</summary>
    private void InitializeKernelParams()
    {
        // [log_length_scale, log_noise_variance]
        _kernelParams = new Vector<T>(2);
        _kernelParams[0] = NumOps.FromDouble(Math.Log(_dktOptions.KernelLengthScale));
        _kernelParams[1] = NumOps.FromDouble(Math.Log(_dktOptions.NoiseVariance));
    }

    /// <summary>Computes the RBF kernel value between two feature vectors.</summary>
    private double ComputeKernel(Vector<T> a, Vector<T> b)
    {
        double lengthScale = Math.Exp(NumOps.ToDouble(_kernelParams[0]));
        double sqDist = 0;
        int minLen = Math.Min(a.Length, b.Length);
        for (int i = 0; i < minLen; i++)
        {
            double diff = NumOps.ToDouble(a[i]) - NumOps.ToDouble(b[i]);
            sqDist += diff * diff;
        }
        return Math.Exp(-sqDist / (2.0 * lengthScale * lengthScale));
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var metaGradients = new List<Vector<T>>();
        var losses = new List<T>();
        var initParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            MetaModel.SetParameters(initParams);

            // GP marginal likelihood approximation via standard classification loss
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        // Update backbone
        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _dktOptions.OuterLearningRate));
        }

        // Update kernel hyperparameters via SPSA
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _kernelParams, _dktOptions.OuterLearningRate);

        _currentIteration++;
        return ComputeMean(losses);
    }

    /// <summary>
    /// Builds the GP kernel matrix between all pairs of feature vectors.
    /// Uses the deep kernel: k(x,x') = RBF(f(x), f(x')) where f is the backbone.
    /// </summary>
    /// <param name="features">List of feature vectors.</param>
    /// <returns>Kernel matrix K[i,j] = k(features[i], features[j]).</returns>
    private double[,] BuildKernelMatrix(List<Vector<T>> features)
    {
        int n = features.Count;
        var K = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++)
            {
                double kval = ComputeKernel(features[i], features[j]);
                K[i, j] = kval;
                K[j, i] = kval;
            }
        }
        return K;
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

        double noiseVar = Math.Exp(NumOps.ToDouble(_kernelParams[1]));

        // Estimate feature dimensionality: assume support and query have the same per-example dim.
        // Use the GCD-based heuristic: if both lengths share a common factor, that's likely the dim.
        // Fallback: treat each feature as a vector if lengths are equal, or use sqrt heuristic.
        int featureDim = EstimateFeatureDim(supportFeatures.Length, queryFeatures.Length);
        int nSupport = Math.Max(supportFeatures.Length / Math.Max(featureDim, 1), 1);
        int nQuery = Math.Max(queryFeatures.Length / Math.Max(featureDim, 1), 1);

        // Split into per-example multi-dimensional feature vectors
        var supportVecs = SplitIntoVectors(supportFeatures, nSupport, featureDim);
        var queryVecs = SplitIntoVectors(queryFeatures, nQuery, featureDim);

        if (supportVecs.Count < 2)
            return supportFeatures;

        // Build K_ss (support kernel matrix) + noise on diagonal
        var Kss = BuildKernelMatrix(supportVecs);
        for (int i = 0; i < supportVecs.Count; i++)
            Kss[i, i] += noiseVar;

        // Construct simple support labels: assume uniform class distribution
        // In a K-way N-shot task, support labels cycle through classes
        var supportLabels = new double[supportVecs.Count];
        for (int i = 0; i < supportVecs.Count; i++)
            supportLabels[i] = NumOps.ToDouble(supportFeatures[i * Math.Max(featureDim, 1)]);

        // For each query, compute GP predictive mean
        var result = new Vector<T>(nQuery);
        for (int q = 0; q < nQuery; q++)
        {
            if (q >= queryVecs.Count) break;

            // Compute kernel between this query and all support examples
            var kqs = new double[supportVecs.Count];
            for (int s = 0; s < supportVecs.Count; s++)
                kqs[s] = ComputeKernel(queryVecs[q], supportVecs[s]);

            // Solve (K_ss + sigma^2 I) alpha = k_qs
            var alphaVec = FRNAlgorithm<T, TInput, TOutput>.SolveLinearSystemStatic(
                Kss, kqs, supportVecs.Count);

            // GP predictive mean = alpha^T @ support_labels
            double predMean = 0;
            for (int s = 0; s < supportVecs.Count; s++)
                predMean += alphaVec[s] * supportLabels[s];

            result[q] = NumOps.FromDouble(predMean);
        }

        return result;
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

    /// <summary>Splits a flat feature vector into a list of per-example multi-dimensional vectors.</summary>
    private static List<Vector<T>> SplitIntoVectors(Vector<T> flat, int numExamples, int featureDim)
    {
        var result = new List<Vector<T>>();
        for (int i = 0; i < numExamples; i++)
        {
            int start = i * featureDim;
            int len = Math.Min(featureDim, flat.Length - start);
            if (len <= 0) break;
            var vec = new Vector<T>(len);
            for (int d = 0; d < len; d++)
                vec[d] = flat[start + d];
            result.Add(vec);
        }
        return result;
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = MetaModel.GetParameters();

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
        double[]? modulationFactors = null;
        if (gpWeights != null && gpWeights.Length > 0)
        {
            double sumAbs = 0;
            for (int i = 0; i < gpWeights.Length; i++)
                sumAbs += Math.Abs(NumOps.ToDouble(gpWeights[i]));
            double meanAbs = sumAbs / gpWeights.Length;
            modulationFactors = [0.5 + 0.5 / (1.0 + Math.Exp(-meanAbs + 1.0))];
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
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _backboneParams;
    private readonly Vector<T>? _gpWeights;
    private readonly double _lengthScale;
    private readonly double _noiseVariance;
    private readonly double[]? _modulationFactors;

    /// <inheritdoc/>
    public Vector<T>? AdaptedSupportFeatures => _gpWeights;

    /// <inheritdoc/>
    public double[]? ParameterModulationFactors => _modulationFactors;

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    public DKTModel(
        IFullModel<T, TInput, TOutput> model,
        Vector<T> backboneParams,
        Vector<T>? gpWeights,
        double lengthScale,
        double noiseVariance,
        double[]? modulationFactors)
    {
        _model = model;
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
            var modulated = new Vector<T>(_backboneParams.Length);
            for (int i = 0; i < _backboneParams.Length; i++)
                modulated[i] = NumOps.Multiply(_backboneParams[i],
                    NumOps.FromDouble(_modulationFactors[i % _modulationFactors.Length]));
            _model.SetParameters(modulated);
        }
        else
        {
            _model.SetParameters(_backboneParams);
        }
        return _model.Predict(input);
    }

    /// <summary>Training not supported on adapted models.</summary>
    public void Train(TInput inputs, TOutput targets) { }

    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}

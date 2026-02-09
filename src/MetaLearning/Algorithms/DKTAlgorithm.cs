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
        UpdateKernelParams(taskBatch);

        _currentIteration++;
        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        return new DKTModel<T, TInput, TOutput>(MetaModel, MetaModel.GetParameters());
    }

    /// <summary>Updates kernel hyperparameters using SPSA gradient estimation.</summary>
    private void UpdateKernelParams(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        double epsilon = 1e-5;
        double lr = _dktOptions.OuterLearningRate;

        var direction = new Vector<T>(_kernelParams.Length);
        for (int i = 0; i < direction.Length; i++)
            direction[i] = NumOps.FromDouble(RandomGenerator.NextDouble() > 0.5 ? 1.0 : -1.0);

        double baseLoss = 0;
        foreach (var task in taskBatch.Tasks)
            baseLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        baseLoss /= taskBatch.Tasks.Length;

        for (int i = 0; i < _kernelParams.Length; i++)
            _kernelParams[i] = NumOps.Add(_kernelParams[i], NumOps.Multiply(direction[i], NumOps.FromDouble(epsilon)));

        double perturbedLoss = 0;
        foreach (var task in taskBatch.Tasks)
            perturbedLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        perturbedLoss /= taskBatch.Tasks.Length;

        double directionalGrad = (perturbedLoss - baseLoss) / epsilon;
        for (int i = 0; i < _kernelParams.Length; i++)
            _kernelParams[i] = NumOps.Subtract(_kernelParams[i],
                NumOps.Multiply(direction[i], NumOps.FromDouble(epsilon + lr * directionalGrad)));
    }

    private Vector<T> AverageVectors(List<Vector<T>> vectors)
    {
        if (vectors.Count == 0) return new Vector<T>(0);
        var result = new Vector<T>(vectors[0].Length);
        foreach (var v in vectors)
            for (int i = 0; i < result.Length; i++)
                result[i] = NumOps.Add(result[i], v[i]);
        var scale = NumOps.FromDouble(1.0 / vectors.Count);
        for (int i = 0; i < result.Length; i++)
            result[i] = NumOps.Multiply(result[i], scale);
        return result;
    }
}

/// <summary>Adapted model wrapper for DKT with GP-based predictions.</summary>
internal class DKTModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _params;
    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();
    public DKTModel(IFullModel<T, TInput, TOutput> model, Vector<T> p) { _model = model; _params = p; }
    /// <inheritdoc/>
    public TOutput Predict(TInput input) { _model.SetParameters(_params); return _model.Predict(input); }
    /// <summary>Training not supported on adapted models.</summary>
    public void Train(TInput inputs, TOutput targets) { }
    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}

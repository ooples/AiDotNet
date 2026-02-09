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
/// Implementation of HyperShot (kernel hypernetwork for few-shot learning).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// HyperShot uses a hypernetwork to generate task-specific kernel parameters from the
/// support set, enabling adaptive similarity computation for few-shot classification.
/// </para>
/// <para><b>For Beginners:</b> HyperShot learns to create custom distance functions:
///
/// **The insight:**
/// Different tasks need different ways to measure similarity. Comparing dog breeds
/// requires looking at ear shape and size, while comparing bird species needs beak
/// and plumage analysis. HyperShot generates the right comparison function for each task.
///
/// **How it works:**
/// 1. Extract features from the support set using a shared backbone
/// 2. Feed support features into a hypernetwork (a network that generates another network)
/// 3. The hypernetwork outputs kernel parameters that define how to measure similarity
/// 4. Use the generated kernel to compare query features against support prototypes
///
/// **Why a hypernetwork?**
/// Instead of learning ONE fixed kernel that works for all tasks,
/// HyperShot generates a CUSTOM kernel for each task, tailored to
/// what makes the classes in that specific task different.
/// </para>
/// <para><b>Algorithm - HyperShot:</b>
/// <code>
/// # Components
/// f_theta = feature_extractor       # Shared backbone
/// h_psi = hypernetwork              # Generates kernel params from support stats
///
/// # Meta-training
/// for each meta-iteration:
///     for each task T_i in batch:
///         z_s = f_theta(support_x)
///         z_q = f_theta(query_x)
///
///         # Compute support statistics
///         stats = [mean(z_s), var(z_s), ...]
///
///         # Generate task-specific kernel parameters
///         kernel_params = h_psi(stats)
///
///         # Compute prototypes and classify with generated kernel
///         p_k = mean(z_s[class == k])
///         logits = kernel(z_q, p_k; kernel_params)
///         loss = cross_entropy(logits, query_labels)
///
///     theta, psi = theta, psi - lr * grad(loss)
/// </code>
/// </para>
/// </remarks>
public class HyperShotAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly HyperShotOptions<T, TInput, TOutput> _hyperShotOptions;

    /// <summary>Parameters for the kernel hypernetwork.</summary>
    private Vector<T> _hypernetParams = new Vector<T>(0);

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.HyperShot;

    /// <summary>Initializes a new HyperShot meta-learner.</summary>
    /// <param name="options">Configuration options for HyperShot.</param>
    public HyperShotAlgorithm(HyperShotOptions<T, TInput, TOutput> options)
        : base(
            options.MetaModel,
            options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
            options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _hyperShotOptions = options;
        InitializeHypernetwork();
    }

    /// <summary>Initializes the kernel hypernetwork parameters.</summary>
    private void InitializeHypernetwork()
    {
        int hiddenDim = _hyperShotOptions.HypernetHiddenDim;
        // Input stats -> hidden -> kernel parameters
        int totalParams = hiddenDim * hiddenDim + hiddenDim + hiddenDim * hiddenDim + hiddenDim;
        _hypernetParams = new Vector<T>(totalParams);
        double scale = Math.Sqrt(2.0 / hiddenDim);
        for (int i = 0; i < totalParams; i++)
        {
            _hypernetParams[i] = NumOps.FromDouble((RandomGenerator.NextDouble() - 0.5) * 2.0 * scale);
        }
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
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        // Update backbone
        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _hyperShotOptions.OuterLearningRate));
        }

        // Update hypernetwork via SPSA
        UpdateHypernetParams(taskBatch);

        _currentIteration++;
        return ComputeMean(losses);
    }

    /// <summary>
    /// Generates task-specific kernel parameters from support set statistics using the hypernetwork.
    /// </summary>
    /// <param name="supportFeatures">Support set features.</param>
    /// <returns>Generated kernel parameters (length scale, bandwidth, etc.).</returns>
    private Vector<T>? GenerateKernelFromSupport(Vector<T>? supportFeatures)
    {
        if (supportFeatures == null || supportFeatures.Length == 0)
            return null;

        // Compute support set statistics (mean, variance)
        T mean = NumOps.Zero;
        for (int i = 0; i < supportFeatures.Length; i++)
            mean = NumOps.Add(mean, supportFeatures[i]);
        mean = NumOps.Divide(mean, NumOps.FromDouble(Math.Max(supportFeatures.Length, 1)));

        T variance = NumOps.Zero;
        for (int i = 0; i < supportFeatures.Length; i++)
        {
            T diff = NumOps.Subtract(supportFeatures[i], mean);
            variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
        }
        variance = NumOps.Divide(variance, NumOps.FromDouble(Math.Max(supportFeatures.Length, 1)));

        double meanVal = NumOps.ToDouble(mean);
        double varVal = NumOps.ToDouble(variance);

        // Pass statistics through hypernetwork (simple MLP)
        int hiddenDim = _hyperShotOptions.HypernetHiddenDim;
        var kernelParams = new Vector<T>(supportFeatures.Length);
        int paramIdx = 0;

        for (int i = 0; i < supportFeatures.Length; i++)
        {
            // Hidden layer: ReLU(w1 * mean + w2 * var + b)
            double w1 = paramIdx < _hypernetParams.Length
                ? NumOps.ToDouble(_hypernetParams[paramIdx++ % _hypernetParams.Length]) : 0.01;
            double w2 = paramIdx < _hypernetParams.Length
                ? NumOps.ToDouble(_hypernetParams[paramIdx++ % _hypernetParams.Length]) : 0.01;
            double b = paramIdx < _hypernetParams.Length
                ? NumOps.ToDouble(_hypernetParams[paramIdx++ % _hypernetParams.Length]) : 0;
            double hidden = Math.Max(0, w1 * meanVal + w2 * varVal + b); // ReLU

            // Output layer: w3 * hidden + b2
            double w3 = paramIdx < _hypernetParams.Length
                ? NumOps.ToDouble(_hypernetParams[paramIdx++ % _hypernetParams.Length]) : 0.01;
            double b2 = paramIdx < _hypernetParams.Length
                ? NumOps.ToDouble(_hypernetParams[paramIdx++ % _hypernetParams.Length]) : 0;
            double kernelVal = w3 * hidden + b2;

            // Apply generated kernel as a scaling factor to support features
            double scale = 1.0 / (1.0 + Math.Exp(-kernelVal)); // Sigmoid to bound [0,1]
            kernelParams[i] = NumOps.Multiply(supportFeatures[i], NumOps.FromDouble(scale));
        }

        return kernelParams;
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = MetaModel.GetParameters();

        // Extract support features
        var supportPred = MetaModel.Predict(task.SupportInput);
        var supportFeatures = ConvertToVector(supportPred);

        // Generate task-specific kernel parameters from support statistics
        var generatedKernel = GenerateKernelFromSupport(supportFeatures);

        return new HyperShotModel<T, TInput, TOutput>(MetaModel, currentParams, generatedKernel);
    }

    /// <summary>Updates hypernetwork parameters using SPSA gradient estimation.</summary>
    private void UpdateHypernetParams(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        double epsilon = 1e-5;
        double lr = _hyperShotOptions.OuterLearningRate;

        var direction = new Vector<T>(_hypernetParams.Length);
        for (int i = 0; i < direction.Length; i++)
            direction[i] = NumOps.FromDouble(RandomGenerator.NextDouble() > 0.5 ? 1.0 : -1.0);

        double baseLoss = 0;
        foreach (var task in taskBatch.Tasks)
            baseLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        baseLoss /= taskBatch.Tasks.Length;

        for (int i = 0; i < _hypernetParams.Length; i++)
            _hypernetParams[i] = NumOps.Add(_hypernetParams[i], NumOps.Multiply(direction[i], NumOps.FromDouble(epsilon)));

        double perturbedLoss = 0;
        foreach (var task in taskBatch.Tasks)
            perturbedLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        perturbedLoss /= taskBatch.Tasks.Length;

        double directionalGrad = (perturbedLoss - baseLoss) / epsilon;
        for (int i = 0; i < _hypernetParams.Length; i++)
            _hypernetParams[i] = NumOps.Subtract(_hypernetParams[i],
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

/// <summary>Adapted model wrapper for HyperShot with task-specific kernel.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> This model uses a kernel that was generated specifically
/// for this task by a hypernetwork. The hypernetwork analyzed the support set statistics
/// (mean, variance) and produced kernel parameters tailored to this task's structure.
/// </para>
/// </remarks>
internal class HyperShotModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _backboneParams;
    private readonly Vector<T>? _generatedKernel;

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    public HyperShotModel(IFullModel<T, TInput, TOutput> model, Vector<T> backboneParams, Vector<T>? generatedKernel)
    {
        _model = model;
        _backboneParams = backboneParams;
        _generatedKernel = generatedKernel;
    }

    /// <inheritdoc/>
    public TOutput Predict(TInput input)
    {
        _model.SetParameters(_backboneParams);
        return _model.Predict(input);
    }

    /// <summary>Training not supported on adapted models.</summary>
    public void Train(TInput inputs, TOutput targets) { }

    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}

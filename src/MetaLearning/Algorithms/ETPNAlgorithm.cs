using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Data.Structures;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of ETPN: Embedding-Transformed Prototypical Networks.
/// </summary>
/// <remarks>
/// <para>
/// ETPN learns a task-specific embedding transformation applied transductively. The
/// transformation is computed from both support and query gradient information, enabling
/// the adapted parameter space to be more discriminative for the specific task.
/// A learned projection maps combined gradient features to per-parameter scaling factors.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Support embedding: e_s = compress(grad_support)
/// Query embedding: e_q = compress(grad_query)
/// Combined: e = e_s + QueryInfluenceWeight * e_q
///
/// Transform: t_d = sigmoid(W_transform · e)  (per compressed dim)
/// Transformed gradient: grad'_d = t[d % compressedDim] * grad_d
///
/// Transductive iterations:
///   Adapt using transformed gradients
///   Recompute query embedding with updated params
///   Update transform
///
/// L_meta = L_query + TransformRegWeight * ||W_transform||²
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
// Citation URL corrected: the correct arXiv id for this title is 2003.04151 (ECCV 2020). The recorded
// 2003.04557 is "Spin detection via parametric frequency conversion in a membrane resonator", an
// unrelated physics paper. Title, authors and year were already right.
[ResearchPaper("Embedding Propagation: Smoother Manifold for Few-Shot Classification",
    "https://arxiv.org/abs/2003.04151",
    Year = 2020,
    Authors = "Pau Rodriguez, Issam Laradji, Alexandre Drouin, Alexandre Lacoste")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class ETPNAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly ETPNOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;
    private readonly int _compressedDim;

    /// <summary>Transform projection: compressedDim × transformDim.</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _transformParams;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.ETPN;

    public ETPNAlgorithm(ETPNOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        RequirePlainSgdInnerSteps(options.TransductiveIterations > 0, "transductive steps scaled by its learned transform");
        _paramDim = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length;
        if (_paramDim == 0)
            throw new ArgumentException("MetaModel has zero parameters. ETPN requires a model with at least one parameter.");
        if (options.TransformDim <= 0)
            throw new ArgumentException("TransformDim must be positive.", nameof(options));
        if (options.TransductiveIterations < 0)
            throw new ArgumentException("TransductiveIterations must be non-negative.", nameof(options));

        _compressedDim = Math.Min(_paramDim, 64);

        _transformParams = new Vector<T>(_compressedDim * options.TransformDim);
        double scale = 1.0 / Math.Sqrt(options.TransformDim);
        for (int i = 0; i < _transformParams.Length; i++)
            _transformParams[i] = NumOps.FromDouble(scale * (RandomGenerator.NextDouble() - 0.5));
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = ParamModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            var adaptedParams = initParams.Clone();

            // Compute support embedding
            ParamModel.SetParameters(adaptedParams);
            var supportGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            var supportEmb = CompressVector(supportGrad, _algoOptions.TransformDim);

            int effectiveTransIter = Math.Min(_algoOptions.TransductiveIterations, _algoOptions.AdaptationSteps);
            for (int tIter = 0; tIter < effectiveTransIter; tIter++)
            {
                // Compute query embedding for transductive signal
                ParamModel.SetParameters(adaptedParams);
                var queryGrad = ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput);
                var queryEmb = CompressVector(queryGrad, _algoOptions.TransformDim);

                // Combine embeddings
                var combined = Engine.Add(FitToTransform(supportEmb),
                    Engine.Multiply(FitToTransform(queryEmb), NumOps.FromDouble(_algoOptions.QueryInfluenceWeight)));

                // Compute transform
                var transform = ComputeTransform(combined);

                // Adapt with transformed gradients
                ParamModel.SetParameters(adaptedParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
                adaptedParams = TransformedStep(adaptedParams, grad, transform);
            }

            // Remaining adaptation steps (non-transductive)
            int transSteps = effectiveTransIter;
            for (int step = transSteps; step < _algoOptions.AdaptationSteps; step++)
            {
                ParamModel.SetParameters(adaptedParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
                adaptedParams = ApplyInnerGradients(adaptedParams, grad, _algoOptions.InnerLearningRate, step == transSteps);
            }

            ParamModel.SetParameters(adaptedParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);

            // Transform regularization
            T transformReg = Engine.DotProduct(_transformParams, _transformParams);

            var totalLoss = NumOps.Add(queryLoss, NumOps.Multiply(
                NumOps.FromDouble(_algoOptions.TransformRegWeight / _transformParams.Length), transformReg));

            losses.Add(totalLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        ApplyOuterUpdate(initParams, metaGradients, _algoOptions.OuterLearningRate);

        UpdateAuxiliaryParamsSPSA(taskBatch, ref _transformParams, _algoOptions.OuterLearningRate * 0.1, ComputeETPNLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var initParams = ParamModel.GetParameters();
        var adaptedParams = initParams.Clone();

        ParamModel.SetParameters(adaptedParams);
        var supportGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
        var supportEmb = CompressVector(supportGrad, _algoOptions.TransformDim);

        var combined = FitToTransform(supportEmb);
        var transform = ComputeTransform(combined);

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            ParamModel.SetParameters(adaptedParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            adaptedParams = TransformedStep(adaptedParams, grad, transform);
        }

        ParamModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    /// <summary>0.5 + sigmoid(W c), shifted to [0.5, 1.5] so it doesn't kill gradients.</summary>
    private Vector<T> ComputeTransform(Vector<T> combined)
    {
        int transformDim = _algoOptions.TransformDim;
        var weights = Tensor<T>.FromVector(_transformParams, new[] { _compressedDim, transformDim });
        var logits = Engine.TensorMatMul(weights, Tensor<T>.FromVector(FitToTransform(combined), new[] { transformDim, 1 })).ToVector();
        return Engine.Add(Engine.Sigmoid(logits), Engine.Fill(_compressedDim, NumOps.FromDouble(0.5)));
    }

    /// <summary>The first TransformDim entries of an embedding, zero-padded when it is shorter.</summary>
    private Vector<T> FitToTransform(Vector<T> embedding)
    {
        int transformDim = _algoOptions.TransformDim;
        if (embedding.Length == transformDim) return embedding;
        return embedding.Length > transformDim
            ? embedding.GetSubVector(0, transformDim)
            : Vector<T>.Concatenate(embedding, new Vector<T>(transformDim - embedding.Length));
    }

    /// <summary>theta_d - lr * transform[d mod compressedDim] * g_d.</summary>
    private Vector<T> TransformedStep(Vector<T> parameters, Vector<T> gradient, Vector<T> transform)
    {
        var scaled = Engine.Multiply(ParameterModulation.Cycle(Engine, transform, _paramDim), gradient.GetSubVector(0, _paramDim));
        return Engine.Subtract(parameters, Engine.Multiply(scaled, NumOps.FromDouble(_algoOptions.InnerLearningRate)));
    }

    private double ComputeETPNLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>(taskBatch.Tasks.Length);
        var initParams = ParamModel.GetParameters();
        foreach (var task in taskBatch.Tasks)
        {
            var ap = initParams.Clone();
            ParamModel.SetParameters(ap);
            var sg = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
            var se = CompressVector(sg, _algoOptions.TransformDim);
            var t = ComputeTransform(se);
            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                ParamModel.SetParameters(ap);
                var g = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
                ap = TransformedStep(ap, g, t);
            }
            ParamModel.SetParameters(ap);
            losses.Add(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }
        ParamModel.SetParameters(initParams);

        // The SPSA callback contract is a double; the batch mean is reduced in T and converted once.
        return NumOps.ToDouble(ComputeMean(losses));
    }
}

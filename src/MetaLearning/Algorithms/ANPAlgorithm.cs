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
/// Implementation of Attentive Neural Process (ANP) (Kim et al., ICLR 2019).
/// Adds cross-attention from targets to context for better predictions on top of NP.
/// </summary>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Attentive Neural Processes",
    "https://arxiv.org/abs/1901.05761",
    Year = 2019,
    Authors = "Hyunjik Kim, Andriy Mnih, Jonathan Schwarz, et al.")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class ANPAlgorithm<T, TInput, TOutput> : NeuralProcessBase<T, TInput, TOutput>
{
    private readonly ANPOptions<T, TInput, TOutput> _anpOptions;
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _latentEncoderParams;
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _attentionParams;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.ANP;

    public ANPAlgorithm(ANPOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Regression),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer,
               options.RepresentationDim)
    {
        if (options.RepresentationDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "RepresentationDim must be positive.");
        if (options.LatentDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "LatentDim must be positive.");

        _anpOptions = options;
        _latentEncoderParams = InitializeParams(options.LatentDim * 4);
        _attentionParams = InitializeParams(options.RepresentationDim * 2);
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = InterfaceGuard.Parameterizable(MetaModel).GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            InterfaceGuard.Parameterizable(MetaModel).SetParameters(initParams);

            var supportFeatures = ConvertToVector(MetaModel.Predict(task.SupportInput));
            var supportLabels = ConvertToVector(task.SupportOutput);
            var queryFeatures = ConvertToVector(MetaModel.Predict(task.QueryInput));

            // Deterministic path: cross-attention from query to context
            var contextReps = BuildContextRepresentations(supportFeatures, supportLabels);
            var attendedRep = CrossAttend(queryFeatures, contextReps);

            // Latent path: aggregate and sample
            var aggRep = AggregateRepresentations(contextReps);
            var (priorMean, priorLogVar) = ComputeLatent(aggRep);

            // Posterior from full set
            var queryLabels = ConvertToVector(task.QueryOutput);
            var fullReps = BuildContextRepresentations(
                ConcatVectors(supportFeatures, queryFeatures),
                ConcatVectors(supportLabels, queryLabels));
            var fullAgg = AggregateRepresentations(fullReps);
            var (postMean, postLogVar) = ComputeLatent(fullAgg);

            var z = ReparameterizeSample(postMean, postLogVar);

            // Modulate backbone with attended rep + latent
            ModulateParameters(initParams, ComputeScale(z, attendedRep));

            double reconLoss = NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
            double klLoss = KLDivergenceGaussian(postMean, postLogVar, priorMean, priorLogVar);
            losses.Add(NumOps.FromDouble(reconLoss + _anpOptions.KLWeight * klLoss));
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        ApplyOuterUpdate(initParams, metaGradients, _anpOptions.OuterLearningRate);

        UpdateAuxiliaryParamsSPSA(taskBatch, ref EncoderParams, _anpOptions.OuterLearningRate, ComputeAuxLoss);
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _attentionParams, _anpOptions.OuterLearningRate, ComputeAuxLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = InterfaceGuard.Parameterizable(MetaModel).GetParameters();
        var supportFeatures = ConvertToVector(MetaModel.Predict(task.SupportInput));
        var supportLabels = ConvertToVector(task.SupportOutput);

        var contextReps = BuildContextRepresentations(supportFeatures, supportLabels);
        var aggRep = AggregateRepresentations(contextReps);
        var (mean, _) = ComputeLatent(aggRep);

        return new NeuralProcessModel<T, TInput, TOutput>(MetaModel, ScaleVector(currentParams, ComputeScale(mean, aggRep)), aggRep);
    }

    private Vector<T> CrossAttend(Vector<T>? queryFeatures, List<Vector<T>> contextReps)
    {
        if (queryFeatures == null || contextReps.Count == 0)
            return new Vector<T>(RepresentationDim);

        int count = contextReps.Count;
        int width = contextReps[0].Length;
        int used = Math.Min(RepresentationDim, width);
        if (used == 0) return new Vector<T>(RepresentationDim);
        var context = Engine.TensorStack(contextReps.Select(r => Tensor<T>.FromVector(r).Cast<double>()).ToArray(), 0);
        int m = Math.Min(queryFeatures.Length, width);
        Tensor<double> weights;
        if (m == 0)
        {
            // An empty query has zero dot products, hence uniform attention over the context.
            weights = Engine.TensorFullLike(new Tensor<double>(new[] { count, 1 }), 1.0);
        }
        else
        {
            var query = Tensor<T>.FromVector(queryFeatures.GetSubVector(0, m)).Cast<double>().ToVector();
            var attention = Tensor<T>.FromVector(ParameterModulation.Cycle(Engine, _attentionParams, m)).Cast<double>().ToVector();
            var scaledQuery = Engine.Multiply(query, attention);
            var scores = Engine.TensorMultiplyScalar(Engine.TensorMatMul(
                Engine.TensorSlice(context, new[] { 0, 0 }, new[] { count, m }),
                Engine.Reshape(Tensor<double>.FromVector(scaledQuery), new[] { m, 1 })),
                1.0 / Math.Sqrt(RepresentationDim));
            // Subtracting a shared maximum preserves normalized attention and prevents exponential overflow.
            weights = Engine.TensorExp(Engine.TensorSubtractScalar(scores, Engine.TensorMaxValue(scores)));
        }
        double sumW = Engine.TensorSum(weights);
        if (!(sumW > 0)) return new Vector<T>(RepresentationDim);
        var attended = Engine.TensorMatMul(
            Engine.Reshape(Engine.TensorDivideScalar(weights, sumW), new[] { 1, count }),
            Engine.TensorSlice(context, new[] { 0, 0 }, new[] { count, used })).Cast<T>().ToVector();
        return used == RepresentationDim
            ? attended
            : Vector<T>.Concatenate(attended, new Vector<T>(RepresentationDim - used));
    }


    private (Vector<T> mean, Vector<T> logvar) ComputeLatent(Vector<T> representation)
    {
        int latentDim = _anpOptions.LatentDim;
        int m = Math.Min(representation.Length, RepresentationDim);
        if (latentDim <= 0 || m == 0)
        {
            return (VectorTanh(new Vector<T>(Math.Max(latentDim, 0))), new Vector<T>(Math.Max(latentDim, 0)));
        }

        // The encoder reads its parameters cyclically: the mean weight for (i, j) is P[(i R + j) mod |P|] and the
        // log-variance weight is P[(latentDim R + i R + j) mod |P|], so both are row-major [latentDim, R] blocks of
        // P repeated to 2 latentDim R entries.
        int block = latentDim * RepresentationDim;
        var cycled = Tensor<T>.FromVector(ParameterModulation.Cycle(Engine, _latentEncoderParams, 2 * block));
        var meanWeights = Engine.TensorSlice(
            Engine.Reshape(Engine.TensorSlice(cycled, new[] { 0 }, new[] { block }), new[] { latentDim, RepresentationDim }),
            new[] { 0, 0 }, new[] { latentDim, m });
        var logVarWeights = Engine.TensorSlice(
            Engine.Reshape(Engine.TensorSlice(cycled, new[] { block }, new[] { block }), new[] { latentDim, RepresentationDim }),
            new[] { 0, 0 }, new[] { latentDim, m });
        var rep = Engine.Reshape(Tensor<T>.FromVector(representation.GetSubVector(0, m)), new[] { m, 1 });

        var meanPre = Engine.TensorMatMul(meanWeights, rep).ToVector();
        var logvar = Engine.Clamp(
            Engine.TensorMatMul(logVarWeights, rep).ToVector(), NumOps.FromDouble(-4), NumOps.FromDouble(2));

        // SIMD tanh on entire mean vector
        var mean = VectorTanh(meanPre);
        return (mean, logvar);
    }

    private double ComputeScale(Vector<T> z, Vector<T> attended)
    {
        var wideZ = Tensor<T>.FromVector(z).Cast<double>().ToVector();
        var wideAttended = Tensor<T>.FromVector(attended).Cast<double>().ToVector();
        double squared = Engine.DotProduct(wideZ, wideZ) + Engine.DotProduct(wideAttended, wideAttended);
        double norm = Math.Sqrt(squared / Math.Max((long)z.Length + attended.Length, 1));
        return 0.5 + 0.5 / (1 + Math.Exp(1 - norm));
    }

    private double ComputeAuxLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var initParams = InterfaceGuard.Parameterizable(MetaModel).GetParameters();
        double totalLoss = 0;
        foreach (var task in taskBatch.Tasks)
        {
            InterfaceGuard.Parameterizable(MetaModel).SetParameters(initParams);
            var sf = ConvertToVector(MetaModel.Predict(task.SupportInput));
            var sl = ConvertToVector(task.SupportOutput);
            var cr = BuildContextRepresentations(sf, sl);
            var att = CrossAttend(ConvertToVector(MetaModel.Predict(task.QueryInput)), cr);
            var agg = AggregateRepresentations(cr);
            var (m, lv) = ComputeLatent(agg);
            var z = ReparameterizeSample(m, lv);
            ModulateParameters(initParams, ComputeScale(z, att));
            totalLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }
        InterfaceGuard.Parameterizable(MetaModel).SetParameters(initParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }
}

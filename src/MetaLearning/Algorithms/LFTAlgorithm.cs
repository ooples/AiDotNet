using AiDotNet.Attributes;
using AiDotNet.Data.Structures;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Learned Feature-Wise Transformation (LFT): train a metric-based few-shot learner against
/// SIMULATED domain shift by perturbing features with affine transforms whose spread is itself
/// learned.
/// </summary>
/// <remarks>
/// <para>
/// Tseng, Lee, Huang and Yang, "Cross-Domain Few-Shot Classification via Learned Feature-Wise
/// Transformation" (ICLR 2020, arXiv:2001.08735). The problem: metric-based few-shot methods "often
/// fail to generalize to unseen domains due to large discrepancy of the feature distribution across
/// domains". The insight is to stop trying to make the encoder produce domain-invariant features and
/// instead make the METRIC function insensitive to which distribution it is handed, by showing it
/// many during training.
/// </para>
/// <code>
///   gamma_c ~ N(1, softplus(theta_gamma_c))     beta_c ~ N(0, softplus(theta_beta_c))
///   zhat_(c,h,w) = gamma_c * z_(c,h,w) + beta_c            (per channel, resampled every use)
///
///   learning to learn, per iteration:
///     split the batch into non-overlapping pseudo-seen T^ps and pseudo-unseen T^pu
///     stage 1:  (theta_e, theta_m) &lt;- (theta_e, theta_m) - a * grad L_cls(T^ps)   WITH transforms
///     stage 2:  theta_f            &lt;- theta_f            - a * grad L^pu(T^pu)   WITHOUT transforms
/// </code>
/// <para>
/// <b>Four properties carry the method, and all four are implemented rather than described.</b>
/// </para>
/// <list type="number">
/// <item><description><b>The transformation is training-only.</b> The paper removes the layers at
/// test time, so inference costs nothing and is deterministic. <see cref="Adapt"/> here never
/// applies them.</description></item>
/// <item><description><b>The second stage measures WITHOUT the transformations.</b> That is the
/// whole objective: the hyper-parameters are judged by how well the model generalizes to a domain it
/// did not train on and where the perturbation is absent. Measuring the pseudo-unseen loss WITH the
/// transformations still applied would reward the model for fitting its own noise.</description></item>
/// <item><description><b>The two domains must not overlap.</b> If the pseudo-unseen tasks appear in
/// the pseudo-seen update, stage 2 measures training loss and the hyper-parameters have no reason to
/// help generalization.</description></item>
/// <item><description><b>The scale is centred on 1 and the bias on 0.</b> The expected transform is
/// the identity, so this perturbs the feature distribution without shifting it — see
/// <see cref="FeatureWiseTransformation{T}"/>.</description></item>
/// </list>
/// <para>
/// <b>Structure.</b> Following this library's ANIL and MbPA convention, the configured meta-model is
/// the feature ENCODER and the algorithm owns the metric head. The transformation sits BETWEEN them,
/// which is what puts it in the gradient path: the head's update is computed from transformed
/// features, so theta_f genuinely changes what the model learns.
/// <para>
/// That placement is load-bearing, and getting it wrong is silent. An earlier revision here applied
/// the transformation only to the reported loss while the model's own <c>ComputeGradients</c>
/// computed its loss internally and never saw it. Training was then completely independent of
/// theta_f, the pseudo-unseen loss was identical for every hyper-parameter value, and the
/// learning-to-learn gradient was exactly zero — a model that cites the paper and implements none of
/// it. <c>LearningToLearn_MovesTheHyperparameters</c> exists to catch precisely that.
/// </para>
/// <para>
/// The paper inserts a transformation after each batch-normalization layer inside a residual
/// encoder; one insertion point at the encoder output is the same mechanism where the generic
/// <c>IFullModel</c> contract can express it, and is documented as such rather than presented as the
/// full multi-layer placement.
/// </para>
/// <para>
/// <b>Estimator.</b> Stage 2's gradient with respect to <c>theta_f</c> flows through stage 1's
/// update, so an exact computation needs second-order autodiff through an arbitrary model, which
/// this generic contract cannot provide. The hyper-parameters are therefore updated by SPSA — the
/// same estimator this library already uses for auxiliary parameters elsewhere. The OBJECTIVE is the
/// paper's exactly; only the way its gradient is estimated differs.
/// </para>
/// <para>
/// <b>For Beginners:</b> A model trained on photographs usually stumbles on sketches, because the
/// numbers its feature extractor produces look different. Instead of forcing those numbers to match,
/// this deliberately jiggles them while training — differently each time — so the comparison step
/// learns to ignore their exact scale. To decide how much jiggle helps, each round holds one group
/// of tasks out, trains on the rest with jiggling, and then checks the held-out group with the
/// jiggling switched off.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input type.</typeparam>
/// <typeparam name="TOutput">The output type.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Cross-Domain Few-Shot Classification via Learned Feature-Wise Transformation",
    "https://arxiv.org/abs/2001.08735",
    Year = 2020,
    Authors = "Hung-Yu Tseng, Hsin-Ying Lee, Jia-Bin Huang, Ming-Hsuan Yang")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class LFTAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private IParameterizable<T, TInput, TOutput>? _cachedParamModel;
    private IParameterizable<T, TInput, TOutput> ParamModel => _cachedParamModel ??= InterfaceGuard.Parameterizable(MetaModel);

    private readonly LFTOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;

    /// <summary>
    /// The feature-wise transformation and its hyper-parameters theta_f.
    /// </summary>
    private readonly FeatureWiseTransformation<T> _transformation;

    /// <summary>
    /// The metric head theta_m, as a flat [outputDim * featureDim | outputDim] vector. The
    /// transformation is applied to the encoder's features BEFORE this head consumes them, so the
    /// head's closed-form gradient carries theta_f's influence into the update.
    /// </summary>
    [AiDotNet.Attributes.Buffer]
    private Vector<T> _metricHead;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.LFT;

    /// <summary>
    /// Gets the transformation, so its learned hyper-parameters can be inspected.
    /// </summary>
    public FeatureWiseTransformation<T> Transformation => _transformation;

    /// <summary>Gets the metric head the transformation feeds.</summary>
    public Vector<T> MetricHead => _metricHead;

    /// <summary>
    /// Gets whether the transformation is currently being applied. It is on only inside the
    /// pseudo-seen stage of <see cref="MetaTrain"/>, and never during <see cref="Adapt"/>.
    /// </summary>
    public bool TransformationActive { get; private set; }

    /// <summary>
    /// Initializes LFT over a feature encoder.
    /// </summary>
    /// <param name="options">LFT options; defaults use the paper's hyper-parameter initialization.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="options"/> is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when a dimension or fraction is out of range.</exception>
    /// <exception cref="ArgumentException">Thrown when the meta-model has no parameters.</exception>
    public LFTAlgorithm(LFTOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        if (options.FeatureDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "FeatureDimension (channel count C) must be positive.");
        if (options.PseudoSeenFraction <= 0.0 || options.PseudoSeenFraction >= 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(options),
                "PseudoSeenFraction must lie strictly between 0 and 1: the pseudo-seen and " +
                "pseudo-unseen domains must both be non-empty for the second stage to measure anything.");
        }

        _algoOptions = options;
        _paramDim = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length;
        if (_paramDim == 0)
            throw new ArgumentException("MetaModel has zero parameters.", nameof(options));

        _transformation = new FeatureWiseTransformation<T>(
            options.FeatureDimension,
            options.InitialScaleHyperparameter,
            options.InitialBiasHyperparameter,
            RandomGenerator);

        int weightCount = options.OutputDimension * options.FeatureDimension;
        _metricHead = new Vector<T>(weightCount + options.OutputDimension);
        double headScale = Math.Sqrt(2.0 / (options.FeatureDimension + options.OutputDimension));
        for (int i = 0; i < weightCount; i++)
        {
            _metricHead[i] = NumOps.FromDouble((RandomGenerator.NextDouble() * 2.0 - 1.0) * headScale);
        }
    }

    #region Meta-training

    /// <inheritdoc/>
    /// <remarks>
    /// One learning-to-learn iteration: split the batch into non-overlapping pseudo-seen and
    /// pseudo-unseen domains, update the model on the pseudo-seen tasks WITH the transformation
    /// applied, then update the transformation hyper-parameters against the pseudo-unseen loss
    /// measured WITHOUT it.
    /// </remarks>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var (pseudoSeen, pseudoUnseen) = SplitDomains(taskBatch);

        // ---- Stage 1: update (theta_e, theta_m) on the pseudo-seen domain, WITH transforms ----
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = ParamModel.GetParameters();

        RunPseudoSeenStage(pseudoSeen, initParams, losses, metaGradients);

        // ---- Stage 2: update theta_f against the pseudo-unseen loss, WITHOUT transforms ----
        if (_algoOptions.LearnTransformationHyperparameters && pseudoUnseen.Count > 0 && pseudoSeen.Count > 0)
        {
            var trainedParams = ParamModel.GetParameters();
            UpdateTransformationHyperparameters(pseudoSeen, pseudoUnseen, initParams);
            // Restore the REAL stage-1 outcome: the probes below re-ran stage 1 under perturbed
            // hyper-parameters and left the model wherever the last probe finished.
            ParamModel.SetParameters(trainedParams);
        }

        return losses.Count > 0 ? ComputeMean(losses) : NumOps.Zero;
    }

    /// <summary>
    /// Stage 1: adapt and meta-update the model on the pseudo-seen domain, WITH the transformation
    /// applied. Leaves the model at the updated parameters.
    /// </summary>
    /// <remarks>
    /// Factored out because stage 2 has to re-run it. The hyper-parameters influence the
    /// pseudo-unseen loss ONLY through this update, so a probe that perturbs them without redoing
    /// this pass measures a quantity that does not depend on them at all — the gradient estimate is
    /// then identically zero and the hyper-parameters never move. That was a real defect here,
    /// caught by LearningToLearn_MovesTheHyperparameters.
    /// </remarks>
    private void RunPseudoSeenStage(
        List<IMetaLearningTask<T, TInput, TOutput>> pseudoSeen,
        Vector<T> initParams,
        List<T>? losses,
        List<Vector<T>> metaGradients)
    {
        TransformationActive = true;
        try
        {
            foreach (var task in pseudoSeen)
            {
                ParamModel.SetParameters(initParams);

                var adaptedParams = initParams.Clone();
                for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
                {
                    ParamModel.SetParameters(adaptedParams);
                    var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
                    for (int d = 0; d < _paramDim; d++)
                    {
                        adaptedParams[d] = NumOps.Subtract(adaptedParams[d],
                            NumOps.FromDouble(_algoOptions.InnerLearningRate * NumOps.ToDouble(grad[d])));
                    }
                }

                ParamModel.SetParameters(adaptedParams);

                // Train the metric head on features that HAVE passed through the transformation.
                // This is the only place theta_f enters the model update, and therefore the only
                // reason the pseudo-unseen loss depends on it at all.
                var supportFeatures = EncodeWithOptionalTransform(task.SupportInput);
                var supportTargets = TargetVectors(task.SupportOutput, supportFeatures.Count);
                for (int i = 0; i < supportFeatures.Count; i++)
                {
                    var headGrad = MbPAOutputNetwork<T>.Gradient(
                        _metricHead, supportFeatures[i], supportTargets[i], weight: 1.0,
                        _algoOptions.FeatureDimension, _algoOptions.OutputDimension,
                        MbPAOutputDistribution.Categorical);
                    for (int d = 0; d < _metricHead.Length; d++)
                    {
                        _metricHead[d] = NumOps.Subtract(_metricHead[d],
                            NumOps.FromDouble(_algoOptions.InnerLearningRate * NumOps.ToDouble(headGrad[d])));
                    }
                }

                losses?.Add(TransformedQueryLoss(task));
                metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
            }

            ApplyOuterUpdate(initParams, metaGradients, _algoOptions.OuterLearningRate);
        }
        finally
        {
            TransformationActive = false;
        }
    }

    /// <summary>
    /// Splits a batch into non-overlapping pseudo-seen and pseudo-unseen domains.
    /// </summary>
    /// <remarks>
    /// Non-overlap is enforced structurally by partitioning one list, not by sampling twice — two
    /// independent samples could share tasks, and a shared task makes stage 2 measure training loss.
    /// A batch too small to split leaves the pseudo-unseen side empty, in which case stage 2 is
    /// skipped rather than run against a degenerate measurement.
    /// </remarks>
    private (List<IMetaLearningTask<T, TInput, TOutput>> Seen, List<IMetaLearningTask<T, TInput, TOutput>> Unseen)
        SplitDomains(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var all = new List<IMetaLearningTask<T, TInput, TOutput>>(taskBatch.Tasks);
        var seen = new List<IMetaLearningTask<T, TInput, TOutput>>();
        var unseen = new List<IMetaLearningTask<T, TInput, TOutput>>();

        if (all.Count <= 1)
        {
            // Nothing to hold out. Train on what there is; stage 2 is skipped by the empty unseen set.
            seen.AddRange(all);
            return (seen, unseen);
        }

        int seenCount = (int)Math.Round(all.Count * _algoOptions.PseudoSeenFraction);
        seenCount = Math.Max(1, Math.Min(all.Count - 1, seenCount));

        for (int i = 0; i < all.Count; i++)
        {
            if (i < seenCount) seen.Add(all[i]); else unseen.Add(all[i]);
        }
        return (seen, unseen);
    }

    /// <summary>
    /// SPSA update of theta_f against the pseudo-unseen loss.
    /// </summary>
    /// <remarks>
    /// Simultaneous perturbation: probe theta_f in one random direction, measure the pseudo-unseen
    /// loss either side, and step along the estimated gradient. Two loss evaluations regardless of
    /// how many hyper-parameters there are, which is what makes it affordable here.
    /// </remarks>
    private void UpdateTransformationHyperparameters(
        List<IMetaLearningTask<T, TInput, TOutput>> pseudoSeen,
        List<IMetaLearningTask<T, TInput, TOutput>> pseudoUnseen,
        Vector<T> initParams)
    {
        int c = _transformation.FeatureDimension;
        const double perturbation = 1e-2;

        var scale = _transformation.ScaleHyperparameters;
        var bias = _transformation.BiasHyperparameters;

        // Rademacher probe direction, the standard SPSA choice.
        var deltaScale = new double[c];
        var deltaBias = new double[c];
        for (int i = 0; i < c; i++)
        {
            deltaScale[i] = RandomGenerator.NextDouble() < 0.5 ? -1.0 : 1.0;
            deltaBias[i] = RandomGenerator.NextDouble() < 0.5 ? -1.0 : 1.0;
        }

        Vector<T> Shift(Vector<T> baseVec, double[] delta, double sign)
        {
            var shifted = new Vector<T>(baseVec.Length);
            for (int i = 0; i < baseVec.Length; i++)
            {
                shifted[i] = NumOps.FromDouble(NumOps.ToDouble(baseVec[i]) + sign * perturbation * delta[i]);
            }
            return shifted;
        }

        // THE HEAD IS PART OF THE PROBE STATE, NOT JUST THE MODEL. RunPseudoSeenStage mutates
        // _metricHead IN PLACE, and ProbeLoss reset only ParamModel -- so ProbeLoss(-1) trained on
        // top of the head ProbeLoss(+1) had already moved. The two probes then measured different
        // head trajectories and the SPSA difference mixed the hyper-parameter effect with the extra
        // head training, biasing the gradient estimate. PseudoUnseenLoss scores THROUGH this head, so
        // the contamination lands directly in the quantity being differenced.
        //
        // _metricHead at entry IS the real stage-1 outcome (stage 1 has already run), so it is both
        // the correct per-probe starting point and the correct value to leave behind -- which also
        // fixes the second half: the head previously kept the mutations from BOTH probes, so the real
        // stage-1 head was lost on every iteration where stage 2 ran.
        var stageOneHead = _metricHead.Clone();

        // Each probe must REDO stage 1 under the perturbed hyper-parameters, then measure the
        // pseudo-unseen loss on the model that produced. Measuring without re-running stage 1 scores
        // a quantity independent of theta_f.
        double ProbeLoss(double sign)
        {
            _transformation.SetHyperparameters(Shift(scale, deltaScale, sign), Shift(bias, deltaBias, sign));
            ParamModel.SetParameters(initParams);
            _metricHead = stageOneHead.Clone();
            RunPseudoSeenStage(pseudoSeen, initParams, losses: null, metaGradients: new List<Vector<T>>());
            return PseudoUnseenLoss(pseudoUnseen);
        }

        double lossPlus;
        double lossMinus;
        try
        {
            lossPlus = ProbeLoss(+1.0);
            lossMinus = ProbeLoss(-1.0);
        }
        finally
        {
            // Restored on every exit, including a throw from inside a probe, which would otherwise
            // leave a perturbed transformation and a probe-contaminated head behind silently.
            _transformation.SetHyperparameters(scale, bias);
            _metricHead = stageOneHead;
        }

        double scaledDifference = (lossPlus - lossMinus) / (2.0 * perturbation);
        if (double.IsNaN(scaledDifference) || double.IsInfinity(scaledDifference)) return;

        var newScale = new Vector<T>(c);
        var newBias = new Vector<T>(c);
        for (int i = 0; i < c; i++)
        {
            newScale[i] = NumOps.FromDouble(NumOps.ToDouble(scale[i])
                - _algoOptions.HyperparameterLearningRate * scaledDifference / deltaScale[i]);
            newBias[i] = NumOps.FromDouble(NumOps.ToDouble(bias[i])
                - _algoOptions.HyperparameterLearningRate * scaledDifference / deltaBias[i]);
        }
        _transformation.SetHyperparameters(newScale, newBias);
    }

    /// <summary>
    /// The pseudo-unseen classification loss, measured with the transformation OFF.
    /// </summary>
    private double PseudoUnseenLoss(List<IMetaLearningTask<T, TInput, TOutput>> pseudoUnseen)
    {
        // Measured through the metric head — which stage 1 just trained on TRANSFORMED features —
        // but with the transformation itself switched off. That combination is the paper's stage-2
        // objective: score the hyper-parameters by the clean generalization of the model they
        // produced, never by how well it fits its own injected noise.
        bool restore = TransformationActive;
        TransformationActive = false;
        try
        {
            double total = 0.0;
            int count = 0;
            foreach (var task in pseudoUnseen)
            {
                var features = EncodeWithOptionalTransform(task.QueryInput);
                var targets = TargetVectors(task.QueryOutput, features.Count);
                for (int i = 0; i < features.Count; i++)
                {
                    var probs = MbPAOutputNetwork<T>.Forward(
                        _metricHead, features[i], _algoOptions.FeatureDimension,
                        _algoOptions.OutputDimension, MbPAOutputDistribution.Categorical);
                    for (int k = 0; k < _algoOptions.OutputDimension; k++)
                    {
                        double t = k < targets[i].Length ? NumOps.ToDouble(targets[i][k]) : 0.0;
                        if (t > 0.0) total -= t * Math.Log(Math.Max(NumOps.ToDouble(probs[k]), 1e-15));
                    }
                    count++;
                }
            }
            return count > 0 ? total / count : 0.0;
        }
        finally
        {
            TransformationActive = restore;
        }
    }

    /// <summary>
    /// Encodes each example of a batch, applying the transformation when it is active.
    /// </summary>
    private List<Vector<T>> EncodeWithOptionalTransform(TInput inputs)
    {
        int batch = MbPAConversions<T>.GetBatchSize(inputs);
        var features = new List<Vector<T>>(batch);
        for (int i = 0; i < batch; i++)
        {
            var single = MbPAConversions<T>.SliceExample(inputs, i);
            var encoded = MbPAConversions<T>.ResizeTo(
                ConvertToVector(MetaModel.Predict(single)), _algoOptions.FeatureDimension);
            features.Add(TransformationActive ? _transformation.Apply(encoded) : encoded);
        }
        return features;
    }

    private List<Vector<T>> TargetVectors(TOutput targets, int count)
    {
        var values = new List<Vector<T>>(count);
        for (int i = 0; i < count; i++)
        {
            values.Add(MbPAConversions<T>.ResizeTo(
                MbPAConversions<T>.SliceTargetRow(targets, i), _algoOptions.OutputDimension));
        }
        return values;
    }

    /// <summary>
    /// Query loss with the feature-wise transformation applied to the encoder's output.
    /// </summary>
    private T TransformedQueryLoss(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var prediction = MetaModel.Predict(task.QueryInput);
        if (!TransformationActive) return ComputeLossFromOutput(prediction, task.QueryOutput);

        var features = ConvertToVector(prediction);
        if (features is null) return ComputeLossFromOutput(prediction, task.QueryOutput);

        var transformed = _transformation.Apply(features);
        var asOutput = ConvertVectorToOutput(transformed, prediction);
        return ComputeLossFromOutput(asOutput, task.QueryOutput);
    }

    /// <summary>
    /// Packs a transformed feature vector back into the model's output type, preserving the
    /// original when the shape cannot be reconstructed.
    /// </summary>
    private TOutput ConvertVectorToOutput(Vector<T> values, TOutput template)
    {
        if (template is Vector<T>) return (TOutput)(object)values;

        if (template is Tensor<T> tensor)
        {
            var packed = new Tensor<T>(tensor._shape);
            int n = Math.Min(values.Length, packed.Length);
            for (int i = 0; i < n; i++) packed[i] = values[i];
            for (int i = n; i < packed.Length; i++) packed[i] = tensor[i];
            return (TOutput)(object)packed;
        }

        return template;
    }

    #endregion

    #region Adaptation

    /// <inheritdoc/>
    /// <remarks>
    /// Standard metric-based adaptation with the transformation OFF. The paper is explicit that the
    /// layers are removed before the model is used, so applying them here would inject noise into
    /// inference — the opposite of what the method is for.
    /// </remarks>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null) throw new ArgumentNullException(nameof(task));

        var initParams = ParamModel.GetParameters();
        var adaptedParams = initParams.Clone();

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            ParamModel.SetParameters(adaptedParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            for (int d = 0; d < _paramDim; d++)
            {
                adaptedParams[d] = NumOps.Subtract(adaptedParams[d],
                    NumOps.FromDouble(_algoOptions.InnerLearningRate * NumOps.ToDouble(grad[d])));
            }
        }

        ParamModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    #endregion
}

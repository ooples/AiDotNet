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
/// Implementation of CAML (Context-Aware Meta-Learning) (Fifty et al., NeurIPS 2023).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// CAML uses a frozen pretrained backbone with a lightweight context module that adapts
/// features based on the support set. Classification is performed by comparing query features
/// to context-adapted class prototypes.
/// </para>
/// <para><b>For Beginners:</b> CAML is built on a simple but powerful insight:
///
/// **The insight:**
/// Modern pretrained models (CLIP, DINO) produce such good features that you don't
/// need to fine-tune them. Instead, learn a lightweight context module that adapts
/// how you USE the features for each specific task.
///
/// **How it works:**
/// 1. Extract features using a frozen pretrained backbone (no gradient computation needed)
/// 2. Compute class prototypes from support features (like ProtoNets)
/// 3. Pass prototypes through a context module that adjusts them based on the task
/// 4. Classify queries by distance to context-adapted prototypes
///
/// **Why freeze the backbone?**
/// - Much faster training (no backbone gradients)
/// - Avoids overfitting on small support sets
/// - Preserves the rich representations learned during pretraining
/// - Only the small context module needs to be meta-learned
/// </para>
/// <para><b>Algorithm - CAML:</b>
/// <code>
/// # Components
/// f_theta = frozen_backbone           # Pretrained feature extractor (NOT updated)
/// g_phi = context_module              # Lightweight context adaptation (meta-learned)
///
/// # Meta-training
/// for each meta-iteration:
///     for each task T_i in batch:
///         # 1. Extract features (no gradient needed for backbone)
///         z_s = f_theta(support_x)
///         z_q = f_theta(query_x)
///
///         # 2. Compute class prototypes
///         p_k = mean(z_s[class == k])
///
///         # 3. Context adaptation
///         context = aggregate(z_s)           # Summarize support set
///         p_adapted = g_phi(p_k, context)    # Adapt prototypes using context
///
///         # 4. Classify queries
///         logits = cosine_similarity(z_q, p_adapted)
///         loss = cross_entropy(logits, query_labels)
///
///     # Only update context module
///     phi = phi - lr * grad(loss)
/// </code>
/// </para>
/// <para>
/// Reference: Fifty, C., Duan, D., Junkins, R.G., Amid, E., Leskovec, J.,
/// Re, C., &amp; Thrun, S. (2023). Context-Aware Meta-Learning. NeurIPS 2023.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Context-Aware Meta-Learning",
    "https://arxiv.org/abs/2310.10971",
    Year = 2023,
    Authors = "Christopher Fifty, Dennis Duan, Ronald G. Junkins, et al.")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class CAMLAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly CAMLOptions<T, TInput, TOutput> _camlOptions;

    /// <summary>Parameters for the lightweight context module.</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _contextParams = new Vector<T>(0);

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.CAML;

    /// <summary>Initializes a new CAML meta-learner.</summary>
    /// <param name="options">Configuration options for CAML.</param>
    public CAMLAlgorithm(CAMLOptions<T, TInput, TOutput> options)
        : base(
            options.MetaModel,
            options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
            options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer,
            optimizerUse: MetaLearnerOptimizerUse.Outer)
    {
        _camlOptions = options;
        InitializeContextModule();
    }

    /// <summary>Initializes the context module parameters.</summary>
    private void InitializeContextModule()
    {
        int ctxDim = Math.Max(_camlOptions.ContextDimension, 1);
        // Context aggregator + context-conditioned projection
        int totalParams = ctxDim * ctxDim + ctxDim + ctxDim * ctxDim + ctxDim;
        _contextParams = new Vector<T>(totalParams);
        double scale = Math.Sqrt(2.0 / ctxDim);
        for (int i = 0; i < totalParams; i++)
        {
            _contextParams[i] = NumOps.FromDouble((RandomGenerator.NextDouble() - 0.5) * 2.0 * scale);
        }
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
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);

            if (!_camlOptions.FreezeBackbone)
            {
                metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
            }
        }

        // Update backbone only if not frozen
        if (!_camlOptions.FreezeBackbone && metaGradients.Count > 0)
        {
            ParamModel.SetParameters(initParams);
            var avgGrad = AverageVectors(metaGradients);
            ParamModel.SetParameters(ApplyOuterGradients(initParams, avgGrad, _camlOptions.OuterLearningRate));
        }

        // Update context module via SPSA
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _contextParams, _camlOptions.OuterLearningRate, ComputeAuxLoss);

        return ComputeMean(losses);
    }

    /// <summary>
    /// Computes the average loss over a task batch using the context module.
    /// Called by SPSA to measure how perturbed context params affect loss.
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
            var adaptedPrototypes = ApplyContextModule(supportFeatures);

            if (supportFeatures != null && adaptedPrototypes != null)
            {
                var ratio = ParameterModulation.MeanClampedRatio(Engine, supportFeatures, adaptedPrototypes, 0.5, 2.0, 1e-10);
                if (ratio is not null)
                {
                    var currentParams = ParamModel.GetParameters();
                    ParamModel.SetParameters(ParameterModulation.Apply(Engine, currentParams, ratio));
                }
            }

            var queryPred = MetaModel.Predict(task.QueryInput);
            totalLoss += NumOps.ToDouble(ComputeLossFromOutput(queryPred, task.QueryOutput));
        }

        ParamModel.SetParameters(initParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }

    /// <summary>
    /// Applies the context module to adapt prototypes based on the support set context.
    /// </summary>
    /// <param name="supportFeatures">Support set features (prototypes).</param>
    /// <returns>Context-adapted prototypes.</returns>
    private Vector<T>? ApplyContextModule(Vector<T>? supportFeatures)
    {
        if (supportFeatures == null || supportFeatures.Length == 0)
            return supportFeatures;

        int n = supportFeatures.Length;

        // Aggregate support features into a context vector (mean pooling)
        T contextMean = Engine.Mean(supportFeatures);

        // Each feature reads four context parameters in order - gate weight, gate bias, projection weight,
        // projection bias - and once the parameters run out every read takes its default (0.01, 0, 0.01, 0).
        T small = NumOps.FromDouble(0.01);
        var table = ParameterModulation.DefaultedTable(
            Engine, _contextParams, n, new Vector<T>(new[] { small, NumOps.Zero, small, NumOps.Zero }));
        Vector<T> Column(int column) => ParameterModulation.Column(Engine, table, column);

        // Context gate: sigmoid(w * context + b)
        var gate = Engine.Sigmoid(Engine.Add(Engine.Multiply(Column(0), contextMean), Column(1)));

        // Context-adapted feature: gate * projection(feature) + (1-gate) * feature = f + gate * (projection - f)
        var projected = Engine.Add(Engine.Multiply(supportFeatures, Column(2)), Column(3));
        return Engine.Add(supportFeatures, Engine.Multiply(gate, Engine.Subtract(projected, supportFeatures)));
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = ParamModel.GetParameters();

        // Extract support features
        var supportPred = MetaModel.Predict(task.SupportInput);
        var supportFeatures = ConvertToVector(supportPred);

        // Apply context module to adapt prototypes
        var adaptedPrototypes = ApplyContextModule(supportFeatures);

        // Compute modulation factors from adapted vs raw prototypes
        Vector<T>? modulationFactors = null;
        if (supportFeatures != null && adaptedPrototypes != null)
        {
            modulationFactors = ParameterModulation.MeanClampedRatio(Engine, supportFeatures, adaptedPrototypes, 0.5, 2.0, 1e-10);
        }

        return new CAMLModel<T, TInput, TOutput>(MetaModel, currentParams, adaptedPrototypes, modulationFactors);
    }

}

/// <summary>Adapted model wrapper for CAML with context-adapted prototypes.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> This model uses prototypes that have been adapted
/// by a lightweight context module. The frozen backbone provides rich features,
/// and the context module adjusts prototypes based on the specific task context.
/// </para>
/// </remarks>
internal class CAMLModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>, IAdaptedMetaModel<T>
{
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _backboneParams;
    private readonly Vector<T>? _adaptedPrototypes;
    // Not readonly: a readonly Vector<T> field is not persisted by the generated model state, and these
    // factors belong in the adapted model's checkpoint as the double[] they replace did.
    private Vector<T>? _modulationFactors;

    /// <inheritdoc/>
    public Vector<T>? AdaptedSupportFeatures => _adaptedPrototypes;

    /// <inheritdoc/>
    public Vector<T>? ParameterModulationFactors => _modulationFactors;

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    public CAMLModel(IFullModel<T, TInput, TOutput> model, Vector<T> backboneParams,
        Vector<T>? adaptedPrototypes, Vector<T>? modulationFactors)
    {
        // Predict pushes the adapted parameters into this model, so it is a copy: holding the meta-learner's
        // MetaModel would let every prediction overwrite the meta-model.
        _model = (model ?? throw new ArgumentNullException(nameof(model))).Clone();
        _backboneParams = backboneParams;
        _adaptedPrototypes = adaptedPrototypes;
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

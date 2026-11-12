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
/// Implementation of ICM-Fusion: In-Context Meta-Optimized LoRA Fusion (Shao et al., AAAI 2026).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// ICM-Fusion fuses task-specific adapters through a Fusion VAE (F-VAE). The pretrained weights stay frozen;
/// what is encoded, fused and reconstructed is the ADAPTER - the offset a task's fine-tuning puts on top of
/// them, which is what a LoRA is (W = W_pt + BA). Here the adapter of task i is
/// <c>l_i = theta_i - theta_pt</c>, where <c>theta_pt</c> is the meta-model's parameters at construction.
/// </para>
/// <para><b>Meta-training, Algorithm 1 of the paper, per task:</b>
/// <code>
///   4:  l_i                  = ObtainLoRAParams(T_i)   inner-loop fine-tuning, as an offset from theta_pt
///   5:  v_i                  = ComputeTaskVector(T_i)  theta_i - theta_start (Figure 2, step 2)
///   6:  (mu_i, log sigma_i^2) = E([l_i; v_i])
///   7:  z_i ~ N(mu_i, diag(sigma_i^2))
///   8:  l_init               = D([z_i; v_i])
///   9:  l_adapt              = l_init - beta * grad L_task(l_init, D_i)      (K steps, Eq. 8)
///  10:  (mu_i, log sigma_i^2) = E([l_adapt; v_i])
///  11:  l_recon              = D([z_i; v_i])            z_i redrawn from the step-10 posterior (Eq. 10)
///  12:  L_recon              = MSE(l_i, l_recon)
///  13:  L_KL                 = KL(N(mu_i, sigma_i^2) || N(0, I))           (Eq. 11)
///  14:  L_meta               = L_recon + lambda_KL * L_KL                  (Eq. 12)
///  16-17: phi, psi          -= gamma * grad L_meta                         (Eqs. 13-14)
/// </code>
/// </para>
/// <para>
/// <b>Gradients are exact, not estimated.</b> The encoder and decoder are affine, so L_meta is differentiated in
/// closed form, reparameterization included. Step 9 makes l_adapt depend on the decoder and, through z_i, on the
/// encoder; with <c>UseFirstOrder = false</c> that dependence is back-propagated through every refinement step as
/// <c>(I - beta H)</c> products, each Hessian-vector product taken as a central difference of task gradients.
/// With <c>UseFirstOrder = true</c> (the default) l_adapt is held constant, exactly as first-order MAML holds the
/// adapted weights constant.
/// </para>
/// <para>
/// <b>Beyond the paper: the meta-model is the fused model.</b> The paper fuses at inference only. Here, after each
/// VAE update, the batch's posterior means are pushed into a decaying component history (<c>NumFusionComponents</c>,
/// <c>FusionDecay</c>), averaged in latent space, decoded, and written back as
/// <c>theta = theta_pt + D([z_fused; v_fused])</c>. The decoder starts at zero - the LoRA convention (Hu et al. 2022)
/// that an untrained adapter is the identity - so the meta-model starts exactly at the pretrained weights and moves
/// only as far as the VAE has learned to reconstruct real adapters. The next batch then fine-tunes from the fused
/// model, so the history accumulates rather than restarting.
/// </para>
/// <para>
/// Parameter vectors wider than 128 are compressed for the VAE into contiguous blocks: a block's value is the mean
/// of its entries, a decoded block is written to all of its entries, and a gradient reaches a block as the sum over
/// its entries - the chain rule of that write.
/// </para>
/// <para><b>For Beginners:</b> Fine-tuning a model on one task gives a small set of changes to its weights (an
/// adapter). Fine-tuning on another task gives different changes, and simply averaging the two often makes the
/// model worse at both. ICM-Fusion trains a small variational autoencoder to squeeze each adapter into a short
/// code, blends the codes, and turns the blend back into one adapter that works for both tasks. The original
/// weights are never overwritten: the model you get is always "original weights + blended adapter".</para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("ICM-Fusion: In-Context Meta-Optimized LoRA Fusion for Multi-Task Adaptation",
    "https://arxiv.org/abs/2508.04153",
    Year = 2025,
    Authors = "Yihua Shao, Xiaofeng Lin, et al.")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class ICMFusionAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly ICMFusionOptions<T, TInput, TOutput> _algoOptions;

    private const int MaxCompressedDim = 128;

    /// <summary>Upper and lower bound on an encoded log-variance, for numerical stability.</summary>
    private const double LogVarianceBound = 10.0;

    /// <summary>
    /// Encoder E: [l; v] (2c) to (mu, log sigma^2) (2d). Layout: mu weights (d x 2c), log-variance weights
    /// (d x 2c), mu bias (d), log-variance bias (d).
    /// </summary>
    [TrainableParameter]
    private Vector<T> _encoderParams;

    /// <summary>Decoder D: [z; v] (d + c) to an adapter (c). Layout: weights (c x (d + c)), bias (c).</summary>
    [TrainableParameter]
    private Vector<T> _decoderParams;

    /// <summary>theta_pt: the meta-model's parameters at construction, frozen for the learner's lifetime.</summary>
    [Buffer]
    private Vector<T> _pretrainedParams;

    /// <summary>Circular history of fused posterior means, NumFusionComponents x LatentDim.</summary>
    [Buffer]
    private Vector<T> _fusionLatents;

    /// <summary>Circular history of the task vectors those means were encoded with, NumFusionComponents x c.</summary>
    [Buffer]
    private Vector<T> _fusionTaskVectors;

    /// <summary>[next write slot, filled slot count] of the fusion history.</summary>
    [Buffer]
    private Vector<T> _fusionCursor;

    private readonly int _paramDim;
    private readonly int _latentDim;
    private readonly int _compressedDim;
    private readonly int _stride;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.ICMFusion;

    public ICMFusionAlgorithm(ICMFusionOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer,
               optimizerUse: MetaLearnerOptimizerUse.Inner)
    {
        if (options.LatentDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "LatentDim must be positive.");
        if (options.NumFusionComponents <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "NumFusionComponents must be positive.");
        if (options.AdaptationSteps < 0)
            throw new ArgumentOutOfRangeException(nameof(options), "AdaptationSteps cannot be negative.");

        _algoOptions = options;
        RequirePlainSgdInnerSteps(!options.UseFirstOrder, "second-order meta-gradients through the inner trajectory");
        _pretrainedParams = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Clone();
        _paramDim = _pretrainedParams.Length;
        if (_paramDim == 0)
            throw new ArgumentException("MetaModel has zero parameters.", nameof(options));
        _latentDim = options.LatentDim;
        _compressedDim = Math.Min(_paramDim, MaxCompressedDim);
        _stride = _paramDim / _compressedDim;

        int c = _compressedDim, d = _latentDim;
        _encoderParams = new Vector<T>((2 * d) * (2 * c) + 2 * d);
        double encoderScale = 1.0 / Math.Sqrt(2.0 * c);
        for (int i = 0; i < (2 * d) * (2 * c); i++)
            _encoderParams[i] = NumOps.FromDouble(SampleNormal() * encoderScale);

        // Zero decoder: an untrained adapter is the identity, so the fused model starts at theta_pt.
        _decoderParams = new Vector<T>(c * (d + c) + c);

        _fusionLatents = new Vector<T>(options.NumFusionComponents * d);
        _fusionTaskVectors = new Vector<T>(options.NumFusionComponents * c);
        _fusionCursor = new Vector<T>(2);
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null) throw new ArgumentNullException(nameof(taskBatch));

        var thetaStart = ParamModel.GetParameters();
        int taskCount = taskBatch.Tasks.Length;
        if (taskCount == 0) return NumOps.Zero;

        var encoder = Tensor<T>.FromVector(_encoderParams.Clone());
        var decoder = Tensor<T>.FromVector(_decoderParams.Clone());
        var encoderGradient = new Tensor<T>(new[] { encoder.Length });
        var decoderGradient = new Tensor<T>(new[] { decoder.Length });
        var adapters = new List<Tensor<T>>(taskCount);
        var taskVectors = new List<Tensor<T>>(taskCount);
        bool secondOrder = !_algoOptions.UseFirstOrder;
        var losses = new T[taskCount];
        int index = 0;

        foreach (var task in taskBatch.Tasks)
        {
            var firstNoise = SampleLatentNoise();
            var secondNoise = SampleLatentNoise();
            var result = TaskObjective(task, thetaStart, encoder, decoder, firstNoise, secondNoise, secondOrder,
                withGradient: true, frozenAdaptedAdapter: null);
            losses[index++] = result.Loss;
            if (result.EncoderGradient is not null) encoderGradient = Engine.TensorAdd(encoderGradient, result.EncoderGradient);
            if (result.DecoderGradient is not null) decoderGradient = Engine.TensorAdd(decoderGradient, result.DecoderGradient);
            adapters.Add(result.Adapter);
            taskVectors.Add(result.TaskVector);
        }

        // Eqs. 13-14: one SGD step on the batch-mean meta-loss gradient, clipped as one vector.
        T count = NumOps.FromDouble(taskCount);
        var combined = ClipGradients(Engine.TensorDivideScalar(
            Engine.TensorConcatenate(new[] { encoderGradient, decoderGradient }, 0), count).ToVector());
        var step = Engine.TensorMultiplyScalar(Tensor<T>.FromVector(combined), NumOps.FromDouble(_algoOptions.OuterLearningRate));
        encoder = Engine.TensorSubtract(encoder, Engine.TensorSlice(step, new[] { 0 }, new[] { encoder.Length }));
        decoder = Engine.TensorSubtract(decoder, Engine.TensorSlice(step, new[] { encoder.Length }, new[] { decoder.Length }));
        encoder.ToVector().AsSpan().CopyTo(_encoderParams.AsWritableSpan());
        decoder.ToVector().AsSpan().CopyTo(_decoderParams.AsWritableSpan());

        // Fuse in latent space with the updated encoder, decode with the updated decoder.
        for (int t = 0; t < taskCount; t++)
        {
            StoreComponent(Encode(encoder, adapters[t], taskVectors[t]).Mean, taskVectors[t]);
        }

        var fused = FuseComponents();
        ParamModel.SetParameters(PretrainedPlus(Decode(decoder, fused.Latent, fused.TaskVector)));

        return NumOps.Divide(Engine.TensorSum(new Tensor<T>(losses, new[] { taskCount })), count);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// In-context generation of task-specialized parameters (Figure 2, step 3): the task's adapter and task vector are
    /// encoded, the posterior mean is decoded into an initial adapter, and that adapter is refined on the support set
    /// (Eq. 8). The meta-model is left exactly as it was.
    /// </remarks>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null) throw new ArgumentNullException(nameof(task));

        var thetaStart = ParamModel.GetParameters();
        var encoder = Tensor<T>.FromVector(_encoderParams.Clone());
        var decoder = Tensor<T>.FromVector(_decoderParams.Clone());

        var thetaTask = ObtainTaskParameters(thetaStart, task);
        var adapter = CompressDifference(thetaTask, _pretrainedParams);
        var taskVector = CompressDifference(thetaTask, thetaStart);
        var encoded = Encode(encoder, adapter, taskVector);
        var refined = RefineAdapter(Decode(decoder, encoded.Mean, taskVector), task, trajectory: null);
        var finalParams = PretrainedPlus(refined);

        ParamModel.SetParameters(thetaStart);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, finalParams);
    }

    /// <summary>One encoding (Algorithm 1, lines 6 and 10): the posterior mean and clamped log-variance.</summary>
    private sealed class Encoding
    {
        public Encoding(Tensor<T> mean, Tensor<T> logVar, Tensor<T> clamped)
        {
            Mean = mean;
            LogVar = logVar;
            Clamped = clamped;
        }

        public Tensor<T> Mean { get; }

        public Tensor<T> LogVar { get; }

        /// <summary>1 where the raw log-variance fell outside the bound and was clamped (its gradient is zero).</summary>
        public Tensor<T> Clamped { get; }
    }

    /// <summary>L_meta for one task, the quantities MetaTrain reuses, and its exact gradient when requested.</summary>
    private sealed class ObjectiveResult
    {
        public ObjectiveResult(T loss, Tensor<T> adapter, Tensor<T> taskVector, Tensor<T> adaptedAdapter,
            Tensor<T>? encoderGradient, Tensor<T>? decoderGradient)
        {
            Loss = loss;
            Adapter = adapter;
            TaskVector = taskVector;
            AdaptedAdapter = adaptedAdapter;
            EncoderGradient = encoderGradient;
            DecoderGradient = decoderGradient;
        }

        public T Loss { get; }

        public Tensor<T> Adapter { get; }

        public Tensor<T> TaskVector { get; }

        public Tensor<T> AdaptedAdapter { get; }

        public Tensor<T>? EncoderGradient { get; }

        public Tensor<T>? DecoderGradient { get; }
    }

    /// <summary>
    /// L_meta for one task (Algorithm 1, lines 4-14), with its gradient with respect to the encoder and decoder
    /// when <paramref name="withGradient"/> is set.
    /// </summary>
    private ObjectiveResult TaskObjective(
        IMetaLearningTask<T, TInput, TOutput> task,
        Vector<T> thetaStart,
        Tensor<T> encoder,
        Tensor<T> decoder,
        Tensor<T> firstNoise,
        Tensor<T> secondNoise,
        bool secondOrder,
        bool withGradient,
        Tensor<T>? frozenAdaptedAdapter)
    {
        int c = _compressedDim, d = _latentDim;
        T klWeight = NumOps.FromDouble(_algoOptions.KLWeight);
        T half = NumOps.FromDouble(0.5);

        // Lines 4-5. Neither depends on the VAE.
        var thetaTask = ObtainTaskParameters(thetaStart, task);
        var adapter = CompressDifference(thetaTask, _pretrainedParams);
        var taskVector = CompressDifference(thetaTask, thetaStart);

        // Lines 6-8.
        var first = Encode(encoder, adapter, taskVector);
        var firstLatent = Reparameterize(first, firstNoise);
        var initialAdapter = Decode(decoder, firstLatent, taskVector);

        // Line 9.
        List<Tensor<T>>? trajectory = secondOrder && frozenAdaptedAdapter == null ? new List<Tensor<T>>() : null;
        var adaptedAdapter = frozenAdaptedAdapter ?? RefineAdapter(initialAdapter, task, trajectory);

        // Lines 10-14.
        var second = Encode(encoder, adaptedAdapter, taskVector);
        var latent = Reparameterize(second, secondNoise);
        var reconstruction = Decode(decoder, latent, taskVector);

        var residual = Engine.TensorSubtract(reconstruction, adapter);
        T recon = NumOps.Divide(Engine.TensorSumOfSquares(residual), NumOps.FromDouble(c));
        var variance = Engine.TensorExp(second.LogVar);
        var klTerms = Engine.TensorSubtract(
            Engine.TensorSubtract(Engine.TensorAddScalar(second.LogVar, NumOps.One), Engine.TensorMultiply(second.Mean, second.Mean)),
            variance);
        T kl = NumOps.Multiply(NumOps.FromDouble(-0.5), Engine.TensorSum(klTerms));
        T loss = NumOps.Add(recon, NumOps.Multiply(klWeight, kl));

        if (!withGradient)
        {
            return new ObjectiveResult(loss, adapter, taskVector, adaptedAdapter, null, null);
        }

        var zeros = new Tensor<T>(new[] { d });
        var reconstructionGradient = Engine.TensorMultiplyScalar(residual, NumOps.FromDouble(2.0 / c));
        var decoderGradient = DecoderGradient(reconstructionGradient, latent, taskVector);
        var latentGradient = DecoderLatentBackward(decoder, reconstructionGradient);

        var meanGradient = Engine.TensorAdd(latentGradient, Engine.TensorMultiplyScalar(second.Mean, klWeight));
        var pathGradient = Engine.TensorMultiply(Engine.TensorMultiply(latentGradient, secondNoise), HalfStd(second.LogVar));
        var klGradient = Engine.TensorMultiplyScalar(
            Engine.TensorAddScalar(Engine.TensorNegate(variance), NumOps.One), NumOps.Multiply(klWeight, half));
        var logVarGradient = Engine.TensorWhere(second.Clamped, zeros, Engine.TensorSubtract(pathGradient, klGradient));

        var encoderGradient = EncoderGradient(meanGradient, logVarGradient, adaptedAdapter, taskVector);

        if (trajectory is not null)
        {
            // dL/dl_adapt through the step-10 encoding, then back through every refinement step.
            var adaptedGradient = EncoderAdapterBackward(encoder, meanGradient, logVarGradient);
            var initialGradient = BackpropagateRefinement(adaptedGradient, trajectory, task);

            decoderGradient = Engine.TensorAdd(decoderGradient, DecoderGradient(initialGradient, firstLatent, taskVector));
            var firstLatentGradient = DecoderLatentBackward(decoder, initialGradient);
            var firstLogVarGradient = Engine.TensorWhere(first.Clamped, zeros,
                Engine.TensorMultiply(Engine.TensorMultiply(firstLatentGradient, firstNoise), HalfStd(first.LogVar)));

            encoderGradient = Engine.TensorAdd(encoderGradient,
                EncoderGradient(firstLatentGradient, firstLogVarGradient, adapter, taskVector));
        }

        return new ObjectiveResult(loss, adapter, taskVector, adaptedAdapter, encoderGradient, decoderGradient);
    }

    /// <summary>ObtainLoRAParams: inner-loop fine-tuning of the current meta-model on the support set.</summary>
    private Vector<T> ObtainTaskParameters(Vector<T> thetaStart, IMetaLearningTask<T, TInput, TOutput> task)
    {
        var adapted = thetaStart.Clone();
        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            ParamModel.SetParameters(adapted);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            adapted = ApplyInnerGradients(adapted, grad, _algoOptions.InnerLearningRate, step == 0);
        }

        ParamModel.SetParameters(thetaStart);
        return adapted;
    }

    /// <summary>Eq. 8: K plain gradient steps on the adapter, recording each point the step was taken from.</summary>
    private Tensor<T> RefineAdapter(Tensor<T> initialAdapter, IMetaLearningTask<T, TInput, TOutput> task, List<Tensor<T>>? trajectory)
    {
        var current = initialAdapter;
        T beta = NumOps.FromDouble(_algoOptions.InnerLearningRate);
        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            trajectory?.Add(current);
            var gradient = AdapterGradient(current, task);
            current = Engine.TensorSubtract(current, Engine.TensorMultiplyScalar(gradient, beta));
        }

        return current;
    }

    /// <summary>
    /// Pulls dL/dl_K back to dL/dl_0 through l_{k+1} = l_k - beta g(l_k): a_k = a_{k+1} - beta H(l_k) a_{k+1}.
    /// </summary>
    private Tensor<T> BackpropagateRefinement(Tensor<T> adaptedGradient, List<Tensor<T>> trajectory, IMetaLearningTask<T, TInput, TOutput> task)
    {
        var current = adaptedGradient;
        T beta = NumOps.FromDouble(_algoOptions.InnerLearningRate);
        for (int step = trajectory.Count - 1; step >= 0; step--)
        {
            var hessianVector = HessianVectorProduct(trajectory[step], current, task);
            current = Engine.TensorSubtract(current, Engine.TensorMultiplyScalar(hessianVector, beta));
        }

        return current;
    }

    /// <summary>H(point) * direction as a central difference of task gradients along the direction.</summary>
    private Tensor<T> HessianVectorProduct(Tensor<T> point, Tensor<T> direction, IMetaLearningTask<T, TInput, TOutput> task)
    {
        T directionNorm = NumOps.Sqrt(Engine.TensorSumOfSquares(direction));
        if (NumOps.Equals(directionNorm, NumOps.Zero)) return new Tensor<T>(new[] { direction.Length });

        T h = NumOps.Divide(
            NumOps.Multiply(NumOps.FromDouble(1e-4), NumOps.Add(NumOps.One, NumOps.Sqrt(Engine.TensorSumOfSquares(point)))),
            directionNorm);
        var offset = Engine.TensorMultiplyScalar(direction, h);
        var gradientPlus = AdapterGradient(Engine.TensorAdd(point, offset), task);
        var gradientMinus = AdapterGradient(Engine.TensorSubtract(point, offset), task);
        return Engine.TensorDivideScalar(Engine.TensorSubtract(gradientPlus, gradientMinus), NumOps.Multiply(NumOps.FromDouble(2.0), h));
    }

    /// <summary>dL_task/dl at theta_pt + expand(l), summed per block.</summary>
    private Tensor<T> AdapterGradient(Tensor<T> adapter, IMetaLearningTask<T, TInput, TOutput> task)
    {
        ParamModel.SetParameters(PretrainedPlus(adapter));
        var full = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
        return BlockSums(Tensor<T>.FromVector(full));
    }

    /// <summary>E([l; v]): weights <c>[2d, 2c]</c> (mean rows, then log-variance rows) and a bias <c>[2d]</c>.</summary>
    private Encoding Encode(Tensor<T> encoder, Tensor<T> adapter, Tensor<T> taskVector)
    {
        int c = _compressedDim, d = _latentDim, width = 2 * c;
        var weights = EncoderWeights(encoder);
        var bias = Engine.TensorSlice(encoder, new[] { 2 * d * width }, new[] { 2 * d });
        var input = Engine.Reshape(Engine.TensorConcatenate(new[] { adapter, taskVector }, 0), new[] { width, 1 });
        var output = Engine.TensorAdd(Engine.Reshape(Engine.TensorMatMul(weights, input), new[] { 2 * d }), bias);

        var mean = Engine.TensorSlice(output, new[] { 0 }, new[] { d });
        var raw = Engine.TensorSlice(output, new[] { d }, new[] { d });
        T bound = NumOps.FromDouble(LogVarianceBound);
        var clamped = Engine.TensorGreaterThan(Engine.TensorAbs(raw), bound);
        var logVar = Engine.TensorClamp(raw, NumOps.Negate(bound), bound);
        return new Encoding(mean, logVar, clamped);
    }

    /// <summary>D([z; v]): weights <c>[c, d + c]</c> and a bias <c>[c]</c>.</summary>
    private Tensor<T> Decode(Tensor<T> decoder, Tensor<T> latent, Tensor<T> taskVector)
    {
        int c = _compressedDim, d = _latentDim, width = d + c;
        var weights = DecoderWeights(decoder);
        var bias = Engine.TensorSlice(decoder, new[] { c * width }, new[] { c });
        var input = Engine.Reshape(Engine.TensorConcatenate(new[] { latent, taskVector }, 0), new[] { width, 1 });
        return Engine.TensorAdd(Engine.Reshape(Engine.TensorMatMul(weights, input), new[] { c }), bias);
    }

    private Tensor<T> EncoderWeights(Tensor<T> encoder)
    {
        int c = _compressedDim, d = _latentDim, width = 2 * c;
        return Engine.Reshape(Engine.TensorSlice(encoder, new[] { 0 }, new[] { 2 * d * width }), new[] { 2 * d, width });
    }

    private Tensor<T> DecoderWeights(Tensor<T> decoder)
    {
        int c = _compressedDim, d = _latentDim, width = d + c;
        return Engine.Reshape(Engine.TensorSlice(decoder, new[] { 0 }, new[] { c * width }), new[] { c, width });
    }

    /// <summary>The decoder's gradient for one decoding: outer(g, [z; v]) for the weights, g for the bias.</summary>
    private Tensor<T> DecoderGradient(Tensor<T> outputGradient, Tensor<T> latent, Tensor<T> taskVector)
    {
        int c = _compressedDim, d = _latentDim;
        var input = Engine.TensorConcatenate(new[] { latent, taskVector }, 0);
        var weights = Engine.Reshape(Engine.TensorOuter(outputGradient, input), new[] { c * (d + c) });
        return Engine.TensorConcatenate(new[] { weights, outputGradient }, 0);
    }

    /// <summary>The gradient a decoding passes back to its latent input (the first d inputs).</summary>
    private Tensor<T> DecoderLatentBackward(Tensor<T> decoder, Tensor<T> outputGradient)
    {
        int c = _compressedDim, d = _latentDim;
        var latentWeights = Engine.TensorSlice(DecoderWeights(decoder), new[] { 0, 0 }, new[] { c, d });
        return Engine.Reshape(Engine.TensorMatMul(Engine.Reshape(outputGradient, new[] { 1, c }), latentWeights), new[] { d });
    }

    /// <summary>The encoder's gradient for one encoding: outer([g_mu; g_logvar], [l; v]) and the bias gradient.</summary>
    private Tensor<T> EncoderGradient(Tensor<T> meanGradient, Tensor<T> logVarGradient, Tensor<T> adapter, Tensor<T> taskVector)
    {
        int c = _compressedDim, d = _latentDim;
        var outputGradient = Engine.TensorConcatenate(new[] { meanGradient, logVarGradient }, 0);
        var input = Engine.TensorConcatenate(new[] { adapter, taskVector }, 0);
        var weights = Engine.Reshape(Engine.TensorOuter(outputGradient, input), new[] { (2 * d) * (2 * c) });
        return Engine.TensorConcatenate(new[] { weights, outputGradient }, 0);
    }

    /// <summary>The gradient an encoding passes back to its adapter input (the first c inputs).</summary>
    private Tensor<T> EncoderAdapterBackward(Tensor<T> encoder, Tensor<T> meanGradient, Tensor<T> logVarGradient)
    {
        int c = _compressedDim, d = _latentDim;
        var outputGradient = Engine.Reshape(Engine.TensorConcatenate(new[] { meanGradient, logVarGradient }, 0), new[] { 1, 2 * d });
        var adapterWeights = Engine.TensorSlice(EncoderWeights(encoder), new[] { 0, 0 }, new[] { 2 * d, c });
        return Engine.Reshape(Engine.TensorMatMul(outputGradient, adapterWeights), new[] { c });
    }

    /// <summary>d sigma / d log sigma^2 = 0.5 exp(0.5 log sigma^2).</summary>
    private Tensor<T> HalfStd(Tensor<T> logVar)
    {
        T half = NumOps.FromDouble(0.5);
        return Engine.TensorMultiplyScalar(Engine.TensorExp(Engine.TensorMultiplyScalar(logVar, half)), half);
    }

    private Tensor<T> Reparameterize(Encoding encoding, Tensor<T> noise)
        => Engine.TensorAdd(encoding.Mean,
            Engine.TensorMultiply(Engine.TensorExp(Engine.TensorMultiplyScalar(encoding.LogVar, NumOps.FromDouble(0.5))), noise));

    /// <summary>Standard normal latent noise, one seeded engine draw from the learner's generator.</summary>
    private Tensor<T> SampleLatentNoise()
    {
        var noise = new Tensor<T>(new[] { _latentDim });
        Engine.TensorNormalInPlace(noise, 0.0, 1.0, RandomGenerator.Next());
        return noise;
    }

    /// <summary>Writes one posterior mean and its task vector into the next slot of the circular history.</summary>
    private void StoreComponent(Tensor<T> latent, Tensor<T> taskVector)
    {
        int slots = _algoOptions.NumFusionComponents;
        int slot = (int)NumOps.ToDouble(_fusionCursor[0]);
        int filled = (int)NumOps.ToDouble(_fusionCursor[1]);
        latent.ToVector().AsSpan().CopyTo(_fusionLatents.AsWritableSpan().Slice(slot * _latentDim, _latentDim));
        taskVector.ToVector().AsSpan().CopyTo(_fusionTaskVectors.AsWritableSpan().Slice(slot * _compressedDim, _compressedDim));
        _fusionCursor[0] = NumOps.FromDouble((slot + 1) % slots);
        _fusionCursor[1] = NumOps.FromDouble(Math.Min(filled + 1, slots));
    }

    /// <summary>Latent-space task arithmetic: the newest component weighs 1, each older one FusionDecay times less.</summary>
    private (Tensor<T> Latent, Tensor<T> TaskVector) FuseComponents()
    {
        int filled = (int)NumOps.ToDouble(_fusionCursor[1]);
        if (filled == 0)
        {
            return (new Tensor<T>(new[] { _latentDim }), new Tensor<T>(new[] { _compressedDim }));
        }

        // Age weights 1, decay, decay^2, ... for the newest-first rows; the newest weighs 1, so the total is positive.
        var factors = new Tensor<T>(new[] { filled });
        factors.Fill(NumOps.FromDouble(_algoOptions.FusionDecay));
        factors[0] = NumOps.One;
        var weights = Engine.TensorCumProd(factors, 0);
        T totalWeight = Engine.TensorSum(weights);
        var row = Engine.Reshape(weights, new[] { 1, filled });

        Tensor<T> Fuse(Vector<T> history, int width)
        {
            var newest = NewestFirst(history, width, filled);
            var sum = Engine.Reshape(Engine.TensorMatMul(row, newest), new[] { width });
            return Engine.TensorDivideScalar(sum, totalWeight);
        }

        return (Fuse(_fusionLatents, _latentDim), Fuse(_fusionTaskVectors, _compressedDim));
    }

    /// <summary>The <paramref name="count"/> newest rows of a circular history, newest first.</summary>
    private Tensor<T> NewestFirst(Vector<T> history, int width, int count)
    {
        int slots = _algoOptions.NumFusionComponents;
        int next = (int)NumOps.ToDouble(_fusionCursor[0]);

        // Flipped, row p holds slot (slots - 1 - p); rolling by next brings slot next - 1 to row 0.
        var rows = Engine.Reshape(Tensor<T>.FromVector(history), new[] { slots, width });
        var ordered = Engine.TensorRoll(Engine.TensorFlip(rows, new[] { 0 }), new[] { next }, new[] { 0 });
        return Engine.TensorSlice(ordered, new[] { 0, 0 }, new[] { count, width });
    }

    /// <summary>Each block's entry count: <c>stride</c>, and the remainder for the last block.</summary>
    private Tensor<T> BlockSizes()
    {
        var sizes = new Tensor<T>(new[] { _compressedDim });
        sizes.Fill(NumOps.FromDouble(_stride));
        sizes[_compressedDim - 1] = NumOps.FromDouble(_paramDim - (_compressedDim - 1) * _stride);
        return sizes;
    }

    /// <summary>The sum of a full parameter-space vector over each contiguous block, <c>[c]</c>.</summary>
    private Tensor<T> BlockSums(Tensor<T> full)
    {
        int c = _compressedDim, head = (c - 1) * _stride;
        var tail = Engine.ReduceSum(Engine.TensorSlice(full, new[] { head }, new[] { _paramDim - head }), new[] { 0 }, keepDims: true);
        if (c == 1) return Engine.Reshape(tail, new[] { 1 });

        var heads = Engine.ReduceSum(
            Engine.Reshape(Engine.TensorSlice(full, new[] { 0 }, new[] { head }), new[] { c - 1, _stride }), new[] { 1 }, keepDims: false);
        return Engine.TensorConcatenate(new[] { heads, Engine.Reshape(tail, new[] { 1 }) }, 0);
    }

    /// <summary>A block vector written to every entry of its block, <c>[paramDim]</c>.</summary>
    private Tensor<T> Expand(Tensor<T> adapter)
    {
        int c = _compressedDim, head = (c - 1) * _stride;
        var tail = Engine.TensorTile(Engine.TensorSlice(adapter, new[] { c - 1 }, new[] { 1 }), new[] { _paramDim - head });
        if (c == 1) return tail;

        var heads = Engine.TensorRepeatInterleave(Engine.TensorSlice(adapter, new[] { 0 }, new[] { c - 1 }), _stride, 0);
        return Engine.TensorConcatenate(new[] { heads, tail }, 0);
    }

    private Tensor<T> CompressDifference(Vector<T> minuend, Vector<T> subtrahend)
        => Engine.TensorDivide(
            BlockSums(Engine.TensorSubtract(Tensor<T>.FromVector(minuend), Tensor<T>.FromVector(subtrahend))),
            BlockSizes());

    private Vector<T> PretrainedPlus(Tensor<T> adapter)
        => Engine.TensorAdd(Tensor<T>.FromVector(_pretrainedParams), Expand(adapter)).ToVector();

    /// <summary>L_meta for one task under the given VAE parameters and noise; the meta-model is restored.</summary>
    internal T MetaObjectiveForTesting(IMetaLearningTask<T, TInput, TOutput> task, Vector<T> encoder, Vector<T> decoder,
        Vector<T> firstNoise, Vector<T> secondNoise, Vector<T>? frozenAdaptedAdapter)
    {
        var thetaStart = ParamModel.GetParameters();
        var result = TaskObjective(task, thetaStart, Tensor<T>.FromVector(encoder.Clone()), Tensor<T>.FromVector(decoder.Clone()),
            Tensor<T>.FromVector(firstNoise.Clone()), Tensor<T>.FromVector(secondNoise.Clone()), secondOrder: false,
            withGradient: false,
            frozenAdaptedAdapter is null ? null : Tensor<T>.FromVector(frozenAdaptedAdapter.Clone()));
        ParamModel.SetParameters(thetaStart);
        return result.Loss;
    }

    /// <summary>The analytic gradient of <see cref="MetaObjectiveForTesting"/> at the learner's current VAE.</summary>
    internal (Vector<T> Encoder, Vector<T> Decoder, Vector<T> AdaptedAdapter) MetaGradientForTesting(
        IMetaLearningTask<T, TInput, TOutput> task, Vector<T> firstNoise, Vector<T> secondNoise, bool secondOrder)
    {
        var thetaStart = ParamModel.GetParameters();
        var result = TaskObjective(task, thetaStart, Tensor<T>.FromVector(_encoderParams.Clone()),
            Tensor<T>.FromVector(_decoderParams.Clone()), Tensor<T>.FromVector(firstNoise.Clone()),
            Tensor<T>.FromVector(secondNoise.Clone()), secondOrder, withGradient: true, frozenAdaptedAdapter: null);
        ParamModel.SetParameters(thetaStart);
        var encoderGradient = result.EncoderGradient ?? new Tensor<T>(new[] { _encoderParams.Length });
        var decoderGradient = result.DecoderGradient ?? new Tensor<T>(new[] { _decoderParams.Length });
        return (encoderGradient.ToVector(), decoderGradient.ToVector(), result.AdaptedAdapter.ToVector());
    }

    internal Vector<T> EncoderParametersForTesting => _encoderParams.Clone();

    internal Vector<T> DecoderParametersForTesting => _decoderParams.Clone();

    internal Vector<T> PretrainedParametersForTesting() => _pretrainedParams.Clone();
}

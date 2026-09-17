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
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        if (options.LatentDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "LatentDim must be positive.");
        if (options.NumFusionComponents <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "NumFusionComponents must be positive.");
        if (options.AdaptationSteps < 0)
            throw new ArgumentOutOfRangeException(nameof(options), "AdaptationSteps cannot be negative.");

        _algoOptions = options;
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

        var encoder = ToDoubles(_encoderParams);
        var decoder = ToDoubles(_decoderParams);
        var encoderGradient = new double[encoder.Length];
        var decoderGradient = new double[decoder.Length];
        var adapters = new List<double[]>(taskCount);
        var taskVectors = new List<double[]>(taskCount);
        bool secondOrder = !_algoOptions.UseFirstOrder;
        double totalLoss = 0;

        foreach (var task in taskBatch.Tasks)
        {
            var firstNoise = SampleLatentNoise();
            var secondNoise = SampleLatentNoise();
            totalLoss += TaskObjective(task, thetaStart, encoder, decoder, firstNoise, secondNoise, secondOrder,
                encoderGradient, decoderGradient, frozenAdaptedAdapter: null,
                out var adapter, out var taskVector, out _);
            adapters.Add(adapter);
            taskVectors.Add(taskVector);
        }

        // Eqs. 13-14: one SGD step on the batch-mean meta-loss gradient, clipped as one vector.
        var combined = new Vector<T>(encoder.Length + decoder.Length);
        for (int i = 0; i < encoder.Length; i++)
            combined[i] = NumOps.FromDouble(encoderGradient[i] / taskCount);
        for (int i = 0; i < decoder.Length; i++)
            combined[encoder.Length + i] = NumOps.FromDouble(decoderGradient[i] / taskCount);
        combined = ClipGradients(combined);

        double gamma = _algoOptions.OuterLearningRate;
        for (int i = 0; i < encoder.Length; i++)
        {
            encoder[i] -= gamma * NumOps.ToDouble(combined[i]);
            _encoderParams[i] = NumOps.FromDouble(encoder[i]);
        }

        for (int i = 0; i < decoder.Length; i++)
        {
            decoder[i] -= gamma * NumOps.ToDouble(combined[encoder.Length + i]);
            _decoderParams[i] = NumOps.FromDouble(decoder[i]);
        }

        // Fuse in latent space with the updated encoder, decode with the updated decoder.
        for (int t = 0; t < taskCount; t++)
        {
            Encode(encoder, adapters[t], taskVectors[t], out var mean, out _, out _);
            StoreComponent(mean, taskVectors[t]);
        }

        FuseComponents(out var fusedLatent, out var fusedTaskVector);
        ParamModel.SetParameters(PretrainedPlus(Decode(decoder, fusedLatent, fusedTaskVector)));

        return NumOps.FromDouble(totalLoss / taskCount);
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
        var encoder = ToDoubles(_encoderParams);
        var decoder = ToDoubles(_decoderParams);

        var thetaTask = ObtainTaskParameters(thetaStart, task);
        var adapter = CompressDifference(thetaTask, _pretrainedParams);
        var taskVector = CompressDifference(thetaTask, thetaStart);
        Encode(encoder, adapter, taskVector, out var mean, out _, out _);
        var refined = RefineAdapter(Decode(decoder, mean, taskVector), task, trajectory: null);
        var finalParams = PretrainedPlus(refined);

        ParamModel.SetParameters(thetaStart);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, finalParams);
    }

    /// <summary>
    /// L_meta for one task (Algorithm 1, lines 4-14), accumulating its gradient into the supplied buffers when they
    /// are given.
    /// </summary>
    private double TaskObjective(
        IMetaLearningTask<T, TInput, TOutput> task,
        Vector<T> thetaStart,
        double[] encoder,
        double[] decoder,
        double[] firstNoise,
        double[] secondNoise,
        bool secondOrder,
        double[]? encoderGradient,
        double[]? decoderGradient,
        double[]? frozenAdaptedAdapter,
        out double[] adapter,
        out double[] taskVector,
        out double[] adaptedAdapter)
    {
        int c = _compressedDim, d = _latentDim;
        double klWeight = _algoOptions.KLWeight;

        // Lines 4-5. Neither depends on the VAE.
        var thetaTask = ObtainTaskParameters(thetaStart, task);
        adapter = CompressDifference(thetaTask, _pretrainedParams);
        taskVector = CompressDifference(thetaTask, thetaStart);

        // Lines 6-8.
        Encode(encoder, adapter, taskVector, out var firstMean, out var firstLogVar, out var firstClamped);
        var firstLatent = Reparameterize(firstMean, firstLogVar, firstNoise);
        var initialAdapter = Decode(decoder, firstLatent, taskVector);

        // Line 9.
        List<double[]>? trajectory = secondOrder && frozenAdaptedAdapter == null ? new List<double[]>() : null;
        adaptedAdapter = frozenAdaptedAdapter ?? RefineAdapter(initialAdapter, task, trajectory);

        // Lines 10-14.
        Encode(encoder, adaptedAdapter, taskVector, out var mean, out var logVar, out var clamped);
        var latent = Reparameterize(mean, logVar, secondNoise);
        var reconstruction = Decode(decoder, latent, taskVector);

        double recon = 0;
        for (int j = 0; j < c; j++)
        {
            double diff = reconstruction[j] - adapter[j];
            recon += diff * diff;
        }

        recon /= c;
        double kl = 0;
        for (int o = 0; o < d; o++)
            kl += -0.5 * (1.0 + logVar[o] - mean[o] * mean[o] - Math.Exp(logVar[o]));

        if (encoderGradient is not null && decoderGradient is not null)
        {
            var reconstructionGradient = new double[c];
            for (int j = 0; j < c; j++)
                reconstructionGradient[j] = 2.0 * (reconstruction[j] - adapter[j]) / c;

            AccumulateDecoderGradient(decoderGradient, reconstructionGradient, latent, taskVector);
            var latentGradient = DecoderLatentBackward(decoder, reconstructionGradient);

            var meanGradient = new double[d];
            var logVarGradient = new double[d];
            for (int o = 0; o < d; o++)
            {
                meanGradient[o] = latentGradient[o] + klWeight * mean[o];
                logVarGradient[o] = clamped[o]
                    ? 0.0
                    : latentGradient[o] * secondNoise[o] * 0.5 * Math.Exp(0.5 * logVar[o])
                      - klWeight * 0.5 * (1.0 - Math.Exp(logVar[o]));
            }

            AccumulateEncoderGradient(encoderGradient, meanGradient, logVarGradient, adaptedAdapter, taskVector);

            if (trajectory is not null)
            {
                // dL/dl_adapt through the step-10 encoding, then back through every refinement step.
                var adaptedGradient = EncoderAdapterBackward(encoder, meanGradient, logVarGradient);
                var initialGradient = BackpropagateRefinement(adaptedGradient, trajectory, task);

                AccumulateDecoderGradient(decoderGradient, initialGradient, firstLatent, taskVector);
                var firstLatentGradient = DecoderLatentBackward(decoder, initialGradient);
                var firstLogVarGradient = new double[d];
                for (int o = 0; o < d; o++)
                {
                    firstLogVarGradient[o] = firstClamped[o]
                        ? 0.0
                        : firstLatentGradient[o] * firstNoise[o] * 0.5 * Math.Exp(0.5 * firstLogVar[o]);
                }

                AccumulateEncoderGradient(encoderGradient, firstLatentGradient, firstLogVarGradient, adapter, taskVector);
            }
        }

        return recon + klWeight * kl;
    }

    /// <summary>ObtainLoRAParams: inner-loop fine-tuning of the current meta-model on the support set.</summary>
    private Vector<T> ObtainTaskParameters(Vector<T> thetaStart, IMetaLearningTask<T, TInput, TOutput> task)
    {
        var adapted = thetaStart.Clone();
        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            ParamModel.SetParameters(adapted);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            adapted = ApplyGradients(adapted, grad, _algoOptions.InnerLearningRate);
        }

        ParamModel.SetParameters(thetaStart);
        return adapted;
    }

    /// <summary>Eq. 8: K plain gradient steps on the adapter, recording each point the step was taken from.</summary>
    private double[] RefineAdapter(double[] initialAdapter, IMetaLearningTask<T, TInput, TOutput> task, List<double[]>? trajectory)
    {
        var current = (double[])initialAdapter.Clone();
        double beta = _algoOptions.InnerLearningRate;
        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            trajectory?.Add((double[])current.Clone());
            var gradient = AdapterGradient(current, task);
            for (int j = 0; j < current.Length; j++) current[j] -= beta * gradient[j];
        }

        return current;
    }

    /// <summary>
    /// Pulls dL/dl_K back to dL/dl_0 through l_{k+1} = l_k - beta g(l_k): a_k = a_{k+1} - beta H(l_k) a_{k+1}.
    /// </summary>
    private double[] BackpropagateRefinement(double[] adaptedGradient, List<double[]> trajectory, IMetaLearningTask<T, TInput, TOutput> task)
    {
        var current = (double[])adaptedGradient.Clone();
        double beta = _algoOptions.InnerLearningRate;
        for (int step = trajectory.Count - 1; step >= 0; step--)
        {
            var hessianVector = HessianVectorProduct(trajectory[step], current, task);
            for (int j = 0; j < current.Length; j++) current[j] -= beta * hessianVector[j];
        }

        return current;
    }

    /// <summary>H(point) * direction as a central difference of task gradients along the direction.</summary>
    private double[] HessianVectorProduct(double[] point, double[] direction, IMetaLearningTask<T, TInput, TOutput> task)
    {
        double directionNorm = Norm(direction);
        if (directionNorm == 0.0) return new double[direction.Length];

        double h = 1e-4 * (1.0 + Norm(point)) / directionNorm;
        var plus = new double[point.Length];
        var minus = new double[point.Length];
        for (int j = 0; j < point.Length; j++)
        {
            plus[j] = point[j] + h * direction[j];
            minus[j] = point[j] - h * direction[j];
        }

        var gradientPlus = AdapterGradient(plus, task);
        var gradientMinus = AdapterGradient(minus, task);
        var result = new double[point.Length];
        for (int j = 0; j < point.Length; j++)
            result[j] = (gradientPlus[j] - gradientMinus[j]) / (2.0 * h);
        return result;
    }

    /// <summary>dL_task/dl at theta_pt + expand(l), summed per block.</summary>
    private double[] AdapterGradient(double[] adapter, IMetaLearningTask<T, TInput, TOutput> task)
    {
        ParamModel.SetParameters(PretrainedPlus(adapter));
        var full = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
        var result = new double[_compressedDim];
        for (int block = 0; block < _compressedDim; block++)
        {
            BlockRange(block, out int start, out int end);
            double sum = 0;
            for (int p = start; p < end; p++) sum += NumOps.ToDouble(full[p]);
            result[block] = sum;
        }

        return result;
    }

    private void Encode(double[] encoder, double[] adapter, double[] taskVector,
        out double[] mean, out double[] logVar, out bool[] clamped)
    {
        int c = _compressedDim, d = _latentDim, width = 2 * c;
        int biasOffset = (2 * d) * width;
        mean = new double[d];
        logVar = new double[d];
        clamped = new bool[d];
        for (int o = 0; o < d; o++)
        {
            double sumMean = encoder[biasOffset + o];
            double sumLogVar = encoder[biasOffset + d + o];
            int meanRow = o * width, logVarRow = (d + o) * width;
            for (int i = 0; i < c; i++)
            {
                sumMean += encoder[meanRow + i] * adapter[i] + encoder[meanRow + c + i] * taskVector[i];
                sumLogVar += encoder[logVarRow + i] * adapter[i] + encoder[logVarRow + c + i] * taskVector[i];
            }

            mean[o] = sumMean;
            clamped[o] = sumLogVar > LogVarianceBound || sumLogVar < -LogVarianceBound;
            logVar[o] = Math.Max(-LogVarianceBound, Math.Min(LogVarianceBound, sumLogVar));
        }
    }

    private double[] Decode(double[] decoder, double[] latent, double[] taskVector)
    {
        int c = _compressedDim, d = _latentDim, width = d + c;
        int biasOffset = c * width;
        var output = new double[c];
        for (int o = 0; o < c; o++)
        {
            double sum = decoder[biasOffset + o];
            int row = o * width;
            for (int i = 0; i < d; i++) sum += decoder[row + i] * latent[i];
            for (int i = 0; i < c; i++) sum += decoder[row + d + i] * taskVector[i];
            output[o] = sum;
        }

        return output;
    }

    private void AccumulateDecoderGradient(double[] gradient, double[] outputGradient, double[] latent, double[] taskVector)
    {
        int c = _compressedDim, d = _latentDim, width = d + c;
        int biasOffset = c * width;
        for (int o = 0; o < c; o++)
        {
            int row = o * width;
            for (int i = 0; i < d; i++) gradient[row + i] += outputGradient[o] * latent[i];
            for (int i = 0; i < c; i++) gradient[row + d + i] += outputGradient[o] * taskVector[i];
            gradient[biasOffset + o] += outputGradient[o];
        }
    }

    private double[] DecoderLatentBackward(double[] decoder, double[] outputGradient)
    {
        int c = _compressedDim, d = _latentDim, width = d + c;
        var latentGradient = new double[d];
        for (int o = 0; o < c; o++)
        {
            int row = o * width;
            for (int i = 0; i < d; i++) latentGradient[i] += decoder[row + i] * outputGradient[o];
        }

        return latentGradient;
    }

    private void AccumulateEncoderGradient(double[] gradient, double[] meanGradient, double[] logVarGradient,
        double[] adapter, double[] taskVector)
    {
        int c = _compressedDim, d = _latentDim, width = 2 * c;
        int biasOffset = (2 * d) * width;
        for (int o = 0; o < d; o++)
        {
            int meanRow = o * width, logVarRow = (d + o) * width;
            for (int i = 0; i < c; i++)
            {
                gradient[meanRow + i] += meanGradient[o] * adapter[i];
                gradient[meanRow + c + i] += meanGradient[o] * taskVector[i];
                gradient[logVarRow + i] += logVarGradient[o] * adapter[i];
                gradient[logVarRow + c + i] += logVarGradient[o] * taskVector[i];
            }

            gradient[biasOffset + o] += meanGradient[o];
            gradient[biasOffset + d + o] += logVarGradient[o];
        }
    }

    /// <summary>The gradient an encoding passes back to its adapter input (the first c inputs).</summary>
    private double[] EncoderAdapterBackward(double[] encoder, double[] meanGradient, double[] logVarGradient)
    {
        int c = _compressedDim, d = _latentDim, width = 2 * c;
        var adapterGradient = new double[c];
        for (int o = 0; o < d; o++)
        {
            int meanRow = o * width, logVarRow = (d + o) * width;
            for (int i = 0; i < c; i++)
                adapterGradient[i] += meanGradient[o] * encoder[meanRow + i] + logVarGradient[o] * encoder[logVarRow + i];
        }

        return adapterGradient;
    }

    private static double[] Reparameterize(double[] mean, double[] logVar, double[] noise)
    {
        var latent = new double[mean.Length];
        for (int i = 0; i < mean.Length; i++)
            latent[i] = mean[i] + Math.Exp(0.5 * logVar[i]) * noise[i];
        return latent;
    }

    private double[] SampleLatentNoise()
    {
        var noise = new double[_latentDim];
        for (int i = 0; i < noise.Length; i++) noise[i] = SampleNormal();
        return noise;
    }

    private void StoreComponent(double[] latent, double[] taskVector)
    {
        int slots = _algoOptions.NumFusionComponents;
        int slot = (int)NumOps.ToDouble(_fusionCursor[0]);
        int filled = (int)NumOps.ToDouble(_fusionCursor[1]);
        for (int i = 0; i < _latentDim; i++)
            _fusionLatents[slot * _latentDim + i] = NumOps.FromDouble(latent[i]);
        for (int i = 0; i < _compressedDim; i++)
            _fusionTaskVectors[slot * _compressedDim + i] = NumOps.FromDouble(taskVector[i]);
        _fusionCursor[0] = NumOps.FromDouble((slot + 1) % slots);
        _fusionCursor[1] = NumOps.FromDouble(Math.Min(filled + 1, slots));
    }

    /// <summary>Latent-space task arithmetic: the newest component weighs 1, each older one FusionDecay times less.</summary>
    private void FuseComponents(out double[] latent, out double[] taskVector)
    {
        int slots = _algoOptions.NumFusionComponents;
        int next = (int)NumOps.ToDouble(_fusionCursor[0]);
        int filled = (int)NumOps.ToDouble(_fusionCursor[1]);
        latent = new double[_latentDim];
        taskVector = new double[_compressedDim];
        double weight = 1.0, totalWeight = 0.0;
        for (int k = 0; k < filled; k++)
        {
            int slot = ((next - 1 - k) % slots + slots) % slots;
            for (int i = 0; i < _latentDim; i++)
                latent[i] += weight * NumOps.ToDouble(_fusionLatents[slot * _latentDim + i]);
            for (int i = 0; i < _compressedDim; i++)
                taskVector[i] += weight * NumOps.ToDouble(_fusionTaskVectors[slot * _compressedDim + i]);
            totalWeight += weight;
            weight *= _algoOptions.FusionDecay;
        }

        if (totalWeight <= 0.0) return;
        for (int i = 0; i < _latentDim; i++) latent[i] /= totalWeight;
        for (int i = 0; i < _compressedDim; i++) taskVector[i] /= totalWeight;
    }

    private void BlockRange(int block, out int start, out int end)
    {
        start = block * _stride;
        end = block == _compressedDim - 1 ? _paramDim : (block + 1) * _stride;
    }

    private double[] CompressDifference(Vector<T> minuend, Vector<T> subtrahend)
    {
        var result = new double[_compressedDim];
        for (int block = 0; block < _compressedDim; block++)
        {
            BlockRange(block, out int start, out int end);
            double sum = 0;
            for (int p = start; p < end; p++)
                sum += NumOps.ToDouble(minuend[p]) - NumOps.ToDouble(subtrahend[p]);
            result[block] = sum / (end - start);
        }

        return result;
    }

    private Vector<T> PretrainedPlus(double[] adapter)
    {
        var result = new Vector<T>(_paramDim);
        for (int block = 0; block < _compressedDim; block++)
        {
            BlockRange(block, out int start, out int end);
            for (int p = start; p < end; p++)
                result[p] = NumOps.FromDouble(NumOps.ToDouble(_pretrainedParams[p]) + adapter[block]);
        }

        return result;
    }

    private double[] ToDoubles(Vector<T> vector)
    {
        var result = new double[vector.Length];
        for (int i = 0; i < vector.Length; i++) result[i] = NumOps.ToDouble(vector[i]);
        return result;
    }

    private static double Norm(double[] vector)
    {
        double sum = 0;
        for (int i = 0; i < vector.Length; i++) sum += vector[i] * vector[i];
        return Math.Sqrt(sum);
    }

    /// <summary>L_meta for one task under the given VAE parameters and noise; the meta-model is restored.</summary>
    internal double MetaObjectiveForTesting(IMetaLearningTask<T, TInput, TOutput> task, double[] encoder, double[] decoder,
        double[] firstNoise, double[] secondNoise, double[]? frozenAdaptedAdapter)
    {
        var thetaStart = ParamModel.GetParameters();
        double loss = TaskObjective(task, thetaStart, encoder, decoder, firstNoise, secondNoise, secondOrder: false,
            null, null, frozenAdaptedAdapter, out _, out _, out _);
        ParamModel.SetParameters(thetaStart);
        return loss;
    }

    /// <summary>The analytic gradient of <see cref="MetaObjectiveForTesting"/> at the learner's current VAE.</summary>
    internal (double[] Encoder, double[] Decoder, double[] AdaptedAdapter) MetaGradientForTesting(
        IMetaLearningTask<T, TInput, TOutput> task, double[] firstNoise, double[] secondNoise, bool secondOrder)
    {
        var thetaStart = ParamModel.GetParameters();
        var encoder = ToDoubles(_encoderParams);
        var decoder = ToDoubles(_decoderParams);
        var encoderGradient = new double[encoder.Length];
        var decoderGradient = new double[decoder.Length];
        TaskObjective(task, thetaStart, encoder, decoder, firstNoise, secondNoise, secondOrder,
            encoderGradient, decoderGradient, null, out _, out _, out var adapted);
        ParamModel.SetParameters(thetaStart);
        return (encoderGradient, decoderGradient, adapted);
    }

    internal double[] EncoderParametersForTesting => ToDoubles(_encoderParams);

    internal double[] DecoderParametersForTesting => ToDoubles(_decoderParams);

    internal double[] PretrainedParametersForTesting() => ToDoubles(_pretrainedParams);
}

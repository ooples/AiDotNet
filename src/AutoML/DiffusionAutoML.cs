using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Diffusion;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Diffusion.ImageEditing;
using AiDotNet.Diffusion.Video;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Diffusion.ThreeD;
using AiDotNet.Diffusion.Control;
using AiDotNet.Diffusion.SuperResolution;
using AiDotNet.Diffusion.FastGeneration;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.AutoML
{
    /// <summary>
    /// Represents the type of noise predictor architecture.
    /// </summary>
    public enum NoisePredictorType
    {
        /// <summary>U-Net architecture.</summary>
        UNet,
        /// <summary>Diffusion Transformer (DiT) architecture.</summary>
        DiT,
        /// <summary>U-Vision Transformer (UViT) architecture.</summary>
        UViT
    }

    /// <summary>
    /// Represents the type of scheduler for diffusion sampling.
    /// </summary>
    public enum DiffusionSchedulerType
    {
        /// <summary>DDPM (Denoising Diffusion Probabilistic Models) scheduler.</summary>
        DDPM,
        /// <summary>DDIM (Denoising Diffusion Implicit Models) scheduler.</summary>
        DDIM,
        /// <summary>Euler discrete scheduler.</summary>
        Euler,
        /// <summary>Euler ancestral discrete scheduler.</summary>
        EulerAncestral,
        /// <summary>DPM-Solver++ multi-step scheduler.</summary>
        DPMSolver,
        /// <summary>LCM (Latent Consistency Models) scheduler.</summary>
        LCM
    }

    /// <summary>
    /// Configuration for a diffusion model trial in AutoML.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class DiffusionTrialConfig<T>
    {
        /// <summary>Gets or sets the noise predictor type.</summary>
        public NoisePredictorType NoisePredictorType { get; set; } = NoisePredictorType.UNet;

        /// <summary>Gets or sets the scheduler type.</summary>
        public DiffusionSchedulerType SchedulerType { get; set; } = DiffusionSchedulerType.DDIM;

        /// <summary>Gets or sets the number of inference steps.</summary>
        public int InferenceSteps { get; set; } = 50;

        /// <summary>Gets or sets the guidance scale for classifier-free guidance.</summary>
        public double GuidanceScale { get; set; } = 7.5;

        /// <summary>Gets or sets the learning rate for training.</summary>
        public double LearningRate { get; set; } = 1e-4;

        /// <summary>Gets or sets the base channels for the noise predictor.</summary>
        public int BaseChannels { get; set; } = 128;

        /// <summary>Gets or sets the number of residual blocks per level.</summary>
        public int NumResBlocks { get; set; } = 2;

        /// <summary>Gets or sets the latent dimension (channels).</summary>
        public int LatentDim { get; set; } = 4;

        /// <summary>Gets or sets the latent spatial height (default: 64 for 512x512 images with 8x downscaling).</summary>
        public int LatentHeight { get; set; } = 64;

        /// <summary>Gets or sets the latent spatial width (default: 64 for 512x512 images with 8x downscaling).</summary>
        public int LatentWidth { get; set; } = 64;

        /// <summary>Gets or sets the optional random seed.</summary>
        public int? Seed { get; set; }

        /// <summary>
        /// Converts the configuration to a dictionary of parameters.
        /// </summary>
        public Dictionary<string, object> ToDictionary()
        {
            return new Dictionary<string, object>
            {
                ["NoisePredictorType"] = NoisePredictorType.ToString(),
                ["SchedulerType"] = SchedulerType.ToString(),
                ["InferenceSteps"] = InferenceSteps,
                ["GuidanceScale"] = GuidanceScale,
                ["LearningRate"] = LearningRate,
                ["BaseChannels"] = BaseChannels,
                ["NumResBlocks"] = NumResBlocks,
                ["LatentDim"] = LatentDim,
                ["LatentHeight"] = LatentHeight,
                ["LatentWidth"] = LatentWidth,
                ["Seed"] = Seed ?? 0
            };
        }

        /// <summary>
        /// Creates a configuration from a dictionary of parameters.
        /// </summary>
        public static DiffusionTrialConfig<T> FromDictionary(Dictionary<string, object> parameters)
        {
            var config = new DiffusionTrialConfig<T>();

            if (parameters.TryGetValue("NoisePredictorType", out var npt) && npt is string nptStr)
            {
                if (Enum.TryParse<NoisePredictorType>(nptStr, out var parsed))
                    config.NoisePredictorType = parsed;
            }

            if (parameters.TryGetValue("SchedulerType", out var st) && st is string stStr)
            {
                if (Enum.TryParse<DiffusionSchedulerType>(stStr, out var parsed))
                    config.SchedulerType = parsed;
            }

            if (parameters.TryGetValue("InferenceSteps", out var steps))
                config.InferenceSteps = Convert.ToInt32(steps);

            if (parameters.TryGetValue("GuidanceScale", out var gs))
                config.GuidanceScale = Convert.ToDouble(gs);

            if (parameters.TryGetValue("LearningRate", out var lr))
                config.LearningRate = Convert.ToDouble(lr);

            if (parameters.TryGetValue("BaseChannels", out var bc))
                config.BaseChannels = Convert.ToInt32(bc);

            if (parameters.TryGetValue("NumResBlocks", out var nrb))
                config.NumResBlocks = Convert.ToInt32(nrb);

            if (parameters.TryGetValue("LatentDim", out var ld))
                config.LatentDim = Convert.ToInt32(ld);

            if (parameters.TryGetValue("LatentHeight", out var lh))
                config.LatentHeight = Convert.ToInt32(lh);

            if (parameters.TryGetValue("LatentWidth", out var lw))
                config.LatentWidth = Convert.ToInt32(lw);

            if (parameters.TryGetValue("Seed", out var seed))
            {
                int seedVal = Convert.ToInt32(seed);
                config.Seed = seedVal != 0 ? seedVal : null;
            }

            return config;
        }
    }

    /// <summary>
    /// AutoML for diffusion models with automatic hyperparameter optimization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// DiffusionAutoML automatically searches for optimal diffusion model configurations,
    /// including noise predictor architecture, scheduler type, and training hyperparameters.
    /// </para>
    /// <para><b>For Beginners:</b> This class automatically finds the best settings for your diffusion model.
    ///
    /// When using diffusion models, there are many choices to make:
    /// - What type of neural network architecture (U-Net, DiT, etc.)
    /// - What sampling scheduler (DDIM, Euler, DPM-Solver, etc.)
    /// - How many inference steps to use
    /// - What guidance scale for conditional generation
    /// - Training hyperparameters like learning rate
    ///
    /// DiffusionAutoML tries different combinations automatically and finds
    /// what works best for your specific data and use case.
    /// </para>
    /// </remarks>
    /// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
    public class DiffusionAutoML<T> : AutoMLModelBase<T, Tensor<T>, Tensor<T>>
    {
        private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
        private readonly Random _random;
        private readonly int? _seed;

        /// <summary>
        /// Gets the best diffusion configuration found during search.
        /// </summary>
        public DiffusionTrialConfig<T>? BestConfig { get; private set; }

        /// <summary>
        /// Gets the list of noise predictor types to try during search.
        /// </summary>
        public List<NoisePredictorType> NoisePredictorTypesToTry { get; } = new List<NoisePredictorType>
        {
            NoisePredictorType.UNet,
            NoisePredictorType.DiT
        };

        /// <summary>
        /// Gets the list of scheduler types to try during search.
        /// </summary>
        public List<DiffusionSchedulerType> SchedulerTypesToTry { get; } = new List<DiffusionSchedulerType>
        {
            DiffusionSchedulerType.DDIM,
            DiffusionSchedulerType.Euler,
            DiffusionSchedulerType.DPMSolver
        };

        /// <summary>
        /// Initializes a new instance of the DiffusionAutoML class.
        /// </summary>
        /// <param name="seed">Optional random seed for reproducibility.</param>
        public DiffusionAutoML(int? seed = null)
        {
            _seed = seed;
            _random = seed.HasValue
                ? RandomHelper.CreateSeededRandom(seed.Value)
                : RandomHelper.CreateSecureRandom();

            // Set default search space
            SetSearchSpace(GetDefaultDiffusionSearchSpace());

            // Set default optimization metric (FID-like score - lower is better)
            SetOptimizationMetric(MetricType.MeanSquaredError, maximize: false);
        }

        /// <summary>
        /// Searches for the best diffusion model configuration.
        /// </summary>
        public override async Task<IFullModel<T, Tensor<T>, Tensor<T>>> SearchAsync(
            Tensor<T> inputs,
            Tensor<T> targets,
            Tensor<T> validationInputs,
            Tensor<T> validationTargets,
            TimeSpan timeLimit,
            CancellationToken cancellationToken = default)
        {
            Status = AutoMLStatus.Running;
            var stopwatch = Stopwatch.StartNew();

            try
            {
                int trialCount = 0;
                TimeSpan totalTimeLimit = timeLimit;

                while (stopwatch.Elapsed < totalTimeLimit && trialCount < TrialLimit)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    // Check early stopping
                    if (ShouldStop())
                    {
                        break;
                    }

                    // Get next trial parameters
                    var parameters = await SuggestNextTrialAsync();
                    var trialStopwatch = Stopwatch.StartNew();

                    try
                    {
                        // Create and evaluate model
                        var model = await CreateModelAsync(ModelType.NeuralNetwork, parameters);

                        // Train model if training data provided
                        if (inputs.Shape[0] > 0)
                        {
                            await TrainModelAsync(model, inputs, targets, cancellationToken);
                        }

                        // Evaluate model
                        var score = await EvaluateModelAsync(model, validationInputs, validationTargets);

                        trialStopwatch.Stop();
                        await ReportTrialResultAsync(parameters, score, trialStopwatch.Elapsed);

                        // Update best model
                        bool isBetter = _maximize ? score > BestScore : score < BestScore;
                        if (isBetter)
                        {
                            BestScore = score;
                            BestModel = model;
                            BestConfig = DiffusionTrialConfig<T>.FromDictionary(parameters);
                        }
                    }
                    catch (Exception ex)
                    {
                        trialStopwatch.Stop();
                        await ReportTrialFailureAsync(parameters, ex, trialStopwatch.Elapsed);
                    }

                    trialCount++;
                }

                Status = AutoMLStatus.Completed;

                if (BestModel is null)
                {
                    // Create a default model if no successful trials
                    var defaultParams = GetDefaultParameters();
                    BestModel = await CreateModelAsync(ModelType.NeuralNetwork, defaultParams);
                    BestConfig = DiffusionTrialConfig<T>.FromDictionary(defaultParams);
                }

                return BestModel;
            }
            catch (OperationCanceledException)
            {
                Status = AutoMLStatus.Cancelled;
                throw;
            }
            catch (Exception)
            {
                Status = AutoMLStatus.Failed;
                throw;
            }
        }

        /// <summary>
        /// Suggests the next trial parameters based on search history.
        /// </summary>
        public override Task<Dictionary<string, object>> SuggestNextTrialAsync()
        {
            var parameters = new Dictionary<string, object>();

            // Sample noise predictor type
            var nptIndex = _random.Next(NoisePredictorTypesToTry.Count);
            parameters["NoisePredictorType"] = NoisePredictorTypesToTry[nptIndex].ToString();

            // Sample scheduler type
            var stIndex = _random.Next(SchedulerTypesToTry.Count);
            parameters["SchedulerType"] = SchedulerTypesToTry[stIndex].ToString();

            // Sample continuous hyperparameters
            foreach (var kvp in _searchSpace)
            {
                switch (kvp.Key)
                {
                    case "InferenceSteps":
                        parameters[kvp.Key] = SampleIntParameter(kvp.Value);
                        break;
                    case "GuidanceScale":
                    case "LearningRate":
                        parameters[kvp.Key] = SampleFloatParameter(kvp.Value);
                        break;
                    case "BaseChannels":
                    case "NumResBlocks":
                    case "LatentDim":
                        parameters[kvp.Key] = SampleIntParameter(kvp.Value);
                        break;
                }
            }

            // Add seed if configured
            if (_seed.HasValue)
            {
                parameters["Seed"] = _seed.Value + _trialHistory.Count;
            }

            return Task.FromResult(parameters);
        }

        /// <summary>
        /// Creates a diffusion model based on the specified parameters.
        /// </summary>
        protected override async Task<IFullModel<T, Tensor<T>, Tensor<T>>> CreateModelAsync(
            ModelType modelType,
            Dictionary<string, object> parameters)
        {
            return await Task.Run(() =>
            {
                var config = DiffusionTrialConfig<T>.FromDictionary(parameters);

                // Create noise predictor based on config
                var noisePredictor = CreateNoisePredictor(config);

                // Create VAE
                var vae = CreateVAE(config);

                // Create scheduler
                var scheduler = CreateScheduler(config);

                // Create conditioner (simple text embedding)
                var conditioner = new SimpleConditioner<T>(config.BaseChannels * 4);

                // Create the latent diffusion model wrapper
                var model = new DiffusionAutoMLModel<T>(
                    noisePredictor,
                    vae,
                    scheduler,
                    conditioner,
                    config,
                    _seed);

                return model;
            });
        }

        /// <summary>
        /// Gets the default search space for diffusion models.
        /// </summary>
        protected override Dictionary<string, ParameterRange> GetDefaultSearchSpace(ModelType modelType)
        {
            return GetDefaultDiffusionSearchSpace();
        }

        /// <summary>
        /// Creates an instance for deep copy.
        /// </summary>
        protected override AutoMLModelBase<T, Tensor<T>, Tensor<T>> CreateInstanceForCopy()
        {
            return new DiffusionAutoML<T>(_seed);
        }

        private Dictionary<string, ParameterRange> GetDefaultDiffusionSearchSpace()
        {
            return new Dictionary<string, ParameterRange>
            {
                ["InferenceSteps"] = new ParameterRange
                {
                    Type = ParameterType.Integer,
                    MinValue = 10,
                    MaxValue = 100,
                    Step = 10
                },
                ["GuidanceScale"] = new ParameterRange
                {
                    Type = ParameterType.Float,
                    MinValue = 1.0,
                    MaxValue = 15.0,
                    Step = 0.5
                },
                ["LearningRate"] = new ParameterRange
                {
                    Type = ParameterType.Float,
                    MinValue = 1e-6,
                    MaxValue = 1e-3,
                    UseLogScale = true
                },
                ["BaseChannels"] = new ParameterRange
                {
                    Type = ParameterType.Integer,
                    MinValue = 64,
                    MaxValue = 512,
                    Step = 64
                },
                ["NumResBlocks"] = new ParameterRange
                {
                    Type = ParameterType.Integer,
                    MinValue = 1,
                    MaxValue = 4,
                    Step = 1
                },
                ["LatentDim"] = new ParameterRange
                {
                    Type = ParameterType.Integer,
                    MinValue = 4,
                    MaxValue = 16,
                    Step = 4
                }
            };
        }

        private Dictionary<string, object> GetDefaultParameters()
        {
            return new Dictionary<string, object>
            {
                ["NoisePredictorType"] = NoisePredictorType.UNet.ToString(),
                ["SchedulerType"] = DiffusionSchedulerType.DDIM.ToString(),
                ["InferenceSteps"] = 50,
                ["GuidanceScale"] = 7.5,
                ["LearningRate"] = 1e-4,
                ["BaseChannels"] = 128,
                ["NumResBlocks"] = 2,
                ["LatentDim"] = 4,
                ["Seed"] = _seed ?? 42
            };
        }

        private int SampleIntParameter(ParameterRange range)
        {
            int min = Convert.ToInt32(range.MinValue);
            int max = Convert.ToInt32(range.MaxValue);
            int step = range.Step.HasValue && range.Step.Value > 0 ? (int)range.Step.Value : 1;

            int numSteps = (max - min) / step + 1;
            int selectedStep = _random.Next(numSteps);
            return min + selectedStep * step;
        }

        private double SampleFloatParameter(ParameterRange range)
        {
            double min = Convert.ToDouble(range.MinValue);
            double max = Convert.ToDouble(range.MaxValue);

            if (range.UseLogScale)
            {
                // Log-uniform sampling
                double logMin = Math.Log(min);
                double logMax = Math.Log(max);
                double logValue = logMin + _random.NextDouble() * (logMax - logMin);
                return Math.Exp(logValue);
            }
            else
            {
                // Uniform sampling
                return min + _random.NextDouble() * (max - min);
            }
        }

        private UNetNoisePredictor<T> CreateNoisePredictor(DiffusionTrialConfig<T> config)
        {
            return new UNetNoisePredictor<T>(
                inputChannels: config.LatentDim,
                outputChannels: config.LatentDim,
                baseChannels: config.BaseChannels,
                numResBlocks: config.NumResBlocks,
                seed: config.Seed);
        }

        private StandardVAE<T> CreateVAE(DiffusionTrialConfig<T> config)
        {
            return new StandardVAE<T>(
                inputChannels: 3,
                latentChannels: config.LatentDim,
                baseChannels: config.BaseChannels / 2,
                numResBlocksPerLevel: config.NumResBlocks,
                seed: config.Seed);
        }

        private INoiseScheduler<T> CreateScheduler(DiffusionTrialConfig<T> config)
        {
            int numSteps = config.InferenceSteps;
            var schedulerConfig = SchedulerConfig<T>.CreateDefault();

            INoiseScheduler<T> scheduler;
            switch (config.SchedulerType)
            {
                case DiffusionSchedulerType.DDPM:
                    // DDPM is the original slow scheduler; DDIM is a more efficient equivalent
                    scheduler = new DDIMScheduler<T>(schedulerConfig);
                    break;
                case DiffusionSchedulerType.DDIM:
                    scheduler = new DDIMScheduler<T>(schedulerConfig);
                    break;
                case DiffusionSchedulerType.Euler:
                case DiffusionSchedulerType.EulerAncestral:
                case DiffusionSchedulerType.DPMSolver:
                    // PNDM uses pseudo numerical methods similar to Euler/DPM-Solver
                    // and provides high quality with fewer steps
                    scheduler = new PNDMScheduler<T>(schedulerConfig);
                    break;
                case DiffusionSchedulerType.LCM:
                    // LCM uses few steps with high quality - PNDM is best suited for few-step inference
                    scheduler = new PNDMScheduler<T>(schedulerConfig);
                    numSteps = Math.Min(4, numSteps);
                    break;
                default:
                    scheduler = new DDIMScheduler<T>(schedulerConfig);
                    break;
            }

            scheduler.SetTimesteps(numSteps);
            return scheduler;
        }

        private async Task TrainModelAsync(
            IFullModel<T, Tensor<T>, Tensor<T>> model,
            Tensor<T> inputs,
            Tensor<T> targets,
            CancellationToken cancellationToken)
        {
            await Task.Run(() =>
            {
                // Train the model for a fixed number of iterations
                // In practice, this would be more sophisticated
                int numIterations = 100;

                for (int i = 0; i < numIterations; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    model.Train(inputs, targets);
                }
            }, cancellationToken);
        }
    }

    /// <summary>
    /// Simple conditioner that creates conditioning embeddings from input tensors.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    internal class SimpleConditioner<T> : IConditioningModule<T>
    {
        private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
        private readonly int _embeddingDim;

        public int EmbeddingDimension => _embeddingDim;
        public ConditioningType ConditioningType => ConditioningType.Text;
        public bool ProducesPooledOutput => true;
        public int MaxSequenceLength => 77;

        public SimpleConditioner(int embeddingDim)
        {
            _embeddingDim = embeddingDim;
        }

        public Tensor<T> Encode(Tensor<T> condition)
        {
            // Simple pass-through or projection
            if (condition.Shape.Length == 1 && condition.Shape[0] == _embeddingDim)
            {
                return condition;
            }

            // Create a simple embedding by averaging or padding
            var result = new Tensor<T>(new[] { _embeddingDim });
            var resultSpan = result.AsWritableSpan();
            var condSpan = condition.AsSpan();

            int srcLen = condSpan.Length;
            for (int i = 0; i < _embeddingDim; i++)
            {
                if (i < srcLen)
                {
                    resultSpan[i] = condSpan[i];
                }
                else
                {
                    resultSpan[i] = NumOps.Zero;
                }
            }

            return result;
        }

        public Tensor<T> EncodeText(Tensor<T> tokenIds, Tensor<T>? attentionMask = null)
        {
            return Encode(tokenIds);
        }

        public Tensor<T> GetPooledEmbedding(Tensor<T> sequenceEmbeddings)
        {
            // For pooled output, take mean across sequence dimension
            if (sequenceEmbeddings.Shape.Length < 2)
            {
                return sequenceEmbeddings;
            }

            int batchSize = sequenceEmbeddings.Shape[0];
            int seqLen = sequenceEmbeddings.Shape[1];
            int embDim = sequenceEmbeddings.Shape.Length > 2 ? sequenceEmbeddings.Shape[2] : 1;

            var result = new Tensor<T>(new[] { batchSize, embDim });
            var resultSpan = result.AsWritableSpan();
            var srcSpan = sequenceEmbeddings.AsSpan();

            for (int b = 0; b < batchSize; b++)
            {
                for (int d = 0; d < embDim; d++)
                {
                    T sum = NumOps.Zero;
                    for (int s = 0; s < seqLen; s++)
                    {
                        int srcIdx = b * seqLen * embDim + s * embDim + d;
                        if (srcIdx < srcSpan.Length)
                        {
                            sum = NumOps.Add(sum, srcSpan[srcIdx]);
                        }
                    }
                    resultSpan[b * embDim + d] = NumOps.Divide(sum, NumOps.FromDouble(seqLen));
                }
            }

            return result;
        }

        public Tensor<T> GetUnconditionalEmbedding(int batchSize)
        {
            // Return zero embedding for unconditional generation
            return new Tensor<T>(new[] { batchSize, MaxSequenceLength, _embeddingDim });
        }

        public Tensor<T> Tokenize(string text)
        {
            // Simple character-level tokenization
            var result = new Tensor<T>(new[] { 1, MaxSequenceLength });
            var span = result.AsWritableSpan();

            int len = Math.Min(text.Length, MaxSequenceLength);
            for (int i = 0; i < len; i++)
            {
                span[i] = NumOps.FromDouble(text[i]);
            }

            return result;
        }

        public Tensor<T> TokenizeBatch(string[] texts)
        {
            var result = new Tensor<T>(new[] { texts.Length, MaxSequenceLength });
            var span = result.AsWritableSpan();

            for (int b = 0; b < texts.Length; b++)
            {
                string text = texts[b];
                int len = Math.Min(text.Length, MaxSequenceLength);
                for (int i = 0; i < len; i++)
                {
                    span[b * MaxSequenceLength + i] = NumOps.FromDouble(text[i]);
                }
            }

            return result;
        }
    }

    /// <summary>
    /// Wrapper model that combines diffusion components into an IFullModel.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    internal class DiffusionAutoMLModel<T> : IFullModel<T, Tensor<T>, Tensor<T>>
    {
        private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
        private readonly UNetNoisePredictor<T> _noisePredictor;
        private readonly StandardVAE<T> _vae;
        private readonly INoiseScheduler<T> _scheduler;
        private readonly IConditioningModule<T> _conditioner;
        private readonly DiffusionTrialConfig<T> _config;
        private readonly int? _seed;
        private Random _random;

        public ModelType Type => ModelType.NeuralNetwork;

        public int ParameterCount => _noisePredictor.ParameterCount + _vae.ParameterCount;

        public string[] FeatureNames { get; set; } = Array.Empty<string>();

        public bool SupportsJitCompilation => false;

        public ILossFunction<T> DefaultLossFunction => new MeanSquaredErrorLoss<T>();

        public DiffusionAutoMLModel(
            UNetNoisePredictor<T> noisePredictor,
            StandardVAE<T> vae,
            INoiseScheduler<T> scheduler,
            IConditioningModule<T> conditioner,
            DiffusionTrialConfig<T> config,
            int? seed)
        {
            _noisePredictor = noisePredictor;
            _vae = vae;
            _scheduler = scheduler;
            _conditioner = conditioner;
            _config = config;
            _seed = seed;
            _random = seed.HasValue
                ? RandomHelper.CreateSeededRandom(seed.Value)
                : RandomHelper.CreateSecureRandom();
        }

        public Tensor<T> Predict(Tensor<T> input)
        {
            // Encode conditioning
            var condition = _conditioner.Encode(input);

            // Start from noise
            int latentSize = _config.LatentDim;
            var latent = SampleNoise(new[] { 1, latentSize, _config.LatentHeight, _config.LatentWidth });

            // Set scheduler timesteps
            _scheduler.SetTimesteps(_config.InferenceSteps);
            var timesteps = _scheduler.Timesteps;

            // Denoising loop
            foreach (var t in timesteps)
            {
                var noisePred = _noisePredictor.PredictNoise(latent, t, condition);

                // Apply classifier-free guidance if scale > 1
                if (_config.GuidanceScale > 1.0)
                {
                    // Simplified guidance - in practice would need unconditional prediction
                    var scale = NumOps.FromDouble(_config.GuidanceScale);
                    var noisePredSpan = noisePred.AsWritableSpan();
                    for (int i = 0; i < noisePredSpan.Length; i++)
                    {
                        noisePredSpan[i] = NumOps.Multiply(noisePredSpan[i], scale);
                    }
                }

                // Convert tensors to vectors for scheduler step
                var noisePredVec = new Vector<T>(noisePred.AsSpan().ToArray());
                var latentVec = new Vector<T>(latent.AsSpan().ToArray());
                var eta = NumOps.Zero; // deterministic mode (DDIM-style)

                var newLatentVec = _scheduler.Step(noisePredVec, t, latentVec, eta);

                // Convert back to tensor
                var newLatent = new Tensor<T>(latent.Shape);
                var newLatentSpan = newLatent.AsWritableSpan();
                for (int j = 0; j < newLatentVec.Length && j < newLatentSpan.Length; j++)
                    newLatentSpan[j] = newLatentVec[j];
                latent = newLatent;
            }

            // Decode from latent space
            return _vae.Decode(latent);
        }

        public void Train(Tensor<T> input, Tensor<T> expectedOutput)
        {
            // Encode target to latent space
            var latent = _vae.Encode(expectedOutput);

            // Sample random timestep
            int maxT = _config.InferenceSteps;
            int t = _random.Next(1, maxT + 1);

            // Sample noise and add to latent
            var noise = SampleNoise(latent.Shape);
            var noisyLatent = AddNoise(latent, noise, t);

            // Encode conditioning
            var condition = _conditioner.Encode(input);

            // Predict noise
            var predictedNoise = _noisePredictor.PredictNoise(noisyLatent, t, condition);

            // Compute loss and update (simplified - actual training would use optimizer)
            var loss = ComputeMSELoss(predictedNoise, noise);

            // Gradient computation would happen here in a full implementation
        }

        public Vector<T> GetParameters()
        {
            var noisePredParams = _noisePredictor.GetParameters();
            var vaeParams = _vae.GetParameters();

            int totalLen = noisePredParams.Length + vaeParams.Length;
            var combined = new T[totalLen];

            for (int i = 0; i < noisePredParams.Length; i++)
                combined[i] = noisePredParams[i];

            for (int i = 0; i < vaeParams.Length; i++)
                combined[noisePredParams.Length + i] = vaeParams[i];

            return new Vector<T>(combined);
        }

        public void SetParameters(Vector<T> parameters)
        {
            var noisePredLen = _noisePredictor.ParameterCount;
            var noisePredParams = new T[noisePredLen];
            var vaeParams = new T[_vae.ParameterCount];

            for (int i = 0; i < noisePredLen; i++)
                noisePredParams[i] = parameters[i];

            for (int i = 0; i < vaeParams.Length; i++)
                vaeParams[i] = parameters[noisePredLen + i];

            _noisePredictor.SetParameters(new Vector<T>(noisePredParams));
            _vae.SetParameters(new Vector<T>(vaeParams));
        }

        public IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
        {
            var copy = DeepCopy();
            copy.SetParameters(parameters);
            return copy;
        }

        public IFullModel<T, Tensor<T>, Tensor<T>> Clone()
        {
            return new DiffusionAutoMLModel<T>(
                _noisePredictor,
                _vae,
                _scheduler,
                _conditioner,
                _config,
                _seed);
        }

        public IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
        {
            return new DiffusionAutoMLModel<T>(
                (UNetNoisePredictor<T>)_noisePredictor.DeepCopy(),
                (StandardVAE<T>)_vae.DeepCopy(),
                _scheduler,
                _conditioner,
                _config,
                _seed);
        }

        public Dictionary<string, T> GetFeatureImportance()
        {
            return new Dictionary<string, T>();
        }

        public IEnumerable<int> GetActiveFeatureIndices()
        {
            return Enumerable.Range(0, _config.LatentDim);
        }

        public bool IsFeatureUsed(int featureIndex)
        {
            return featureIndex >= 0 && featureIndex < _config.LatentDim;
        }

        public void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
        {
            // Not applicable for diffusion models
        }

        public ModelMetadata<T> GetModelMetadata()
        {
            var metadata = new ModelMetadata<T>
            {
                Name = "DiffusionAutoMLModel",
                Description = "Diffusion model created by AutoML search",
                Version = "1.0",
                ModelType = ModelType.NeuralNetwork,
                Complexity = ParameterCount
            };

            metadata.SetProperty("NoisePredictorType", _config.NoisePredictorType.ToString());
            metadata.SetProperty("SchedulerType", _config.SchedulerType.ToString());
            metadata.SetProperty("InferenceSteps", _config.InferenceSteps);
            metadata.SetProperty("GuidanceScale", _config.GuidanceScale);
            metadata.SetProperty("BaseChannels", _config.BaseChannels);
            metadata.SetProperty("LatentDim", _config.LatentDim);

            return metadata;
        }

        public void SaveModel(string filePath)
        {
            var data = Serialize();
            File.WriteAllBytes(filePath, data);
        }

        public void LoadModel(string filePath)
        {
            var data = File.ReadAllBytes(filePath);
            Deserialize(data);
        }

        public byte[] Serialize()
        {
            // Serialize parameters as doubles for portability across numeric types.
            // This allows models trained with float to be loaded as double and vice versa.
            // The format is: [version byte] [parameter count (4 bytes)] [parameters as doubles]
            var parameters = GetParameters();
            int headerSize = 1 + sizeof(int); // version + count
            var data = new byte[headerSize + parameters.Length * sizeof(double)];

            // Version byte (for future format changes)
            data[0] = 1;

            // Parameter count
            Buffer.BlockCopy(BitConverter.GetBytes(parameters.Length), 0, data, 1, sizeof(int));

            // Parameters as doubles
            for (int i = 0; i < parameters.Length; i++)
            {
                double value = NumOps.ToDouble(parameters[i]);
                var bytes = BitConverter.GetBytes(value);
                Buffer.BlockCopy(bytes, 0, data, headerSize + i * sizeof(double), sizeof(double));
            }

            return data;
        }

        public void Deserialize(byte[] data)
        {
            // Check minimum header size
            int headerSize = 1 + sizeof(int);
            if (data.Length < headerSize)
                throw new InvalidDataException("Invalid serialized data: too short for header.");

            // Read version (currently only version 1 supported)
            byte version = data[0];
            if (version != 1)
                throw new InvalidDataException($"Unsupported serialization version: {version}");

            // Read parameter count
            int paramCount = BitConverter.ToInt32(data, 1);
            if (data.Length < headerSize + paramCount * sizeof(double))
                throw new InvalidDataException("Invalid serialized data: truncated parameter data.");

            var parameters = new T[paramCount];
            for (int i = 0; i < paramCount; i++)
            {
                double value = BitConverter.ToDouble(data, headerSize + i * sizeof(double));
                parameters[i] = NumOps.FromDouble(value);
            }

            SetParameters(new Vector<T>(parameters));
        }

        public void SaveState(Stream stream)
        {
            if (stream is null)
                throw new ArgumentNullException(nameof(stream));

            var data = Serialize();
            stream.Write(data, 0, data.Length);
            stream.Flush();
        }

        public void LoadState(Stream stream)
        {
            if (stream is null)
                throw new ArgumentNullException(nameof(stream));

            using var ms = new MemoryStream();
            stream.CopyTo(ms);
            var data = ms.ToArray();
            Deserialize(data);
        }

        public Vector<T> ComputeGradients(Tensor<T> input, Tensor<T> target, ILossFunction<T>? lossFunction = null)
        {
            // Simplified gradient computation
            var parameters = GetParameters();
            return new Vector<T>(new T[parameters.Length]);
        }

        public void ApplyGradients(Vector<T> gradients, T learningRate)
        {
            var parameters = GetParameters();
            var newParams = new T[parameters.Length];

            for (int i = 0; i < parameters.Length; i++)
            {
                var update = NumOps.Multiply(gradients[i], learningRate);
                newParams[i] = NumOps.Subtract(parameters[i], update);
            }

            SetParameters(new Vector<T>(newParams));
        }

        public ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
        {
            throw new NotSupportedException("JIT compilation not supported for diffusion models");
        }

        private Tensor<T> SampleNoise(int[] shape)
        {
            var tensor = new Tensor<T>(shape);
            var span = tensor.AsWritableSpan();

            for (int i = 0; i < span.Length; i += 2)
            {
                double u1 = _random.NextDouble();
                double u2 = _random.NextDouble();
                while (u1 <= double.Epsilon) u1 = _random.NextDouble();

                double mag = Math.Sqrt(-2.0 * Math.Log(u1));
                double z0 = mag * Math.Cos(2 * Math.PI * u2);
                double z1 = mag * Math.Sin(2 * Math.PI * u2);

                span[i] = NumOps.FromDouble(z0);
                if (i + 1 < span.Length)
                    span[i + 1] = NumOps.FromDouble(z1);
            }

            return tensor;
        }

        private Tensor<T> AddNoise(Tensor<T> latent, Tensor<T> noise, int timestep)
        {
            double t = timestep / (double)_config.InferenceSteps;
            double alpha = 1.0 - t;
            double sigma = t;

            var result = new Tensor<T>(latent.Shape);
            var resultSpan = result.AsWritableSpan();
            var latentSpan = latent.AsSpan();
            var noiseSpan = noise.AsSpan();

            var alphaT = NumOps.FromDouble(Math.Sqrt(alpha));
            var sigmaT = NumOps.FromDouble(Math.Sqrt(sigma));

            for (int i = 0; i < resultSpan.Length; i++)
            {
                var scaledLatent = NumOps.Multiply(latentSpan[i], alphaT);
                var scaledNoise = NumOps.Multiply(noiseSpan[i], sigmaT);
                resultSpan[i] = NumOps.Add(scaledLatent, scaledNoise);
            }

            return result;
        }

        private T ComputeMSELoss(Tensor<T> predicted, Tensor<T> target)
        {
            var predSpan = predicted.AsSpan();
            var targetSpan = target.AsSpan();

            T sum = NumOps.Zero;
            int count = Math.Min(predSpan.Length, targetSpan.Length);

            for (int i = 0; i < count; i++)
            {
                var diff = NumOps.Subtract(predSpan[i], targetSpan[i]);
                sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
            }

            return NumOps.Divide(sum, NumOps.FromDouble(count));
        }
    }
}

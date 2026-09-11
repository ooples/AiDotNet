using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using AiDotNet.Attributes;
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
using AiDotNet.NeuralNetworks;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
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

        /// <summary>Gets or sets the width of the conditioning vector, which is the context width of the noise
        /// predictor's cross-attention.</summary>
        /// <remarks>DiffusionAutoML sets it from the training inputs. The default, 768, is the width of CLIP ViT-L/14,
        /// Stable Diffusion 1.x's text encoder.</remarks>
        public int ConditioningDim { get; set; } = 768;

        /// <summary>Gets or sets the number of image channels the autoencoder reconstructs.</summary>
        /// <remarks>DiffusionAutoML sets it from the training targets.</remarks>
        public int ImageChannels { get; set; } = 3;

        /// <summary>Gets or sets the number of attention heads in the noise predictor.</summary>
        /// <remarks>It must divide <see cref="BaseChannels"/>, which the search space keeps a multiple of 64.</remarks>
        public int NumHeads { get; set; } = 8;

        /// <summary>Gets or sets the number of transformer blocks in the DiT and U-ViT noise predictors.</summary>
        /// <remarks>12 is the depth of DiT-S and DiT-B (Peebles and Xie 2023, Table 1).</remarks>
        public int TransformerDepth { get; set; } = 12;

        /// <summary>Gets or sets the probability that a training step drops its condition.</summary>
        /// <remarks>Classifier-free guidance combines a conditional and an unconditional prediction, so one network
        /// has to learn both: Ho and Salimans (2022, Algorithm 1) discard the conditioning with probability p_uncond,
        /// and report that 0.5 "consistently performs worse" than 0.1 or 0.2.</remarks>
        public double ConditioningDropoutProbability { get; set; } = 0.1;

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
                ["ConditioningDim"] = ConditioningDim,
                ["ImageChannels"] = ImageChannels,
                ["NumHeads"] = NumHeads,
                ["TransformerDepth"] = TransformerDepth,
                ["ConditioningDropoutProbability"] = ConditioningDropoutProbability,
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

            if (parameters.TryGetValue("ConditioningDim", out var cd))
                config.ConditioningDim = Convert.ToInt32(cd);

            if (parameters.TryGetValue("ImageChannels", out var ic))
                config.ImageChannels = Convert.ToInt32(ic);

            if (parameters.TryGetValue("NumHeads", out var nh))
                config.NumHeads = Convert.ToInt32(nh);

            if (parameters.TryGetValue("TransformerDepth", out var td))
                config.TransformerDepth = Convert.ToInt32(td);

            if (parameters.TryGetValue("ConditioningDropoutProbability", out var cdp))
                config.ConditioningDropoutProbability = Convert.ToDouble(cdp);

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
    /// <example>
    /// <code>
    /// var automl = new DiffusionAutoML&lt;float&gt;(imageSize: 256, channels: 3);
    /// var bestModel = await automl.SearchAsync(
    ///     trainImages, trainLabels,
    ///     valImages, valLabels,
    ///     maxTrials: 10,
    ///     timeLimit: TimeSpan.FromHours(1));
    /// </code>
    /// </example>
    [ModelDomain(ModelDomain.Generative)]
    [ModelCategory(ModelCategory.Diffusion)]
    [ModelCategory(ModelCategory.Optimization)]
    [ModelTask(ModelTask.Generation)]
    [ModelTask(ModelTask.TextToImage)]
    [ModelComplexity(ModelComplexity.VeryHigh)]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("Denoising Diffusion Probabilistic Models", "https://arxiv.org/abs/2006.11239")]
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
                    AddDataShape(parameters, inputs, targets);
                    var trialStopwatch = Stopwatch.StartNew();

                    try
                    {
                        // Create and evaluate model
                        var model = await CreateModelWithHookAsync(typeof(NeuralNetworks.NeuralNetworkBase<T>), parameters);

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
                    AddDataShape(defaultParams, inputs, targets);
                    BestModel = await CreateModelWithHookAsync(typeof(NeuralNetworks.NeuralNetworkBase<T>), defaultParams);
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
            Type modelType,
            Dictionary<string, object> parameters)
        {
            return await Task.Run(() =>
            {
                var config = DiffusionTrialConfig<T>.FromDictionary(parameters);

                // The model builds the noise predictor, autoencoder, scheduler and conditioner the trial names.
                return (IFullModel<T, Tensor<T>, Tensor<T>>)new DiffusionAutoMLModel<T>(config, config.Seed ?? _seed);
            });
        }

        /// <summary>
        /// Gets the default search space for diffusion models.
        /// </summary>
        protected override Dictionary<string, ParameterRange> GetDefaultSearchSpace(Type modelType)
        {
            _ = modelType; // Diffusion models use a fixed search space independent of model type
            return GetDefaultDiffusionSearchSpace();
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

        /// <summary>
        /// Adds the sizes a trial takes from the data rather than from the search: the conditioning width the
        /// noise predictor's cross-attention reads, and the image the autoencoder reconstructs.
        /// </summary>
        /// <remarks>
        /// Left at their defaults, the noise predictor's context width (768) never matched the conditioner's
        /// (four times the base channels), and the latent size never matched the images.
        /// </remarks>
        private static void AddDataShape(Dictionary<string, object> parameters, Tensor<T> inputs, Tensor<T> targets)
        {
            if (inputs.Length > 0)
            {
                int batch = inputs.Shape.Length > 1 ? Math.Max(1, inputs.Shape[0]) : 1;
                parameters["ConditioningDim"] = Math.Max(1, inputs.Length / batch);
            }

            if (targets.Shape.Length == 4)
            {
                int factor = DiffusionAutoMLModel<T>.AutoencoderDownsampling;
                if (targets.Shape[2] % factor != 0 || targets.Shape[3] % factor != 0)
                {
                    throw new ArgumentException(
                        $"Image sides must be multiples of {factor}, the autoencoder's downsampling; got " +
                        $"{targets.Shape[2]} x {targets.Shape[3]}.", nameof(targets));
                }

                parameters["ImageChannels"] = targets.Shape[1];
                parameters["LatentHeight"] = targets.Shape[2] / factor;
                parameters["LatentWidth"] = targets.Shape[3] / factor;
            }
        }

        /// <summary>Gets or sets the autoencoder training steps each trial runs before its denoiser trains.</summary>
        /// <remarks>Latent diffusion trains in two phases (Rombach et al. 2022): "First, we train an autoencoder",
        /// then the diffusion model in its latent space.</remarks>
        public int AutoencoderTrainingIterations { get; set; } = 100;

        /// <summary>Gets or sets the denoiser training steps each trial runs.</summary>
        public int DiffusionTrainingIterations { get; set; } = 100;

        private async Task TrainModelAsync(
            IFullModel<T, Tensor<T>, Tensor<T>> model,
            Tensor<T> inputs,
            Tensor<T> targets,
            CancellationToken cancellationToken)
        {
            await Task.Run(() =>
            {
                // Phase one: the autoencoder, when the targets are images. Latent targets need none.
                if (model is DiffusionAutoMLModel<T> diffusion && diffusion.IsImage(targets))
                {
                    for (int i = 0; i < AutoencoderTrainingIterations; i++)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        diffusion.TrainAutoencoder(targets);
                    }
                }

                // Phase two: the denoiser on the autoencoder's latents, mini-batched by NeuralBatchHelper (#1296).
                for (int i = 0; i < DiffusionTrainingIterations; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    NeuralBatchHelper.TrainMaybeBatched(model, inputs, targets);
                }
            }, cancellationToken);
        }

        /// <summary>
        /// Scores a trial by how close its conditioned samples come to the paired validation images.
        /// </summary>
        /// <remarks>
        /// The inherited scorer compares Predict with the targets, but a latent diffusion model's Predict denoises
        /// a latent: it takes no condition and returns no image. This runs the conditioned sampler instead, so the
        /// sampler, step count and guidance scale under search all affect the score. The starting noise is seeded,
        /// so every trial starts from the same noise.
        /// </remarks>
        protected override Task<double> EvaluateModelAsync(
            IFullModel<T, Tensor<T>, Tensor<T>> model,
            Tensor<T> validationInputs,
            Tensor<T> validationTargets)
        {
            if (model is not DiffusionAutoMLModel<T> diffusion)
                return base.EvaluateModelAsync(model, validationInputs, validationTargets);

            return Task.Run(() =>
            {
                var samples = diffusion.GenerateConditioned(validationInputs, _seed ?? 0);
                if (samples.Length != validationTargets.Length)
                {
                    throw new ArgumentException(
                        $"The conditioned samples hold {samples.Length} values and the validation targets " +
                        $"{validationTargets.Length}; the targets must be images of the configured size.",
                        nameof(validationTargets));
                }

                double sum = 0.0;
                for (int i = 0; i < samples.Length; i++)
                {
                    double difference = NumOps.ToDouble(samples[i]) - NumOps.ToDouble(validationTargets[i]);
                    sum += difference * difference;
                }

                return samples.Length == 0 ? 0.0 : sum / samples.Length;
            });
        }
    }

    /// <summary>
    /// Turns a conditioning vector into the one-token context a noise predictor's cross-attention reads.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// Cross-attention takes context of shape [batch, tokens, width]. This conditioner has no weights: each row
    /// of the condition becomes one token, zero-padded or truncated to the width, and the noise predictor's key
    /// and value projections learn what to read from it. The null condition for classifier-free guidance is the
    /// zero token, so an all-zero condition is the unconditional one. It used to return a rank-1 vector for a
    /// condition and a [batch, 77, width] tensor for the null condition.
    /// </remarks>
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
            if (embeddingDim <= 0)
                throw new ArgumentOutOfRangeException(nameof(embeddingDim), embeddingDim, "The conditioning width must be positive.");

            _embeddingDim = embeddingDim;
        }

        /// <summary>Encodes a condition of shape [features] or [batch, ...] as context [batch, 1, EmbeddingDimension].</summary>
        public Tensor<T> Encode(Tensor<T> condition)
        {
            if (condition is null)
                throw new ArgumentNullException(nameof(condition));
            if (condition.Shape.Length == 3 && condition.Shape[1] == 1 && condition.Shape[2] == _embeddingDim)
                return condition;

            int batch = condition.Shape.Length > 1 ? condition.Shape[0] : 1;
            if (batch <= 0 || condition.Length == 0)
                throw new ArgumentException("A condition needs at least one value per sample.", nameof(condition));

            int features = condition.Length / batch;
            int copied = Math.Min(features, _embeddingDim);
            var context = new Tensor<T>(new[] { batch, 1, _embeddingDim });
            var target = context.AsWritableSpan();
            var source = condition.AsSpan();
            for (int b = 0; b < batch; b++)
            {
                for (int f = 0; f < copied; f++)
                    target[b * _embeddingDim + f] = source[b * features + f];
            }

            return context;
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

        /// <summary>The null condition: one zero token per sample, the shape <see cref="Encode"/> returns.</summary>
        public Tensor<T> GetUnconditionalEmbedding(int batchSize)
        {
            return new Tensor<T>(new[] { Math.Max(1, batchSize), 1, _embeddingDim });
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
    /// The latent diffusion model a <see cref="DiffusionAutoML{T}"/> trial builds: a noise predictor trained on an
    /// autoencoder's latents and conditioned through cross-attention on a conditioning vector.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// <para>
    /// It derives from <see cref="LatentDiffusionModelBase{T}"/>, so the denoiser trains with the shared DDPM step
    /// on the gradient tape (Ho et al. 2020, Algorithm 1), noised and timestepped by the trial's own scheduler.
    /// Train takes the condition as its input and the image, or an already-encoded latent, as its target.
    /// </para>
    /// <para>
    /// It replaces a hand-written wrapper whose Train computed the denoising loss, discarded it, and stepped on
    /// an SPSA estimate taken through the whole sampling loop against the target image. That wrapper noised
    /// with a schedule of its own and "guided" by multiplying the noise prediction by the scale. It handed
    /// cross-attention a rank-1 condition of the wrong width, and it built a U-Net and a DDIM or PNDM scheduler
    /// whichever predictor and scheduler the trial named.
    /// </para>
    /// <para>
    /// Predict keeps the latent-diffusion contract and denoises a latent. <see cref="GenerateConditioned"/> turns
    /// conditions into images.
    /// </para>
    /// </remarks>
    [ModelDomain(ModelDomain.Generative)]
    [ModelCategory(ModelCategory.Diffusion)]
    [ModelTask(ModelTask.Generation)]
    [ModelTask(ModelTask.Denoising)]
    [ModelComplexity(ModelComplexity.VeryHigh)]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("Denoising Diffusion Probabilistic Models", "https://arxiv.org/abs/2006.11239")]
    [ResearchPaper("High-Resolution Image Synthesis with Latent Diffusion Models", "https://arxiv.org/abs/2112.10752")]
    [ResearchPaper("Classifier-Free Diffusion Guidance", "https://arxiv.org/abs/2207.12598")]
    internal partial class DiffusionAutoMLModel<T> : LatentDiffusionModelBase<T>
    {
        /// <summary>
        /// The autoencoder's downsampling: four levels, 2^3 = 8, the factor Latent Diffusion and DiT use.
        /// </summary>
        internal const int AutoencoderDownsampling = 8;

        private static readonly int[] AutoencoderChannelMultipliers = { 1, 2, 4, 4 };

        /// <inheritdoc />
        /// <remarks>The noise predictor then the autoencoder, the previous wrapper's serialization order.</remarks>
        protected override void RegisterComponents()
        {
            RegisterParameterComponent(_noisePredictor);
            RegisterParameterComponent(_vae);
        }

        // Stored under the constructor's own parameter names so the clone plan replays the constructor.
        private readonly DiffusionTrialConfig<T> _config;
        private readonly int? _seed;

        private readonly NoisePredictorBase<T> _noisePredictor;
        private readonly StandardVAE<T> _vae;
        private readonly SimpleConditioner<T> _conditioner;

        /// <inheritdoc />
        public override INoisePredictor<T> NoisePredictor => _noisePredictor;

        /// <inheritdoc />
        public override IVAEModel<T> VAE => _vae;

        /// <inheritdoc />
        public override IConditioningModule<T>? Conditioner => _conditioner;

        /// <inheritdoc />
        /// <remarks>Guarded because the base constructor runs before this one assigns the configuration.</remarks>
        public override int LatentChannels => _config?.LatentDim ?? new DiffusionTrialConfig<T>().LatentDim;

        /// <summary>Gets the trial configuration the model was built from.</summary>
        public DiffusionTrialConfig<T> Config => _config;

        /// <summary>
        /// Builds the noise predictor, autoencoder, scheduler and conditioner a trial configuration names.
        /// </summary>
        /// <param name="config">The trial configuration, or null for its defaults.</param>
        /// <param name="seed">Seeds weight initialization and the training noise and timesteps, or null.</param>
        public DiffusionAutoMLModel(DiffusionTrialConfig<T>? config = null, int? seed = null)
            : base(CreateOptions(config, seed), CreateScheduler(config))
        {
            _config = config ?? new DiffusionTrialConfig<T>();
            _seed = seed;
            _noisePredictor = CreateNoisePredictor(_config, seed);
            _vae = new StandardVAE<T>(
                inputChannels: _config.ImageChannels,
                latentChannels: _config.LatentDim,
                baseChannels: Math.Max(1, _config.BaseChannels / 2),
                channelMultipliers: (int[])AutoencoderChannelMultipliers.Clone(),
                numResBlocksPerLevel: _config.NumResBlocks,
                seed: seed);
            _conditioner = new SimpleConditioner<T>(_config.ConditioningDim);
            SetGuidanceScale(_config.GuidanceScale);
        }

        private static DiffusionModelOptions<T> CreateOptions(DiffusionTrialConfig<T>? config, int? seed)
        {
            var resolved = config ?? new DiffusionTrialConfig<T>();

            // The options' own schedule defaults are DDPM's (Ho et al. 2020): T = 1000, betas linear from 1e-4 to 0.02.
            return new DiffusionModelOptions<T>
            {
                LearningRate = resolved.LearningRate,
                DefaultInferenceSteps = resolved.InferenceSteps,
                LatentChannels = resolved.LatentDim,
                Seed = seed,
            };
        }

        private static INoiseScheduler<T> CreateScheduler(DiffusionTrialConfig<T>? config)
        {
            // The same DDPM schedule the options declare, so training and sampling agree.
            var schedule = SchedulerConfig<T>.CreateDefault();
            var type = (config ?? new DiffusionTrialConfig<T>()).SchedulerType;
            switch (type)
            {
                case DiffusionSchedulerType.DDPM:
                    return new DDPMScheduler<T>(schedule);
                case DiffusionSchedulerType.DDIM:
                    return new DDIMScheduler<T>(schedule);
                case DiffusionSchedulerType.Euler:
                    return new EulerDiscreteScheduler<T>(schedule);
                case DiffusionSchedulerType.EulerAncestral:
                    return new EulerAncestralDiscreteScheduler<T>(schedule);
                case DiffusionSchedulerType.DPMSolver:
                    return new DPMSolverMultistepScheduler<T>(schedule);
                case DiffusionSchedulerType.LCM:
                    // LCM's few-step sampler assumes a consistency-distilled model (Luo et al. 2023); a denoiser
                    // trained here is not one. It is built because the search space names it.
                    return new LCMScheduler<T>(schedule);
                default:
                    throw new NotSupportedException($"Scheduler type {type} is not supported.");
            }
        }

        private static NoisePredictorBase<T> CreateNoisePredictor(DiffusionTrialConfig<T> config, int? seed)
        {
            switch (config.NoisePredictorType)
            {
                case NoisePredictorType.UNet:
                    return new UNetNoisePredictor<T>(
                        inputChannels: config.LatentDim,
                        outputChannels: config.LatentDim,
                        baseChannels: config.BaseChannels,
                        numResBlocks: config.NumResBlocks,
                        contextDim: config.ConditioningDim,
                        numHeads: config.NumHeads,
                        inputHeight: config.LatentHeight,
                        seed: seed);
                case NoisePredictorType.DiT:
                    RequireSquareLatent(config);
                    return new DiTNoisePredictor<T>(
                        inputChannels: config.LatentDim,
                        hiddenSize: config.BaseChannels,
                        numLayers: config.TransformerDepth,
                        numHeads: config.NumHeads,
                        patchSize: PatchSize(config),
                        contextDim: config.ConditioningDim,
                        latentSpatialSize: config.LatentHeight,
                        seed: seed);
                case NoisePredictorType.UViT:
                    RequireSquareLatent(config);
                    return new UViTNoisePredictor<T>(
                        inputChannels: config.LatentDim,
                        hiddenSize: config.BaseChannels,
                        numLayers: config.TransformerDepth,
                        numHeads: config.NumHeads,
                        patchSize: PatchSize(config),
                        contextDim: config.ConditioningDim,
                        latentSpatialSize: config.LatentHeight,
                        seed: seed);
                default:
                    throw new NotSupportedException($"Noise predictor type {config.NoisePredictorType} is not supported.");
            }
        }

        /// <summary>Patch size 2, DiT's "/2" configurations (Peebles and Xie 2023), unless a latent side is odd.</summary>
        private static int PatchSize(DiffusionTrialConfig<T> config)
            => config.LatentHeight % 2 == 0 && config.LatentWidth % 2 == 0 ? 2 : 1;

        private static void RequireSquareLatent(DiffusionTrialConfig<T> config)
        {
            if (config.LatentHeight != config.LatentWidth)
            {
                throw new NotSupportedException(
                    $"The {config.NoisePredictorType} noise predictor takes a square latent; got {config.LatentHeight} x {config.LatentWidth}.");
            }
        }

        /// <summary>
        /// Whether a tensor is an image batch for the autoencoder: [batch, image channels, height, width], with a
        /// channel count the latent does not share.
        /// </summary>
        public bool IsImage(Tensor<T> tensor)
            => tensor is not null
               && tensor.Shape.Length == 4
               && tensor.Shape[1] == _vae.InputChannels
               && tensor.Shape[1] != LatentChannels;

        /// <summary>
        /// One step of the first training phase: the autoencoder learns to reconstruct the images.
        /// </summary>
        /// <remarks>
        /// Rombach et al. (2022) "separate training into two distinct phases: First, we train an autoencoder",
        /// then the denoiser on its latents. The denoiser's latents are encoded before its gradient tape opens, so
        /// the autoencoder stays fixed while the denoiser trains.
        /// </remarks>
        public void TrainAutoencoder(Tensor<T> images)
        {
            if (images is null)
                throw new ArgumentNullException(nameof(images));
            if (!IsImage(images))
            {
                throw new ArgumentException(
                    $"Autoencoder training takes images of shape [batch, {_vae.InputChannels}, height, width].", nameof(images));
            }

            EnsureOwnWeights();
            _vae.Train(images, images);
        }

        /// <inheritdoc />
        /// <remarks>The denoiser trains on z = E(x), the scaled latent of the image (Rombach et al. 2022), and a
        /// target that is already a latent is used as it is.</remarks>
        protected override Tensor<T> PrepareTrainingSample(Tensor<T> input, Tensor<T> expectedOutput)
        {
            if (expectedOutput is null)
                throw new ArgumentNullException(nameof(expectedOutput));

            return IsImage(expectedOutput) ? EncodeToLatent(expectedOutput, sampleMode: true) : expectedOutput;
        }

        /// <inheritdoc />
        /// <remarks>The condition is the Train input, dropped per sample with the configured probability so the
        /// network also learns the unconditional prediction guidance needs.</remarks>
        protected override Tensor<T> PredictTrainingNoise(
            Tensor<T> noisySample,
            int[] timesteps,
            bool isBatched,
            Tensor<T> input,
            Tensor<T> expectedOutput)
        {
            var latents = ToLatentLayout(noisySample);
            int batch = latents.Shape[0];
            var context = TrainingContext(input, batch);

            Tensor<T> prediction;
            if (batch == 1 || timesteps.Length == 1)
            {
                prediction = _noisePredictor.PredictNoise(latents, timesteps[0], context);
            }
            else
            {
                // A timestep per sample (Ho et al. 2020, Algorithm 1), so each sample is its own forward.
                int channels = latents.Shape[1], height = latents.Shape[2], width = latents.Shape[3];
                int tokens = context.Shape[1], contextWidth = context.Shape[2];
                var parts = new Tensor<T>[batch];
                for (int b = 0; b < batch; b++)
                {
                    parts[b] = _noisePredictor.PredictNoise(
                        Engine.TensorSlice(latents, new[] { b, 0, 0, 0 }, new[] { 1, channels, height, width }),
                        timesteps[b],
                        Engine.TensorSlice(context, new[] { b, 0, 0 }, new[] { 1, tokens, contextWidth }));
                }

                prediction = Engine.TensorConcatenate(parts, axis: 0);
            }

            return noisySample.Shape.Length >= 4 ? prediction : Engine.Reshape(prediction, noisySample._shape);
        }

        private Tensor<T> TrainingContext(Tensor<T> condition, int batch)
        {
            var context = _conditioner.Encode(condition);
            if (context.Shape[0] != batch)
            {
                if (batch == 1)
                {
                    // One sample: its condition is every value given, whatever its shape.
                    context = _conditioner.Encode(condition.Reshape(new[] { 1, condition.Length }));
                }
                else if (context.Shape[0] == 1)
                {
                    context = Engine.TensorTile(context, new[] { batch, 1, 1 });
                }
                else
                {
                    throw new ArgumentException(
                        $"{context.Shape[0]} conditions were given for a batch of {batch} samples.", nameof(condition));
                }
            }

            double dropout = _config.ConditioningDropoutProbability;
            if (dropout <= 0.0)
                return context;

            // Ho and Salimans 2022, Algorithm 1: "with probability p_uncond", discard the conditioning.
            var unconditional = _conditioner.GetUnconditionalEmbedding(1);
            var rows = new Tensor<T>[batch];
            bool anyDropped = false;
            for (int b = 0; b < batch; b++)
            {
                bool drop = RandomGenerator.NextDouble() < dropout;
                anyDropped |= drop;
                rows[b] = drop
                    ? unconditional
                    : Engine.TensorSlice(context, new[] { b, 0, 0 }, new[] { 1, context.Shape[1], context.Shape[2] });
            }

            return anyDropped ? Engine.TensorConcatenate(rows, axis: 0) : context;
        }

        /// <summary>Gives a latent the [batch, channels, height, width] layout the noise predictor takes.</summary>
        private Tensor<T> ToLatentLayout(Tensor<T> sample)
        {
            if (sample.Shape.Length >= 4)
                return sample;

            // [channels, height, width] is one unbatched latent; [batch, values] carries its batch first.
            if (sample.Shape.Length == 3)
                return Engine.Reshape(sample, new[] { 1, sample.Shape[0], sample.Shape[1], sample.Shape[2] });

            int batch = sample.Shape.Length == 2 ? Math.Max(1, sample.Shape[0]) : 1;
            int perSample = sample.Length / batch;
            int channels = LatentChannels;
            if (perSample % channels != 0)
            {
                throw new ArgumentException(
                    $"A latent of {perSample} values per sample cannot hold {channels} channels.", nameof(sample));
            }

            int spatial = perSample / channels;
            int side = (int)Math.Sqrt(spatial);
            return side * side == spatial
                ? Engine.Reshape(sample, new[] { batch, channels, side, side })
                : Engine.Reshape(sample, new[] { batch, channels, 1, spatial });
        }

        /// <summary>
        /// Samples one image per condition with classifier-free guidance (Ho and Salimans 2022): the guided noise
        /// is eps(z, null) + w * (eps(z, c) - eps(z, null)) at the configured guidance scale w.
        /// </summary>
        /// <param name="condition">One conditioning vector per image: [features] or [batch, features].</param>
        /// <param name="seed">Seeds the starting noise, or null for a fresh draw.</param>
        /// <returns>Images of shape [batch, image channels, latent height * 8, latent width * 8].</returns>
        public Tensor<T> GenerateConditioned(Tensor<T> condition, int? seed = null)
        {
            if (condition is null)
                throw new ArgumentNullException(nameof(condition));

            var context = _conditioner.Encode(condition);
            int batch = context.Shape[0];
            double scale = GuidanceScale;

            // w = 1 is plain conditional sampling, so the unconditional pass is only paid for when it changes the
            // result - the same test the base's text-to-image sampler makes.
            var unconditional = scale > 1.0 && _noisePredictor.SupportsCFG
                ? _conditioner.GetUnconditionalEmbedding(batch)
                : null;

            var latentShape = new[] { batch, LatentChannels, _config.LatentHeight, _config.LatentWidth };
            var latents = SampleNoiseTensor(latentShape, CreateInferenceRng(seed));
            Scheduler.SetTimesteps(_config.InferenceSteps);
            foreach (var timestep in Scheduler.Timesteps)
            {
                var noise = _noisePredictor.PredictNoise(latents, timestep, context);
                if (unconditional is not null)
                    noise = ApplyGuidance(_noisePredictor.PredictNoise(latents, timestep, unconditional), noise, scale);

                var stepped = Scheduler.Step(noise.ToVector(), timestep, latents.ToVector(), NumOps.Zero);
                latents = new Tensor<T>(latentShape, stepped);
            }

            return DecodeFromLatent(latents);
        }

        /// <inheritdoc />
        public override ModelMetadata<T> GetModelMetadata()
        {
            var metadata = base.GetModelMetadata();
            metadata.SetProperty("NoisePredictorType", _config.NoisePredictorType.ToString());
            metadata.SetProperty("SchedulerType", _config.SchedulerType.ToString());
            metadata.SetProperty("InferenceSteps", _config.InferenceSteps);
            metadata.SetProperty("GuidanceScale", _config.GuidanceScale);
            metadata.SetProperty("BaseChannels", _config.BaseChannels);
            metadata.SetProperty("LatentDim", _config.LatentDim);
            metadata.SetProperty("ConditioningDim", _config.ConditioningDim);
            metadata.SetProperty("Seed", _seed ?? 0);
            return metadata;
        }
    }
}

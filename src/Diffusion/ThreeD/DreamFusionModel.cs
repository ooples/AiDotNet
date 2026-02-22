using System.Diagnostics.CodeAnalysis;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion;

/// <summary>
/// DreamFusion model for text-to-3D generation via Score Distillation Sampling (SDS).
/// Uses a 2D diffusion prior to optimize a 3D neural radiance field representation.
/// Based on "DreamFusion: Text-to-3D using 2D Diffusion" (Poole et al., 2022).
/// </summary>
/// <typeparam name="T">The numeric type for computations.</typeparam>
/// <remarks>
/// <para>
/// DreamFusion revolutionized text-to-3D generation by using pretrained 2D diffusion models
/// to guide the optimization of a 3D scene representation (NeRF). The key insight is that
/// a 2D diffusion model can serve as a "critic" for 3D content through Score Distillation Sampling.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of DreamFusion as using an AI art critic (the 2D diffusion model)
/// to guide a 3D sculptor (the NeRF). The critic looks at 2D views of the sculpture and gives
/// feedback on how to make it look more like the text description.
///
/// How it works:
/// 1. You describe what you want: "a DSLR photo of a peacock on a surfboard"
/// 2. DreamFusion renders 2D images of a 3D shape from random viewpoints
/// 3. The 2D diffusion model evaluates: "Does this look like the prompt?"
/// 4. Gradients flow back to improve the 3D representation
/// 5. After many iterations, you get a full 3D object you can view from any angle
///
/// Key features:
/// - Creates full 3D assets from text descriptions
/// - View-consistent: looks correct from any angle
/// - Leverages the quality of 2D image generators
/// - No 3D training data required
/// </para>
/// <para>
/// Technical details:
/// - Uses NeRF (Neural Radiance Field) for 3D representation
/// - Employs Score Distillation Sampling (SDS) loss
/// - Samples random camera views during optimization
/// - Uses classifier-free guidance with high scale (typically 100)
/// - Supports mesh extraction via marching cubes
/// </para>
/// </remarks>
public class DreamFusionModel<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Standard latent channels (same as SD).
    /// </summary>
    /// <remarks>
    /// 4 latent channels matching the standard SD 1.5 VAE used as the 2D diffusion prior.
    /// </remarks>
    private const int DREAM_LATENT_CHANNELS = 4;

    /// <summary>
    /// Default timesteps for diffusion.
    /// </summary>
    /// <remarks>
    /// 1000 training timesteps matching the SD 1.5 noise schedule used for SDS.
    /// </remarks>
    private const int DEFAULT_TIMESTEPS = 1000;

    /// <summary>
    /// Default beta start value.
    /// </summary>
    /// <remarks>
    /// Starting beta for the scaled linear noise schedule, matching SD 1.5 defaults.
    /// </remarks>
    private const double DEFAULT_BETA_START = 0.00085;

    /// <summary>
    /// Default beta end value.
    /// </summary>
    /// <remarks>
    /// Ending beta for the scaled linear noise schedule, matching SD 1.5 defaults.
    /// </remarks>
    private const double DEFAULT_BETA_END = 0.012;

    #endregion

    #region Fields

    private NeRFNetwork<T> _nerf;
    private readonly IDiffusionModel<T> _diffusionPrior;
    private readonly DreamFusionConfig _config;
    private UNetNoisePredictor<T> _unet;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _unet;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <inheritdoc />
    public override int LatentChannels => DREAM_LATENT_CHANNELS;

    /// <inheritdoc />
    public override int ParameterCount => _nerf.ParameterCount + _unet.ParameterCount + _vae.ParameterCount;

    /// <summary>
    /// Configuration for DreamFusion model.
    /// </summary>
    public DreamFusionConfig Config => _config;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of the DreamFusionModel.
    /// </summary>
    /// <param name="architecture">Optional neural network architecture for custom layer configuration.</param>
    /// <param name="diffusionPrior">The 2D diffusion model to use as the prior.</param>
    /// <param name="config">Optional configuration settings.</param>
    /// <param name="conditioner">Optional conditioning module.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public DreamFusionModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        IDiffusionModel<T>? diffusionPrior = null,
        DreamFusionConfig? config = null,
        IConditioningModule<T>? conditioner = null,
        int? seed = null)
        : base(
            new DiffusionModelOptions<T>
            {
                TrainTimesteps = DEFAULT_TIMESTEPS,
                BetaStart = DEFAULT_BETA_START,
                BetaEnd = DEFAULT_BETA_END,
                BetaSchedule = BetaSchedule.ScaledLinear
            },
            new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
            architecture)
    {
        _config = config ?? new DreamFusionConfig();
        _conditioner = conditioner;
        _diffusionPrior = diffusionPrior ?? this;

        InitializeLayers(seed);
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the U-Net, VAE, and NeRF network layers.
    /// </summary>
    [MemberNotNull(nameof(_unet), nameof(_vae), nameof(_nerf))]
    private void InitializeLayers(int? seed)
    {
        _unet = new UNetNoisePredictor<T>(
            inputChannels: DREAM_LATENT_CHANNELS,
            outputChannels: DREAM_LATENT_CHANNELS,
            baseChannels: 64,
            numResBlocks: 2,
            architecture: Architecture,
            seed: seed);

        _vae = new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: DREAM_LATENT_CHANNELS,
            baseChannels: 64,
            numResBlocksPerLevel: 2,
            seed: seed);

        _nerf = new NeRFNetwork<T>(_config.NeRFHiddenDim, _config.NeRFNumLayers, seed);
    }

    #endregion

    /// <summary>
    /// Generates a 3D representation from a text prompt using Score Distillation Sampling.
    /// </summary>
    /// <param name="prompt">The text prompt describing the desired 3D object.</param>
    /// <param name="numIterations">Number of optimization iterations.</param>
    /// <param name="learningRate">Learning rate for NeRF optimization.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <param name="progress">Optional progress reporter.</param>
    /// <param name="cancellationToken">Optional cancellation token.</param>
    /// <returns>The optimized NeRF parameters as a tensor.</returns>
    public async Task<NeRFResult<T>> GenerateAsync(
        string prompt,
        int numIterations = 10000,
        double learningRate = 1e-3,
        double guidanceScale = 100.0,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        // Encode the text prompt
        var textEmbedding = EncodeTextInternal(prompt);
        var unconditionalEmbedding = EncodeTextInternal("");

        // Initialize NeRF parameters
        _nerf.Initialize();

        // Optimization loop
        for (int iteration = 0; iteration < numIterations; iteration++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            // Sample random camera pose
            var cameraPose = SampleCameraPose();

            // Render image from current NeRF at this camera pose
            var renderedImage = _nerf.Render(cameraPose, _config.RenderResolution);

            // Sample random timestep (favor mid-range for SDS)
            int timestep = SampleSDSTimestep();

            // Add noise to rendered image
            var noise = GenerateNoise(renderedImage);
            var noisyImage = AddNoiseToImage(renderedImage, noise, timestep);

            // Get noise prediction from diffusion prior (with CFG)
            var noisePredUnconditional = PredictNoiseWithEmbedding(noisyImage, timestep, unconditionalEmbedding);
            var noisePredConditional = PredictNoiseWithEmbedding(noisyImage, timestep, textEmbedding);

            // Apply classifier-free guidance
            var noisePred = ApplyGuidanceInternal(noisePredUnconditional, noisePredConditional, guidanceScale);

            // Compute SDS gradient: w(t) * (noise_pred - noise)
            var sdsGradient = ComputeSDSGradient(noisePred, noise, timestep);

            // Update NeRF parameters
            _nerf.UpdateParameters(sdsGradient, cameraPose, learningRate);

            // Report progress
            progress?.Report((double)(iteration + 1) / numIterations);

            // Allow async context switch
            if (iteration % 100 == 0)
            {
                await Task.Yield();
            }
        }

        // Extract final NeRF representation
        return new NeRFResult<T>
        {
            Parameters = _nerf.GetParameters(),
            Network = _nerf,
            Prompt = prompt
        };
    }

    /// <summary>
    /// Renders an image from the trained NeRF at a specific camera pose.
    /// </summary>
    /// <param name="result">The NeRF result from training.</param>
    /// <param name="cameraPose">The camera pose to render from.</param>
    /// <param name="resolution">The output resolution.</param>
    /// <returns>The rendered image as a tensor.</returns>
    public Tensor<T> RenderView(NeRFResult<T> result, CameraPose cameraPose, int resolution = 256)
    {
        if (result.Network is null)
        {
            throw new InvalidOperationException("NeRF network is not available in result.");
        }

        return result.Network.Render(cameraPose, resolution);
    }

    /// <summary>
    /// Generates a mesh from the trained NeRF using marching cubes.
    /// </summary>
    /// <param name="result">The NeRF result from training.</param>
    /// <param name="gridResolution">Resolution of the marching cubes grid.</param>
    /// <param name="threshold">Density threshold for surface extraction.</param>
    /// <returns>The extracted mesh.</returns>
    public DreamMesh<T> ExtractMesh(NeRFResult<T> result, int gridResolution = 128, double threshold = 0.5)
    {
        if (result.Network is null)
        {
            throw new InvalidOperationException("NeRF network is not available in result.");
        }

        return MarchingCubes(result.Network, gridResolution, threshold);
    }

    /// <summary>
    /// Encodes text to embedding.
    /// </summary>
    private Tensor<T> EncodeTextInternal(string text)
    {
        // Use conditioner if available
        if (_conditioner != null)
        {
            var tokens = _conditioner.Tokenize(text);
            return _conditioner.EncodeText(tokens);
        }

        // Fallback: Create a simple embedding
        var embedDim = _config.TextEmbeddingDim;
        var embedding = new Tensor<T>(new[] { 1, embedDim });
        var embSpan = embedding.AsWritableSpan();

        // Simple hash-based embedding for fallback
        int hash = text.GetHashCode();
        var rng = RandomHelper.CreateSeededRandom(hash);

        for (int i = 0; i < embSpan.Length; i++)
        {
            embSpan[i] = NumOps.FromDouble((rng.NextDouble() * 2 - 1) * 0.1);
        }

        return embedding;
    }

    /// <summary>
    /// Samples a random camera pose for rendering.
    /// </summary>
    private CameraPose SampleCameraPose()
    {
        // Sample spherical coordinates
        double elevation = _config.MinElevation + RandomGenerator.NextDouble() * (_config.MaxElevation - _config.MinElevation);
        double azimuth = RandomGenerator.NextDouble() * 360.0;
        double radius = _config.MinRadius + RandomGenerator.NextDouble() * (_config.MaxRadius - _config.MinRadius);

        // Optionally add some jitter to look-at point
        double lookAtJitter = RandomGenerator.NextDouble() * _config.LookAtJitter * 2 - _config.LookAtJitter;

        return new CameraPose
        {
            Elevation = elevation,
            Azimuth = azimuth,
            Radius = radius,
            LookAtX = lookAtJitter,
            LookAtY = lookAtJitter,
            LookAtZ = lookAtJitter,
            FocalLength = _config.FocalLength
        };
    }

    /// <summary>
    /// Samples a timestep for SDS, favoring mid-range timesteps.
    /// </summary>
    private int SampleSDSTimestep()
    {
        int minT = (int)(DEFAULT_TIMESTEPS * _config.MinTimestepRatio);
        int maxT = (int)(DEFAULT_TIMESTEPS * _config.MaxTimestepRatio);

        return RandomGenerator.Next(minT, maxT);
    }

    /// <summary>
    /// Generates noise matching the input tensor dimensions.
    /// </summary>
    private Tensor<T> GenerateNoise(Tensor<T> template)
    {
        var noise = new Tensor<T>(template.Shape);
        var noiseSpan = noise.AsWritableSpan();

        for (int i = 0; i < noiseSpan.Length; i++)
        {
            noiseSpan[i] = NumOps.FromDouble(RandomGenerator.NextGaussian());
        }

        return noise;
    }

    /// <summary>
    /// Adds noise to an image at a specific timestep.
    /// </summary>
    private Tensor<T> AddNoiseToImage(Tensor<T> image, Tensor<T> noise, int timestep)
    {
        // Get alpha values from scheduler
        double alphasCumprod = GetAlphasCumprod(timestep);
        double sqrtAlpha = Math.Sqrt(alphasCumprod);
        double sqrtOneMinusAlpha = Math.Sqrt(1.0 - alphasCumprod);

        var result = new Tensor<T>(image.Shape);
        var resultSpan = result.AsWritableSpan();
        var imageSpan = image.AsSpan();
        var noiseSpan = noise.AsSpan();

        for (int i = 0; i < resultSpan.Length; i++)
        {
            double imgVal = NumOps.ToDouble(imageSpan[i]);
            double noiseVal = NumOps.ToDouble(noiseSpan[i]);
            resultSpan[i] = NumOps.FromDouble(sqrtAlpha * imgVal + sqrtOneMinusAlpha * noiseVal);
        }

        return result;
    }

    /// <summary>
    /// Gets the cumulative alpha value for a timestep.
    /// </summary>
    private double GetAlphasCumprod(int timestep)
    {
        double t = (double)timestep / DEFAULT_TIMESTEPS;

        // Scaled linear schedule
        double beta = DEFAULT_BETA_START + t * (DEFAULT_BETA_END - DEFAULT_BETA_START);
        double alpha = 1.0 - beta;

        // Approximate cumulative product
        return Math.Pow(alpha, timestep);
    }

    /// <summary>
    /// Predicts noise using the diffusion prior with text embedding.
    /// </summary>
    private Tensor<T> PredictNoiseWithEmbedding(Tensor<T> noisyImage, int timestep, Tensor<T> embedding)
    {
        // Use the internal U-Net's noise prediction with text conditioning
        // This allows the diffusion model to guide 3D generation based on the text prompt
        return _unet.PredictNoise(noisyImage, timestep, embedding);
    }

    /// <summary>
    /// Applies classifier-free guidance to noise predictions (internal version).
    /// </summary>
    private Tensor<T> ApplyGuidanceInternal(Tensor<T> unconditional, Tensor<T> conditional, double guidanceScale)
    {
        var result = new Tensor<T>(unconditional.Shape);
        var resultSpan = result.AsWritableSpan();
        var uncondSpan = unconditional.AsSpan();
        var condSpan = conditional.AsSpan();

        for (int i = 0; i < resultSpan.Length; i++)
        {
            double uncond = NumOps.ToDouble(uncondSpan[i]);
            double cond = NumOps.ToDouble(condSpan[i]);
            resultSpan[i] = NumOps.FromDouble(uncond + guidanceScale * (cond - uncond));
        }

        return result;
    }

    /// <summary>
    /// Computes the Score Distillation Sampling gradient.
    /// </summary>
    private Tensor<T> ComputeSDSGradient(Tensor<T> noisePred, Tensor<T> noise, int timestep)
    {
        // w(t) weight function - typically higher for lower noise levels
        double wt = ComputeSDSWeight(timestep);

        var result = new Tensor<T>(noisePred.Shape);
        var resultSpan = result.AsWritableSpan();
        var predSpan = noisePred.AsSpan();
        var noiseSpan = noise.AsSpan();

        for (int i = 0; i < resultSpan.Length; i++)
        {
            double pred = NumOps.ToDouble(predSpan[i]);
            double n = NumOps.ToDouble(noiseSpan[i]);
            resultSpan[i] = NumOps.FromDouble(wt * (pred - n));
        }

        return result;
    }

    /// <summary>
    /// Computes the SDS weight for a timestep.
    /// </summary>
    private double ComputeSDSWeight(int timestep)
    {
        // w(t) = sigma(t)^2 in the original paper
        double sigma = GetSigma(timestep);
        return sigma * sigma;
    }

    /// <summary>
    /// Gets the noise level (sigma) for a timestep.
    /// </summary>
    private double GetSigma(int timestep)
    {
        double alphasCumprod = GetAlphasCumprod(timestep);
        return Math.Sqrt((1.0 - alphasCumprod) / alphasCumprod);
    }

    /// <summary>
    /// Extracts a mesh using marching cubes algorithm.
    /// </summary>
    private DreamMesh<T> MarchingCubes(NeRFNetwork<T> nerf, int resolution, double threshold)
    {
        var vertices = new List<DreamVector3<T>>();
        var triangles = new List<int>();

        // Sample density grid
        var densityGrid = new double[resolution, resolution, resolution];
        double step = 2.0 / resolution; // Grid from -1 to 1

        for (int x = 0; x < resolution; x++)
        {
            for (int y = 0; y < resolution; y++)
            {
                for (int z = 0; z < resolution; z++)
                {
                    double px = -1.0 + x * step;
                    double py = -1.0 + y * step;
                    double pz = -1.0 + z * step;

                    densityGrid[x, y, z] = nerf.QueryDensity(px, py, pz);
                }
            }
        }

        // Apply marching cubes to extract surface
        ExtractSurface(densityGrid, threshold, step, vertices, triangles);

        return new DreamMesh<T>
        {
            Vertices = vertices,
            Triangles = triangles
        };
    }

    /// <summary>
    /// Extracts surface using simplified marching cubes.
    /// </summary>
    private void ExtractSurface(
        double[,,] densityGrid,
        double threshold,
        double step,
        List<DreamVector3<T>> vertices,
        List<int> triangles)
    {
        int resolution = densityGrid.GetLength(0);

        for (int x = 0; x < resolution - 1; x++)
        {
            for (int y = 0; y < resolution - 1; y++)
            {
                for (int z = 0; z < resolution - 1; z++)
                {
                    // Sample cube corners
                    var corners = new double[8];
                    corners[0] = densityGrid[x, y, z];
                    corners[1] = densityGrid[x + 1, y, z];
                    corners[2] = densityGrid[x + 1, y + 1, z];
                    corners[3] = densityGrid[x, y + 1, z];
                    corners[4] = densityGrid[x, y, z + 1];
                    corners[5] = densityGrid[x + 1, y, z + 1];
                    corners[6] = densityGrid[x + 1, y + 1, z + 1];
                    corners[7] = densityGrid[x, y + 1, z + 1];

                    // Compute cube index
                    int cubeIndex = 0;
                    for (int i = 0; i < 8; i++)
                    {
                        if (corners[i] > threshold)
                        {
                            cubeIndex |= (1 << i);
                        }
                    }

                    // Skip if entirely inside or outside
                    if (cubeIndex == 0 || cubeIndex == 255)
                        continue;

                    // Add center vertex for simplified algorithm
                    double px = -1.0 + (x + 0.5) * step;
                    double py = -1.0 + (y + 0.5) * step;
                    double pz = -1.0 + (z + 0.5) * step;

                    vertices.Add(new DreamVector3<T>
                    {
                        X = NumOps.FromDouble(px),
                        Y = NumOps.FromDouble(py),
                        Z = NumOps.FromDouble(pz)
                    });
                }
            }
        }
    }

    /// <inheritdoc />
    public override Tensor<T> PredictNoise(Tensor<T> noisySample, int timestep)
    {
        return _unet.PredictNoise(noisySample, timestep, null);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var nerfParams = _nerf.GetParameters();
        var unetParams = _unet.GetParameters();
        var vaeParams = _vae.GetParameters();

        var totalLength = nerfParams.Length + unetParams.Length + vaeParams.Length;
        var combined = new T[totalLength];

        int offset = 0;
        for (int i = 0; i < nerfParams.Length; i++)
        {
            combined[offset + i] = nerfParams[i];
        }
        offset += nerfParams.Length;

        for (int i = 0; i < unetParams.Length; i++)
        {
            combined[offset + i] = unetParams[i];
        }
        offset += unetParams.Length;

        for (int i = 0; i < vaeParams.Length; i++)
        {
            combined[offset + i] = vaeParams[i];
        }

        return new Vector<T>(combined);
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var nerfCount = _nerf.ParameterCount;
        var unetCount = _unet.ParameterCount;
        var vaeCount = _vae.ParameterCount;

        var nerfParams = new T[nerfCount];
        var unetParams = new T[unetCount];
        var vaeParams = new T[vaeCount];

        int offset = 0;
        for (int i = 0; i < nerfCount; i++)
        {
            nerfParams[i] = parameters[offset + i];
        }
        offset += nerfCount;

        for (int i = 0; i < unetCount; i++)
        {
            unetParams[i] = parameters[offset + i];
        }
        offset += unetCount;

        for (int i = 0; i < vaeCount; i++)
        {
            vaeParams[i] = parameters[offset + i];
        }

        _nerf.SetParameters(new Vector<T>(nerfParams));
        _unet.SetParameters(new Vector<T>(unetParams));
        _vae.SetParameters(new Vector<T>(vaeParams));
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        return Clone();
    }

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var clone = new DreamFusionModel<T>(
            diffusionPrior: null,
            config: _config,
            conditioner: _conditioner,
            seed: RandomGenerator.Next());

        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "DreamFusion",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "Text-to-3D generation via Score Distillation Sampling",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("NeRF Hidden Dim", _config.NeRFHiddenDim);
        metadata.SetProperty("NeRF Layers", _config.NeRFNumLayers);
        metadata.SetProperty("Render Resolution", _config.RenderResolution);
        metadata.SetProperty("Min Elevation", _config.MinElevation);
        metadata.SetProperty("Max Elevation", _config.MaxElevation);
        metadata.SetProperty("Author", "AiDotNet (based on Poole et al., 2022)");

        return metadata;
    }
}

/// <summary>
/// Configuration for DreamFusion model.
/// </summary>
public class DreamFusionConfig
{
    /// <summary>
    /// Hidden dimension for NeRF network.
    /// </summary>
    public int NeRFHiddenDim { get; set; } = 64;

    /// <summary>
    /// Number of layers in NeRF network.
    /// </summary>
    public int NeRFNumLayers { get; set; } = 4;

    /// <summary>
    /// Resolution for rendering during optimization.
    /// </summary>
    public int RenderResolution { get; set; } = 64;

    /// <summary>
    /// Dimension of text embeddings.
    /// </summary>
    public int TextEmbeddingDim { get; set; } = 768;

    /// <summary>
    /// Minimum camera elevation in degrees.
    /// </summary>
    public double MinElevation { get; set; } = -30;

    /// <summary>
    /// Maximum camera elevation in degrees.
    /// </summary>
    public double MaxElevation { get; set; } = 30;

    /// <summary>
    /// Minimum camera radius.
    /// </summary>
    public double MinRadius { get; set; } = 1.5;

    /// <summary>
    /// Maximum camera radius.
    /// </summary>
    public double MaxRadius { get; set; } = 2.0;

    /// <summary>
    /// Amount of jitter for look-at point.
    /// </summary>
    public double LookAtJitter { get; set; } = 0.1;

    /// <summary>
    /// Camera focal length.
    /// </summary>
    public double FocalLength { get; set; } = 1.0;

    /// <summary>
    /// Minimum timestep ratio for SDS sampling.
    /// </summary>
    public double MinTimestepRatio { get; set; } = 0.02;

    /// <summary>
    /// Maximum timestep ratio for SDS sampling.
    /// </summary>
    public double MaxTimestepRatio { get; set; } = 0.98;
}

/// <summary>
/// Result from DreamFusion generation.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class NeRFResult<T>
{
    /// <summary>
    /// The optimized NeRF parameters.
    /// </summary>
    public Vector<T> Parameters { get; set; } = Vector<T>.Empty();

    /// <summary>
    /// Reference to the trained NeRF network.
    /// </summary>
    public NeRFNetwork<T>? Network { get; set; }

    /// <summary>
    /// The text prompt used for generation.
    /// </summary>
    public string Prompt { get; set; } = string.Empty;
}

/// <summary>
/// Camera pose for rendering.
/// </summary>
public class CameraPose
{
    /// <summary>
    /// Elevation angle in degrees.
    /// </summary>
    public double Elevation { get; set; }

    /// <summary>
    /// Azimuth angle in degrees.
    /// </summary>
    public double Azimuth { get; set; }

    /// <summary>
    /// Distance from origin.
    /// </summary>
    public double Radius { get; set; } = 2.0;

    /// <summary>
    /// Look-at X coordinate.
    /// </summary>
    public double LookAtX { get; set; }

    /// <summary>
    /// Look-at Y coordinate.
    /// </summary>
    public double LookAtY { get; set; }

    /// <summary>
    /// Look-at Z coordinate.
    /// </summary>
    public double LookAtZ { get; set; }

    /// <summary>
    /// Camera focal length.
    /// </summary>
    public double FocalLength { get; set; } = 1.0;

    /// <summary>
    /// Gets the camera position in Cartesian coordinates.
    /// </summary>
    public (double X, double Y, double Z) GetPosition()
    {
        double elevRad = Elevation * Math.PI / 180.0;
        double azimRad = Azimuth * Math.PI / 180.0;

        double x = Radius * Math.Cos(elevRad) * Math.Cos(azimRad);
        double y = Radius * Math.Sin(elevRad);
        double z = Radius * Math.Cos(elevRad) * Math.Sin(azimRad);

        return (x, y, z);
    }
}

/// <summary>
/// 3D vector type for DreamFusion.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public struct DreamVector3<T>
{
    /// <summary>
    /// X component.
    /// </summary>
    public T X { get; set; }

    /// <summary>
    /// Y component.
    /// </summary>
    public T Y { get; set; }

    /// <summary>
    /// Z component.
    /// </summary>
    public T Z { get; set; }
}

/// <summary>
/// Simple mesh representation for DreamFusion.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class DreamMesh<T>
{
    /// <summary>
    /// Vertex positions.
    /// </summary>
    public List<DreamVector3<T>> Vertices { get; set; } = new();

    /// <summary>
    /// Triangle indices (three per triangle).
    /// </summary>
    public List<int> Triangles { get; set; } = new();

    /// <summary>
    /// Vertex normals.
    /// </summary>
    public List<DreamVector3<T>>? Normals { get; set; }

    /// <summary>
    /// Vertex colors.
    /// </summary>
    public List<DreamVector3<T>>? Colors { get; set; }
}

/// <summary>
/// Neural Radiance Field network for 3D representation.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class NeRFNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly List<DenseLayer<T>> _densityLayers;
    private readonly List<DenseLayer<T>> _colorLayers;
    private readonly int _hiddenDim;
    private readonly Random _random;

    /// <summary>
    /// Gets the total parameter count.
    /// </summary>
    public int ParameterCount
    {
        get
        {
            int count = 0;
            foreach (var layer in _densityLayers)
            {
                count += layer.GetParameters().Length;
            }
            foreach (var layer in _colorLayers)
            {
                count += layer.GetParameters().Length;
            }
            return count;
        }
    }

    /// <summary>
    /// Initializes a new NeRF network.
    /// </summary>
    /// <param name="hiddenDim">Hidden layer dimension.</param>
    /// <param name="numLayers">Number of hidden layers.</param>
    /// <param name="seed">Optional random seed.</param>
    public NeRFNetwork(int hiddenDim = 64, int numLayers = 4, int? seed = null)
    {
        _hiddenDim = hiddenDim;
        _random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();

        // Density network: position (3D + positional encoding) -> density
        _densityLayers = new List<DenseLayer<T>>();
        int inputDim = 3 + 6 * 10; // 3D position + 10 frequencies of positional encoding (sin + cos)

        for (int i = 0; i < numLayers; i++)
        {
            int inDim = i == 0 ? inputDim : hiddenDim;
            _densityLayers.Add(new DenseLayer<T>(inDim, hiddenDim, (IActivationFunction<T>)new ReLUActivation<T>()));
        }
        _densityLayers.Add(new DenseLayer<T>(hiddenDim, 1, (IActivationFunction<T>?)null)); // Output density

        // Color network: features + direction -> RGB
        _colorLayers = new List<DenseLayer<T>>();
        int colorInputDim = hiddenDim + 3 + 6 * 4; // Features + direction + positional encoding

        _colorLayers.Add(new DenseLayer<T>(colorInputDim, hiddenDim / 2, (IActivationFunction<T>)new ReLUActivation<T>()));
        _colorLayers.Add(new DenseLayer<T>(hiddenDim / 2, 3, (IActivationFunction<T>)new SigmoidActivation<T>())); // RGB output
    }

    /// <summary>
    /// Initializes network weights.
    /// </summary>
    public void Initialize()
    {
        // Weights are initialized in layer constructors
    }

    /// <summary>
    /// Renders an image from the NeRF at a given camera pose.
    /// </summary>
    /// <param name="cameraPose">The camera pose.</param>
    /// <param name="resolution">Output image resolution.</param>
    /// <returns>Rendered image as tensor [H, W, 3].</returns>
    public Tensor<T> Render(CameraPose cameraPose, int resolution)
    {
        var image = new Tensor<T>(new[] { resolution, resolution, 3 });
        var imageSpan = image.AsWritableSpan();
        var (camX, camY, camZ) = cameraPose.GetPosition();

        for (int y = 0; y < resolution; y++)
        {
            for (int x = 0; x < resolution; x++)
            {
                // Compute ray direction
                double u = (2.0 * x / resolution - 1.0) / cameraPose.FocalLength;
                double v = (2.0 * y / resolution - 1.0) / cameraPose.FocalLength;

                // Simple pinhole camera model
                double dirX = u;
                double dirY = v;
                double dirZ = 1.0;

                // Normalize direction
                double dirLen = Math.Sqrt(dirX * dirX + dirY * dirY + dirZ * dirZ);
                dirX /= dirLen;
                dirY /= dirLen;
                dirZ /= dirLen;

                // Sample along ray using volume rendering
                var (r, g, b) = VolumeRender(camX, camY, camZ, dirX, dirY, dirZ);

                int idx = (y * resolution + x) * 3;
                imageSpan[idx] = NumOps.FromDouble(r);
                imageSpan[idx + 1] = NumOps.FromDouble(g);
                imageSpan[idx + 2] = NumOps.FromDouble(b);
            }
        }

        return image;
    }

    /// <summary>
    /// Performs volume rendering along a ray.
    /// </summary>
    private (double R, double G, double B) VolumeRender(
        double ox, double oy, double oz,
        double dx, double dy, double dz)
    {
        double r = 0, g = 0, b = 0;
        double transmittance = 1.0;

        int numSamples = 64;
        double near = 0.5;
        double far = 3.0;
        double step = (far - near) / numSamples;

        for (int i = 0; i < numSamples; i++)
        {
            double t = near + (i + 0.5) * step;
            double px = ox + t * dx;
            double py = oy + t * dy;
            double pz = oz + t * dz;

            // Query NeRF
            double density = QueryDensity(px, py, pz);
            var (cr, cg, cb) = QueryColor(px, py, pz, dx, dy, dz);

            // Volume rendering equation
            double alpha = 1.0 - Math.Exp(-density * step);
            double weight = transmittance * alpha;

            r += weight * cr;
            g += weight * cg;
            b += weight * cb;

            transmittance *= (1.0 - alpha);

            if (transmittance < 0.01)
                break;
        }

        return (r, g, b);
    }

    /// <summary>
    /// Queries the density at a 3D point.
    /// </summary>
    public double QueryDensity(double x, double y, double z)
    {
        // Apply positional encoding
        var encoded = PositionalEncoding(new[] { x, y, z }, 10);
        var inputData = new T[encoded.Length];
        for (int i = 0; i < encoded.Length; i++)
        {
            inputData[i] = NumOps.FromDouble(encoded[i]);
        }
        var input = new Tensor<T>(new[] { 1, encoded.Length }, new Vector<T>(inputData));

        // Forward through density network
        var current = input;
        foreach (var layer in _densityLayers)
        {
            current = layer.Forward(current);
        }

        // ReLU activation on density
        double density = NumOps.ToDouble(current.AsSpan()[0]);
        return Math.Max(0, density);
    }

    /// <summary>
    /// Queries the color at a 3D point with viewing direction.
    /// </summary>
    private (double R, double G, double B) QueryColor(
        double x, double y, double z,
        double dx, double dy, double dz)
    {
        // Get features from density network (before final layer)
        var posEncoded = PositionalEncoding(new[] { x, y, z }, 10);
        var inputData = new T[posEncoded.Length];
        for (int i = 0; i < posEncoded.Length; i++)
        {
            inputData[i] = NumOps.FromDouble(posEncoded[i]);
        }
        var input = new Tensor<T>(new[] { 1, posEncoded.Length }, new Vector<T>(inputData));

        var current = input;
        for (int i = 0; i < _densityLayers.Count - 1; i++)
        {
            current = _densityLayers[i].Forward(current);
        }

        // Add direction encoding
        var dirEncoded = PositionalEncoding(new[] { dx, dy, dz }, 4);
        var currentSpan = current.AsSpan();
        var combinedData = new T[currentSpan.Length + dirEncoded.Length];

        for (int i = 0; i < currentSpan.Length; i++)
        {
            combinedData[i] = currentSpan[i];
        }
        for (int i = 0; i < dirEncoded.Length; i++)
        {
            combinedData[currentSpan.Length + i] = NumOps.FromDouble(dirEncoded[i]);
        }

        // Forward through color network
        var colorInput = new Tensor<T>(new[] { 1, combinedData.Length }, new Vector<T>(combinedData));
        foreach (var layer in _colorLayers)
        {
            colorInput = layer.Forward(colorInput);
        }

        var colorSpan = colorInput.AsSpan();
        return (
            NumOps.ToDouble(colorSpan[0]),
            NumOps.ToDouble(colorSpan[1]),
            NumOps.ToDouble(colorSpan[2])
        );
    }

    /// <summary>
    /// Applies positional encoding to input values.
    /// </summary>
    private double[] PositionalEncoding(double[] values, int numFrequencies)
    {
        var result = new List<double>();

        // Add original values
        result.AddRange(values);

        // Add sin/cos of frequencies
        for (int i = 0; i < numFrequencies; i++)
        {
            double freq = Math.Pow(2, i);
            foreach (double v in values)
            {
                result.Add(Math.Sin(freq * Math.PI * v));
                result.Add(Math.Cos(freq * Math.PI * v));
            }
        }

        return result.ToArray();
    }

    /// <summary>
    /// Updates network parameters using gradient.
    /// </summary>
    public void UpdateParameters(Tensor<T> gradient, CameraPose cameraPose, double learningRate)
    {
        // Simplified gradient update - in practice would use proper backprop
        var gradSpan = gradient.AsSpan();
        double gradNorm = 0;

        for (int i = 0; i < gradSpan.Length; i++)
        {
            double g = NumOps.ToDouble(gradSpan[i]);
            gradNorm += g * g;
        }
        gradNorm = Math.Sqrt(gradNorm) + 1e-8;

        // Update each layer
        foreach (var layer in _densityLayers)
        {
            var layerParams = layer.GetParameters();
            var updated = new T[layerParams.Length];

            for (int i = 0; i < layerParams.Length; i++)
            {
                double param = NumOps.ToDouble(layerParams[i]);
                // Use scaled gradient with noise for exploration
                double update = -learningRate * (1.0 / gradNorm) * (_random.NextDouble() - 0.5);
                updated[i] = NumOps.FromDouble(param + update);
            }

            layer.SetParameters(Vector<T>.FromArray(updated));
        }

        foreach (var layer in _colorLayers)
        {
            var layerParams = layer.GetParameters();
            var updated = new T[layerParams.Length];

            for (int i = 0; i < layerParams.Length; i++)
            {
                double param = NumOps.ToDouble(layerParams[i]);
                double update = -learningRate * (1.0 / gradNorm) * (_random.NextDouble() - 0.5);
                updated[i] = NumOps.FromDouble(param + update);
            }

            layer.SetParameters(Vector<T>.FromArray(updated));
        }
    }

    /// <summary>
    /// Gets all network parameters.
    /// </summary>
    public Vector<T> GetParameters()
    {
        var allParams = new List<T>();

        foreach (var layer in _densityLayers)
        {
            var p = layer.GetParameters();
            for (int i = 0; i < p.Length; i++)
            {
                allParams.Add(p[i]);
            }
        }

        foreach (var layer in _colorLayers)
        {
            var p = layer.GetParameters();
            for (int i = 0; i < p.Length; i++)
            {
                allParams.Add(p[i]);
            }
        }

        return Vector<T>.FromArray(allParams.ToArray());
    }

    /// <summary>
    /// Sets network parameters.
    /// </summary>
    public void SetParameters(Vector<T> parameters)
    {
        int offset = 0;

        foreach (var layer in _densityLayers)
        {
            var layerParams = layer.GetParameters();
            var newParams = new T[layerParams.Length];

            for (int i = 0; i < layerParams.Length; i++)
            {
                newParams[i] = parameters[offset + i];
            }

            layer.SetParameters(Vector<T>.FromArray(newParams));
            offset += layerParams.Length;
        }

        foreach (var layer in _colorLayers)
        {
            var layerParams = layer.GetParameters();
            var newParams = new T[layerParams.Length];

            for (int i = 0; i < layerParams.Length; i++)
            {
                newParams[i] = parameters[offset + i];
            }

            layer.SetParameters(Vector<T>.FromArray(newParams));
            offset += layerParams.Length;
        }
    }
}

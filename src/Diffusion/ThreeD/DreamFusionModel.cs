using System.Diagnostics.CodeAnalysis;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.ThreeD;

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
/// <example>
/// <code>
/// var options = new DiffusionModelOptions&lt;float&gt; { LatentChannels = 4, DefaultInferenceSteps = 50 };
/// var noise = Tensor&lt;float&gt;.CreateRandom(new[] { 1, 4, 8, 8 });
/// var trainX = Tensor&lt;float&gt;.CreateRandom(4, 8);
/// var trainY = Tensor&lt;float&gt;.CreateRandom(4, 2);
/// var result = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
///     .ConfigureModel(new DreamFusionModel&lt;float&gt;())
///     .Build(trainX, trainY);
/// var rendering = result.Predict(noise);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.ThreeD)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Generation)]
[ModelTask(ModelTask.ThreeDGeneration)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("DreamFusion: Text-to-3D using 2D Diffusion", "https://arxiv.org/abs/2209.14988", Year = 2022, Authors = "Poole et al.")]
public partial class DreamFusionModel<T> : LatentDiffusionModelBase<T>
{
    /// <inheritdoc />
    /// <remarks>Registration order is serialization order, and matches the
    /// concatenation the previous hand-written GetParameters performed.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_nerf);
        RegisterParameterComponent(_unet);
        RegisterParameterComponent(_vae);
    }

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
        UNetNoisePredictor<T>? unet = null,
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

        InitializeLayers(unet, seed);
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the U-Net, VAE, and NeRF network layers.
    /// </summary>
    [MemberNotNull(nameof(_unet), nameof(_vae), nameof(_nerf))]
    private void InitializeLayers(UNetNoisePredictor<T>? unet, int? seed)
    {
        _unet = unet ?? new UNetNoisePredictor<T>(
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

        // Simple hash-based embedding for fallback
        // A stable hash: string.GetHashCode() changes per process (#2361).
        int hash = AiDotNet.Helpers.StableHash.Seed(text);
        var rng = RandomHelper.CreateSeededRandom(hash);

        var values = new double[embedDim];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = (rng.NextDouble() * 2 - 1) * 0.1;
        }

        return new Tensor<double>(new[] { 1, embedDim }, new Vector<double>(values)).Cast<T>();
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
        var values = new double[template.Length];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = RandomGenerator.NextGaussian();
        }

        return new Tensor<double>(template._shape, new Vector<double>(values)).Cast<T>();
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

        // result = sqrtAlpha * image + sqrtOneMinusAlpha * noise
        var sqrtAlphaT = NumOps.FromDouble(sqrtAlpha);
        var sqrtOneMinusAlphaT = NumOps.FromDouble(sqrtOneMinusAlpha);
        var scaledImage = Engine.TensorMultiplyScalar<T>(image, sqrtAlphaT);
        var scaledNoise = Engine.TensorMultiplyScalar<T>(noise, sqrtOneMinusAlphaT);
        return Engine.TensorAdd<T>(scaledImage, scaledNoise);
    }

    /// <summary>
    /// The cumulative product alpha_bar_t = prod_{i &lt;= t} (1 - beta_i) of the scaled-linear schedule
    /// beta_i = (sqrt(beta_start) + i / (T - 1) * (sqrt(beta_end) - sqrt(beta_start)))^2 (the Stable Diffusion
    /// prior's schedule), computed once in double.
    /// </summary>
    internal static double GetAlphasCumprod(int timestep)
    {
        int index = Math.Max(0, Math.Min(DEFAULT_TIMESTEPS - 1, timestep));
        return AlphasCumprodTable[index];
    }

    private static readonly double[] AlphasCumprodTable = BuildAlphasCumprodTable();

    private static double[] BuildAlphasCumprodTable()
    {
        var table = new double[DEFAULT_TIMESTEPS];
        double sqrtStart = Math.Sqrt(DEFAULT_BETA_START);
        double sqrtEnd = Math.Sqrt(DEFAULT_BETA_END);
        double product = 1.0;
        for (int i = 0; i < DEFAULT_TIMESTEPS; i++)
        {
            double sqrtBeta = sqrtStart + ((sqrtEnd - sqrtStart) * i / (DEFAULT_TIMESTEPS - 1));
            product *= 1.0 - (sqrtBeta * sqrtBeta);
            table[i] = product;
        }

        return table;
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
        // CFG: guided = uncond + scale * (cond - uncond)
        var scaleT = NumOps.FromDouble(guidanceScale);
        var diff = Engine.TensorSubtract<T>(conditional, unconditional);
        var scaled = Engine.TensorMultiplyScalar<T>(diff, scaleT);
        return Engine.TensorAdd<T>(unconditional, scaled);
    }

    /// <summary>
    /// Computes the Score Distillation Sampling gradient.
    /// </summary>
    private Tensor<T> ComputeSDSGradient(Tensor<T> noisePred, Tensor<T> noise, int timestep)
    {
        // w(t) weight function - typically higher for lower noise levels
        double wt = ComputeSDSWeight(timestep);

        // gradient = wt * (noisePred - noise)
        var wtT = NumOps.FromDouble(wt);
        var diff = Engine.TensorSubtract<T>(noisePred, noise);
        return Engine.TensorMultiplyScalar<T>(diff, wtT);
    }

    /// <summary>
    /// The SDS weight w(t) = sigma_t^2 = 1 - alpha_bar_t of DreamFusion (Poole et al., 2022, eq. 3), with the
    /// variance-preserving sigma_t^2 = 1 - alpha_t^2 of the diffusion prior.
    /// </summary>
    internal static double ComputeSDSWeight(int timestep) => 1.0 - GetAlphasCumprod(timestep);


    /// <summary>
    /// Extracts a mesh using marching cubes algorithm.
    /// </summary>
    private DreamMesh<T> MarchingCubes(NeRFNetwork<T> nerf, int resolution, double threshold)
    {
        var vertices = new List<DreamVector3<T>>();
        var triangles = new List<int>();

        // Sample the density grid (one batched query of every grid point)
        double step = 2.0 / resolution; // Grid from -1 to 1
        var densityGrid = nerf.QueryDensityGrid(resolution, -1.0, step);

        // Apply marching cubes to extract surface
        ExtractSurface(densityGrid, threshold, step, vertices, triangles);

        return new DreamMesh<T>
        {
            Vertices = vertices,
            Triangles = triangles
        };
    }

    /// <summary>
    /// Extracts the iso-surface {density = threshold} of the grid (grid index i at coordinate -1 + i * step) by
    /// marching tetrahedra, appending its vertices and triangles.
    /// </summary>
    internal static void ExtractSurface(
        double[,,] densityGrid,
        double threshold,
        double step,
        List<DreamVector3<T>> vertices,
        List<int> triangles)
    {
        int vertexOffset = vertices.Count;
        var (coordinates, surfaceTriangles) = MarchingTetrahedra.Extract(densityGrid, threshold, -1.0, step);
        foreach (int index in surfaceTriangles)
            triangles.Add(vertexOffset + index);
        if (coordinates.Count == 0)
            return;

        // One bulk conversion of all vertex coordinates to T.
        var converted = new Tensor<double>(new[] { coordinates.Count }, new Vector<double>(coordinates.ToArray()))
            .Cast<T>().ToArray();
        for (int v = 0; v + 2 < converted.Length; v += 3)
            vertices.Add(new DreamVector3<T> { X = converted[v], Y = converted[v + 1], Z = converted[v + 2] });
    }

    /// <inheritdoc />
    public override Tensor<T> PredictNoise(Tensor<T> noisySample, int timestep)
    {
        return _unet.PredictNoise(noisySample, timestep, null);
    }



    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "DreamFusion",
            Version = "1.0",
            Description = "Text-to-3D generation via Score Distillation Sampling",
            FeatureCount = (int)System.Math.Min((long)int.MaxValue, ParameterCount),
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
/// <remarks>
/// A NeRF MLP (Mildenhall et al., 2020) rendered by batched volume rendering: every ray of a chunk is sampled at
/// <see cref="SamplesPerRay"/> stratified depths in [0.5, 3], the MLP is evaluated on all samples at once, and the
/// colour is composited with <c>C = sum_i T_i (1 - exp(-sigma_i delta)) c_i</c>,
/// <c>T_i = exp(-sum_{j &lt; i} sigma_j delta)</c>. Because the whole render is made of engine ops, the SDS update
/// back-propagates the score-distillation gradient through the renderer into the MLP weights (DreamFusion,
/// Poole et al., 2022), instead of perturbing them at random.
/// </remarks>
public class NeRFNetwork<T> : IParameterSource<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>Samples taken along every ray.</summary>
    internal const int SamplesPerRay = 64;

    private const double NearPlane = 0.5;
    private const double FarPlane = 3.0;
    private const int PositionFrequencies = 10;
    private const int DirectionFrequencies = 4;
    private const int RaysPerChunk = 256;
    private const int PointsPerChunk = 16384;

    private readonly List<DenseLayer<T>> _densityLayers;
    private readonly List<DenseLayer<T>> _colorLayers;
    private readonly int _hiddenDim;

    // Sample depths [1, S, 1] and the strictly-upper-triangular [S, S] matrix whose product with the per-sample
    // optical depth gives the exclusive prefix sum sum_{j < i} sigma_j delta.
    private readonly Tensor<T> _sampleDepths;
    private readonly Tensor<T> _exclusivePrefix;

    private IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Gets the total parameter count.
    /// </summary>
    public long ParameterCount
    {
        get
        {
            long count = 0;
            foreach (var layer in AllLayers())
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
    /// <param name="seed">Optional random seed; when set, every layer's initialization is seeded from it.</param>
    public NeRFNetwork(int hiddenDim = 64, int numLayers = 4, int? seed = null)
    {
        _hiddenDim = hiddenDim;
        var seedSource = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : null;
        DenseLayer<T> Seeded(DenseLayer<T> layer)
        {
            if (seedSource is not null)
                layer.RandomSeed = seedSource.Next();
            return layer;
        }

        // Density network: position (3D + positional encoding) -> density
        _densityLayers = new List<DenseLayer<T>>();
        int inputDim = EncodedDim(PositionFrequencies);

        for (int i = 0; i < numLayers; i++)
        {
            int inDim = i == 0 ? inputDim : hiddenDim;
            var layer = Seeded(new DenseLayer<T>(hiddenDim, (IActivationFunction<T>)new ReLUActivation<T>()));
            // Pre-resolve from the ctor-known input dim so ParameterCount, GetParameters,
            // SetParameters, and Clone reflect the correct layout before any render.
            layer.ResolveFromShape(new[] { 1, inDim });
            _densityLayers.Add(layer);
        }
        var densityOut = Seeded(new DenseLayer<T>(1, (IActivationFunction<T>?)null)); // Output density
        densityOut.ResolveFromShape(new[] { 1, hiddenDim });
        _densityLayers.Add(densityOut);

        // Color network: features + direction -> RGB
        _colorLayers = new List<DenseLayer<T>>();
        int colorInputDim = hiddenDim + EncodedDim(DirectionFrequencies); // Features + encoded direction

        var colorMid = Seeded(new DenseLayer<T>(hiddenDim / 2, (IActivationFunction<T>)new ReLUActivation<T>()));
        colorMid.ResolveFromShape(new[] { 1, colorInputDim });
        _colorLayers.Add(colorMid);

        var colorOut = Seeded(new DenseLayer<T>(3, (IActivationFunction<T>)new SigmoidActivation<T>())); // RGB output
        colorOut.ResolveFromShape(new[] { 1, hiddenDim / 2 });
        _colorLayers.Add(colorOut);

        double delta = SampleSpacing;
        _sampleDepths = Engine.Reshape(
            Engine.TensorAddScalar(
                Engine.TensorMultiplyScalar(Engine.TensorArange<T>(0, SamplesPerRay, 1), NumOps.FromDouble(delta)),
                NumOps.FromDouble(NearPlane + (0.5 * delta))),
            new[] { 1, SamplesPerRay, 1 });

        var prefix = new double[SamplesPerRay * SamplesPerRay];
        for (int j = 0; j < SamplesPerRay; j++)
        {
            for (int i = j + 1; i < SamplesPerRay; i++)
                prefix[(j * SamplesPerRay) + i] = 1.0;
        }
        _exclusivePrefix = new Tensor<double>(new[] { SamplesPerRay, SamplesPerRay }, new Vector<double>(prefix)).Cast<T>();
    }

    /// <summary>The spacing between consecutive samples along a ray.</summary>
    internal static double SampleSpacing => (FarPlane - NearPlane) / SamplesPerRay;

    /// <summary>The width of the positional encoding of a 3-vector with <paramref name="frequencies"/> octaves.</summary>
    internal static int EncodedDim(int frequencies) => 3 + (6 * frequencies);

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
    /// <returns>Rendered image as tensor [H, W, 3] (row 0 at the top).</returns>
    public Tensor<T> Render(CameraPose cameraPose, int resolution)
    {
        var (origin, directions) = BuildRays(cameraPose, resolution);
        int rays = resolution * resolution;
        var chunks = new List<Tensor<T>>();
        for (int start = 0; start < rays; start += RaysPerChunk)
        {
            int count = Math.Min(RaysPerChunk, rays - start);
            chunks.Add(RenderRays(origin, Engine.TensorNarrow(directions, 0, start, count)));
        }

        return Engine.Reshape(Engine.TensorConcatenate(chunks.ToArray(), 0), new[] { resolution, resolution, 3 });
    }

    /// <summary>
    /// Builds the camera origin [1, 1, 3] and unit ray directions [resolution * resolution, 3] (row-major pixels)
    /// of a pinhole camera at the pose's position looking at its look-at point, with world +Y up.
    /// </summary>
    internal (Tensor<T> Origin, Tensor<T> Directions) BuildRays(CameraPose cameraPose, int resolution)
    {
        if (resolution <= 0)
            throw new ArgumentOutOfRangeException(nameof(resolution), "Resolution must be positive.");

        var (px, py, pz) = cameraPose.GetPosition();
        var forward = Normalize(cameraPose.LookAtX - px, cameraPose.LookAtY - py, cameraPose.LookAtZ - pz);

        // right = forward x worldUp with worldUp = +Y; looking straight up or down falls back to worldUp = +Z.
        var right = Normalize(-forward[2], 0.0, forward[0]);
        if (Math.Abs(forward[0]) < 1e-9 && Math.Abs(forward[2]) < 1e-9)
            right = Normalize(forward[1], -forward[0], 0.0);
        var up = new[]
        {
            (right[1] * forward[2]) - (right[2] * forward[1]),
            (right[2] * forward[0]) - (right[0] * forward[2]),
            (right[0] * forward[1]) - (right[1] * forward[0])
        };

        // Pixel centres: u = (2 (x + 0.5) / R - 1) / f to the right, v = (1 - 2 (y + 0.5) / R) / f upwards.
        double scale = 2.0 / (resolution * cameraPose.FocalLength);
        double offset = ((1.0 / resolution) - 1.0) / cameraPose.FocalLength;
        var pixels = Engine.TensorArange<T>(0, resolution, 1);
        var u = Engine.TensorAddScalar(Engine.TensorMultiplyScalar(pixels, NumOps.FromDouble(scale)), NumOps.FromDouble(offset));
        var v = Engine.TensorNegate(u);

        var grid = new[] { resolution, resolution, 3 };
        var directions = Engine.TensorAdd(
            Engine.TensorAdd(
                Engine.TensorBroadcastTo(
                    Engine.TensorMultiply(Engine.Reshape(u, new[] { 1, resolution, 1 }), Vector3Tensor(right)), grid),
                Engine.TensorBroadcastTo(
                    Engine.TensorMultiply(Engine.Reshape(v, new[] { resolution, 1, 1 }), Vector3Tensor(up)), grid)),
            Engine.TensorBroadcastTo(Vector3Tensor(forward), grid));
        var norm = Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorMultiply(directions, directions), new[] { 2 }, true));
        var unit = Engine.TensorDivide(directions, Engine.TensorBroadcastTo(norm, grid));

        var origin = new Tensor<double>(new[] { 1, 1, 3 }, new Vector<double>(new[] { px, py, pz })).Cast<T>();
        return (origin, Engine.Reshape(unit, new[] { resolution * resolution, 3 }));
    }

    private static double[] Normalize(double x, double y, double z)
    {
        double length = Math.Sqrt((x * x) + (y * y) + (z * z));
        if (length < 1e-12)
            return new[] { 0.0, 0.0, -1.0 };
        return new[] { x / length, y / length, z / length };
    }

    private static Tensor<T> Vector3Tensor(double[] vector)
        => new Tensor<double>(new[] { 1, 1, 3 }, new Vector<double>(vector)).Cast<T>();

    /// <summary>
    /// Volume-renders the rays from <paramref name="origin"/> [1, 1, 3] along <paramref name="directions"/>
    /// [rays, 3] into colours [rays, 3]; differentiable with respect to the MLP weights.
    /// </summary>
    internal Tensor<T> RenderRays(Tensor<T> origin, Tensor<T> directions)
    {
        int rays = directions.Shape[0];
        int samples = rays * SamplesPerRay;

        // Sample points o + t_i d: [rays, S, 3].
        var sampleShape = new[] { rays, SamplesPerRay, 3 };
        var points = Engine.TensorAdd(
            Engine.TensorBroadcastTo(origin, sampleShape),
            Engine.TensorBroadcastTo(
                Engine.TensorMultiply(Engine.Reshape(directions, new[] { rays, 1, 3 }), _sampleDepths), sampleShape));
        var features = DensityFeatures(Encode(Engine.Reshape(points, new[] { samples, 3 }), PositionFrequencies));
        var sigma = Engine.Reshape(
            Engine.TensorReLU(_densityLayers[_densityLayers.Count - 1].Forward(features)), new[] { rays, SamplesPerRay });

        int directionDim = EncodedDim(DirectionFrequencies);
        var encodedDirections = Engine.TensorBroadcastTo(
            Engine.Reshape(Encode(directions, DirectionFrequencies), new[] { rays, 1, directionDim }),
            new[] { rays, SamplesPerRay, directionDim });
        var colour = ColorHead(Engine.TensorConcatenate(
            new[] { features, Engine.Reshape(encodedDirections.Contiguous(), new[] { samples, directionDim }) }, 1));

        // w_i = T_i (1 - exp(-sigma_i delta)), T_i = exp(-sum_{j<i} sigma_j delta).
        var opticalDepth = Engine.TensorMultiplyScalar(sigma, NumOps.FromDouble(SampleSpacing));
        var alpha = Engine.ScalarMinusTensor(NumOps.One, Engine.TensorExp(Engine.TensorNegate(opticalDepth)));
        var transmittance = Engine.TensorExp(Engine.TensorNegate(Engine.TensorMatMul(opticalDepth, _exclusivePrefix)));
        var weights = Engine.Reshape(Engine.TensorMultiply(transmittance, alpha), new[] { rays, SamplesPerRay, 1 });

        return Engine.ReduceSum(
            Engine.TensorMultiply(weights, Engine.Reshape(colour, new[] { rays, SamplesPerRay, 3 })), new[] { 1 }, false);
    }

    /// <summary>
    /// The NeRF positional encoding of <paramref name="values"/> [N, 3]: the values followed, for every octave
    /// k &lt; <paramref name="frequencies"/>, by (sin, cos)(2^k pi v) of each coordinate, giving [N, 3 + 6L].
    /// </summary>
    internal Tensor<T> Encode(Tensor<T> values, int frequencies)
        => NeRFPositionalEncoding.Encode(Engine, values, frequencies);

    private Tensor<T> DensityFeatures(Tensor<T> encoded)
    {
        var current = encoded;
        for (int i = 0; i < _densityLayers.Count - 1; i++)
            current = _densityLayers[i].Forward(current);
        return current;
    }

    private Tensor<T> ColorHead(Tensor<T> input)
    {
        var current = input;
        foreach (var layer in _colorLayers)
            current = layer.Forward(current);
        return current;
    }

    /// <summary>
    /// Queries the density at a 3D point.
    /// </summary>
    public double QueryDensity(double x, double y, double z)
    {
        var point = new Tensor<double>(new[] { 1, 3 }, new Vector<double>(new[] { x, y, z })).Cast<T>();
        return NumOps.ToDouble(QueryDensities(point).AsSpan()[0]);
    }

    /// <summary>
    /// The densities max(0, sigma) [N] at the points [N, 3], evaluated in batches.
    /// </summary>
    internal Tensor<T> QueryDensities(Tensor<T> points)
    {
        int total = points.Shape[0];
        var chunks = new List<Tensor<T>>();
        for (int start = 0; start < total; start += PointsPerChunk)
        {
            int count = Math.Min(PointsPerChunk, total - start);
            var features = DensityFeatures(Encode(Engine.TensorNarrow(points, 0, start, count), PositionFrequencies));
            chunks.Add(Engine.TensorReLU(_densityLayers[_densityLayers.Count - 1].Forward(features)));
        }

        return Engine.Reshape(Engine.TensorConcatenate(chunks.ToArray(), 0), new[] { total });
    }

    /// <summary>
    /// Samples the density on the grid point (origin + i step, origin + j step, origin + k step) for
    /// i, j, k &lt; <paramref name="resolution"/>.
    /// </summary>
    internal double[,,] QueryDensityGrid(int resolution, double origin, double step)
    {
        var axis = Engine.TensorAddScalar(
            Engine.TensorMultiplyScalar(Engine.TensorArange<T>(0, resolution, 1), NumOps.FromDouble(step)),
            NumOps.FromDouble(origin));
        var shape = new[] { resolution, resolution, resolution, 1 };
        var xs = Engine.TensorBroadcastTo(Engine.Reshape(axis, new[] { resolution, 1, 1, 1 }), shape);
        var ys = Engine.TensorBroadcastTo(Engine.Reshape(axis, new[] { 1, resolution, 1, 1 }), shape);
        var zs = Engine.TensorBroadcastTo(Engine.Reshape(axis, new[] { 1, 1, resolution, 1 }), shape);
        var points = Engine.Reshape(
            Engine.TensorConcatenate(new[] { xs, ys, zs }, 3), new[] { resolution * resolution * resolution, 3 });

        var densities = QueryDensities(points).Cast<double>().ToArray();
        var grid = new double[resolution, resolution, resolution];
        Buffer.BlockCopy(densities, 0, grid, 0, densities.Length * sizeof(double));
        return grid;
    }

    /// <summary>
    /// Applies one SDS step: back-propagates the image-space gradient <paramref name="gradient"/>
    /// [R, R, 3] (the score-distillation residual w(t) (eps_hat - eps)) through the volume renderer at
    /// <paramref name="cameraPose"/> and descends the MLP weights by <paramref name="learningRate"/> times it.
    /// </summary>
    /// <remarks>
    /// SDS (Poole et al., 2022, eq. 3) defines grad_theta L = E[w(t) (eps_hat - eps) dx/dtheta]; the loss
    /// sum(x(theta) * g) with g held constant has exactly that gradient, so the tape computes it per ray chunk.
    /// </remarks>
    public void UpdateParameters(Tensor<T> gradient, CameraPose cameraPose, double learningRate)
    {
        int pixels = gradient.Length / 3;
        int resolution = (int)Math.Round(Math.Sqrt(pixels));
        if (resolution * resolution * 3 != gradient.Length)
            throw new ArgumentException(
                $"The SDS gradient must hold a square RGB image [R, R, 3]; got {gradient.Length} values.", nameof(gradient));

        var (origin, directions) = BuildRays(cameraPose, resolution);
        var pixelGradient = Engine.Reshape(gradient, new[] { pixels, 3 });
        var parameters = TrainableParameters();
        var accumulated = new Tensor<T>?[parameters.Count];

        for (int start = 0; start < pixels; start += RaysPerChunk)
        {
            int count = Math.Min(RaysPerChunk, pixels - start);
            using var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<T>();
            var colour = RenderRays(origin, Engine.TensorNarrow(directions, 0, start, count));
            var loss = Engine.ReduceSum(
                Engine.TensorMultiply(colour, Engine.TensorNarrow(pixelGradient, 0, start, count)), null);
            var grads = tape.ComputeGradients(loss, parameters);
            for (int p = 0; p < parameters.Count; p++)
            {
                if (!grads.TryGetValue(parameters[p], out var parameterGradient))
                    continue;
                var previous = accumulated[p];
                accumulated[p] = previous is null ? parameterGradient.Clone() : Engine.TensorAdd(previous, parameterGradient);
            }
        }

        var rate = NumOps.FromDouble(learningRate);
        for (int p = 0; p < parameters.Count; p++)
        {
            var step = accumulated[p];
            if (step is null)
                continue;
            var updated = Engine.TensorSubtract(parameters[p], Engine.TensorMultiplyScalar(step, rate));
            updated.AsSpan().CopyTo(parameters[p].AsWritableSpan());
        }
    }

    private List<Tensor<T>> TrainableParameters()
    {
        var parameters = new List<Tensor<T>>();
        foreach (var layer in AllLayers())
            parameters.AddRange(layer.GetTrainableParameters());
        return parameters;
    }

    private IEnumerable<DenseLayer<T>> AllLayers() => _densityLayers.Concat(_colorLayers);

    /// <summary>
    /// Gets all network parameters.
    /// </summary>
    public Vector<T> GetParameters()
        => Vector<T>.Concatenate(AllLayers().Select(layer => layer.GetParameters()).ToArray());

    /// <summary>
    /// Sets network parameters.
    /// </summary>
    public void SetParameters(Vector<T> parameters)
    {
        int offset = 0;
        foreach (var layer in AllLayers())
        {
            int length = layer.GetParameters().Length;
            layer.SetParameters(parameters.GetSubVector(offset, length));
            offset += length;
        }
    }
}

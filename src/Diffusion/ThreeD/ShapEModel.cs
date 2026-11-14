using System.Diagnostics.CodeAnalysis;
using AiDotNet.Attributes;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Diffusion.Schedulers;

namespace AiDotNet.Diffusion.ThreeD;

/// <summary>
/// Shap-E model for text-to-3D and image-to-3D generation with implicit neural representations.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Shap-E is OpenAI's model for generating 3D objects as implicit neural representations
/// (NeRFs). Unlike Point-E which generates point clouds, Shap-E generates parameters
/// for a neural network that represents the 3D shape, which can then be rendered
/// from any angle or converted to meshes.
/// </para>
/// <para>
/// <b>For Beginners:</b> Shap-E creates 3D objects that you can view from any angle:
///
/// What is an Implicit Neural Representation (NeRF)?
/// - A neural network that knows the 3D shape
/// - Input: 3D coordinates (x, y, z)
/// - Output: Color and density at that point
/// - Can render views from ANY angle without artifacts
///
/// Comparison with Point-E:
/// | Feature        | Point-E      | Shap-E        |
/// |----------------|--------------|---------------|
/// | Output         | Point cloud  | Neural field  |
/// | Quality        | Good         | Better        |
/// | Rendering      | Fast         | Slower        |
/// | Mesh export    | Reconstruction | Direct SDF |
/// | Memory         | Lower        | Higher        |
///
/// Example: "A red chair"
/// 1. Shap-E generates network weights (latent representation)
/// 2. These weights define a neural network
/// 3. Query (x,y,z) -> neural network -> color, density
/// 4. Render from any view or extract mesh via marching cubes
///
/// Use cases:
/// - High-quality 3D assets
/// - Novel view synthesis
/// - Direct mesh export with SDF
/// - View-consistent 3D models
/// </para>
/// <para>
/// Technical specifications:
/// - Latent dimension: 1024 parameters per shape
/// - Output: NeRF weights or SDF (Signed Distance Function)
/// - Rendering: Differentiable volumetric rendering
/// - Mesh export: Marching cubes on SDF
/// - Inference: ~64 steps
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Shap-E model
/// var shapE = new ShapEModel&lt;float&gt;();
///
/// // Generate a 3D shape from text
/// var latent = shapE.GenerateLatent(
///     prompt: "A wooden chair",
///     numInferenceSteps: 64,
///     guidanceScale: 15.0);
///
/// // Render from a specific view
/// var image = shapE.RenderView(latent, cameraPosition: (0, 0, 2), lookAt: (0, 0, 0));
///
/// // Export to mesh
/// var (vertices, faces) = shapE.ExtractMesh(latent, resolution: 64);
///
/// // Or use the high-level API
/// var mesh = shapE.GenerateMesh(
///     prompt: "A red sports car",
///     resolution: 128);
///
/// // mesh is the result tensor — write it out with your own encoder.
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.ThreeD)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelTask(ModelTask.ThreeDGeneration)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Shap-E: Generating Conditional 3D Implicit Functions", "https://arxiv.org/abs/2305.02463", Year = 2023, Authors = "Jun & Nichol")]
public partial class ShapEModel<T> : ThreeDDiffusionModelBase<T>
{
    /// <inheritdoc />
    /// <remarks>Registration order is serialization order, and matches the
    /// concatenation the previous hand-written GetParameters performed.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_latentPredictor);
    }

    #region Constants

    /// <summary>
    /// Standard Shap-E latent dimension.
    /// </summary>
    /// <remarks>
    /// Each shape is represented as a 1024-dimensional latent vector that encodes
    /// the complete parameters of a NeRF/SDF network. This compact representation
    /// enables efficient diffusion over the space of 3D shapes.
    /// </remarks>
    private const int SHAPE_LATENT_DIM = 1024;

    /// <summary>
    /// Number of MLP layers for NeRF decoder.
    /// </summary>
    /// <remarks>
    /// The NeRF decoder uses 8 MLP layers following standard NeRF architecture
    /// practices. This depth allows learning complex geometry and appearance.
    /// </remarks>
    private const int NERF_MLP_LAYERS = 8;

    /// <summary>
    /// MLP hidden dimension.
    /// </summary>
    /// <remarks>
    /// 256-dimensional hidden layers in the NeRF MLP, following the original
    /// NeRF paper conventions for shape representation networks.
    /// </remarks>
    private const int NERF_MLP_HIDDEN = 256;

    /// <summary>
    /// Hidden size for the latent diffusion transformer.
    /// </summary>
    /// <remarks>
    /// Shap-E uses a 768-dimensional hidden space in the DiT transformer,
    /// matching ViT-Base architecture for processing shape latent tokens.
    /// </remarks>
    private const int SHAPE_HIDDEN_SIZE = 768;

    /// <summary>
    /// Number of transformer layers.
    /// </summary>
    /// <remarks>
    /// 16 transformer layers provide deep processing capacity for learning
    /// the distribution over shape latents. More layers than Point-E due to
    /// the higher complexity of implicit neural representations.
    /// </remarks>
    private const int SHAPE_NUM_LAYERS = 16;

    /// <summary>
    /// Number of attention heads.
    /// </summary>
    /// <remarks>
    /// 12 attention heads (768 / 12 = 64 dims per head), following ViT-Base
    /// configuration for multi-head self-attention over shape tokens.
    /// </remarks>
    private const int SHAPE_NUM_HEADS = 12;

    /// <summary>
    /// Context dimension for conditioning (CLIP embedding size).
    /// </summary>
    /// <remarks>
    /// Matches the CLIP ViT-L/14 output dimension (1024) for rich semantic
    /// conditioning from text and image inputs.
    /// </remarks>
    private const int SHAPE_CONTEXT_DIM = 1024;

    #endregion

    #region Fields

    /// <summary>
    /// The latent diffusion transformer.
    /// </summary>
    private DiTNoisePredictor<T> _latentPredictor;

    /// <summary>
    /// The conditioning module (CLIP for text/image encoding).
    /// </summary>
    private readonly IConditioningModule<T>? _conditioner;

    /// <summary>
    /// Standard VAE for image encoding.
    /// </summary>
    private StandardVAE<T>? _imageVAE;

    /// <summary>
    /// Whether to generate SDF (Signed Distance Function) or NeRF.
    /// </summary>
    private readonly bool _useSDFMode;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _latentPredictor;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _imageVAE ?? CreateDummyVAE();

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <inheritdoc />
    public override int LatentChannels => SHAPE_LATENT_DIM;

    /// <inheritdoc />
    public override bool SupportsPointCloud => true;

    /// <inheritdoc />
    public override bool SupportsMesh => true;

    /// <inheritdoc />
    public override bool SupportsTexture => true;

    /// <inheritdoc />
    public override bool SupportsNovelView => true;

    /// <inheritdoc />
    public override bool SupportsScoreDistillation => true;

    /// <summary>
    /// Gets whether this model uses SDF mode.
    /// </summary>
    public bool UseSDFMode => _useSDFMode;

    /// <summary>
    /// Gets the latent dimension.
    /// </summary>
    public int LatentDimension => SHAPE_LATENT_DIM;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new Shap-E model with full customization support.
    /// </summary>
    /// <param name="architecture">Optional neural network architecture for custom layer configuration.</param>
    /// <param name="options">Configuration options.</param>
    /// <param name="scheduler">Optional custom scheduler.</param>
    /// <param name="latentPredictor">Optional custom latent predictor.</param>
    /// <param name="conditioner">Optional conditioning module.</param>
    /// <param name="useSDFMode">Whether to use SDF mode.</param>
    /// <param name="defaultPointCount">Default point count for point cloud extraction.</param>
    /// <param name="seed">Optional random seed.</param>
    public ShapEModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        DiTNoisePredictor<T>? latentPredictor = null,
        IConditioningModule<T>? conditioner = null,
        bool useSDFMode = true,
        int defaultPointCount = 4096,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1024,
                BetaStart = 0.0001,
                BetaEnd = 0.02,
                BetaSchedule = BetaSchedule.Linear
            },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
            defaultPointCount,
            architecture)
    {
        _useSDFMode = useSDFMode;
        _conditioner = conditioner;

        InitializeLayers(latentPredictor, seed);
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the latent predictor and image VAE layers.
    /// </summary>
    [MemberNotNull(nameof(_latentPredictor))]
    private void InitializeLayers(DiTNoisePredictor<T>? latentPredictor, int? seed)
    {
        _latentPredictor = latentPredictor ?? new DiTNoisePredictor<T>(
            inputChannels: SHAPE_LATENT_DIM,
            hiddenSize: SHAPE_HIDDEN_SIZE,
            numLayers: SHAPE_NUM_LAYERS,
            numHeads: SHAPE_NUM_HEADS,
            patchSize: 1,
            contextDim: SHAPE_CONTEXT_DIM,
            seed: seed);

        _imageVAE = new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: 4,
            baseChannels: 64,
            seed: seed);
    }

    #endregion

    /// <summary>
    /// Creates a dummy VAE for interface compliance.
    /// </summary>
    private StandardVAE<T> CreateDummyVAE()
    {
        return new StandardVAE<T>();
    }

    #region Generation Methods

    /// <summary>
    /// Generates a latent representation of a 3D shape from text.
    /// </summary>
    /// <param name="prompt">Text description of the 3D object.</param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Latent tensor representing the 3D shape [1, 1, latentDim].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The latent is a compressed representation of the 3D shape.
    /// It contains the "recipe" for rendering the object from any angle.
    ///
    /// After generating a latent, you can:
    /// - Render views with RenderView()
    /// - Extract a mesh with ExtractMesh()
    /// - Get a point cloud with SamplePointCloud()
    /// </para>
    /// </remarks>
    public virtual Tensor<T> GenerateLatent(
        string prompt,
        string? negativePrompt = null,
        int numInferenceSteps = 64,
        double guidanceScale = 15.0,
        int? seed = null)
    {
        var useCFG = guidanceScale > 1.0 && _latentPredictor.SupportsCFG;

        // Get conditioning
        Tensor<T>? promptEmbedding = null;
        Tensor<T>? negativeEmbedding = null;

        if (_conditioner != null)
        {
            var promptTokens = _conditioner.Tokenize(prompt);
            promptEmbedding = _conditioner.EncodeText(promptTokens);

            if (useCFG)
            {
                negativeEmbedding = !string.IsNullOrEmpty(negativePrompt)
                    ? _conditioner.EncodeText(_conditioner.Tokenize(negativePrompt ?? string.Empty))
                    : _conditioner.GetUnconditionalEmbedding(1);
            }
        }

        // Generate latent
        var latentShape = new[] { 1, 1, SHAPE_LATENT_DIM };
        var rng = CreateInferenceRng(seed);
        var latent = SampleNoiseTensor(latentShape, rng);

        Scheduler.SetTimesteps(numInferenceSteps);

        foreach (var timestep in Scheduler.Timesteps)
        {
            Tensor<T> noisePrediction;

            if (useCFG && negativeEmbedding != null)
            {
                var condPred = _latentPredictor.PredictNoise(latent, timestep, promptEmbedding);
                var uncondPred = _latentPredictor.PredictNoise(latent, timestep, negativeEmbedding);
                noisePrediction = ApplyGuidance(uncondPred, condPred, guidanceScale);
            }
            else
            {
                noisePrediction = _latentPredictor.PredictNoise(latent, timestep, promptEmbedding);
            }

            var latentVector = latent.ToVector();
            var noiseVector = noisePrediction.ToVector();
            latentVector = Scheduler.Step(noiseVector, timestep, latentVector, NumOps.Zero);
            latent = new Tensor<T>(latentShape, latentVector);
        }

        return latent;
    }

    /// <summary>
    /// Generates a latent from an image.
    /// </summary>
    /// <param name="image">Input image [batch, channels, height, width].</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Guidance scale.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Latent tensor representing the 3D shape.</returns>
    public virtual Tensor<T> GenerateLatentFromImage(
        Tensor<T> image,
        int numInferenceSteps = 64,
        double guidanceScale = 3.0,
        int? seed = null)
    {
        if (_imageVAE == null)
            throw new InvalidOperationException("Image VAE not initialized.");

        // Encode image for conditioning
        var imageLatent = _imageVAE.Encode(image, sampleMode: false);
        var imageCondition = FlattenToCondition(imageLatent);

        // Generate shape latent
        var latentShape = new[] { 1, 1, SHAPE_LATENT_DIM };
        var rng = CreateInferenceRng(seed);
        var latent = SampleNoiseTensor(latentShape, rng);

        Scheduler.SetTimesteps(numInferenceSteps);

        foreach (var timestep in Scheduler.Timesteps)
        {
            var noisePrediction = _latentPredictor.PredictNoise(latent, timestep, imageCondition);

            var latentVector = latent.ToVector();
            var noiseVector = noisePrediction.ToVector();
            latentVector = Scheduler.Step(noiseVector, timestep, latentVector, NumOps.Zero);
            latent = new Tensor<T>(latentShape, latentVector);
        }

        return latent;
    }

    /// <summary>
    /// Renders a view of the shape from a camera position.
    /// </summary>
    /// <param name="latent">Shape latent representation.</param>
    /// <param name="cameraPosition">Camera position (x, y, z).</param>
    /// <param name="lookAt">Look-at target (x, y, z).</param>
    /// <param name="imageSize">Output image size.</param>
    /// <param name="numSamples">Number of ray samples for rendering.</param>
    /// <returns>Rendered image tensor [1, 3, imageSize, imageSize].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This renders what the 3D object looks like from a specific viewpoint:
    ///
    /// cameraPosition: Where the "camera" is located in 3D space
    /// lookAt: What point the camera is looking at
    ///
    /// Example:
    /// - cameraPosition: (0, 0, 2) - camera is 2 units in front
    /// - lookAt: (0, 0, 0) - looking at the center
    /// - Result: Front view of the object
    ///
    /// Change cameraPosition to render from different angles!
    /// </para>
    /// </remarks>
    public virtual Tensor<T> RenderView(
        Tensor<T> latent,
        (double x, double y, double z) cameraPosition,
        (double x, double y, double z) lookAt,
        int imageSize = 256,
        int numSamples = 64)
    {
        if (imageSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(imageSize), "Image size must be positive.");
        if (numSamples <= 0)
            throw new ArgumentOutOfRangeException(nameof(numSamples), "The sample count must be positive.");

        var field = new ShapEImplicitFunction<T>(latent);
        var directions = RayDirections(cameraPosition, lookAt, imageSize);
        var origin = new Tensor<double>(new[] { 1, 1, 3 },
            new Vector<double>(new[] { cameraPosition.x, cameraPosition.y, cameraPosition.z })).Cast<T>();

        // Stratified depths t_i = 0.1 + 1.9 i / n with spacing delta = 1.9 / n, and the strictly-upper-triangular
        // matrix that turns per-sample optical depths into the exclusive prefix sums sum_{j<i} sigma_j delta.
        double delta = (RenderFar - RenderNear) / numSamples;
        var depths = Engine.Reshape(
            Engine.TensorAddScalar(
                Engine.TensorMultiplyScalar(Engine.TensorArange<T>(0, numSamples, 1), NumOps.FromDouble(delta)),
                NumOps.FromDouble(RenderNear)),
            new[] { 1, numSamples, 1 });
        var prefixValues = new double[numSamples * numSamples];
        for (int j = 0; j < numSamples; j++)
        {
            for (int i = j + 1; i < numSamples; i++)
                prefixValues[(j * numSamples) + i] = 1.0;
        }
        var exclusivePrefix = new Tensor<double>(new[] { numSamples, numSamples }, new Vector<double>(prefixValues)).Cast<T>();

        var deltaT = NumOps.FromDouble(delta);
        var background = NumOps.FromDouble(RenderBackground);
        int pixels = imageSize * imageSize;
        int raysPerChunk = Math.Max(1, RenderPointsPerChunk / numSamples);
        var chunks = new List<Tensor<T>>();
        for (int start = 0; start < pixels; start += raysPerChunk)
        {
            int rays = Math.Min(raysPerChunk, pixels - start);
            var chunkDirections = Engine.Reshape(Engine.TensorNarrow(directions, 0, start, rays), new[] { rays, 1, 3 });
            var sampleShape = new[] { rays, numSamples, 3 };
            var points = Engine.Reshape(
                Engine.TensorAdd(
                    Engine.TensorBroadcastTo(origin, sampleShape),
                    Engine.TensorBroadcastTo(Engine.TensorMultiply(chunkDirections, depths), sampleShape)),
                new[] { rays * numSamples, 3 });
            var outputs = field.Evaluate(points);

            // C = sum_i T_i (1 - exp(-sigma_i delta)) c_i + T_n * background.
            var opticalDepth = Engine.TensorMultiplyScalar(
                Engine.Reshape(ShapEImplicitFunction<T>.Density(outputs), new[] { rays, numSamples }), deltaT);
            var alpha = Engine.TensorSubtract(
                Engine.TensorOnesLike(opticalDepth), Engine.TensorExp(Engine.TensorNegate(opticalDepth)));
            var transmittance = Engine.TensorExp(Engine.TensorNegate(Engine.TensorMatMul(opticalDepth, exclusivePrefix)));
            var weights = Engine.Reshape(Engine.TensorMultiply(transmittance, alpha), new[] { rays, numSamples, 1 });
            var colour = Engine.Reshape(ShapEImplicitFunction<T>.Colour(outputs), new[] { rays, numSamples, 3 });
            var composited = Engine.ReduceSum(Engine.TensorMultiply(weights, colour), new[] { 1 }, false);
            var remaining = Engine.TensorExp(Engine.TensorNegate(Engine.ReduceSum(opticalDepth, new[] { 1 }, true)));
            chunks.Add(Engine.TensorAdd(composited, Engine.TensorBroadcastTo(
                Engine.TensorMultiplyScalar(remaining, background), new[] { rays, 3 })));
        }

        // [pixels, 3] -> [1, 3, H, W]
        var image = Engine.TensorConcatenate(chunks.ToArray(), 0);
        return Engine.Reshape(Engine.TensorPermute(image, new[] { 1, 0 }).Contiguous(), new[] { 1, 3, imageSize, imageSize });
    }

    #endregion

    #region Helper Methods

    private const double RenderNear = 0.1;
    private const double RenderFar = 2.0;
    private const double RenderBackground = 0.8; // Light gray background
    private const int RenderPointsPerChunk = 65536;
    private const int GridPointsPerChunk = 65536;

    /// <summary>
    /// Unit ray directions [size * size, 3] (row-major pixels, row 0 at the top) of a camera at
    /// <paramref name="cameraPosition"/> looking at <paramref name="lookAt"/> with world +Y up and a half-extent of
    /// 0.5 on the image plane at unit distance.
    /// </summary>
    internal Tensor<T> RayDirections(
        (double x, double y, double z) cameraPosition, (double x, double y, double z) lookAt, int size)
    {
        var forward = Normalize(lookAt.x - cameraPosition.x, lookAt.y - cameraPosition.y, lookAt.z - cameraPosition.z);

        // right = forward x up (+Y), up' = right x forward: x to the right of the image, y to its top.
        var right = Normalize(-forward[2], 0.0, forward[0]);
        if (Math.Abs(forward[0]) < 1e-9 && Math.Abs(forward[2]) < 1e-9)
            right = new[] { 1.0, 0.0, 0.0 };
        var up = new[]
        {
            (right[1] * forward[2]) - (right[2] * forward[1]),
            (right[2] * forward[0]) - (right[0] * forward[2]),
            (right[0] * forward[1]) - (right[1] * forward[0])
        };

        // Pixel centres: u = 2 (x + 0.5) / size - 1, v = 1 - 2 (y + 0.5) / size, each scaled by 0.5.
        var u = Engine.TensorAddScalar(
            Engine.TensorMultiplyScalar(Engine.TensorArange<T>(0, size, 1), NumOps.FromDouble(1.0 / size)),
            NumOps.FromDouble((0.5 / size) - 0.5));
        var v = Engine.TensorNegate(u);
        Tensor<T> Basis(double[] vector) => new Tensor<double>(new[] { 1, 1, 3 }, new Vector<double>(vector)).Cast<T>();

        var grid = new[] { size, size, 3 };
        var directions = Engine.TensorAdd(
            Engine.TensorAdd(
                Engine.TensorBroadcastTo(Engine.TensorMultiply(Engine.Reshape(u, new[] { 1, size, 1 }), Basis(right)), grid),
                Engine.TensorBroadcastTo(Engine.TensorMultiply(Engine.Reshape(v, new[] { size, 1, 1 }), Basis(up)), grid)),
            Engine.TensorBroadcastTo(Basis(forward), grid));
        var norm = Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorMultiply(directions, directions), new[] { 2 }, true));
        var unit = Engine.TensorDivide(directions, Engine.TensorBroadcastTo(norm, grid));
        return Engine.Reshape(unit, new[] { size * size, 3 });
    }

    private static double[] Normalize(double x, double y, double z)
    {
        double length = Math.Sqrt((x * x) + (y * y) + (z * z));
        if (length < 1e-12)
            return new[] { 0.0, 0.0, -1.0 };
        return new[] { x / length, y / length, z / length };
    }

    /// <summary>
    /// Evaluates the latent's implicit MLP at <paramref name="points"/> [N, 3] in batches, returning [N, 5].
    /// </summary>
    private Tensor<T> EvaluateField(ShapEImplicitFunction<T> field, Tensor<T> points)
    {
        int total = points.Shape[0];
        var chunks = new List<Tensor<T>>();
        for (int start = 0; start < total; start += GridPointsPerChunk)
        {
            int count = Math.Min(GridPointsPerChunk, total - start);
            chunks.Add(field.Evaluate(Engine.TensorNarrow(points, 0, start, count)));
        }

        return chunks.Count == 1 ? chunks[0] : Engine.TensorConcatenate(chunks.ToArray(), 0);
    }

    /// <summary>
    /// Extracts a mesh from the latent using marching cubes.
    /// </summary>
    /// <param name="latent">Shape latent representation.</param>
    /// <param name="resolution">Grid resolution for marching cubes.</param>
    /// <returns>Tuple of vertices [numVerts, 3] and faces [numFaces, 3].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This converts the neural representation to a triangle mesh:
    ///
    /// Marching cubes algorithm:
    /// 1. Create a 3D grid of points
    /// 2. Evaluate SDF (signed distance) at each point
    /// 3. Find where surface crosses grid cells (SDF = 0)
    /// 4. Generate triangles for those crossings
    ///
    /// Higher resolution = more triangles = more detail but slower
    /// </para>
    /// <para>
    /// The latent's implicit MLP is evaluated on a [-1, 1]^3 grid and its zero level set is extracted by marching
    /// tetrahedra (a crack-free marching-cubes variant). With no surface in the grid both tensors hold one zero row.
    /// </para>
    /// </remarks>
    public virtual (Tensor<T> Vertices, Tensor<T> Faces) ExtractMesh(
        Tensor<T> latent,
        int resolution = 64)
    {
        var (coordinates, triangles) = ExtractSurface(latent, resolution);
        return (VertexTensor(coordinates), FaceTensor(triangles));
    }

    /// <summary>
    /// The faces [max(count, 1), 3] of the flat triangle index list as T (one zero row when empty).
    /// </summary>
    /// <exception cref="InvalidOperationException">
    /// An index is too large for T to hold exactly (above 2048 for Half): the face tensor would silently point at
    /// the wrong vertices. <see cref="GenerateMesh"/> returns integer faces and has no such limit.
    /// </exception>
    internal static Tensor<T> FaceTensor(List<int> triangles)
    {
        // Exactness is checked by round trip rather than from INumericOperations.PrecisionBits: that property is the
        // storage width (16 for Half), not the significand width (11), so 2^PrecisionBits never rejects anything.
        int faceCount = triangles.Count / 3;
        var faceValues = new double[Math.Max(faceCount, 1) * 3];
        for (int i = 0; i < triangles.Count; i++)
            faceValues[i] = triangles[i];

        var faces = new Tensor<double>(new[] { Math.Max(faceCount, 1), 3 }, new Vector<double>(faceValues)).Cast<T>();
        double[] stored = faces.Cast<double>().ToArray();
        for (int i = 0; i < triangles.Count; i++)
        {
            if (stored[i] != faceValues[i])
                throw new InvalidOperationException(
                    $"Vertex index {triangles[i]} cannot be stored exactly in {typeof(T).Name}; lower the resolution " +
                    "or use GenerateMesh, whose faces are integers.");
        }

        return faces;
    }

    /// <summary>The vertices [max(count, 1), 3] of the flat coordinate list (one zero row when empty).</summary>
    private static Tensor<T> VertexTensor(List<double> coordinates)
    {
        int vertexCount = Math.Max(coordinates.Count / 3, 1);
        var values = new double[vertexCount * 3];
        coordinates.CopyTo(values);
        return new Tensor<double>(new[] { vertexCount, 3 }, new Vector<double>(values)).Cast<T>();
    }

    /// <summary>
    /// The zero level set of the latent's SDF on the grid (i / (resolution - 1) - 0.5) * 2, i &lt; resolution.
    /// </summary>
    private (List<double> Coordinates, List<int> Triangles) ExtractSurface(Tensor<T> latent, int resolution)
    {
        if (resolution < 2)
            throw new ArgumentOutOfRangeException(nameof(resolution), "The grid resolution must be at least 2.");

        var field = new ShapEImplicitFunction<T>(latent);
        double step = 2.0 / (resolution - 1);
        var outputs = EvaluateField(field, ShapEImplicitFunction<T>.GridPoints(resolution, -1.0, step));

        // Marching tetrahedra treats values above the threshold as inside; inside is SDF < 0.
        var inside = Engine.TensorNegate(ShapEImplicitFunction<T>.SignedDistance(outputs)).Cast<double>().ToArray();
        var grid = new double[resolution, resolution, resolution];
        Buffer.BlockCopy(inside, 0, grid, 0, inside.Length * sizeof(double));
        return MarchingTetrahedra.Extract(grid, 0.0, -1.0, step);
    }

    /// <summary>
    /// Generates a mesh directly from a text prompt.
    /// </summary>
    /// <param name="prompt">Text description of the 3D object.</param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="resolution">Mesh resolution.</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Guidance scale.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Mesh3D containing vertices and faces.</returns>
    public override Mesh3D<T> GenerateMesh(
        string prompt,
        string? negativePrompt = null,
        int resolution = 64,
        int numInferenceSteps = 64,
        double guidanceScale = 15.0,
        int? seed = null)
    {
        var latent = GenerateLatent(prompt, negativePrompt, numInferenceSteps, guidanceScale, seed);
        var (coordinates, triangles) = ExtractSurface(latent, resolution);

        // Faces as int[faceCount, 3] straight from the integer triangle list (one zero row when empty).
        var faceIndices = new int[Math.Max(triangles.Count / 3, 1) * 3];
        triangles.CopyTo(faceIndices);
        var faceArray = new int[faceIndices.Length / 3, 3];
        Buffer.BlockCopy(faceIndices, 0, faceArray, 0, faceIndices.Length * sizeof(int));

        return new Mesh3D<T>
        {
            Vertices = VertexTensor(coordinates),
            Faces = faceArray
        };
    }

    /// <summary>
    /// Samples a point cloud from the shape.
    /// </summary>
    /// <param name="latent">Shape latent representation.</param>
    /// <param name="numPoints">Number of points to sample.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Point cloud tensor [1, numPoints, 6] with XYZ + RGB.</returns>
    /// <remarks>
    /// Rejection sampling of uniform points in [-1, 1]^3 with SDF &lt; 0, coloured by the implicit MLP. Candidates
    /// are drawn (three draws each, in order) and evaluated in blocks and accepted in draw order, up to
    /// <c>numPoints * 100</c> candidates; rows past the accepted count stay zero.
    /// </remarks>
    public virtual Tensor<T> SamplePointCloud(
        Tensor<T> latent,
        int? numPoints = null,
        int? seed = null)
    {
        var effectiveNumPoints = numPoints ?? DefaultPointCount;
        var field = new ShapEImplicitFunction<T>(latent);
        var rng = CreateInferenceRng(seed);

        var cloud = new double[effectiveNumPoints * 6];
        var sampledCount = 0;
        long maxAttempts = (long)effectiveNumPoints * 100;
        long attempts = 0;

        while (sampledCount < effectiveNumPoints && attempts < maxAttempts)
        {
            int block = (int)Math.Min(maxAttempts - attempts, Math.Max(1024, 2L * (effectiveNumPoints - sampledCount)));
            block = Math.Min(block, GridPointsPerChunk);
            var candidates = new double[block * 3];
            for (int i = 0; i < candidates.Length; i++)
                candidates[i] = rng.NextDouble() * 2 - 1;
            attempts += block;

            var outputs = field.Evaluate(
                new Tensor<double>(new[] { block, 3 }, new Vector<double>(candidates)).Cast<T>());
            var sdf = ShapEImplicitFunction<T>.SignedDistance(outputs).Cast<double>().ToArray();
            var colour = ShapEImplicitFunction<T>.Colour(outputs).Cast<double>().ToArray();

            for (int i = 0; i < block && sampledCount < effectiveNumPoints; i++)
            {
                if (sdf[i] >= 0)
                    continue;
                Array.Copy(candidates, i * 3, cloud, sampledCount * 6, 3);
                Array.Copy(colour, i * 3, cloud, (sampledCount * 6) + 3, 3);
                sampledCount++;
            }
        }

        return new Tensor<double>(new[] { 1, effectiveNumPoints, 6 }, new Vector<double>(cloud)).Cast<T>();
    }

    /// <inheritdoc />
    public override Tensor<T> GeneratePointCloud(
        string prompt,
        string? negativePrompt = null,
        int? numPoints = null,
        int numInferenceSteps = 64,
        double guidanceScale = 15.0,
        int? seed = null)
    {
        var latent = GenerateLatent(prompt, negativePrompt, numInferenceSteps, guidanceScale, seed);
        return SamplePointCloud(latent, numPoints, seed);
    }

    /// <summary>
    /// Flattens an image latent to a conditioning vector.
    /// </summary>
    private Tensor<T> FlattenToCondition(Tensor<T> imageLatent)
    {
        var flatSize = imageLatent.Length;
        return new Tensor<T>(new[] { 1, 1, flatSize }, imageLatent.ToVector());
    }

    #endregion

    #region IParameterizable Implementation





    /// <inheritdoc />
    #endregion

    #region ICloneable Implementation

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "Shap-E",
            Version = "1.0",
            Description = "Shap-E text-to-3D generation with implicit neural representations (NeRF/SDF)",
            FeatureCount = (int)System.Math.Min((long)int.MaxValue, ParameterCount),
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "dit-implicit-neural-rep");
        metadata.SetProperty("latent_dim", SHAPE_LATENT_DIM);
        metadata.SetProperty("hidden_size", SHAPE_HIDDEN_SIZE);
        metadata.SetProperty("num_layers", SHAPE_NUM_LAYERS);
        metadata.SetProperty("num_heads", SHAPE_NUM_HEADS);
        metadata.SetProperty("context_dim", SHAPE_CONTEXT_DIM);
        metadata.SetProperty("nerf_mlp_layers", NERF_MLP_LAYERS);
        metadata.SetProperty("nerf_mlp_hidden", NERF_MLP_HIDDEN);
        metadata.SetProperty("sdf_mode", _useSDFMode);
        metadata.SetProperty("default_point_count", DefaultPointCount);

        return metadata;
    }

    #endregion
}

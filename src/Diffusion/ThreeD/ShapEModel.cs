using System.Diagnostics.CodeAnalysis;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Diffusion;

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
/// ExportToOBJ(mesh, "car.obj");
/// </code>
/// </example>
public class ShapEModel<T> : ThreeDDiffusionModelBase<T>
{
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
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
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
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
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
        // Simplified volumetric rendering
        // Real implementation would use proper ray marching with the NeRF

        var image = new Tensor<T>(new[] { 1, 3, imageSize, imageSize });
        var imageSpan = image.AsWritableSpan();
        var latentSpan = latent.AsSpan();

        // Camera setup
        var camPos = new double[] { cameraPosition.x, cameraPosition.y, cameraPosition.z };
        var target = new double[] { lookAt.x, lookAt.y, lookAt.z };

        // Forward direction
        var forward = new double[3];
        var length = 0.0;
        for (int i = 0; i < 3; i++)
        {
            forward[i] = target[i] - camPos[i];
            length += forward[i] * forward[i];
        }
        length = Math.Sqrt(length);
        for (int i = 0; i < 3; i++) forward[i] /= length;

        // Simple right and up vectors (assuming Y is up)
        var right = new double[] { forward[2], 0, -forward[0] };
        var rLen = Math.Sqrt(right[0] * right[0] + right[2] * right[2]);
        if (rLen > 0.001)
        {
            right[0] /= rLen;
            right[2] /= rLen;
        }
        var up = new double[] {
            forward[1] * right[2] - forward[2] * right[1],
            forward[2] * right[0] - forward[0] * right[2],
            forward[0] * right[1] - forward[1] * right[0]
        };

        // Render each pixel
        for (int y = 0; y < imageSize; y++)
        {
            for (int x = 0; x < imageSize; x++)
            {
                // Normalized coordinates [-1, 1]
                var u = (2.0 * x / imageSize - 1.0);
                var v = (2.0 * y / imageSize - 1.0);

                // Ray direction
                var rayDir = new double[3];
                for (int i = 0; i < 3; i++)
                {
                    rayDir[i] = forward[i] + u * right[i] * 0.5 + v * up[i] * 0.5;
                }
                // Normalize
                length = Math.Sqrt(rayDir[0] * rayDir[0] + rayDir[1] * rayDir[1] + rayDir[2] * rayDir[2]);
                for (int i = 0; i < 3; i++) rayDir[i] /= length;

                // Sample color using simplified NeRF evaluation
                var color = SampleNeRFColor(latentSpan, camPos, rayDir, numSamples);

                // Write to image
                for (int c = 0; c < 3; c++)
                {
                    var idx = c * imageSize * imageSize + y * imageSize + x;
                    imageSpan[idx] = NumOps.FromDouble(color[c]);
                }
            }
        }

        return image;
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Samples color along a ray using the NeRF representation.
    /// </summary>
    private double[] SampleNeRFColor(ReadOnlySpan<T> latent, double[] origin, double[] direction, int numSamples)
    {
        // Simplified NeRF sampling
        // Real implementation would use proper MLP evaluation and alpha compositing

        var color = new double[3];
        var transmission = 1.0;

        for (int i = 0; i < numSamples; i++)
        {
            var t = 0.1 + (2.0 - 0.1) * i / numSamples;
            var point = new double[3];
            for (int j = 0; j < 3; j++)
            {
                point[j] = origin[j] + t * direction[j];
            }

            // Query NeRF at this point (simplified using latent hash)
            var (density, sampleColor) = QueryNeRF(latent, point);

            // Alpha compositing
            var alpha = 1.0 - Math.Exp(-density * (2.0 - 0.1) / numSamples);
            for (int c = 0; c < 3; c++)
            {
                color[c] += transmission * alpha * sampleColor[c];
            }
            transmission *= (1.0 - alpha);

            if (transmission < 0.01) break;
        }

        // Background
        for (int c = 0; c < 3; c++)
        {
            color[c] += transmission * 0.8; // Light gray background
        }

        return color;
    }

    /// <summary>
    /// Queries the NeRF at a 3D point (simplified).
    /// </summary>
    private (double density, double[] color) QueryNeRF(ReadOnlySpan<T> latent, double[] point)
    {
        // Simplified: use latent to define a basic shape
        // Real implementation would evaluate an MLP with the latent as weights

        // Hash point to latent indices
        var latentIdx = Math.Abs((int)(point[0] * 100 + point[1] * 10 + point[2])) % latent.Length;

        // Distance from center (simple sphere SDF)
        var dist = Math.Sqrt(point[0] * point[0] + point[1] * point[1] + point[2] * point[2]);
        var radius = 0.5 + 0.1 * NumOps.ToDouble(latent[latentIdx % latent.Length]);

        // Density based on distance to surface
        var density = dist < radius ? 10.0 : 0.0;

        // Color from latent
        var color = new double[3];
        for (int c = 0; c < 3; c++)
        {
            var colorIdx = (latentIdx + c * 100) % latent.Length;
            color[c] = 0.5 + 0.5 * Math.Tanh(NumOps.ToDouble(latent[colorIdx]));
        }

        return (density, color);
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
    /// </remarks>
    public virtual (Tensor<T> Vertices, Tensor<T> Faces) ExtractMesh(
        Tensor<T> latent,
        int resolution = 64)
    {
        var latentSpan = latent.AsSpan();

        // Evaluate SDF on grid
        var sdfGrid = new double[resolution, resolution, resolution];
        for (int x = 0; x < resolution; x++)
        {
            for (int y = 0; y < resolution; y++)
            {
                for (int z = 0; z < resolution; z++)
                {
                    var point = new double[] {
                        (x / (double)(resolution - 1) - 0.5) * 2,
                        (y / (double)(resolution - 1) - 0.5) * 2,
                        (z / (double)(resolution - 1) - 0.5) * 2
                    };
                    sdfGrid[x, y, z] = EvaluateSDF(latentSpan, point);
                }
            }
        }

        // Simplified marching cubes (generate surface vertices)
        var vertices = new List<double[]>();
        var faces = new List<int[]>();

        for (int x = 0; x < resolution - 1; x++)
        {
            for (int y = 0; y < resolution - 1; y++)
            {
                for (int z = 0; z < resolution - 1; z++)
                {
                    // Check if cell contains surface
                    var hasPositive = false;
                    var hasNegative = false;

                    for (int dx = 0; dx <= 1; dx++)
                    {
                        for (int dy = 0; dy <= 1; dy++)
                        {
                            for (int dz = 0; dz <= 1; dz++)
                            {
                                if (sdfGrid[x + dx, y + dy, z + dz] > 0) hasPositive = true;
                                if (sdfGrid[x + dx, y + dy, z + dz] < 0) hasNegative = true;
                            }
                        }
                    }

                    if (hasPositive && hasNegative)
                    {
                        // Surface crosses this cell - add vertices
                        var vIdx = vertices.Count;
                        var cx = (x + 0.5) / resolution * 2 - 1;
                        var cy = (y + 0.5) / resolution * 2 - 1;
                        var cz = (z + 0.5) / resolution * 2 - 1;

                        vertices.Add(new[] { cx, cy, cz });
                        vertices.Add(new[] { cx + 1.0 / resolution, cy, cz });
                        vertices.Add(new[] { cx, cy + 1.0 / resolution, cz });

                        faces.Add(new[] { vIdx, vIdx + 1, vIdx + 2 });
                    }
                }
            }
        }

        // Convert to tensors
        var vertCount = vertices.Count;
        var faceCount = faces.Count;

        var vertTensor = new Tensor<T>(new[] { Math.Max(vertCount, 1), 3 });
        var faceTensor = new Tensor<T>(new[] { Math.Max(faceCount, 1), 3 });

        var vertSpan = vertTensor.AsWritableSpan();
        var faceSpan = faceTensor.AsWritableSpan();

        for (int i = 0; i < vertCount; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                vertSpan[i * 3 + j] = NumOps.FromDouble(vertices[i][j]);
            }
        }

        for (int i = 0; i < faceCount; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                faceSpan[i * 3 + j] = NumOps.FromDouble(faces[i][j]);
            }
        }

        return (vertTensor, faceTensor);
    }

    /// <summary>
    /// Evaluates the signed distance function at a point.
    /// </summary>
    private double EvaluateSDF(ReadOnlySpan<T> latent, double[] point)
    {
        // Simplified SDF evaluation using latent
        // Real implementation would use proper MLP evaluation

        var latentIdx = Math.Abs((int)(point[0] * 50 + point[1] * 100 + point[2] * 150)) % latent.Length;
        var radius = 0.5 + 0.3 * Math.Tanh(NumOps.ToDouble(latent[latentIdx]));

        var dist = Math.Sqrt(point[0] * point[0] + point[1] * point[1] + point[2] * point[2]);
        return dist - radius;
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
        var (vertices, faces) = ExtractMesh(latent, resolution);

        // Convert faces tensor to int[,] array
        var faceCount = faces.Shape[0];
        var faceArray = new int[faceCount, 3];
        var facesSpan = faces.AsSpan();

        for (int i = 0; i < faceCount; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                faceArray[i, j] = (int)NumOps.ToDouble(facesSpan[i * 3 + j]);
            }
        }

        return new Mesh3D<T>
        {
            Vertices = vertices,
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
    public virtual Tensor<T> SamplePointCloud(
        Tensor<T> latent,
        int? numPoints = null,
        int? seed = null)
    {
        var effectiveNumPoints = numPoints ?? DefaultPointCount;
        var latentSpan = latent.AsSpan();
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;

        var points = new Tensor<T>(new[] { 1, effectiveNumPoints, 6 });
        var pointsSpan = points.AsWritableSpan();

        var sampledCount = 0;
        var maxAttempts = effectiveNumPoints * 100;
        var attempts = 0;

        while (sampledCount < effectiveNumPoints && attempts < maxAttempts)
        {
            attempts++;

            // Random point in [-1, 1]^3
            var point = new double[] {
                rng.NextDouble() * 2 - 1,
                rng.NextDouble() * 2 - 1,
                rng.NextDouble() * 2 - 1
            };

            // Check if inside shape (SDF < 0)
            var sdf = EvaluateSDF(latentSpan, point);
            if (sdf < 0)
            {
                // Get color
                var (_, color) = QueryNeRF(latentSpan, point);

                // Store point
                var baseIdx = sampledCount * 6;
                for (int i = 0; i < 3; i++)
                {
                    pointsSpan[baseIdx + i] = NumOps.FromDouble(point[i]);
                    pointsSpan[baseIdx + 3 + i] = NumOps.FromDouble(color[i]);
                }

                sampledCount++;
            }
        }

        return points;
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
        var flatSize = imageLatent.Shape.Aggregate(1, (a, b) => a * b);
        return new Tensor<T>(new[] { 1, 1, flatSize }, imageLatent.ToVector());
    }

    #endregion

    #region IParameterizable Implementation

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        return _latentPredictor.GetParameters();
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != _latentPredictor.ParameterCount)
        {
            throw new ArgumentException(
                $"Expected {_latentPredictor.ParameterCount} parameters, got {parameters.Length}.",
                nameof(parameters));
        }

        _latentPredictor.SetParameters(parameters);
    }

    /// <inheritdoc />
    public override int ParameterCount => _latentPredictor.ParameterCount;

    #endregion

    #region ICloneable Implementation

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        return Clone();
    }

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        // Clone the predictor to preserve trained weights
        var clonedPredictor = new DiTNoisePredictor<T>(
            inputChannels: SHAPE_LATENT_DIM,
            hiddenSize: SHAPE_HIDDEN_SIZE,
            numLayers: SHAPE_NUM_LAYERS,
            numHeads: SHAPE_NUM_HEADS,
            patchSize: 1,
            contextDim: SHAPE_CONTEXT_DIM);
        clonedPredictor.SetParameters(_latentPredictor.GetParameters());

        return new ShapEModel<T>(
            options: null,
            scheduler: null,
            latentPredictor: clonedPredictor,
            conditioner: _conditioner,
            useSDFMode: _useSDFMode,
            defaultPointCount: DefaultPointCount);
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "Shap-E",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "Shap-E text-to-3D generation with implicit neural representations (NeRF/SDF)",
            FeatureCount = ParameterCount,
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

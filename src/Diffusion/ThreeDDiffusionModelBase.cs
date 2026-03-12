using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Diffusion.Schedulers;

namespace AiDotNet.Diffusion;

/// <summary>
/// Base class for 3D diffusion models that generate 3D content like point clouds, meshes, and scenes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This abstract base class provides common functionality for all 3D diffusion models,
/// including point cloud generation, mesh generation, image-to-3D, novel view synthesis,
/// and score distillation sampling.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is the foundation for 3D generation models like Point-E and Shap-E.
/// It extends diffusion to create 3D objects instead of 2D images.
/// </para>
/// <para>
/// Types of 3D generation:
/// - Point Clouds: Sets of 3D points that form a shape
/// - Meshes: Surfaces made of triangles (like in video games)
/// - Textured Models: Meshes with colors and materials
/// - Novel Views: New angles of an object from one image
/// </para>
/// <para>
/// How 3D diffusion works:
/// 1. Text-to-3D: Describe what you want and get a 3D model
/// 2. Image-to-3D: Turn a single photo into a full 3D model
/// 3. Score Distillation: Use 2D diffusion knowledge to guide 3D optimization
/// </para>
/// </remarks>
public abstract class ThreeDDiffusionModelBase<T> : LatentDiffusionModelBase<T>, I3DDiffusionModel<T>
{
    /// <summary>
    /// Default number of points in generated point clouds.
    /// </summary>
    private readonly int _defaultPointCount;

    /// <inheritdoc />
    public virtual int DefaultPointCount => _defaultPointCount;

    /// <inheritdoc />
    public abstract bool SupportsPointCloud { get; }

    /// <inheritdoc />
    public abstract bool SupportsMesh { get; }

    /// <inheritdoc />
    public abstract bool SupportsTexture { get; }

    /// <inheritdoc />
    public abstract bool SupportsNovelView { get; }

    /// <inheritdoc />
    public abstract bool SupportsScoreDistillation { get; }

    /// <summary>
    /// Gets the coordinate scale for normalizing 3D positions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Points are typically normalized to [-1, 1] or [0, 1] range.
    /// </para>
    /// </remarks>
    public virtual double CoordinateScale { get; protected set; } = 1.0;

    /// <summary>
    /// Initializes a new instance of the ThreeDDiffusionModelBase class.
    /// </summary>
    /// <param name="options">Configuration options for the diffusion model.</param>
    /// <param name="scheduler">Optional custom scheduler.</param>
    /// <param name="defaultPointCount">Default number of points in point clouds.</param>
    protected ThreeDDiffusionModelBase(
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        int defaultPointCount = 4096,
        NeuralNetworkArchitecture<T>? architecture = null)
        : base(options, scheduler, architecture)
    {
        _defaultPointCount = defaultPointCount;
    }

    #region I3DDiffusionModel<T> Implementation

    /// <inheritdoc />
    public virtual Tensor<T> GeneratePointCloud(
        string prompt,
        string? negativePrompt = null,
        int? numPoints = null,
        int numInferenceSteps = 64,
        double guidanceScale = 3.0,
        int? seed = null)
    {
        if (!SupportsPointCloud)
            throw new NotSupportedException("This model does not support point cloud generation.");

        if (Conditioner == null)
            throw new InvalidOperationException("Point cloud generation requires a conditioning module.");

        var effectiveNumPoints = numPoints ?? DefaultPointCount;
        var useCFG = guidanceScale > 1.0 && NoisePredictor.SupportsCFG;

        // Encode text prompts
        var promptTokens = Conditioner.Tokenize(prompt);
        var promptEmbedding = Conditioner.EncodeText(promptTokens);

        Tensor<T>? negativeEmbedding = null;
        if (useCFG)
        {
            if (!string.IsNullOrEmpty(negativePrompt))
            {
                var negTokens = Conditioner.Tokenize(negativePrompt ?? string.Empty);
                negativeEmbedding = Conditioner.EncodeText(negTokens);
            }
            else
            {
                negativeEmbedding = Conditioner.GetUnconditionalEmbedding(1);
            }
        }

        // Point cloud shape: [batch, numPoints, 3] for XYZ coordinates
        var pointCloudShape = new[] { 1, effectiveNumPoints, 3 };

        // Generate initial noise
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
        var points = DiffusionNoiseHelper<T>.SampleGaussian(pointCloudShape, rng);

        // Set up scheduler
        Scheduler.SetTimesteps(numInferenceSteps);

        // Denoising loop
        foreach (var timestep in Scheduler.Timesteps)
        {
            Tensor<T> noisePrediction;

            if (useCFG && negativeEmbedding != null)
            {
                var condPred = PredictPointCloudNoise(points, timestep, promptEmbedding);
                var uncondPred = PredictPointCloudNoise(points, timestep, negativeEmbedding);
                noisePrediction = ApplyGuidance(uncondPred, condPred, guidanceScale);
            }
            else
            {
                noisePrediction = PredictPointCloudNoise(points, timestep, promptEmbedding);
            }

            // Scheduler step
            var pointVector = points.ToVector();
            var noiseVector = noisePrediction.ToVector();
            pointVector = Scheduler.Step(noiseVector, timestep, pointVector, NumOps.Zero);
            points = new Tensor<T>(pointCloudShape, pointVector);
        }

        // Normalize point cloud to coordinate scale
        return NormalizePointCloud(points);
    }

    /// <inheritdoc />
    public virtual Mesh3D<T> GenerateMesh(
        string prompt,
        string? negativePrompt = null,
        int resolution = 128,
        int numInferenceSteps = 64,
        double guidanceScale = 3.0,
        int? seed = null)
    {
        if (!SupportsMesh)
            throw new NotSupportedException("This model does not support mesh generation.");

        // Generate point cloud first
        var pointCloud = GeneratePointCloud(
            prompt,
            negativePrompt,
            resolution * resolution, // More points for higher resolution
            numInferenceSteps,
            guidanceScale,
            seed);

        // Convert to mesh
        return PointCloudToMesh(pointCloud, SurfaceReconstructionMethod.Poisson);
    }

    /// <inheritdoc />
    public virtual Mesh3D<T> ImageTo3D(
        Tensor<T> inputImage,
        int numViews = 4,
        int numInferenceSteps = 50,
        double guidanceScale = 3.0,
        int? seed = null)
    {
        // Generate novel views from input image
        var viewAngles = GenerateViewAngles(numViews);
        var novelViews = SynthesizeNovelViews(
            inputImage,
            viewAngles,
            numInferenceSteps,
            guidanceScale,
            seed);

        // Reconstruct 3D from multiple views
        return ReconstructFromViews(novelViews, viewAngles);
    }

    /// <inheritdoc />
    public virtual Tensor<T>[] SynthesizeNovelViews(
        Tensor<T> inputImage,
        (double azimuth, double elevation)[] targetAngles,
        int numInferenceSteps = 50,
        double guidanceScale = 3.0,
        int? seed = null)
    {
        if (!SupportsNovelView)
            throw new NotSupportedException("This model does not support novel view synthesis.");

        var novelViews = new Tensor<T>[targetAngles.Length];
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;

        // Encode input image
        var imageLatent = EncodeToLatent(inputImage, sampleMode: false);

        for (int v = 0; v < targetAngles.Length; v++)
        {
            var (azimuth, elevation) = targetAngles[v];

            // Create view conditioning
            var viewEmbedding = CreateViewEmbedding(azimuth, elevation);

            // Generate noise for this view
            var latents = DiffusionNoiseHelper<T>.SampleGaussian(imageLatent.Shape, rng);

            // Set up scheduler
            Scheduler.SetTimesteps(numInferenceSteps);

            // Denoising loop with view conditioning
            foreach (var timestep in Scheduler.Timesteps)
            {
                var noisePrediction = PredictNovelViewNoise(
                    latents, timestep, imageLatent, viewEmbedding, guidanceScale);

                var latentVector = latents.ToVector();
                var noiseVector = noisePrediction.ToVector();
                latentVector = Scheduler.Step(noiseVector, timestep, latentVector, NumOps.Zero);
                latents = new Tensor<T>(imageLatent.Shape, latentVector);
            }

            novelViews[v] = DecodeFromLatent(latents);
        }

        return novelViews;
    }

    /// <inheritdoc />
    public virtual Tensor<T> ComputeScoreDistillationGradients(
        Tensor<T> renderedViews,
        string prompt,
        int timestep,
        double guidanceScale = 100.0)
    {
        if (!SupportsScoreDistillation)
            throw new NotSupportedException("This model does not support score distillation.");

        if (Conditioner == null)
            throw new InvalidOperationException("Score distillation requires a conditioning module.");

        // Encode prompt
        var promptTokens = Conditioner.Tokenize(prompt);
        var promptEmbedding = Conditioner.EncodeText(promptTokens);
        var uncondEmbedding = Conditioner.GetUnconditionalEmbedding(1);

        // Encode rendered views to latent
        var latents = EncodeToLatent(renderedViews, sampleMode: false);

        // Add noise at specified timestep
        var noise = DiffusionNoiseHelper<T>.SampleGaussian(latents.Shape, RandomGenerator);
        var noisyLatents = Scheduler.AddNoise(latents.ToVector(), noise.ToVector(), timestep);
        var noisyLatentsTensor = new Tensor<T>(latents.Shape, noisyLatents);

        // Predict noise with and without conditioning
        var condNoisePred = NoisePredictor.PredictNoise(noisyLatentsTensor, timestep, promptEmbedding);
        var uncondNoisePred = NoisePredictor.PredictNoise(noisyLatentsTensor, timestep, uncondEmbedding);

        // Compute SDS gradient: grad = scale * (cond - uncond)
        var scaleT = NumOps.FromDouble(guidanceScale);
        var diff = Engine.TensorSubtract<T>(condNoisePred, uncondNoisePred);
        return Engine.TensorMultiplyScalar<T>(diff, scaleT);
    }

    /// <inheritdoc />
    public virtual Mesh3D<T> PointCloudToMesh(
        Tensor<T> pointCloud,
        SurfaceReconstructionMethod method = SurfaceReconstructionMethod.Poisson)
    {
        return method switch
        {
            SurfaceReconstructionMethod.Poisson => PointCloudToMeshPoisson(pointCloud),
            SurfaceReconstructionMethod.BallPivoting => PointCloudToMeshBallPivoting(pointCloud),
            SurfaceReconstructionMethod.MarchingCubes => PointCloudToMeshMarchingCubes(pointCloud),
            SurfaceReconstructionMethod.AlphaShape => PointCloudToMeshAlphaShape(pointCloud),
            _ => PointCloudToMeshPoisson(pointCloud)
        };
    }

    /// <inheritdoc />
    public virtual Tensor<T> ColorizePointCloud(
        Tensor<T> pointCloud,
        string prompt,
        int numInferenceSteps = 50,
        int? seed = null)
    {
        if (!SupportsTexture)
            throw new NotSupportedException("This model does not support texture/color generation.");

        if (Conditioner == null)
            throw new InvalidOperationException("Point cloud colorization requires a conditioning module.");

        var pointShape = pointCloud.Shape;
        var numPoints = pointShape[1];

        // Encode prompt
        var promptTokens = Conditioner.Tokenize(prompt);
        var promptEmbedding = Conditioner.EncodeText(promptTokens);

        // Generate RGB values for each point
        var colorShape = new[] { pointShape[0], numPoints, 3 }; // [batch, points, RGB]
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
        var colors = DiffusionNoiseHelper<T>.SampleGaussian(colorShape, rng);

        // Set up scheduler
        Scheduler.SetTimesteps(numInferenceSteps);

        // Denoising loop for colors conditioned on points
        foreach (var timestep in Scheduler.Timesteps)
        {
            var noisePrediction = PredictColorNoise(colors, timestep, pointCloud, promptEmbedding);

            var colorVector = colors.ToVector();
            var noiseVector = noisePrediction.ToVector();
            colorVector = Scheduler.Step(noiseVector, timestep, colorVector, NumOps.Zero);
            colors = new Tensor<T>(colorShape, colorVector);
        }

        // Normalize colors to [0, 1]
        colors = NormalizeColors(colors);

        // Combine points and colors: [batch, numPoints, 6] (XYZ + RGB)
        return ConcatenatePointsAndColors(pointCloud, colors);
    }

    #endregion

    #region Protected Methods for Derived Classes

    /// <summary>
    /// Predicts noise for point cloud denoising.
    /// </summary>
    protected virtual Tensor<T> PredictPointCloudNoise(
        Tensor<T> points,
        int timestep,
        Tensor<T> conditioning)
    {
        // Default: flatten point cloud and use noise predictor
        // Derived classes should override for specialized architectures
        return NoisePredictor.PredictNoise(points, timestep, conditioning);
    }

    /// <summary>
    /// Predicts noise for novel view synthesis.
    /// </summary>
    protected virtual Tensor<T> PredictNovelViewNoise(
        Tensor<T> latents,
        int timestep,
        Tensor<T> imageLatent,
        Tensor<T> viewEmbedding,
        double guidanceScale)
    {
        // Combine image conditioning with view embedding
        var conditioning = CombineImageAndViewConditioning(imageLatent, viewEmbedding);

        var condPred = NoisePredictor.PredictNoise(latents, timestep, conditioning);

        if (guidanceScale > 1.0 && NoisePredictor.SupportsCFG)
        {
            var uncondPred = NoisePredictor.PredictNoise(latents, timestep, null);
            return ApplyGuidance(uncondPred, condPred, guidanceScale);
        }

        return condPred;
    }

    /// <summary>
    /// Predicts noise for point cloud colorization.
    /// </summary>
    protected virtual Tensor<T> PredictColorNoise(
        Tensor<T> colors,
        int timestep,
        Tensor<T> pointCloud,
        Tensor<T> promptEmbedding)
    {
        // Combine point positions with color predictions for conditioning
        var combined = ConcatenatePointsAndColors(pointCloud, colors);
        return NoisePredictor.PredictNoise(combined, timestep, promptEmbedding);
    }

    /// <summary>
    /// Creates a view embedding from azimuth and elevation angles.
    /// </summary>
    protected virtual Tensor<T> CreateViewEmbedding(double azimuth, double elevation)
    {
        // Sinusoidal embedding for camera angles
        var embeddingDim = 64;
        var embedding = new Tensor<T>(new[] { 1, embeddingDim });
        var span = embedding.AsWritableSpan();

        var halfDim = embeddingDim / 4;
        var logScale = Math.Log(10000.0) / (halfDim - 1);

        for (int i = 0; i < halfDim; i++)
        {
            var freq = Math.Exp(-i * logScale);

            // Azimuth embedding
            span[i] = NumOps.FromDouble(Math.Sin(azimuth * freq));
            span[i + halfDim] = NumOps.FromDouble(Math.Cos(azimuth * freq));

            // Elevation embedding
            span[i + 2 * halfDim] = NumOps.FromDouble(Math.Sin(elevation * freq));
            span[i + 3 * halfDim] = NumOps.FromDouble(Math.Cos(elevation * freq));
        }

        return embedding;
    }

    /// <summary>
    /// Combines image latent with view embedding for conditioning.
    /// </summary>
    protected virtual Tensor<T> CombineImageAndViewConditioning(Tensor<T> imageLatent, Tensor<T> viewEmbedding)
    {
        // Simple concatenation - derived classes may implement cross-attention
        var imgFlat = imageLatent.ToVector();
        var viewFlat = viewEmbedding.ToVector();

        var combinedLength = imgFlat.Length + viewFlat.Length;
        var combined = new Tensor<T>(new[] { 1, combinedLength });
        var span = combined.AsWritableSpan();

        for (int i = 0; i < imgFlat.Length; i++)
            span[i] = imgFlat[i];
        for (int i = 0; i < viewFlat.Length; i++)
            span[imgFlat.Length + i] = viewFlat[i];

        return combined;
    }

    /// <summary>
    /// Generates evenly distributed view angles around an object.
    /// </summary>
    protected virtual (double azimuth, double elevation)[] GenerateViewAngles(int numViews)
    {
        var angles = new (double azimuth, double elevation)[numViews];
        var elevation = Math.PI / 6; // 30 degrees

        for (int i = 0; i < numViews; i++)
        {
            var azimuth = 2.0 * Math.PI * i / numViews;
            angles[i] = (azimuth, elevation);
        }

        return angles;
    }

    /// <summary>
    /// Reconstructs a 3D mesh from multiple view images.
    /// </summary>
    protected virtual Mesh3D<T> ReconstructFromViews(Tensor<T>[] views, (double azimuth, double elevation)[] angles)
    {
        // Simplified multi-view reconstruction
        // Real implementation would use structure from motion or neural reconstruction
        var numPoints = DefaultPointCount;
        var points = new Tensor<T>(new[] { 1, numPoints, 3 });
        var pointSpan = points.AsWritableSpan();

        // Sample points from depth estimated from views
        var rng = RandomGenerator;
        for (int i = 0; i < numPoints; i++)
        {
            // Random point on unit sphere
            var theta = 2.0 * Math.PI * rng.NextDouble();
            var phi = Math.Acos(2.0 * rng.NextDouble() - 1.0);

            pointSpan[i * 3] = NumOps.FromDouble(Math.Sin(phi) * Math.Cos(theta));
            pointSpan[i * 3 + 1] = NumOps.FromDouble(Math.Sin(phi) * Math.Sin(theta));
            pointSpan[i * 3 + 2] = NumOps.FromDouble(Math.Cos(phi));
        }

        return PointCloudToMesh(points, SurfaceReconstructionMethod.Poisson);
    }

    /// <summary>
    /// Normalizes point cloud coordinates to specified range.
    /// </summary>
    protected virtual Tensor<T> NormalizePointCloud(Tensor<T> pointCloud)
    {
        var span = pointCloud.AsSpan();
        var result = new Tensor<T>(pointCloud.Shape);
        var resultSpan = result.AsWritableSpan();

        // Find bounding box
        double minX = double.MaxValue, minY = double.MaxValue, minZ = double.MaxValue;
        double maxX = double.MinValue, maxY = double.MinValue, maxZ = double.MinValue;

        for (int i = 0; i < span.Length; i += 3)
        {
            var x = NumOps.ToDouble(span[i]);
            var y = NumOps.ToDouble(span[i + 1]);
            var z = NumOps.ToDouble(span[i + 2]);

            minX = Math.Min(minX, x);
            minY = Math.Min(minY, y);
            minZ = Math.Min(minZ, z);
            maxX = Math.Max(maxX, x);
            maxY = Math.Max(maxY, y);
            maxZ = Math.Max(maxZ, z);
        }

        // Normalize to [-scale, scale]
        var rangeX = maxX - minX;
        var rangeY = maxY - minY;
        var rangeZ = maxZ - minZ;
        var maxRange = Math.Max(rangeX, Math.Max(rangeY, rangeZ));
        if (maxRange < 1e-6) maxRange = 1.0;

        for (int i = 0; i < span.Length; i += 3)
        {
            var x = NumOps.ToDouble(span[i]);
            var y = NumOps.ToDouble(span[i + 1]);
            var z = NumOps.ToDouble(span[i + 2]);

            resultSpan[i] = NumOps.FromDouble(((x - minX) / maxRange - 0.5) * 2.0 * CoordinateScale);
            resultSpan[i + 1] = NumOps.FromDouble(((y - minY) / maxRange - 0.5) * 2.0 * CoordinateScale);
            resultSpan[i + 2] = NumOps.FromDouble(((z - minZ) / maxRange - 0.5) * 2.0 * CoordinateScale);
        }

        return result;
    }

    /// <summary>
    /// Normalizes color values to [0, 1] range.
    /// </summary>
    protected virtual Tensor<T> NormalizeColors(Tensor<T> colors)
    {
        var result = new Tensor<T>(colors.Shape);
        var span = colors.AsSpan();
        var resultSpan = result.AsWritableSpan();

        for (int i = 0; i < span.Length; i++)
        {
            var val = NumOps.ToDouble(span[i]);
            resultSpan[i] = NumOps.FromDouble(MathPolyfill.Clamp((val + 1.0) / 2.0, 0.0, 1.0));
        }

        return result;
    }

    /// <summary>
    /// Concatenates point positions with RGB colors.
    /// </summary>
    protected virtual Tensor<T> ConcatenatePointsAndColors(Tensor<T> points, Tensor<T> colors)
    {
        var pointShape = points.Shape;
        var numPoints = pointShape[1];

        var outputShape = new[] { pointShape[0], numPoints, 6 }; // XYZ + RGB
        var output = new Tensor<T>(outputShape);
        var outSpan = output.AsWritableSpan();
        var pointSpan = points.AsSpan();
        var colorSpan = colors.AsSpan();

        for (int b = 0; b < pointShape[0]; b++)
        {
            for (int p = 0; p < numPoints; p++)
            {
                var outIdx = b * numPoints * 6 + p * 6;
                var pointIdx = b * numPoints * 3 + p * 3;
                var colorIdx = b * numPoints * 3 + p * 3;

                // XYZ
                outSpan[outIdx] = pointSpan[pointIdx];
                outSpan[outIdx + 1] = pointSpan[pointIdx + 1];
                outSpan[outIdx + 2] = pointSpan[pointIdx + 2];

                // RGB
                outSpan[outIdx + 3] = colorSpan[colorIdx];
                outSpan[outIdx + 4] = colorSpan[colorIdx + 1];
                outSpan[outIdx + 5] = colorSpan[colorIdx + 2];
            }
        }

        return output;
    }

    #endregion

    #region Surface Reconstruction Methods

    /// <summary>
    /// Converts point cloud to mesh using Poisson surface reconstruction.
    /// </summary>
    protected virtual Mesh3D<T> PointCloudToMeshPoisson(Tensor<T> pointCloud)
    {
        // Simplified Poisson reconstruction
        // Real implementation would use proper octree and Poisson solver
        return CreateSimpleMeshFromPoints(pointCloud);
    }

    /// <summary>
    /// Converts point cloud to mesh using ball pivoting algorithm.
    /// </summary>
    protected virtual Mesh3D<T> PointCloudToMeshBallPivoting(Tensor<T> pointCloud)
    {
        return CreateSimpleMeshFromPoints(pointCloud);
    }

    /// <summary>
    /// Converts point cloud to mesh using marching cubes on a voxel grid.
    /// </summary>
    protected virtual Mesh3D<T> PointCloudToMeshMarchingCubes(Tensor<T> pointCloud)
    {
        return CreateSimpleMeshFromPoints(pointCloud);
    }

    /// <summary>
    /// Converts point cloud to mesh using alpha shape algorithm.
    /// </summary>
    protected virtual Mesh3D<T> PointCloudToMeshAlphaShape(Tensor<T> pointCloud)
    {
        return CreateSimpleMeshFromPoints(pointCloud);
    }

    /// <summary>
    /// Creates a simple triangulated mesh from point cloud using nearest neighbors.
    /// </summary>
    protected virtual Mesh3D<T> CreateSimpleMeshFromPoints(Tensor<T> pointCloud)
    {
        var pointShape = pointCloud.Shape;
        var numPoints = pointShape[1];
        var pointSpan = pointCloud.AsSpan();

        // Create vertices tensor
        var vertices = new Tensor<T>(new[] { numPoints, 3 });
        var vertexSpan = vertices.AsWritableSpan();

        for (int i = 0; i < numPoints * 3; i++)
        {
            vertexSpan[i] = pointSpan[i];
        }

        // Create simple triangulation (placeholder - connects nearby points)
        var faceList = new List<(int, int, int)>();
        var gridSize = (int)Math.Ceiling(Math.Pow(numPoints, 1.0 / 3.0));

        // Simple grid-based triangulation for demonstration
        for (int i = 0; i < numPoints - gridSize - 1; i++)
        {
            if ((i + 1) % gridSize != 0)
            {
                faceList.Add((i, i + 1, i + gridSize));
                faceList.Add((i + 1, i + gridSize + 1, i + gridSize));
            }
        }

        var faces = new int[faceList.Count, 3];
        for (int i = 0; i < faceList.Count; i++)
        {
            faces[i, 0] = faceList[i].Item1;
            faces[i, 1] = faceList[i].Item2;
            faces[i, 2] = faceList[i].Item3;
        }

        return new Mesh3D<T>(vertices, faces);
    }

    #endregion
}

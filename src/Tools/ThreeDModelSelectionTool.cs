using AiDotNet.Interfaces;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Tools;

/// <summary>
/// A specialized tool that recommends optimal 3D AI model types based on input data format, task type,
/// dataset characteristics, and computational constraints.
/// </summary>
/// <remarks>
/// <para>
/// This tool provides AI agents with expert-level 3D model selection capabilities for tasks involving
/// point clouds, meshes, voxel grids, and neural radiance fields. It analyzes the input data format,
/// task requirements, dataset size, and available computational resources to recommend the most
/// appropriate 3D deep learning architecture. The tool considers trade-offs between accuracy,
/// training time, memory requirements, and inference speed.
/// </para>
/// <para><b>For Beginners:</b> This tool helps you choose the right 3D AI model for your specific task.
///
/// 3D AI encompasses many different data formats and tasks:
/// - **Point Clouds**: Collections of 3D points (e.g., from LiDAR sensors)
/// - **Meshes**: 3D surfaces made of triangles (e.g., from 3D modeling software)
/// - **Voxel Grids**: 3D arrays of values (like 3D pixels)
/// - **Neural Radiance Fields**: Implicit representations for novel view synthesis
///
/// Common 3D tasks include:
/// - **Classification**: What object is this? (e.g., chair, table, car)
/// - **Segmentation**: What part is each point/face? (semantic or instance)
/// - **Completion**: Fill in missing parts of incomplete 3D data
/// - **Reconstruction**: Create 3D from images or partial scans
/// - **Novel View Synthesis**: Render new viewpoints from images
///
/// Example input (JSON format):
/// <code>
/// {
///   "data_format": "point_cloud",
///   "task_type": "classification",
///   "n_samples": 10000,
///   "n_points": 2048,
///   "has_normals": true,
///   "has_colors": false,
///   "computational_budget": "moderate",
///   "requires_real_time": false
/// }
/// </code>
///
/// Example output:
/// "Recommended Model: PointNet++\n\n" +
/// "Reasoning:\n" +
/// "- Point cloud format with 2048 points per sample\n" +
/// "- Classification task - PointNet++ excels at global feature extraction\n" +
/// "- Normals available - can improve local feature learning\n" +
/// "- Dataset size (10,000) supports hierarchical architecture\n\n" +
/// "Alternative Models:\n" +
/// "- DGCNN: Better for capturing local geometric structures\n" +
/// "- PointNet: Faster and simpler if accuracy requirements are moderate"
/// </para>
/// </remarks>
public class ThreeDModelSelectionTool : ToolBase
{
    /// <inheritdoc/>
    public override string Name => "ThreeDModelSelectionTool";

    /// <inheritdoc/>
    public override string Description =>
        "Recommends optimal 3D AI model types based on data format and task requirements. " +
        "Input should be a JSON object: { \"data_format\": \"point_cloud|mesh|voxel|images\", " +
        "\"task_type\": \"classification|segmentation|completion|reconstruction|novel_view_synthesis\", " +
        "\"n_samples\": number, \"n_points\": number (for point clouds), " +
        "\"resolution\": number (for voxels), \"n_vertices\": number (for meshes), " +
        "\"has_normals\": boolean, \"has_colors\": boolean, " +
        "\"computational_budget\": \"low|moderate|high\", \"requires_real_time\": boolean }. " +
        "Returns recommended model type with detailed reasoning and alternative suggestions.";

    /// <inheritdoc/>
    protected override string ExecuteCore(string input)
    {
        var root = JObject.Parse(input);

        // Extract parameters
        string dataFormat = TryGetString(root, "data_format", "point_cloud").ToLowerInvariant();
        string taskType = TryGetString(root, "task_type", "classification").ToLowerInvariant();
        int nSamples = TryGetInt(root, "n_samples", 1000);
        int nPoints = TryGetInt(root, "n_points", 1024);
        int resolution = TryGetInt(root, "resolution", 32);
        int nVertices = TryGetInt(root, "n_vertices", 5000);
        bool hasNormals = TryGetBool(root, "has_normals", false);
        bool hasColors = TryGetBool(root, "has_colors", false);
        string computationalBudget = TryGetString(root, "computational_budget", "moderate").ToLowerInvariant();
        bool requiresRealTime = TryGetBool(root, "requires_real_time", false);

        // Validate inputs
        if (!new[] { "point_cloud", "mesh", "voxel", "images", "multi_view" }.Contains(dataFormat))
        {
            return $"Error: Unrecognized data_format '{dataFormat}'. " +
                   "Supported: point_cloud, mesh, voxel, images, multi_view.";
        }

        if (!new[] { "classification", "segmentation", "part_segmentation", "semantic_segmentation",
                     "completion", "reconstruction", "novel_view_synthesis", "detection", "depth_estimation" }
            .Contains(taskType))
        {
            return $"Error: Unrecognized task_type '{taskType}'. " +
                   "Supported: classification, segmentation, part_segmentation, completion, " +
                   "reconstruction, novel_view_synthesis, detection, depth_estimation.";
        }

        var recommendation = new System.Text.StringBuilder();
        recommendation.AppendLine("=== 3D MODEL SELECTION RECOMMENDATION ===\n");

        // Route to appropriate recommendation method
        switch (dataFormat)
        {
            case "point_cloud":
                GeneratePointCloudRecommendation(recommendation, taskType, nSamples, nPoints,
                    hasNormals, hasColors, computationalBudget, requiresRealTime);
                break;
            case "mesh":
                GenerateMeshRecommendation(recommendation, taskType, nSamples, nVertices,
                    hasNormals, hasColors, computationalBudget, requiresRealTime);
                break;
            case "voxel":
                GenerateVoxelRecommendation(recommendation, taskType, nSamples, resolution,
                    computationalBudget, requiresRealTime);
                break;
            case "images":
            case "multi_view":
                GenerateImageBasedRecommendation(recommendation, taskType, nSamples,
                    computationalBudget, requiresRealTime);
                break;
        }

        // Add general guidance
        recommendation.AppendLine("\n**Input Characteristics:**");
        recommendation.AppendLine($"  • Data format: {dataFormat}");
        recommendation.AppendLine($"  • Task type: {taskType}");
        recommendation.AppendLine($"  • Dataset size: {nSamples:N0} samples");

        switch (dataFormat)
        {
            case "point_cloud":
                recommendation.AppendLine($"  • Points per sample: {nPoints:N0}");
                break;
            case "mesh":
                recommendation.AppendLine($"  • Vertices per mesh: ~{nVertices:N0}");
                break;
            case "voxel":
                recommendation.AppendLine($"  • Resolution: {resolution}³");
                break;
        }

        recommendation.AppendLine($"  • Normals available: {(hasNormals ? "Yes" : "No")}");
        recommendation.AppendLine($"  • Colors available: {(hasColors ? "Yes" : "No")}");
        recommendation.AppendLine($"  • Computational budget: {computationalBudget}");
        recommendation.AppendLine($"  • Real-time required: {(requiresRealTime ? "Yes" : "No")}");

        recommendation.AppendLine();
        recommendation.AppendLine("**Next Steps:**");
        recommendation.AppendLine("  1. Preprocess data according to model requirements");
        recommendation.AppendLine("  2. Start with recommended model and default hyperparameters");
        recommendation.AppendLine("  3. Use validation set to monitor for overfitting");
        recommendation.AppendLine("  4. Try alternatives if results are unsatisfactory");

        return recommendation.ToString();
    }

    /// <summary>
    /// Generates model recommendations for point cloud data.
    /// </summary>
    /// <param name="sb">StringBuilder to append recommendations to.</param>
    /// <param name="taskType">The type of task to perform.</param>
    /// <param name="nSamples">Number of samples in the dataset.</param>
    /// <param name="nPoints">Number of points per sample.</param>
    /// <param name="hasNormals">Whether normal vectors are available.</param>
    /// <param name="hasColors">Whether color information is available.</param>
    /// <param name="budget">Computational budget level.</param>
    /// <param name="realTime">Whether real-time inference is required.</param>
    private void GeneratePointCloudRecommendation(
        System.Text.StringBuilder sb,
        string taskType,
        int nSamples,
        int nPoints,
        bool hasNormals,
        bool hasColors,
        string budget,
        bool realTime)
    {
        string recommendedModel;
        var reasoning = new List<string>();
        var alternatives = new List<(string Model, string Reason)>();

        // Determine feature dimension
        int featureDim = 3; // xyz
        if (hasNormals) featureDim += 3;
        if (hasColors) featureDim += 3;

        if (realTime || budget == "low")
        {
            // Fast inference required
            recommendedModel = "PointNet";
            reasoning.Add("Real-time or low-budget constraints require efficient architecture");
            reasoning.Add("PointNet processes points independently with shared MLPs - very fast");
            reasoning.Add("Global max pooling captures shape-level features efficiently");
            if (taskType.Contains("segmentation"))
            {
                reasoning.Add("For segmentation, PointNet concatenates global and local features");
            }
            alternatives.Add(("DGCNN with reduced k", "Better local features with k=10-20 for speed"));
            alternatives.Add(("PointNet++ with single scale", "Hierarchical but faster with one MSG scale"));
        }
        else if (taskType == "classification")
        {
            if (nPoints >= 4096 && budget == "high")
            {
                recommendedModel = "DGCNN";
                reasoning.Add("Classification with dense point clouds benefits from edge convolution");
                reasoning.Add("DGCNN dynamically constructs graphs to capture local structure");
                reasoning.Add("EdgeConv layers learn geometric relationships between nearby points");
                alternatives.Add(("PointNet++", "Hierarchical features with multi-scale grouping"));
                alternatives.Add(("PointNet", "Simpler and faster if local structure is less important"));
            }
            else
            {
                recommendedModel = "PointNet++";
                reasoning.Add("Classification task with moderate point density");
                reasoning.Add("PointNet++ hierarchically captures local-to-global features");
                reasoning.Add("Multi-scale grouping (MSG) adapts to varying point densities");
                if (hasNormals)
                {
                    reasoning.Add("Normal vectors will improve local feature learning");
                }
                alternatives.Add(("DGCNN", "Better for capturing fine geometric details"));
                alternatives.Add(("PointNet", "Faster baseline if accuracy requirements are flexible"));
            }
        }
        else if (taskType.Contains("segmentation"))
        {
            recommendedModel = "PointNet++";
            reasoning.Add("Segmentation requires per-point predictions with context");
            reasoning.Add("PointNet++ feature propagation enables dense predictions");
            reasoning.Add("Skip connections preserve fine-grained spatial information");
            if (taskType == "part_segmentation")
            {
                reasoning.Add("Part segmentation benefits from multi-scale feature hierarchy");
            }
            alternatives.Add(("DGCNN", "Edge convolution can better capture part boundaries"));
            alternatives.Add(("PointNet", "Baseline for comparison, may underperform on complex parts"));
        }
        else if (taskType == "completion")
        {
            recommendedModel = "PointNet++ Encoder-Decoder";
            reasoning.Add("Point cloud completion requires understanding global shape and local details");
            reasoning.Add("Encoder-decoder architecture with skip connections preserves details");
            reasoning.Add("Feature propagation reconstructs missing regions");
            alternatives.Add(("PCN (Point Completion Network)", "Specifically designed for completion"));
            alternatives.Add(("FoldingNet", "Generates points by folding 2D grids in 3D"));
        }
        else
        {
            // Default for other tasks
            recommendedModel = "PointNet++";
            reasoning.Add("PointNet++ is a versatile architecture for various point cloud tasks");
            reasoning.Add("Hierarchical feature learning adapts to different task requirements");
            alternatives.Add(("DGCNN", "Alternative with explicit graph structure"));
        }

        sb.AppendLine($"**Primary Recommendation: {recommendedModel}**\n");
        sb.AppendLine("**Reasoning:**");
        foreach (var reason in reasoning)
        {
            sb.AppendLine($"  • {reason}");
        }
        sb.AppendLine();

        sb.AppendLine("**Preprocessing Recommendations:**");
        sb.AppendLine($"  • Sample/pad to consistent point count: {Math.Min(nPoints, 2048)} recommended");
        sb.AppendLine("  • Normalize to unit sphere (center and scale)");
        if (hasNormals)
        {
            sb.AppendLine("  • Include normals as additional features");
        }
        if (hasColors)
        {
            sb.AppendLine("  • Normalize color values to [0,1] range");
        }
        sb.AppendLine($"  • Input feature dimension: {featureDim}");
        sb.AppendLine();

        if (alternatives.Count > 0)
        {
            sb.AppendLine("**Alternative Models:**");
            foreach (var (model, reason) in alternatives)
            {
                sb.AppendLine($"  • {model}: {reason}");
            }
        }
    }

    /// <summary>
    /// Generates model recommendations for mesh data.
    /// </summary>
    /// <param name="sb">StringBuilder to append recommendations to.</param>
    /// <param name="taskType">The type of task to perform.</param>
    /// <param name="nSamples">Number of samples in the dataset.</param>
    /// <param name="nVertices">Average number of vertices per mesh.</param>
    /// <param name="hasNormals">Whether normal vectors are available.</param>
    /// <param name="hasColors">Whether color information is available.</param>
    /// <param name="budget">Computational budget level.</param>
    /// <param name="realTime">Whether real-time inference is required.</param>
    private void GenerateMeshRecommendation(
        System.Text.StringBuilder sb,
        string taskType,
        int nSamples,
        int nVertices,
        bool hasNormals,
        bool hasColors,
        string budget,
        bool realTime)
    {
        string recommendedModel;
        var reasoning = new List<string>();
        var alternatives = new List<(string Model, string Reason)>();

        if (realTime)
        {
            recommendedModel = "Convert to Point Cloud + PointNet";
            reasoning.Add("Real-time constraints - sample vertices as point cloud");
            reasoning.Add("PointNet provides fastest inference on sampled points");
            reasoning.Add("Loses some mesh topology information but enables real-time processing");
            alternatives.Add(("MeshCNN with reduced resolution", "Preserves topology but slower"));
        }
        else if (taskType == "classification")
        {
            if (budget == "high")
            {
                recommendedModel = "DiffusionNet";
                reasoning.Add("DiffusionNet learns features via heat diffusion on mesh surface");
                reasoning.Add("Invariant to mesh discretization - robust to different tessellations");
                reasoning.Add("State-of-the-art for mesh classification when compute is available");
                alternatives.Add(("MeshCNN", "Edge-based convolution with mesh pooling"));
                alternatives.Add(("SpiralNet++", "Fixed spiral ordering for consistent convolution"));
            }
            else
            {
                recommendedModel = "MeshCNN";
                reasoning.Add("MeshCNN operates directly on mesh edges and faces");
                reasoning.Add("Mesh pooling progressively simplifies geometry");
                reasoning.Add("Good balance of accuracy and computational cost");
                alternatives.Add(("DiffusionNet", "Better accuracy if compute budget increases"));
                alternatives.Add(("Convert to point cloud + PointNet++", "Simpler, loses topology"));
            }
        }
        else if (taskType.Contains("segmentation"))
        {
            recommendedModel = "MeshCNN";
            reasoning.Add("MeshCNN designed for mesh segmentation tasks");
            reasoning.Add("Edge-based features capture geometric details at boundaries");
            reasoning.Add("Mesh pooling/unpooling enables encoder-decoder architecture");
            reasoning.Add("Preserves mesh connectivity crucial for part boundaries");
            alternatives.Add(("DiffusionNet", "Better generalization across different mesh resolutions"));
            alternatives.Add(("SpiralNet++", "Consistent vertex ordering simplifies segmentation"));
        }
        else
        {
            // Default
            recommendedModel = "MeshCNN";
            reasoning.Add("MeshCNN is versatile for various mesh learning tasks");
            reasoning.Add("Directly operates on mesh structure without conversion");
            alternatives.Add(("Point cloud conversion + PointNet++", "If mesh topology isn't critical"));
        }

        sb.AppendLine($"**Primary Recommendation: {recommendedModel}**\n");
        sb.AppendLine("**Reasoning:**");
        foreach (var reason in reasoning)
        {
            sb.AppendLine($"  • {reason}");
        }
        sb.AppendLine();

        sb.AppendLine("**Preprocessing Recommendations:**");
        if (nVertices > 10000)
        {
            sb.AppendLine($"  • Simplify meshes to ~{Math.Min(nVertices, 5000)} vertices for efficiency");
        }
        sb.AppendLine("  • Ensure manifold meshes (no holes, consistent normals)");
        sb.AppendLine("  • Normalize to unit bounding box");
        if (!hasNormals)
        {
            sb.AppendLine("  • Compute vertex/face normals from geometry");
        }
        sb.AppendLine();

        if (alternatives.Count > 0)
        {
            sb.AppendLine("**Alternative Models:**");
            foreach (var (model, reason) in alternatives)
            {
                sb.AppendLine($"  • {model}: {reason}");
            }
        }
    }

    /// <summary>
    /// Generates model recommendations for voxel data.
    /// </summary>
    /// <param name="sb">StringBuilder to append recommendations to.</param>
    /// <param name="taskType">The type of task to perform.</param>
    /// <param name="nSamples">Number of samples in the dataset.</param>
    /// <param name="resolution">Voxel grid resolution (one dimension).</param>
    /// <param name="budget">Computational budget level.</param>
    /// <param name="realTime">Whether real-time inference is required.</param>
    private void GenerateVoxelRecommendation(
        System.Text.StringBuilder sb,
        string taskType,
        int nSamples,
        int resolution,
        string budget,
        bool realTime)
    {
        string recommendedModel;
        var reasoning = new List<string>();
        var alternatives = new List<(string Model, string Reason)>();

        // Memory consideration: 64³ = 262K, 128³ = 2M voxels
        bool highResolution = resolution >= 64;

        if (realTime || budget == "low")
        {
            recommendedModel = "VoxelCNN (3D CNN)";
            reasoning.Add("3D CNN with standard convolutions is computationally efficient");
            reasoning.Add("GPU acceleration via cuDNN optimized 3D convolutions");
            if (highResolution)
            {
                reasoning.Add($"⚠️ Resolution {resolution}³ may require reducing to 32³ for real-time");
            }
            alternatives.Add(("Sparse 3D CNN", "If data is sparse (occupancy grids)"));
        }
        else if (taskType == "classification")
        {
            recommendedModel = "VoxelCNN";
            reasoning.Add("3D CNN is the standard for voxel classification");
            reasoning.Add("Hierarchical feature extraction with max pooling");
            reasoning.Add("Straightforward extension of 2D CNN concepts to 3D");
            alternatives.Add(("3D ResNet", "Deeper architecture if dataset is large"));
            alternatives.Add(("Convert to point cloud", "If memory is constrained"));
        }
        else if (taskType.Contains("segmentation"))
        {
            recommendedModel = "3D U-Net";
            reasoning.Add("3D U-Net designed for volumetric segmentation");
            reasoning.Add("Encoder-decoder with skip connections preserves spatial detail");
            reasoning.Add("Widely used in medical imaging (CT/MRI segmentation)");
            if (highResolution)
            {
                reasoning.Add($"⚠️ High resolution ({resolution}³) requires substantial GPU memory");
                reasoning.Add("  Consider patch-based processing or reducing resolution");
            }
            alternatives.Add(("V-Net", "Variant with residual blocks for deeper networks"));
            alternatives.Add(("Sparse U-Net", "If occupancy is sparse (<10% filled)"));
        }
        else if (taskType == "completion")
        {
            recommendedModel = "3D U-Net";
            reasoning.Add("Encoder-decoder architecture suitable for completion");
            reasoning.Add("Skip connections help preserve known voxels");
            reasoning.Add("Can be trained with occupancy or signed distance outputs");
            alternatives.Add(("3D GAN", "For higher quality completions with adversarial training"));
        }
        else
        {
            recommendedModel = "VoxelCNN";
            reasoning.Add("VoxelCNN is the default for voxel-based learning");
            alternatives.Add(("3D U-Net", "If dense spatial outputs are needed"));
        }

        sb.AppendLine($"**Primary Recommendation: {recommendedModel}**\n");
        sb.AppendLine("**Reasoning:**");
        foreach (var reason in reasoning)
        {
            sb.AppendLine($"  • {reason}");
        }
        sb.AppendLine();

        sb.AppendLine("**Preprocessing Recommendations:**");
        sb.AppendLine($"  • Current resolution: {resolution}³ ({(long)resolution * resolution * resolution:N0} voxels)");
        if (resolution > 64)
        {
            sb.AppendLine($"  • Consider reducing to 64³ or 32³ if memory constrained");
        }
        sb.AppendLine("  • Use binary occupancy or signed distance field (SDF)");
        sb.AppendLine("  • Normalize coordinates to unit cube");
        sb.AppendLine($"  • Estimated memory per sample: ~{(long)resolution * resolution * resolution * 4 / 1024:N0} KB (float32)");
        sb.AppendLine();

        if (alternatives.Count > 0)
        {
            sb.AppendLine("**Alternative Models:**");
            foreach (var (model, reason) in alternatives)
            {
                sb.AppendLine($"  • {model}: {reason}");
            }
        }
    }

    /// <summary>
    /// Generates model recommendations for image-based 3D tasks (NeRF, reconstruction).
    /// </summary>
    /// <param name="sb">StringBuilder to append recommendations to.</param>
    /// <param name="taskType">The type of task to perform.</param>
    /// <param name="nSamples">Number of samples (scenes) in the dataset.</param>
    /// <param name="budget">Computational budget level.</param>
    /// <param name="realTime">Whether real-time inference is required.</param>
    private void GenerateImageBasedRecommendation(
        System.Text.StringBuilder sb,
        string taskType,
        int nSamples,
        string budget,
        bool realTime)
    {
        string recommendedModel;
        var reasoning = new List<string>();
        var alternatives = new List<(string Model, string Reason)>();

        if (taskType == "novel_view_synthesis")
        {
            if (realTime)
            {
                recommendedModel = "Gaussian Splatting";
                reasoning.Add("Gaussian Splatting achieves real-time rendering (100+ FPS)");
                reasoning.Add("Explicit 3D Gaussian representation enables fast rasterization");
                reasoning.Add("Quality comparable to NeRF with much faster rendering");
                reasoning.Add("Training time: minutes to hours vs. hours to days for NeRF");
                alternatives.Add(("Instant-NGP", "Fast training and inference, implicit representation"));
                alternatives.Add(("Plenoxels", "Explicit voxel-based, fast but more memory"));
            }
            else if (budget == "low")
            {
                recommendedModel = "Instant-NGP";
                reasoning.Add("Instant-NGP trains in seconds to minutes");
                reasoning.Add("Hash encoding enables compact representation");
                reasoning.Add("Good quality with minimal computational resources");
                alternatives.Add(("Gaussian Splatting", "Better rendering speed, similar training time"));
                alternatives.Add(("NeRF (vanilla)", "Baseline comparison, slower but established"));
            }
            else
            {
                recommendedModel = "NeRF";
                reasoning.Add("NeRF produces highest quality novel views");
                reasoning.Add("Well-studied with many extensions for specific needs");
                reasoning.Add("Good for research or when quality is paramount");
                reasoning.Add("⚠️ Training takes hours to days per scene");
                alternatives.Add(("Instant-NGP", "Much faster training with minor quality trade-off"));
                alternatives.Add(("Gaussian Splatting", "Real-time rendering with explicit geometry"));
                alternatives.Add(("Mip-NeRF", "Better handling of scale variations"));
            }
        }
        else if (taskType == "reconstruction")
        {
            recommendedModel = "Gaussian Splatting";
            reasoning.Add("Gaussian Splatting provides explicit 3D geometry");
            reasoning.Add("Gaussians can be exported to mesh or point cloud");
            reasoning.Add("Faster iteration for reconstruction tasks");
            alternatives.Add(("NeRF + Marching Cubes", "Extract mesh from density field"));
            alternatives.Add(("Instant-NGP", "Fast implicit reconstruction"));
        }
        else if (taskType == "depth_estimation")
        {
            recommendedModel = "Monocular Depth Network";
            reasoning.Add("Single-image depth estimation is well-suited for CNN architectures");
            reasoning.Add("Encoder-decoder with skip connections (U-Net style)");
            reasoning.Add("Can be trained supervised or self-supervised");
            alternatives.Add(("Multi-view Stereo", "If multiple views are available"));
            alternatives.Add(("NeRF depth extraction", "If training full NeRF is acceptable"));
        }
        else
        {
            // Default for image-based 3D
            if (realTime)
            {
                recommendedModel = "Gaussian Splatting";
                reasoning.Add("Best real-time performance for image-based 3D");
            }
            else
            {
                recommendedModel = "Instant-NGP";
                reasoning.Add("Good balance of quality, speed, and flexibility");
            }
            alternatives.Add(("NeRF", "Higher quality if time permits"));
        }

        sb.AppendLine($"**Primary Recommendation: {recommendedModel}**\n");
        sb.AppendLine("**Reasoning:**");
        foreach (var reason in reasoning)
        {
            sb.AppendLine($"  • {reason}");
        }
        sb.AppendLine();

        sb.AppendLine("**Preprocessing Recommendations:**");
        sb.AppendLine("  • Calibrate cameras (intrinsics and extrinsics)");
        sb.AppendLine("  • Use COLMAP or similar for structure-from-motion");
        sb.AppendLine("  • Ensure sufficient view coverage of the scene");
        sb.AppendLine("  • Normalize scene to fit unit cube");
        sb.AppendLine("  • 50-200 images typically needed per scene");
        sb.AppendLine();

        if (alternatives.Count > 0)
        {
            sb.AppendLine("**Alternative Models:**");
            foreach (var (model, reason) in alternatives)
            {
                sb.AppendLine($"  • {model}: {reason}");
            }
        }
    }

    /// <inheritdoc/>
    protected override string GetJsonErrorMessage(JsonReaderException ex)
    {
        return $"Error: Invalid JSON format. {ex.Message}\n" +
               "Expected format: { \"data_format\": \"point_cloud|mesh|voxel|images\", " +
               "\"task_type\": \"classification|segmentation|...\", \"n_samples\": number, ... }";
    }
}

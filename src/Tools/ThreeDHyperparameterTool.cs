using AiDotNet.Interfaces;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Tools;

/// <summary>
/// A specialized tool that suggests optimal hyperparameter values for 3D AI models based on
/// dataset characteristics, model type, and computational constraints.
/// </summary>
/// <remarks>
/// <para>
/// This tool provides AI agents with expert-level hyperparameter tuning guidance specifically
/// for 3D deep learning models including point cloud networks (PointNet, PointNet++, DGCNN),
/// mesh networks (MeshCNN, DiffusionNet), volumetric networks (3D CNN, 3D U-Net), and
/// neural radiance fields (NeRF, Instant-NGP, Gaussian Splatting). The recommendations
/// consider the unique characteristics of 3D data such as point density, mesh resolution,
/// voxel grid size, and rendering requirements.
/// </para>
/// <para><b>For Beginners:</b> This tool helps you configure 3D AI models for best performance.
///
/// 3D models have unique hyperparameters compared to 2D models:
/// - **Point cloud models**: Number of points, sampling radius, k-neighbors
/// - **Mesh models**: Edge features, pooling targets, diffusion time
/// - **Voxel models**: Grid resolution, kernel sizes, downsampling factors
/// - **NeRF/Splatting**: Positional encoding frequencies, rendering samples, learning rates
///
/// Example input (JSON format):
/// <code>
/// {
///   "model_type": "PointNet++",
///   "task_type": "classification",
///   "n_samples": 10000,
///   "n_points": 2048,
///   "has_normals": true,
///   "computational_budget": "moderate"
/// }
/// </code>
///
/// Example output:
/// "Recommended Hyperparameters for PointNet++:\n\n" +
/// "npoint_list: [1024, 256, 64, 16] (points at each level)\n" +
/// "  • Hierarchically reduces points through set abstraction\n" +
/// "  • Start with 1024 for 2048-point input\n\n" +
/// "radius_list: [0.1, 0.2, 0.4, 0.8]\n" +
/// "  • Ball query radius at each level\n" +
/// "  • Increases to capture larger neighborhoods\n"
/// </para>
/// </remarks>
public class ThreeDHyperparameterTool : ToolBase
{
    /// <inheritdoc/>
    public override string Name => "ThreeDHyperparameterTool";

    /// <inheritdoc/>
    public override string Description =>
        "Suggests optimal hyperparameter values for 3D AI models. " +
        "Input should be a JSON object: { \"model_type\": \"PointNet|PointNet++|DGCNN|MeshCNN|VoxelCNN|NeRF|GaussianSplatting|...\", " +
        "\"task_type\": \"classification|segmentation|reconstruction|novel_view_synthesis\", " +
        "\"n_samples\": number, \"n_points\": number (for point clouds), " +
        "\"resolution\": number (for voxels), \"has_normals\": boolean, " +
        "\"computational_budget\": \"low|moderate|high\" }. " +
        "Returns recommended hyperparameter values with explanations and tuning ranges.";

    /// <inheritdoc/>
    protected override string ExecuteCore(string input)
    {
        var root = JObject.Parse(input);

        // Extract parameters
        string modelType = TryGetString(root, "model_type", "PointNet++");
        string taskType = TryGetString(root, "task_type", "classification").ToLowerInvariant();
        int nSamples = TryGetInt(root, "n_samples", 1000);
        int nPoints = TryGetInt(root, "n_points", 1024);
        int resolution = TryGetInt(root, "resolution", 32);
        bool hasNormals = TryGetBool(root, "has_normals", false);
        string budget = TryGetString(root, "computational_budget", "moderate").ToLowerInvariant();

        var recommendations = new System.Text.StringBuilder();
        recommendations.AppendLine("=== 3D MODEL HYPERPARAMETER RECOMMENDATIONS ===\n");
        recommendations.AppendLine($"**Model Type:** {modelType}");
        recommendations.AppendLine($"**Task:** {taskType}");
        recommendations.AppendLine($"**Dataset:** {nSamples:N0} samples");
        recommendations.AppendLine($"**Computational Budget:** {budget}\n");

        // Route to appropriate recommendation method
        switch (modelType.ToLowerInvariant().Replace(" ", "").Replace("++", "plusplus").Replace("+", "plus"))
        {
            case "pointnet":
                GeneratePointNetRecommendations(recommendations, taskType, nSamples, nPoints, hasNormals);
                break;
            case "pointnetplusplus":
            case "pointnet2":
                GeneratePointNetPlusPlusRecommendations(recommendations, taskType, nSamples, nPoints, hasNormals, budget);
                break;
            case "dgcnn":
                GenerateDGCNNRecommendations(recommendations, taskType, nSamples, nPoints, budget);
                break;
            case "meshcnn":
                GenerateMeshCNNRecommendations(recommendations, taskType, nSamples, budget);
                break;
            case "diffusionnet":
                GenerateDiffusionNetRecommendations(recommendations, taskType, nSamples, budget);
                break;
            case "voxelcnn":
            case "3dcnn":
                GenerateVoxelCNNRecommendations(recommendations, taskType, nSamples, resolution, budget);
                break;
            case "3dunet":
            case "unet3d":
                Generate3DUNetRecommendations(recommendations, taskType, nSamples, resolution, budget);
                break;
            case "nerf":
                GenerateNeRFRecommendations(recommendations, nSamples, budget);
                break;
            case "instantngp":
            case "instant-ngp":
                GenerateInstantNGPRecommendations(recommendations, nSamples, budget);
                break;
            case "gaussiansplatting":
            case "3dgs":
                GenerateGaussianSplattingRecommendations(recommendations, nSamples, budget);
                break;
            default:
                return $"Model type '{modelType}' not recognized. Supported models: " +
                       "PointNet, PointNet++, DGCNN, MeshCNN, DiffusionNet, VoxelCNN, 3D U-Net, " +
                       "NeRF, Instant-NGP, GaussianSplatting.";
        }

        recommendations.AppendLine("\n**General Training Advice:**");
        recommendations.AppendLine("  • Use learning rate warmup for first 5-10% of training");
        recommendations.AppendLine("  • Monitor validation loss for early stopping");
        recommendations.AppendLine("  • Data augmentation is crucial for 3D (rotation, jittering, scaling)");
        recommendations.AppendLine("  • Start with recommended values, then fine-tune");

        return recommendations.ToString();
    }

    /// <summary>
    /// Generates hyperparameter recommendations for PointNet.
    /// </summary>
    private void GeneratePointNetRecommendations(
        System.Text.StringBuilder sb,
        string taskType,
        int nSamples,
        int nPoints,
        bool hasNormals)
    {
        int inputDim = hasNormals ? 6 : 3;

        sb.AppendLine("**Recommended Hyperparameters:**\n");

        sb.AppendLine($"**num_points:** {nPoints} (range: 512-4096)");
        sb.AppendLine("  • Number of points to sample from each shape");
        sb.AppendLine($"  • {nPoints} provides good coverage for most objects");
        sb.AppendLine();

        sb.AppendLine($"**input_dim:** {inputDim}");
        sb.AppendLine($"  • XYZ coordinates{(hasNormals ? " + normal vectors" : "")}");
        sb.AppendLine();

        sb.AppendLine("**mlp_dims:** [64, 128, 1024]");
        sb.AppendLine("  • Shared MLP dimensions for point features");
        sb.AppendLine("  • Final 1024 is the global feature dimension");
        sb.AppendLine();

        if (taskType.Contains("segmentation"))
        {
            sb.AppendLine("**seg_mlp_dims:** [512, 256, 128]");
            sb.AppendLine("  • MLP dimensions after concatenating global + local features");
            sb.AppendLine();
        }

        sb.AppendLine("**use_tnet:** true");
        sb.AppendLine("  • Spatial Transformer Network for input alignment");
        sb.AppendLine("  • Improves rotation invariance");
        sb.AppendLine();

        GenerateCommonTrainingParams(sb, nSamples, "moderate");

        sb.AppendLine("**Data Augmentation:**");
        sb.AppendLine("  • Random rotation around vertical axis (0-360°)");
        sb.AppendLine("  • Random jittering: std=0.01, clip=0.05");
        sb.AppendLine("  • Random scaling: [0.8, 1.2]");
        sb.AppendLine("  • Random point dropout: up to 10%");
    }

    /// <summary>
    /// Generates hyperparameter recommendations for PointNet++.
    /// </summary>
    private void GeneratePointNetPlusPlusRecommendations(
        System.Text.StringBuilder sb,
        string taskType,
        int nSamples,
        int nPoints,
        bool hasNormals,
        string budget)
    {
        int inputDim = hasNormals ? 6 : 3;

        // Determine hierarchy based on input points
        var npointList = nPoints >= 2048
            ? new[] { 1024, 256, 64, 16 }
            : nPoints >= 1024
                ? new[] { 512, 128, 32 }
                : new[] { 256, 64, 16 };

        sb.AppendLine("**Recommended Hyperparameters:**\n");

        sb.AppendLine($"**num_points:** {nPoints}");
        sb.AppendLine($"  • Input point count (pad/sample to this)");
        sb.AppendLine();

        sb.AppendLine($"**input_dim:** {inputDim}");
        sb.AppendLine($"  • XYZ{(hasNormals ? " + normals (6D)" : " only (3D)")}");
        sb.AppendLine();

        sb.AppendLine($"**npoint_list:** [{string.Join(", ", npointList)}]");
        sb.AppendLine("  • Number of points at each set abstraction level");
        sb.AppendLine("  • Progressively reduces spatial resolution");
        sb.AppendLine();

        // Radius scales with npoints
        var radiusList = npointList.Select((n, i) => 0.1 * Math.Pow(2, i)).ToArray();
        sb.AppendLine($"**radius_list:** [{string.Join(", ", radiusList.Select(r => r.ToString("F2")))}]");
        sb.AppendLine("  • Ball query radius at each level");
        sb.AppendLine("  • Doubles at each level to capture larger context");
        sb.AppendLine();

        int nsample = budget == "high" ? 32 : 16;
        sb.AppendLine($"**nsample:** {nsample} (range: 8-64)");
        sb.AppendLine("  • Points sampled per local region");
        sb.AppendLine($"  • {nsample} balances detail and computation");
        sb.AppendLine();

        sb.AppendLine("**use_msg:** true (Multi-Scale Grouping)");
        sb.AppendLine("  • Sample at multiple radii per level");
        sb.AppendLine("  • Better handles varying point densities");
        sb.AppendLine();

        if (taskType.Contains("segmentation"))
        {
            sb.AppendLine("**fp_mlp_dims:** [[256, 256], [256, 128], [128, 128, 128]]");
            sb.AppendLine("  • Feature propagation MLP dims (decoder)");
            sb.AppendLine("  • Skip connections from encoder levels");
            sb.AppendLine();
        }

        GenerateCommonTrainingParams(sb, nSamples, budget);

        sb.AppendLine("**Data Augmentation:**");
        sb.AppendLine("  • Random rotation: full SO(3) or just vertical");
        sb.AppendLine("  • Random jittering: std=0.01, clip=0.02");
        sb.AppendLine("  • Random scaling: [0.8, 1.25]");
        sb.AppendLine("  • Random dropout: up to 875 points");
    }

    /// <summary>
    /// Generates hyperparameter recommendations for DGCNN.
    /// </summary>
    private void GenerateDGCNNRecommendations(
        System.Text.StringBuilder sb,
        string taskType,
        int nSamples,
        int nPoints,
        string budget)
    {
        int k = budget == "high" ? 40 : budget == "moderate" ? 20 : 10;

        sb.AppendLine("**Recommended Hyperparameters:**\n");

        sb.AppendLine($"**k:** {k} (range: 10-40)");
        sb.AppendLine("  • Number of nearest neighbors for graph construction");
        sb.AppendLine("  • Higher k captures more context but is slower");
        sb.AppendLine("  • Graph is dynamically constructed at each layer");
        sb.AppendLine();

        sb.AppendLine($"**num_points:** {nPoints}");
        sb.AppendLine("  • All points processed (no sampling)");
        sb.AppendLine();

        sb.AppendLine("**edge_conv_dims:** [64, 64, 128, 256]");
        sb.AppendLine("  • Output dimensions of EdgeConv layers");
        sb.AppendLine("  • Features concatenated for global pooling");
        sb.AppendLine();

        sb.AppendLine("**emb_dims:** 1024");
        sb.AppendLine("  • Global feature embedding dimension");
        sb.AppendLine("  • After concatenation and transformation");
        sb.AppendLine();

        sb.AppendLine("**dynamic_graph:** true");
        sb.AppendLine("  • Recompute k-NN at each layer in feature space");
        sb.AppendLine("  • Key innovation of DGCNN over static graphs");
        sb.AppendLine();

        GenerateCommonTrainingParams(sb, nSamples, budget);

        sb.AppendLine("**Data Augmentation:**");
        sb.AppendLine("  • Random rotation around all axes");
        sb.AppendLine("  • Random translation: [-0.2, 0.2]");
        sb.AppendLine("  • Random anisotropic scaling: [2/3, 3/2]");
    }

    /// <summary>
    /// Generates hyperparameter recommendations for MeshCNN.
    /// </summary>
    private void GenerateMeshCNNRecommendations(
        System.Text.StringBuilder sb,
        string taskType,
        int nSamples,
        string budget)
    {
        int numEdges = budget == "high" ? 750 : 500;

        sb.AppendLine("**Recommended Hyperparameters:**\n");

        sb.AppendLine($"**num_edges:** {numEdges} (range: 300-1000)");
        sb.AppendLine("  • Target number of edges after preprocessing");
        sb.AppendLine("  • Meshes are simplified/refined to this count");
        sb.AppendLine();

        sb.AppendLine("**input_features:** 5");
        sb.AppendLine("  • Edge features: dihedral angle, 2 inner angles, 2 edge-length ratios");
        sb.AppendLine("  • Captures local geometric information");
        sb.AppendLine();

        sb.AppendLine("**conv_dims:** [32, 64, 128, 256]");
        sb.AppendLine("  • Output channels for each convolution layer");
        sb.AppendLine();

        if (taskType.Contains("segmentation"))
        {
            sb.AppendLine("**pool_targets:** [500, 400, 300, 200]");
            sb.AppendLine("  • Target edges after each pooling");
            sb.AppendLine("  • Mesh pooling collapses edges based on learned importance");
        }
        else
        {
            sb.AppendLine("**pool_targets:** [400, 200, 100]");
            sb.AppendLine("  • More aggressive pooling for classification");
        }
        sb.AppendLine();

        sb.AppendLine("**norm:** group (alternatives: batch, instance)");
        sb.AppendLine("  • Group normalization works well for varying mesh sizes");
        sb.AppendLine();

        GenerateCommonTrainingParams(sb, nSamples, budget);

        sb.AppendLine("**Data Augmentation:**");
        sb.AppendLine("  • Random rotation: typically vertical axis only");
        sb.AppendLine("  • Random scaling: [0.9, 1.1]");
        sb.AppendLine("  • Edge feature noise: std=0.01");
    }

    /// <summary>
    /// Generates hyperparameter recommendations for DiffusionNet.
    /// </summary>
    private void GenerateDiffusionNetRecommendations(
        System.Text.StringBuilder sb,
        string taskType,
        int nSamples,
        string budget)
    {
        sb.AppendLine("**Recommended Hyperparameters:**\n");

        sb.AppendLine("**diffusion_time:** 0.001 to 0.01 (learnable)");
        sb.AppendLine("  • Controls spatial extent of diffusion");
        sb.AppendLine("  • Learnable per-layer for adaptive receptive field");
        sb.AppendLine();

        int width = budget == "high" ? 128 : 64;
        sb.AppendLine($"**width:** {width} (range: 32-256)");
        sb.AppendLine("  • Feature dimension throughout network");
        sb.AppendLine();

        int depth = budget == "high" ? 6 : 4;
        sb.AppendLine($"**n_blocks:** {depth} (range: 3-8)");
        sb.AppendLine("  • Number of DiffusionNet blocks");
        sb.AppendLine();

        sb.AppendLine("**laplacian_type:** cotangent");
        sb.AppendLine("  • Cotangent Laplacian for accurate heat diffusion");
        sb.AppendLine("  • Requires manifold mesh");
        sb.AppendLine();

        sb.AppendLine("**k_eig:** 128 (range: 64-256)");
        sb.AppendLine("  • Number of Laplacian eigenvectors");
        sb.AppendLine("  • More captures finer frequencies");
        sb.AppendLine();

        GenerateCommonTrainingParams(sb, nSamples, budget);

        sb.AppendLine("**Data Augmentation:**");
        sb.AppendLine("  • Random rotation (full SO(3))");
        sb.AppendLine("  • Works on any mesh resolution (key advantage)");
    }

    /// <summary>
    /// Generates hyperparameter recommendations for VoxelCNN.
    /// </summary>
    private void GenerateVoxelCNNRecommendations(
        System.Text.StringBuilder sb,
        string taskType,
        int nSamples,
        int resolution,
        string budget)
    {
        sb.AppendLine("**Recommended Hyperparameters:**\n");

        sb.AppendLine($"**resolution:** {resolution}³");
        sb.AppendLine($"  • Voxel grid size (current: {resolution})");
        sb.AppendLine("  • 32³ for efficiency, 64³ for detail");
        sb.AppendLine($"  • Memory: ~{(long)resolution * resolution * resolution * 4 / 1024} KB per sample");
        sb.AppendLine();

        sb.AppendLine("**conv_dims:** [32, 64, 128, 256]");
        sb.AppendLine("  • Output channels for 3D conv layers");
        sb.AppendLine();

        sb.AppendLine("**kernel_size:** 3");
        sb.AppendLine("  • 3×3×3 convolution kernels (standard)");
        sb.AppendLine("  • Occasionally use 5×5×5 for larger receptive field");
        sb.AppendLine();

        sb.AppendLine("**pool_size:** 2");
        sb.AppendLine("  • 2×2×2 max pooling halves each dimension");
        sb.AppendLine();

        if (resolution >= 64)
        {
            sb.AppendLine("**use_sparse:** consider");
            sb.AppendLine("  • Sparse convolutions if occupancy < 10%");
            sb.AppendLine("  • Significant memory savings for large grids");
            sb.AppendLine();
        }

        GenerateCommonTrainingParams(sb, nSamples, budget);

        sb.AppendLine("**Data Augmentation:**");
        sb.AppendLine("  • Random rotation (90° increments for alignment)");
        sb.AppendLine("  • Random translation within grid");
        sb.AppendLine("  • Random flipping along axes");
    }

    /// <summary>
    /// Generates hyperparameter recommendations for 3D U-Net.
    /// </summary>
    private void Generate3DUNetRecommendations(
        System.Text.StringBuilder sb,
        string taskType,
        int nSamples,
        int resolution,
        string budget)
    {
        sb.AppendLine("**Recommended Hyperparameters:**\n");

        sb.AppendLine($"**input_resolution:** {resolution}³");
        sb.AppendLine("  • May need patch-based processing if > 128³");
        sb.AppendLine();

        int baseChannels = budget == "high" ? 64 : 32;
        sb.AppendLine($"**base_channels:** {baseChannels}");
        sb.AppendLine("  • Initial conv output channels");
        sb.AppendLine($"  • Doubles at each level: [{baseChannels}, {baseChannels * 2}, {baseChannels * 4}, {baseChannels * 8}]");
        sb.AppendLine();

        sb.AppendLine("**depth:** 4 (range: 3-5)");
        sb.AppendLine("  • Number of encoder/decoder levels");
        sb.AppendLine("  • Limited by input resolution (halves each level)");
        sb.AppendLine();

        sb.AppendLine("**skip_connections:** true");
        sb.AppendLine("  • Essential for preserving spatial detail");
        sb.AppendLine("  • Concatenate or add encoder features to decoder");
        sb.AppendLine();

        sb.AppendLine("**norm:** instance (alternatives: batch, group)");
        sb.AppendLine("  • Instance norm often works better for segmentation");
        sb.AppendLine();

        GenerateCommonTrainingParams(sb, nSamples, budget);

        sb.AppendLine("**Loss Function:**");
        sb.AppendLine("  • Dice loss + Cross-entropy (combined)");
        sb.AppendLine("  • Handles class imbalance in segmentation");
    }

    /// <summary>
    /// Generates hyperparameter recommendations for NeRF.
    /// </summary>
    private void GenerateNeRFRecommendations(
        System.Text.StringBuilder sb,
        int nSamples,
        string budget)
    {
        sb.AppendLine("**Recommended Hyperparameters:**\n");

        sb.AppendLine("**positional_encoding_L:** 10 (position), 4 (direction)");
        sb.AppendLine("  • Fourier feature frequencies for position/direction");
        sb.AppendLine("  • More frequencies capture higher frequency details");
        sb.AppendLine();

        int networkWidth = budget == "high" ? 256 : 128;
        sb.AppendLine($"**network_width:** {networkWidth}");
        sb.AppendLine("  • MLP hidden layer width");
        sb.AppendLine();

        sb.AppendLine("**network_depth:** 8");
        sb.AppendLine("  • Number of layers (with skip connection at layer 4)");
        sb.AppendLine();

        int nSamplesCoarse = budget == "high" ? 64 : 32;
        int nSamplesFine = budget == "high" ? 128 : 64;
        sb.AppendLine($"**n_samples_coarse:** {nSamplesCoarse}");
        sb.AppendLine($"**n_samples_fine:** {nSamplesFine}");
        sb.AppendLine("  • Samples along each ray for volume rendering");
        sb.AppendLine("  • Hierarchical sampling: coarse + fine network");
        sb.AppendLine();

        sb.AppendLine("**near/far:** scene-dependent");
        sb.AppendLine("  • Clipping planes for ray marching");
        sb.AppendLine("  • Set based on scene bounding box");
        sb.AppendLine();

        sb.AppendLine("**Learning Rate:** 5e-4 (with exponential decay)");
        sb.AppendLine("**Batch Size:** 1024-4096 rays per batch");
        sb.AppendLine("**Iterations:** 200,000-500,000");
        sb.AppendLine("  • ⚠️ Training takes hours to days per scene");
        sb.AppendLine();

        sb.AppendLine("**Data Requirements:**");
        sb.AppendLine("  • 50-200 calibrated images per scene");
        sb.AppendLine("  • Camera intrinsics and extrinsics required");
        sb.AppendLine("  • Use COLMAP for structure-from-motion");
    }

    /// <summary>
    /// Generates hyperparameter recommendations for Instant-NGP.
    /// </summary>
    private void GenerateInstantNGPRecommendations(
        System.Text.StringBuilder sb,
        int nSamples,
        string budget)
    {
        sb.AppendLine("**Recommended Hyperparameters:**\n");

        sb.AppendLine("**hash_table_size:** 2^19 to 2^24");
        sb.AppendLine("  • Size of multi-resolution hash table");
        sb.AppendLine("  • Larger = more capacity, more memory");
        sb.AppendLine();

        sb.AppendLine("**num_levels:** 16");
        sb.AppendLine("  • Number of resolution levels in hash encoding");
        sb.AppendLine();

        sb.AppendLine("**features_per_level:** 2");
        sb.AppendLine("  • Feature dimensions per hash table level");
        sb.AppendLine();

        sb.AppendLine("**base_resolution:** 16");
        sb.AppendLine("**finest_resolution:** 2048-8192");
        sb.AppendLine("  • Multi-resolution grid bounds");
        sb.AppendLine();

        sb.AppendLine("**network_width:** 64");
        sb.AppendLine("**network_depth:** 2");
        sb.AppendLine("  • Small MLP after hash encoding (key to speed)");
        sb.AppendLine();

        sb.AppendLine("**Learning Rate:** 1e-2 (much higher than vanilla NeRF)");
        sb.AppendLine("**Batch Size:** 2^18 rays (262,144)");
        sb.AppendLine("**Training Time:** seconds to minutes");
        sb.AppendLine("  • ~100x faster than vanilla NeRF");
    }

    /// <summary>
    /// Generates hyperparameter recommendations for Gaussian Splatting.
    /// </summary>
    private void GenerateGaussianSplattingRecommendations(
        System.Text.StringBuilder sb,
        int nSamples,
        string budget)
    {
        sb.AppendLine("**Recommended Hyperparameters:**\n");

        sb.AppendLine("**initial_points:** from SfM (COLMAP)");
        sb.AppendLine("  • Initialize Gaussians at sparse point cloud");
        sb.AppendLine("  • Typically 10,000-100,000 initial points");
        sb.AppendLine();

        sb.AppendLine("**spherical_harmonics_degree:** 3");
        sb.AppendLine("  • Degree of SH for view-dependent color");
        sb.AppendLine("  • 0 = diffuse only, 3 = full view-dependent");
        sb.AppendLine();

        sb.AppendLine("**position_lr:** 0.00016");
        sb.AppendLine("**opacity_lr:** 0.05");
        sb.AppendLine("**scaling_lr:** 0.005");
        sb.AppendLine("**rotation_lr:** 0.001");
        sb.AppendLine("**sh_lr:** 0.0025");
        sb.AppendLine("  • Per-parameter learning rates");
        sb.AppendLine();

        sb.AppendLine("**densification_interval:** 100 iterations");
        sb.AppendLine("**densify_until_iter:** 15,000");
        sb.AppendLine("  • Periodically split/clone under-reconstructed Gaussians");
        sb.AppendLine();

        sb.AppendLine("**opacity_reset_interval:** 3000");
        sb.AppendLine("  • Reset opacity to prune floaters");
        sb.AppendLine();

        sb.AppendLine("**densify_grad_threshold:** 0.0002");
        sb.AppendLine("  • Gradient threshold for densification");
        sb.AppendLine();

        sb.AppendLine("**Training Time:** 5-30 minutes per scene");
        sb.AppendLine("**Rendering Speed:** 100+ FPS at 1080p");
        sb.AppendLine("  • Real-time capable after training");
    }

    /// <summary>
    /// Adds common training parameters to recommendations.
    /// </summary>
    private void GenerateCommonTrainingParams(
        System.Text.StringBuilder sb,
        int nSamples,
        string budget)
    {
        sb.AppendLine("**Training Parameters:**");

        double lr = nSamples < 1000 ? 0.001 : 0.0001;
        sb.AppendLine($"  • learning_rate: {lr} (with cosine/step decay)");

        int batchSize = budget == "high" ? 32 : budget == "moderate" ? 16 : 8;
        sb.AppendLine($"  • batch_size: {batchSize}");

        int epochs = nSamples < 1000 ? 300 : nSamples < 10000 ? 200 : 100;
        sb.AppendLine($"  • epochs: {epochs} (with early stopping)");

        sb.AppendLine("  • optimizer: Adam (betas: 0.9, 0.999)");
        sb.AppendLine("  • weight_decay: 1e-4");
        sb.AppendLine();
    }

    /// <inheritdoc/>
    protected override string GetJsonErrorMessage(JsonReaderException ex)
    {
        return $"Error: Invalid JSON format. {ex.Message}\n" +
               "Expected format: { \"model_type\": \"PointNet++\", \"task_type\": \"classification\", ... }";
    }
}

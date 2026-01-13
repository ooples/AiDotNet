using AiDotNet.Interfaces;

namespace AiDotNet.ModelLoading;

/// <summary>
/// Loads pretrained models from various sources.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This class provides methods to load pretrained model weights into our model classes.
/// It supports loading from local SafeTensors files and handles weight name mapping
/// between different model formats.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is your gateway to using pretrained models.
///
/// Instead of training models from scratch (which requires massive datasets and
/// compute resources), you can load pretrained weights that others have trained.
///
/// Example usage:
/// ```csharp
/// var loader = new PretrainedModelLoader&lt;float&gt;();
///
/// // Load a pretrained VAE
/// var vae = new StandardVAE&lt;float&gt;();
/// await loader.LoadVAEWeights(vae, "sd-vae-ft-mse/diffusion_pytorch_model.safetensors");
///
/// // Now your VAE is ready for image encoding/decoding!
/// ```
/// </para>
/// </remarks>
public class PretrainedModelLoader<T>
{
    /// <summary>
    /// Provides numeric operations for the specific type T.
    /// </summary>
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The SafeTensors loader instance.
    /// </summary>
    private readonly SafeTensorsLoader<T> _safeTensorsLoader;

    /// <summary>
    /// Whether to log loading progress.
    /// </summary>
    private readonly bool _verbose;

    /// <summary>
    /// Initializes a new instance of the PretrainedModelLoader class.
    /// </summary>
    /// <param name="verbose">Whether to log loading progress (default: false).</param>
    public PretrainedModelLoader(bool verbose = false)
    {
        _safeTensorsLoader = new SafeTensorsLoader<T>();
        _verbose = verbose;
    }

    /// <summary>
    /// Loads weights from a SafeTensors file into any IWeightLoadable model.
    /// </summary>
    /// <param name="model">The model to load weights into (must implement IWeightLoadable).</param>
    /// <param name="weightsPath">Path to the .safetensors file.</param>
    /// <param name="mapping">Optional custom weight mapping function. Maps source names to target names.</param>
    /// <param name="strict">If true, fails when weights can't be loaded. If false, skips missing weights.</param>
    /// <returns>Load result with statistics about applied weights.</returns>
    /// <exception cref="ArgumentNullException">Thrown when model or weightsPath is null.</exception>
    /// <exception cref="FileNotFoundException">Thrown when weights file doesn't exist.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes pretrained weights from a file and loads them
    /// into your model. The mapping function translates between the weight names in the file
    /// (like "encoder.conv1.weight") and the names your model uses.
    /// </para>
    /// </remarks>
    public LoadResult LoadWeights(IWeightLoadable<T> model, string weightsPath, Func<string, string?>? mapping = null, bool strict = false)
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));
        if (string.IsNullOrWhiteSpace(weightsPath))
            throw new ArgumentNullException(nameof(weightsPath));

        var result = new LoadResult();

        try
        {
            // Load all tensors from file
            var weights = _safeTensorsLoader.Load(weightsPath);
            result.TotalTensors = weights.Count;

            if (_verbose)
            {
                Console.WriteLine($"Loaded {weights.Count} tensors from {weightsPath}");
            }

            // Use IWeightLoadable's LoadWeights method
            var loadResult = model.LoadWeights(weights, mapping, strict);

            result.MappedTensors = loadResult.LoadedCount;
            result.UnmappedTensors = loadResult.SkippedCount + loadResult.FailedCount;
            result.MappedNames = loadResult.LoadedParameters.Select(p => (p, p)).ToList();
            result.UnmappedNames = loadResult.FailedParameters.Select(f => f.Name).ToList();
            result.Success = loadResult.Success;
            result.WeightsApplied = loadResult.Success && loadResult.LoadedCount > 0;
            result.ErrorMessage = loadResult.ErrorMessage;

            if (_verbose)
            {
                Console.WriteLine($"  Applied {loadResult.LoadedCount} weights, skipped {loadResult.SkippedCount}, failed {loadResult.FailedCount}");
            }
        }
        catch (Exception ex)
        {
            result.Success = false;
            result.WeightsApplied = false;
            result.ErrorMessage = ex.Message;
        }

        return result;
    }

    /// <summary>
    /// Loads weights using a WeightMapping instance.
    /// </summary>
    /// <param name="model">The model to load weights into.</param>
    /// <param name="weightsPath">Path to the .safetensors file.</param>
    /// <param name="mapping">Weight mapping to use for name translation.</param>
    /// <param name="strict">If true, fails when weights can't be loaded.</param>
    /// <returns>Load result with statistics.</returns>
    public LoadResult LoadWeights(IWeightLoadable<T> model, string weightsPath, WeightMapping mapping, bool strict = false)
    {
        return LoadWeights(model, weightsPath, mapping.Map, strict);
    }

    /// <summary>
    /// Gets information about tensors in a SafeTensors file.
    /// </summary>
    /// <param name="path">Path to the .safetensors file.</param>
    /// <returns>List of tensor metadata.</returns>
    public List<TensorMetadata> GetTensorInfo(string path)
    {
        return _safeTensorsLoader.GetTensorInfo(path);
    }

    /// <summary>
    /// Loads specific tensors by name from a SafeTensors file.
    /// </summary>
    /// <param name="path">Path to the .safetensors file.</param>
    /// <param name="tensorNames">Names of tensors to load.</param>
    /// <returns>Dictionary of loaded tensors.</returns>
    public Dictionary<string, Tensor<T>> LoadTensors(string path, IEnumerable<string> tensorNames)
    {
        return _safeTensorsLoader.Load(path, tensorNames);
    }

    /// <summary>
    /// Loads all tensors from a SafeTensors file.
    /// </summary>
    /// <param name="path">Path to the .safetensors file.</param>
    /// <returns>Dictionary of all loaded tensors.</returns>
    public Dictionary<string, Tensor<T>> LoadAllTensors(string path)
    {
        return _safeTensorsLoader.Load(path);
    }

    /// <summary>
    /// Validates that required tensors exist in a weights file.
    /// </summary>
    /// <param name="path">Path to the .safetensors file.</param>
    /// <param name="requiredTensorPatterns">Patterns for required tensor names.</param>
    /// <returns>Validation result.</returns>
    public ValidationResult ValidateWeights(string path, IEnumerable<string> requiredTensorPatterns)
    {
        var result = new ValidationResult { Path = path };

        try
        {
            var tensorInfo = GetTensorInfo(path);
            var tensorNames = tensorInfo.Select(t => t.Name).ToHashSet();

            foreach (var pattern in requiredTensorPatterns)
            {
                var regex = new System.Text.RegularExpressions.Regex(
                    pattern,
                    System.Text.RegularExpressions.RegexOptions.None,
                    TimeSpan.FromSeconds(1));
                var matches = tensorNames.Where(name => regex.IsMatch(name)).ToList();

                if (matches.Count > 0)
                {
                    result.FoundPatterns.Add(pattern);
                    result.MatchedTensors.AddRange(matches);
                }
                else
                {
                    result.MissingPatterns.Add(pattern);
                }
            }

            result.IsValid = result.MissingPatterns.Count == 0;
            result.TotalTensors = tensorInfo.Count;
            result.TotalSizeBytes = tensorInfo.Sum(t => t.DataSizeBytes);
        }
        catch (Exception ex)
        {
            result.IsValid = false;
            result.ErrorMessage = ex.Message;
        }

        return result;
    }
}

/// <summary>
/// Result of a model weight loading operation.
/// </summary>
public class LoadResult
{
    /// <summary>
    /// Whether the loading was successful.
    /// </summary>
    public bool Success { get; set; }

    /// <summary>
    /// Whether weights were actually applied to the model.
    /// Currently always false as VAE models don't yet support SetParameterByName.
    /// </summary>
    public bool WeightsApplied { get; set; }

    /// <summary>
    /// Total number of tensors in the file.
    /// </summary>
    public int TotalTensors { get; set; }

    /// <summary>
    /// Number of tensors successfully mapped.
    /// </summary>
    public int MappedTensors { get; set; }

    /// <summary>
    /// Number of tensors that couldn't be mapped.
    /// </summary>
    public int UnmappedTensors { get; set; }

    /// <summary>
    /// List of mapped tensor name pairs (source, target).
    /// </summary>
    public List<(string Source, string Target)> MappedNames { get; set; } = new();

    /// <summary>
    /// List of unmapped tensor names.
    /// </summary>
    public List<string> UnmappedNames { get; set; } = new();

    /// <summary>
    /// Error message if loading failed.
    /// </summary>
    public string? ErrorMessage { get; set; }

    /// <summary>
    /// Gets a summary of the load result.
    /// </summary>
    public override string ToString()
    {
        if (!Success)
        {
            return $"Load failed: {ErrorMessage}";
        }

        var appliedStatus = WeightsApplied ? "applied" : "mapped only";
        return $"Loaded {MappedTensors}/{TotalTensors} tensors ({UnmappedTensors} unmapped, {appliedStatus})";
    }
}

/// <summary>
/// Result of weights file validation.
/// </summary>
public class ValidationResult
{
    /// <summary>
    /// Path to the validated file.
    /// </summary>
    public string Path { get; set; } = string.Empty;

    /// <summary>
    /// Whether the validation passed.
    /// </summary>
    public bool IsValid { get; set; }

    /// <summary>
    /// Total number of tensors in the file.
    /// </summary>
    public int TotalTensors { get; set; }

    /// <summary>
    /// Total size of all tensors in bytes.
    /// </summary>
    public long TotalSizeBytes { get; set; }

    /// <summary>
    /// Patterns that were found.
    /// </summary>
    public List<string> FoundPatterns { get; set; } = new();

    /// <summary>
    /// Patterns that were not found.
    /// </summary>
    public List<string> MissingPatterns { get; set; } = new();

    /// <summary>
    /// Tensor names that matched the patterns.
    /// </summary>
    public List<string> MatchedTensors { get; set; } = new();

    /// <summary>
    /// Error message if validation failed.
    /// </summary>
    public string? ErrorMessage { get; set; }

    /// <summary>
    /// Gets a summary of the validation result.
    /// </summary>
    public override string ToString()
    {
        if (!IsValid && !string.IsNullOrEmpty(ErrorMessage))
        {
            return $"Validation failed: {ErrorMessage}";
        }

        return IsValid
            ? $"Valid: {TotalTensors} tensors, {TotalSizeBytes / 1024 / 1024:N0} MB"
            : $"Invalid: Missing patterns: {string.Join(", ", MissingPatterns)}";
    }
}


using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Interpretability.Explainers;
using AiDotNet.Interpretability.Helpers;
using AiDotNet.LinearAlgebra;
using System.Collections.Concurrent;

namespace AiDotNet.Interpretability;

/// <summary>
/// Automatically selects the optimal explainer based on model type and provides caching for batch explanations.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Different explanation methods work best for different model types:
///
/// <b>Model Type → Best Explainer:</b>
/// - Decision Tree → TreeSHAP (exact, fast)
/// - Random Forest → TreeSHAP (exact, fast)
/// - Gradient Boosting → TreeSHAP (exact, fast)
/// - Neural Network → DeepSHAP or Integrated Gradients (uses gradients)
/// - Any Model → KernelSHAP or LIME (model-agnostic, slower)
///
/// <b>Why auto-selection matters:</b>
/// - TreeSHAP is O(TLD²) exact for trees but doesn't work on neural networks
/// - DeepSHAP is fast for neural networks but needs gradient access
/// - KernelSHAP works everywhere but is O(2^M) approximation
///
/// This class automatically picks the best method for your model type.
///
/// <b>Caching:</b>
/// Computing explanations is expensive. This class caches explanations so repeated
/// requests for the same input return instantly.
/// </para>
/// </remarks>
public class SmartExplainerSelector<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Func<Matrix<T>, Vector<T>> _predictFunction;
    private readonly ExplainableModelType _detectedType;
    private readonly int _numFeatures;
    private readonly string[]? _featureNames;
    private readonly Matrix<T>? _backgroundData;
    private readonly ExplainerCache<T> _cache;
    private readonly SmartExplainerOptions _options;

    // Cached explainers (created on first use)
    private SHAPExplainer<T>? _shapExplainer;
    private IntegratedGradientsExplainer<T>? _igExplainer;
    private LIMEExplainer<T>? _limeExplainer;

    /// <summary>
    /// Gets the detected model type.
    /// </summary>
    public ExplainableModelType DetectedModelType => _detectedType;

    /// <summary>
    /// Gets the cache hit rate.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Higher hit rate means more explanations are served from cache
    /// rather than being recomputed. Monitor this to tune your caching strategy.
    /// </para>
    /// </remarks>
    public double CacheHitRate => _cache.HitRate;

    /// <summary>
    /// Initializes a smart explainer selector with a prediction function.
    /// </summary>
    /// <param name="predictFunction">The model's prediction function.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="modelType">The type of model (auto-detected if not specified).</param>
    /// <param name="featureNames">Optional feature names.</param>
    /// <param name="backgroundData">Background data for SHAP (optional but recommended).</param>
    /// <param name="options">Configuration options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Pass your prediction function and the selector will choose
    /// the best explanation method. If you provide backgroundData, SHAP methods
    /// will use it as the baseline distribution.
    /// </para>
    /// </remarks>
    public SmartExplainerSelector(
        Func<Matrix<T>, Vector<T>> predictFunction,
        int numFeatures,
        ExplainableModelType modelType = ExplainableModelType.BlackBox,
        string[]? featureNames = null,
        Matrix<T>? backgroundData = null,
        SmartExplainerOptions? options = null)
    {
        _predictFunction = predictFunction ?? throw new ArgumentNullException(nameof(predictFunction));
        _numFeatures = numFeatures;
        _featureNames = featureNames;
        _backgroundData = backgroundData ?? CreateDefaultBackground(numFeatures);
        _options = options ?? new SmartExplainerOptions();
        _cache = new ExplainerCache<T>(_options.MaxCacheSize);
        _detectedType = modelType;
    }

    /// <summary>
    /// Creates default background data (zeros).
    /// </summary>
    private static Matrix<T> CreateDefaultBackground(int numFeatures)
    {
        return new Matrix<T>(10, numFeatures); // 10 samples of zeros
    }

    /// <summary>
    /// Gets the recommended explainer type for the detected model.
    /// </summary>
    /// <returns>The recommended explainer type.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns what explainer will be used for this model.
    /// You can override this with the ExplainWith method if you prefer a different approach.
    /// </para>
    /// </remarks>
    public ExplainerType GetRecommendedExplainer()
    {
        return _detectedType switch
        {
            ExplainableModelType.TreeEnsemble => ExplainerType.TreeSHAP,
            ExplainableModelType.NeuralNetwork => _options.PreferDeepSHAP ? ExplainerType.DeepSHAP : ExplainerType.IntegratedGradients,
            ExplainableModelType.Linear => ExplainerType.SHAP,
            ExplainableModelType.KernelBased => ExplainerType.KernelSHAP,
            ExplainableModelType.BlackBox => _options.PreferLIME ? ExplainerType.LIME : ExplainerType.KernelSHAP,
            _ => ExplainerType.KernelSHAP
        };
    }

    /// <summary>
    /// Explains a single instance using the automatically selected method.
    /// </summary>
    /// <param name="instance">The input instance to explain.</param>
    /// <param name="targetClass">Optional target class to explain.</param>
    /// <returns>Feature attributions.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main method to use. Pass your input and
    /// get back feature attributions explaining the prediction.
    /// </para>
    /// </remarks>
    public FeatureAttribution<T> Explain(Vector<T> instance, int? targetClass = null)
    {
        // Check cache first
        var cacheKey = ComputeCacheKey(instance, targetClass);
        if (_options.EnableCaching && _cache.TryGet(cacheKey, out var cached))
        {
            return cached!;
        }

        // Compute explanation
        var attribution = ComputeExplanation(instance, targetClass);

        // Cache result
        if (_options.EnableCaching)
        {
            _cache.Set(cacheKey, attribution);
        }

        return attribution;
    }

    /// <summary>
    /// Explains multiple instances in batch.
    /// </summary>
    /// <param name="instances">Matrix of instances (rows = samples).</param>
    /// <param name="targetClass">Optional target class to explain.</param>
    /// <returns>Array of feature attributions.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Batch explanation is more efficient than explaining
    /// one at a time. The selector will parallelize where possible.
    /// </para>
    /// </remarks>
    public FeatureAttribution<T>[] ExplainBatch(Matrix<T> instances, int? targetClass = null)
    {
        var results = new FeatureAttribution<T>[instances.Rows];

        if (_options.UseParallelProcessing)
        {
            Parallel.For(0, instances.Rows, new ParallelOptions
            {
                MaxDegreeOfParallelism = _options.MaxParallelism
            }, i =>
            {
                results[i] = Explain(instances.GetRow(i), targetClass);
            });
        }
        else
        {
            for (int i = 0; i < instances.Rows; i++)
            {
                results[i] = Explain(instances.GetRow(i), targetClass);
            }
        }

        return results;
    }

    /// <summary>
    /// Forces use of a specific explainer type.
    /// </summary>
    /// <param name="instance">The input instance.</param>
    /// <param name="explainerType">The explainer type to use.</param>
    /// <param name="targetClass">Optional target class.</param>
    /// <returns>Feature attributions.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Override the auto-selection if you want to use
    /// a specific method. Useful for comparing different explanation approaches.
    /// </para>
    /// </remarks>
    public FeatureAttribution<T> ExplainWith(Vector<T> instance, ExplainerType explainerType, int? targetClass = null)
    {
        return ComputeExplanation(instance, targetClass, explainerType);
    }

    /// <summary>
    /// Clears the explanation cache.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Call this if you've retrained the model or want
    /// to free up memory. Cached explanations become invalid after model changes.
    /// </para>
    /// </remarks>
    public void ClearCache()
    {
        _cache.Clear();
    }

    /// <summary>
    /// Computes explanation using the appropriate method.
    /// </summary>
    private FeatureAttribution<T> ComputeExplanation(Vector<T> instance, int? targetClass, ExplainerType? overrideType = null)
    {
        var explainerType = overrideType ?? GetRecommendedExplainer();

        return explainerType switch
        {
            ExplainerType.SHAP or ExplainerType.KernelSHAP or ExplainerType.TreeSHAP or ExplainerType.DeepSHAP => ExplainWithSHAP(instance),
            ExplainerType.IntegratedGradients => ExplainWithIntegratedGradients(instance, targetClass),
            ExplainerType.LIME => ExplainWithLIME(instance),
            _ => ExplainWithSHAP(instance)
        };
    }

    /// <summary>
    /// Explains with Kernel SHAP.
    /// </summary>
    private FeatureAttribution<T> ExplainWithSHAP(Vector<T> instance)
    {
        if (_shapExplainer == null)
        {
            _shapExplainer = new SHAPExplainer<T>(
                _predictFunction,
                _backgroundData!,
                featureNames: _featureNames);
        }

        var explanation = _shapExplainer.Explain(instance);
        return new FeatureAttribution<T>
        {
            FeatureNames = explanation.FeatureNames ?? _featureNames ?? CreateDefaultFeatureNames(),
            Attributions = explanation.ShapValues,
            Instance = instance,
            Method = "SHAP"
        };
    }

    /// <summary>
    /// Explains with Integrated Gradients.
    /// </summary>
    private FeatureAttribution<T> ExplainWithIntegratedGradients(Vector<T> instance, int? targetClass)
    {
        if (_igExplainer == null)
        {
            // Create a Vector predict function from the Matrix predict function
            Vector<T> VectorPredict(Vector<T> input)
            {
                var matrix = new Matrix<T>(new[] { input });
                return _predictFunction(matrix);
            }

            _igExplainer = new IntegratedGradientsExplainer<T>(
                VectorPredict,
                null, // Use numerical gradients
                _numFeatures,
                numSteps: _options.IntegrationSteps,
                featureNames: _featureNames);
        }

        var explanation = _igExplainer.Explain(instance, targetClass ?? 0);
        return new FeatureAttribution<T>
        {
            FeatureNames = explanation.FeatureNames ?? _featureNames ?? CreateDefaultFeatureNames(),
            Attributions = explanation.Attributions,
            Instance = instance,
            Method = "IntegratedGradients"
        };
    }

    /// <summary>
    /// Explains with LIME.
    /// </summary>
    private FeatureAttribution<T> ExplainWithLIME(Vector<T> instance)
    {
        if (_limeExplainer == null)
        {
            _limeExplainer = new LIMEExplainer<T>(
                _predictFunction,
                _numFeatures,
                featureNames: _featureNames);
        }

        var explanation = _limeExplainer.Explain(instance);
        return new FeatureAttribution<T>
        {
            FeatureNames = explanation.FeatureNames ?? _featureNames ?? CreateDefaultFeatureNames(),
            Attributions = explanation.Coefficients,
            Instance = instance,
            Method = "LIME"
        };
    }

    /// <summary>
    /// Creates default feature names.
    /// </summary>
    private string[] CreateDefaultFeatureNames()
    {
        return Enumerable.Range(0, _numFeatures).Select(i => $"Feature {i}").ToArray();
    }

    /// <summary>
    /// Computes cache key for an instance.
    /// </summary>
    private static string ComputeCacheKey(Vector<T> instance, int? targetClass)
    {
        unchecked
        {
            int hash = 17;
            for (int i = 0; i < instance.Length; i++)
            {
                hash = hash * 31 + NumOps.ToDouble(instance[i]).GetHashCode();
            }
            if (targetClass.HasValue)
            {
                hash = hash * 31 + targetClass.Value;
            }
            return hash.ToString();
        }
    }
}

/// <summary>
/// Model types that the smart selector recognizes for choosing the optimal explainer.
/// </summary>
public enum ExplainableModelType
{
    /// <summary>
    /// Decision tree, random forest, gradient boosting.
    /// </summary>
    TreeEnsemble,

    /// <summary>
    /// Neural network with gradient access.
    /// </summary>
    NeuralNetwork,

    /// <summary>
    /// Linear regression, logistic regression.
    /// </summary>
    Linear,

    /// <summary>
    /// SVM, kernel regression.
    /// </summary>
    KernelBased,

    /// <summary>
    /// Any model with only predict access.
    /// </summary>
    BlackBox
}

/// <summary>
/// Explainer types available.
/// </summary>
public enum ExplainerType
{
    /// <summary>
    /// Kernel SHAP (model-agnostic).
    /// </summary>
    SHAP,

    /// <summary>
    /// Kernel SHAP (same as SHAP).
    /// </summary>
    KernelSHAP,

    /// <summary>
    /// TreeSHAP for tree models.
    /// </summary>
    TreeSHAP,

    /// <summary>
    /// DeepSHAP for neural networks.
    /// </summary>
    DeepSHAP,

    /// <summary>
    /// Integrated Gradients for neural networks.
    /// </summary>
    IntegratedGradients,

    /// <summary>
    /// LIME (model-agnostic).
    /// </summary>
    LIME
}

/// <summary>
/// Options for smart explainer selector.
/// </summary>
public class SmartExplainerOptions
{
    /// <summary>
    /// Whether to enable caching of explanations.
    /// </summary>
    public bool EnableCaching { get; set; } = true;

    /// <summary>
    /// Maximum cache size.
    /// </summary>
    public int MaxCacheSize { get; set; } = 1000;

    /// <summary>
    /// Whether to use parallel processing for batch explanations.
    /// </summary>
    public bool UseParallelProcessing { get; set; } = true;

    /// <summary>
    /// Maximum degree of parallelism.
    /// </summary>
    public int MaxParallelism { get; set; } = Environment.ProcessorCount;

    /// <summary>
    /// Number of integration steps for Integrated Gradients.
    /// </summary>
    public int IntegrationSteps { get; set; } = 50;

    /// <summary>
    /// Prefer DeepSHAP over Integrated Gradients for neural networks.
    /// </summary>
    public bool PreferDeepSHAP { get; set; } = false;

    /// <summary>
    /// Prefer LIME over KernelSHAP for black-box models.
    /// </summary>
    public bool PreferLIME { get; set; } = false;
}

/// <summary>
/// Generic feature attribution result.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class FeatureAttribution<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets or sets feature names.
    /// </summary>
    public string[] FeatureNames { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Gets or sets attribution values.
    /// </summary>
    public Vector<T> Attributions { get; set; } = new Vector<T>(0);

    /// <summary>
    /// Gets or sets the input instance.
    /// </summary>
    public Vector<T> Instance { get; set; } = new Vector<T>(0);

    /// <summary>
    /// Gets or sets the explanation method used.
    /// </summary>
    public string Method { get; set; } = string.Empty;

    /// <summary>
    /// Gets attributions sorted by absolute magnitude.
    /// </summary>
    public List<(string Name, T Value, T Attribution)> GetSortedAttributions()
    {
        var result = new List<(string, T, T)>();
        for (int i = 0; i < Attributions.Length && i < FeatureNames.Length && i < Instance.Length; i++)
        {
            result.Add((FeatureNames[i], Instance[i], Attributions[i]));
        }
        return result.OrderByDescending(x => Math.Abs(NumOps.ToDouble(x.Item3))).ToList();
    }

    /// <summary>
    /// Returns a human-readable summary.
    /// </summary>
    public override string ToString()
    {
        var top = GetSortedAttributions().Take(5);
        var lines = new List<string>
        {
            $"Feature Attribution ({Method}):",
        };

        foreach (var (name, value, attr) in top)
        {
            lines.Add($"  {name}: value={NumOps.ToDouble(value):F4}, attr={NumOps.ToDouble(attr):F4}");
        }

        return string.Join(Environment.NewLine, lines);
    }
}

/// <summary>
/// LRU cache for explanations.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
internal class ExplainerCache<T>
{
    private readonly int _maxSize;
    private readonly ConcurrentDictionary<string, FeatureAttribution<T>> _cache;
    private readonly ConcurrentQueue<string> _accessOrder;
    private long _hits;
    private long _misses;

    public ExplainerCache(int maxSize)
    {
        _maxSize = maxSize;
        _cache = new ConcurrentDictionary<string, FeatureAttribution<T>>();
        _accessOrder = new ConcurrentQueue<string>();
    }

    public double HitRate => _hits + _misses > 0 ? (double)_hits / (_hits + _misses) : 0;

    public bool TryGet(string key, out FeatureAttribution<T>? value)
    {
        if (_cache.TryGetValue(key, out value))
        {
            Interlocked.Increment(ref _hits);
            return true;
        }

        Interlocked.Increment(ref _misses);
        value = null;
        return false;
    }

    public void Set(string key, FeatureAttribution<T> value)
    {
        while (_cache.Count >= _maxSize && _accessOrder.TryDequeue(out var oldKey))
        {
            _cache.TryRemove(oldKey, out _);
        }

        _cache[key] = value;
        _accessOrder.Enqueue(key);
    }

    public void Clear()
    {
        _cache.Clear();
        while (_accessOrder.TryDequeue(out _)) { }
        _hits = 0;
        _misses = 0;
    }
}

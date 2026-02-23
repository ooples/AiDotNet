using AiDotNet.Validation;

namespace AiDotNet.Kernels;

/// <summary>
/// Additive Structure Kernel that decomposes the function into additive components.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Additive Structure Kernel assumes that the target function
/// can be decomposed into a sum of lower-dimensional components:
///
/// f(x) = f₁(x₁) + f₂(x₂) + ... + fₘ(x_groupₘ)
///
/// Where each fᵢ depends only on a subset of input features (a "group").
///
/// This is useful because:
/// 1. **Interpretability**: You can understand how each feature (or group) contributes
/// 2. **Efficiency**: Additive structure can reduce computational complexity
/// 3. **Generalization**: Simpler models often generalize better
///
/// Example: Predicting house prices might be additive:
/// price = location_effect(lat, lon) + size_effect(sqft) + age_effect(year_built)
///
/// The kernel for additive structure is:
/// k(x, x') = Σᵢ kᵢ(x_groupᵢ, x'_groupᵢ)
///
/// Each component can have its own kernel (e.g., RBF for smooth components,
/// periodic for cyclical features).
/// </para>
/// <para>
/// Applications:
/// - Feature importance analysis
/// - Structured time series (trend + seasonality)
/// - Scientific modeling with known additive structure
/// - High-dimensional problems where full interactions are intractable
/// </para>
/// </remarks>
public class AdditiveStructureKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The component kernels.
    /// </summary>
    private readonly IKernelFunction<T>[] _componentKernels;

    /// <summary>
    /// The feature indices for each component.
    /// </summary>
    private readonly int[][] _featureGroups;

    /// <summary>
    /// The weights for each component.
    /// </summary>
    private readonly double[] _weights;

    /// <summary>
    /// Number of components.
    /// </summary>
    private readonly int _numComponents;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes an Additive Structure Kernel with specified components.
    /// </summary>
    /// <param name="componentKernels">Kernel for each component.</param>
    /// <param name="featureGroups">Feature indices for each component.</param>
    /// <param name="weights">Weight for each component. If null, all weights are 1.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates an additive kernel with explicit groupings.
    ///
    /// Example:
    /// // Assume features are [x0, x1, x2, x3]
    /// // We want: f(x) = f1(x0, x1) + f2(x2) + f3(x3)
    /// var rbf = new GaussianKernel&lt;double&gt;(1.0);
    /// var kernels = new[] { rbf, rbf, rbf };
    /// var groups = new[] {
    ///     new[] { 0, 1 },  // First component uses features 0 and 1
    ///     new[] { 2 },     // Second component uses feature 2
    ///     new[] { 3 }      // Third component uses feature 3
    /// };
    /// var additive = new AdditiveStructureKernel&lt;double&gt;(kernels, groups);
    /// </para>
    /// </remarks>
    public AdditiveStructureKernel(
        IKernelFunction<T>[] componentKernels,
        int[][] featureGroups,
        double[]? weights = null)
    {
        if (componentKernels is null) throw new ArgumentNullException(nameof(componentKernels));
        if (featureGroups is null) throw new ArgumentNullException(nameof(featureGroups));
        if (componentKernels.Length == 0)
            throw new ArgumentException("Must have at least one component.", nameof(componentKernels));
        if (componentKernels.Length != featureGroups.Length)
            throw new ArgumentException("Number of kernels must match number of feature groups.");

        _numComponents = componentKernels.Length;
        _componentKernels = new IKernelFunction<T>[_numComponents];
        _featureGroups = new int[_numComponents][];

        for (int i = 0; i < _numComponents; i++)
        {
            Guard.NotNull(componentKernels[i], $"componentKernels[{i}]");
            Guard.NotNull(featureGroups[i], $"featureGroups[{i}]");
            _componentKernels[i] = componentKernels[i];
            _featureGroups[i] = featureGroups[i];
            if (_featureGroups[i].Length == 0)
                throw new ArgumentException($"Feature group {i} cannot be empty.");
        }

        if (weights is null)
        {
            _weights = new double[_numComponents];
            for (int i = 0; i < _numComponents; i++) _weights[i] = 1.0;
        }
        else
        {
            if (weights.Length != _numComponents)
                throw new ArgumentException("Weights length must match number of components.");
            // Validate weights to preserve kernel positive semi-definiteness
            for (int i = 0; i < weights.Length; i++)
            {
                if (weights[i] < 0)
                    throw new ArgumentException($"Weight at index {i} is negative ({weights[i]}). Weights must be non-negative to preserve kernel PSD property.");
                if (double.IsNaN(weights[i]) || double.IsInfinity(weights[i]))
                    throw new ArgumentException($"Weight at index {i} is not finite ({weights[i]}). Weights must be finite real numbers.");
            }
            _weights = (double[])weights.Clone();
        }

        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Initializes an Additive Structure Kernel with one kernel per feature (fully additive).
    /// </summary>
    /// <param name="numFeatures">Total number of input features.</param>
    /// <param name="baseKernel">Base kernel to use for all features.</param>
    /// <param name="weights">Weight for each feature. If null, all weights are 1.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a fully additive model where each feature has
    /// its own independent effect:
    ///
    /// f(x) = f₁(x₁) + f₂(x₂) + ... + f_d(x_d)
    ///
    /// This is the simplest additive structure - no interactions between features.
    /// </para>
    /// </remarks>
    public AdditiveStructureKernel(int numFeatures, IKernelFunction<T> baseKernel, double[]? weights = null)
    {
        if (numFeatures < 1)
            throw new ArgumentException("Must have at least one feature.", nameof(numFeatures));
        if (baseKernel is null) throw new ArgumentNullException(nameof(baseKernel));

        _numComponents = numFeatures;
        _componentKernels = new IKernelFunction<T>[_numComponents];
        _featureGroups = new int[_numComponents][];

        for (int i = 0; i < _numComponents; i++)
        {
            _componentKernels[i] = baseKernel;
            _featureGroups[i] = new[] { i };
        }

        if (weights is null)
        {
            _weights = new double[_numComponents];
            for (int i = 0; i < _numComponents; i++) _weights[i] = 1.0;
        }
        else
        {
            if (weights.Length != _numComponents)
                throw new ArgumentException("Weights length must match number of features.");
            // Validate weights to preserve kernel positive semi-definiteness
            for (int i = 0; i < weights.Length; i++)
            {
                if (weights[i] < 0)
                    throw new ArgumentException($"Weight at index {i} is negative ({weights[i]}). Weights must be non-negative to preserve kernel PSD property.");
                if (double.IsNaN(weights[i]) || double.IsInfinity(weights[i]))
                    throw new ArgumentException($"Weight at index {i} is not finite ({weights[i]}). Weights must be finite real numbers.");
            }
            _weights = (double[])weights.Clone();
        }

        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Gets the number of additive components.
    /// </summary>
    public int NumComponents => _numComponents;

    /// <summary>
    /// Gets the feature group for a component.
    /// </summary>
    /// <param name="componentIndex">Index of the component.</param>
    /// <returns>Array of feature indices for that component.</returns>
    public int[] GetFeatureGroup(int componentIndex)
    {
        if (componentIndex < 0 || componentIndex >= _numComponents)
            throw new ArgumentOutOfRangeException(nameof(componentIndex));
        return (int[])_featureGroups[componentIndex].Clone();
    }

    /// <summary>
    /// Gets the weight for a component.
    /// </summary>
    /// <param name="componentIndex">Index of the component.</param>
    /// <returns>The weight for that component.</returns>
    public double GetWeight(int componentIndex)
    {
        if (componentIndex < 0 || componentIndex >= _numComponents)
            throw new ArgumentOutOfRangeException(nameof(componentIndex));
        return _weights[componentIndex];
    }

    /// <summary>
    /// Calculates the additive kernel value.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The sum of component kernel values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Computes the kernel by summing component contributions:
    /// k(x, x') = Σᵢ wᵢ × kᵢ(x_groupᵢ, x'_groupᵢ)
    ///
    /// Each component only "sees" its assigned features.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        double result = 0;

        for (int c = 0; c < _numComponents; c++)
        {
            // Extract features for this component
            var subX1 = ExtractFeatures(x1, _featureGroups[c]);
            var subX2 = ExtractFeatures(x2, _featureGroups[c]);

            // Compute component kernel value
            double kVal = _numOps.ToDouble(_componentKernels[c].Calculate(subX1, subX2));
            result += _weights[c] * kVal;
        }

        return _numOps.FromDouble(result);
    }

    /// <summary>
    /// Calculates the kernel value for a specific component only.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <param name="componentIndex">Index of the component.</param>
    /// <returns>The component's contribution to the kernel.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Computes just one component's contribution.
    /// Useful for understanding how much each component (feature group) contributes.
    /// </para>
    /// </remarks>
    public double CalculateComponent(Vector<T> x1, Vector<T> x2, int componentIndex)
    {
        if (componentIndex < 0 || componentIndex >= _numComponents)
            throw new ArgumentOutOfRangeException(nameof(componentIndex));

        var subX1 = ExtractFeatures(x1, _featureGroups[componentIndex]);
        var subX2 = ExtractFeatures(x2, _featureGroups[componentIndex]);

        return _weights[componentIndex] * _numOps.ToDouble(
            _componentKernels[componentIndex].Calculate(subX1, subX2));
    }

    /// <summary>
    /// Extracts specified features from a vector.
    /// </summary>
    private Vector<T> ExtractFeatures(Vector<T> x, int[] indices)
    {
        var result = new Vector<T>(indices.Length);
        for (int i = 0; i < indices.Length; i++)
        {
            result[i] = x[indices[i]];
        }
        return result;
    }

    /// <summary>
    /// Computes the importance of each component based on its kernel value at the origin.
    /// </summary>
    /// <param name="x">A reference point to evaluate component importance.</param>
    /// <returns>Array of component importances (normalized to sum to 1).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This gives a rough estimate of how important each component is
    /// by looking at its variance contribution (the kernel evaluated at k(x, x)).
    ///
    /// Higher values indicate the component explains more variance.
    /// </para>
    /// </remarks>
    public double[] GetComponentImportances(Vector<T> x)
    {
        var importances = new double[_numComponents];
        double total = 0;

        for (int c = 0; c < _numComponents; c++)
        {
            importances[c] = CalculateComponent(x, x, c);
            total += importances[c];
        }

        if (total > 1e-10)
        {
            for (int c = 0; c < _numComponents; c++)
            {
                importances[c] /= total;
            }
        }

        return importances;
    }

    /// <summary>
    /// Creates a simple additive kernel with RBF components.
    /// </summary>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="lengthScale">Length scale for all components.</param>
    /// <returns>A fully additive RBF kernel.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates the simplest additive GP model with RBF kernels.
    /// Good starting point for additive modeling.
    /// </para>
    /// </remarks>
    public static AdditiveStructureKernel<T> WithRBF(int numFeatures, double lengthScale = 1.0)
    {
        var rbf = new GaussianKernel<T>(lengthScale);
        return new AdditiveStructureKernel<T>(numFeatures, rbf);
    }
}

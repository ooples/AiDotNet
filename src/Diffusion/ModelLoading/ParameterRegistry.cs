using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Diffusion.ModelLoading;

/// <summary>
/// Manages named parameters for weight loading.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class is like a phone book for model parameters.
/// Each parameter has a name (like "encoder.conv1.weight") and we can look up
/// or set parameters by their names.
///
/// When loading pretrained weights, we need to know:
/// 1. What parameters exist in our model
/// 2. What shape each parameter should be
/// 3. Where to actually put the weight data
///
/// This registry provides all three capabilities.
/// </para>
/// </remarks>
public class ParameterRegistry<T>
{
    /// <summary>
    /// Provides numeric operations for the specific type T.
    /// </summary>
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Maps parameter names to their accessors.
    /// </summary>
    private readonly Dictionary<string, ParameterAccessor<T>> _parameters;

    /// <summary>
    /// Initializes a new empty parameter registry.
    /// </summary>
    public ParameterRegistry()
    {
        _parameters = new Dictionary<string, ParameterAccessor<T>>(StringComparer.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Registers a parameter with getter and setter delegates.
    /// </summary>
    /// <param name="name">The parameter name.</param>
    /// <param name="shape">The expected shape.</param>
    /// <param name="getter">Function to get the current tensor.</param>
    /// <param name="setter">Action to set the tensor value.</param>
    public void Register(string name, int[] shape, Func<Tensor<T>?> getter, Action<Tensor<T>> setter)
    {
        _parameters[name] = new ParameterAccessor<T>(name, shape, getter, setter);
    }

    /// <summary>
    /// Registers a layer's weights and biases.
    /// </summary>
    /// <param name="prefix">Name prefix for the layer (e.g., "encoder.conv1").</param>
    /// <param name="layer">The layer to register.</param>
    public void RegisterLayer(string prefix, ILayer<T> layer)
    {
        var weights = layer.GetWeights();
        if (weights != null)
        {
            Register(
                $"{prefix}.weight",
                weights.Shape.ToArray(),
                () => layer.GetWeights(),
                tensor => SetLayerWeights(layer, tensor));
        }

        var biases = layer.GetBiases();
        if (biases != null)
        {
            Register(
                $"{prefix}.bias",
                biases.Shape.ToArray(),
                () => layer.GetBiases(),
                tensor => SetLayerBiases(layer, tensor));
        }
    }

    /// <summary>
    /// Registers multiple layers with a naming pattern.
    /// </summary>
    /// <param name="prefix">Prefix for all layer names.</param>
    /// <param name="layers">The layers to register.</param>
    public void RegisterLayers(string prefix, IEnumerable<(string Name, ILayer<T> Layer)> layers)
    {
        foreach (var (name, layer) in layers)
        {
            RegisterLayer($"{prefix}.{name}", layer);
        }
    }

    /// <summary>
    /// Registers a child ParameterRegistry with a prefix.
    /// </summary>
    /// <param name="prefix">Prefix to add to all child parameter names.</param>
    /// <param name="child">The child registry.</param>
    public void RegisterChild(string prefix, ParameterRegistry<T> child)
    {
        foreach (var kvp in child._parameters)
        {
            var fullName = $"{prefix}.{kvp.Key}";
            _parameters[fullName] = new ParameterAccessor<T>(
                fullName,
                kvp.Value.Shape,
                kvp.Value.Getter,
                kvp.Value.Setter);
        }
    }

    /// <summary>
    /// Gets all registered parameter names.
    /// </summary>
    public IEnumerable<string> GetNames() => _parameters.Keys;

    /// <summary>
    /// Gets the number of registered parameters.
    /// </summary>
    public int Count => _parameters.Count;

    /// <summary>
    /// Tries to get a parameter by name.
    /// </summary>
    public bool TryGet(string name, out Tensor<T>? tensor)
    {
        if (_parameters.TryGetValue(name, out var accessor))
        {
            tensor = accessor.Getter();
            return true;
        }
        tensor = null;
        return false;
    }

    /// <summary>
    /// Sets a parameter by name.
    /// </summary>
    /// <returns>True if set successfully, false if name not found.</returns>
    public bool TrySet(string name, Tensor<T> value)
    {
        if (!_parameters.TryGetValue(name, out var accessor))
        {
            return false;
        }

        // Validate shape
        var expectedShape = accessor.Shape;
        var actualShape = value.Shape.ToArray();
        if (!ShapesMatch(expectedShape, actualShape))
        {
            throw new ArgumentException(
                $"Shape mismatch for '{name}': expected [{string.Join(", ", expectedShape)}], " +
                $"got [{string.Join(", ", actualShape)}]");
        }

        accessor.Setter(value);
        return true;
    }

    /// <summary>
    /// Gets the expected shape for a parameter.
    /// </summary>
    public int[]? GetShape(string name)
    {
        return _parameters.TryGetValue(name, out var accessor)
            ? accessor.Shape
            : null;
    }

    /// <summary>
    /// Validates weights against registered parameters.
    /// </summary>
    public WeightLoadValidation Validate(
        IEnumerable<string> weightNames,
        Func<string, string?>? mapping = null)
    {
        var result = new WeightLoadValidation();
        var registeredNames = new HashSet<string>(_parameters.Keys, StringComparer.OrdinalIgnoreCase);
        var matchedParams = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        foreach (var weightName in weightNames)
        {
            var targetName = mapping?.Invoke(weightName) ?? weightName;
            if (targetName == null)
            {
                result.UnmatchedWeights.Add(weightName);
                continue;
            }

            if (registeredNames.Contains(targetName))
            {
                result.Matched.Add(targetName);
                matchedParams.Add(targetName);
            }
            else
            {
                result.UnmatchedWeights.Add(weightName);
            }
        }

        // Find missing parameters
        foreach (var paramName in registeredNames)
        {
            if (!matchedParams.Contains(paramName))
            {
                result.MissingParameters.Add(paramName);
            }
        }

        return result;
    }

    /// <summary>
    /// Loads weights from a dictionary.
    /// </summary>
    public WeightLoadResult Load(
        Dictionary<string, Tensor<T>> weights,
        Func<string, string?>? mapping = null,
        bool strict = false)
    {
        var result = new WeightLoadResult { Success = true };

        foreach (var kvp in weights)
        {
            var sourceName = kvp.Key;
            var tensor = kvp.Value;

            var targetName = mapping?.Invoke(sourceName) ?? sourceName;
            if (targetName == null)
            {
                result.SkippedCount++;
                continue;
            }

            try
            {
                if (TrySet(targetName, tensor))
                {
                    result.LoadedCount++;
                    result.LoadedParameters.Add(targetName);
                }
                else
                {
                    if (strict)
                    {
                        result.FailedCount++;
                        result.FailedParameters.Add((targetName, "Parameter not found"));
                        result.Success = false;
                    }
                    else
                    {
                        result.SkippedCount++;
                    }
                }
            }
            catch (ArgumentException ex)
            {
                result.FailedCount++;
                result.FailedParameters.Add((targetName, ex.Message));
                if (strict)
                {
                    result.Success = false;
                    result.ErrorMessage = $"Failed to load '{targetName}': {ex.Message}";
                }
            }
        }

        return result;
    }

    private static bool ShapesMatch(int[] expected, int[] actual)
    {
        if (expected.Length != actual.Length)
            return false;
        for (int i = 0; i < expected.Length; i++)
        {
            if (expected[i] != actual[i])
                return false;
        }
        return true;
    }

    private void SetLayerWeights(ILayer<T> layer, Tensor<T> tensor)
    {
        // Get current parameters, replace weights portion, set back
        var currentParams = layer.GetParameters();
        var weights = layer.GetWeights();
        var biases = layer.GetBiases();

        if (weights == null)
            return;

        int weightsLen = weights.Length;
        int biasLen = biases?.Length ?? 0;

        var newParams = new Vector<T>(weightsLen + biasLen);

        // Copy new weights
        for (int i = 0; i < weightsLen; i++)
        {
            newParams[i] = tensor.Data[i];
        }

        // Copy existing biases
        if (biases != null)
        {
            for (int i = 0; i < biasLen; i++)
            {
                newParams[weightsLen + i] = currentParams[weightsLen + i];
            }
        }

        layer.SetParameters(newParams);
    }

    private void SetLayerBiases(ILayer<T> layer, Tensor<T> tensor)
    {
        // Get current parameters, replace biases portion, set back
        var currentParams = layer.GetParameters();
        var weights = layer.GetWeights();
        var biases = layer.GetBiases();

        if (biases == null)
            return;

        int weightsLen = weights?.Length ?? 0;
        int biasLen = biases.Length;

        var newParams = new Vector<T>(weightsLen + biasLen);

        // Copy existing weights
        if (weights != null)
        {
            for (int i = 0; i < weightsLen; i++)
            {
                newParams[i] = currentParams[i];
            }
        }

        // Copy new biases
        for (int i = 0; i < biasLen; i++)
        {
            newParams[weightsLen + i] = tensor.Data[i];
        }

        layer.SetParameters(newParams);
    }
}

/// <summary>
/// Internal accessor for a named parameter.
/// </summary>
internal class ParameterAccessor<T>
{
    public string Name { get; }
    public int[] Shape { get; }
    public Func<Tensor<T>?> Getter { get; }
    public Action<Tensor<T>> Setter { get; }

    public ParameterAccessor(string name, int[] shape, Func<Tensor<T>?> getter, Action<Tensor<T>> setter)
    {
        Name = name;
        Shape = shape;
        Getter = getter;
        Setter = setter;
    }
}

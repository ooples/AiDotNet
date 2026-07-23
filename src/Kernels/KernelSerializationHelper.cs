namespace AiDotNet.Kernels;

/// <summary>
/// Shared utility methods for serializing and deserializing kernel functions.
/// </summary>
internal static class KernelSerializationHelper<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Extracts kernel hyperparameters via reflection for serialization.
    /// Captures <c>double</c>, <c>T</c>, and <c>int</c>-typed private fields.
    /// </summary>
    public static Dictionary<string, double> ExtractKernelParams(IKernelFunction<T> kernel)
    {
        var parameters = new Dictionary<string, double>();
        var type = kernel.GetType();

        foreach (var field in type.GetFields(System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance))
        {
            if (field.FieldType == typeof(double))
            {
                var val = field.GetValue(kernel);
                if (val is double d)
                    parameters[field.Name] = d;
            }
            else if (field.FieldType == typeof(T))
            {
                var val = field.GetValue(kernel);
                if (val is T tVal)
                    parameters[field.Name] = NumOps.ToDouble(tVal);
            }
            else if (field.FieldType == typeof(int))
            {
                var val = field.GetValue(kernel);
                if (val is int i)
                    parameters[field.Name] = i;
            }
        }

        return parameters;
    }

    /// <summary>
    /// Recreates a kernel by type name and restores hyperparameters from serialized values.
    /// </summary>
    public static IKernelFunction<T>? CreateKernelByName(string kernelTypeName, Dictionary<string, double>? kernelParams = null)
    {
        double sigma = kernelParams is not null && kernelParams.TryGetValue("_sigma", out var s) ? s : 1.0;

        return kernelTypeName switch
        {
            "GaussianKernel`1" or "GaussianKernel" => new GaussianKernel<T>(sigma),
            "LaplacianKernel`1" or "LaplacianKernel" => CreateLaplacianKernel(kernelParams),
            "LinearKernel`1" or "LinearKernel" => new LinearKernel<T>(),
            "PolynomialKernel`1" or "PolynomialKernel" => CreatePolynomialKernel(kernelParams),
            _ => null
        };
    }

    private static LaplacianKernel<T> CreateLaplacianKernel(Dictionary<string, double>? kernelParams)
    {
        if (kernelParams is not null && kernelParams.TryGetValue("_sigma", out var sigma))
        {
            return new LaplacianKernel<T>(NumOps.FromDouble(sigma));
        }

        return new LaplacianKernel<T>();
    }

    private static PolynomialKernel<T> CreatePolynomialKernel(Dictionary<string, double>? kernelParams)
    {
        if (kernelParams is null)
        {
            return new PolynomialKernel<T>();
        }

        T? degree = kernelParams.TryGetValue("_degree", out var d) ? NumOps.FromDouble(d) : default;
        T? coef0 = kernelParams.TryGetValue("_coef0", out var c) ? NumOps.FromDouble(c) : default;

        return new PolynomialKernel<T>(degree, coef0);
    }
}

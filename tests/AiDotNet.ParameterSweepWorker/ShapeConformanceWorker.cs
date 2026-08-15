using System.Reflection;
using System.Text.Json;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

internal static class ShapeConformanceWorker
{
    public static int Run(string[] args)
    {
        ShapeProbeMeasurement result;
        try
        {
            if (args.Length != 3
                || !int.TryParse(args[1], out int extent)
                || !int.TryParse(args[2], out int classes)
                || extent <= 0
                || classes <= 0)
            {
                throw new ArgumentException("Usage: shape <type> <extent> <classes>.");
            }

            result = Probe(args[0], extent, classes);
        }
        catch (Exception ex)
        {
            var cause = ex.GetBaseException();
            result = new ShapeProbeMeasurement(
                "error", null, null, null, $"{cause.GetType().Name}: {cause.Message}");
        }

        Console.WriteLine(JsonSerializer.Serialize(result));
        return result.Status == "error" ? 2 : 0;
    }

    private static ShapeProbeMeasurement Probe(string assemblyQualifiedType, int extent, int classes)
    {
        var type = Type.GetType(assemblyQualifiedType, throwOnError: true)!;
        object? model = null;
        int[]? inputShape = null;
        int[]? predictedShape = null;
        BoundInputContract? inputContract = null;
        string? lastError = null;

        try
        {
            foreach (var inputType in new[]
                     { InputType.ThreeDimensional, InputType.TwoDimensional, InputType.OneDimensional })
            {
                object? candidate = null;
                try
                {
                    candidate = Construct(type, inputType, extent, classes);
                    if (candidate is null)
                    {
                        lastError ??= "no usable constructor";
                        continue;
                    }

                    if (candidate is not IShapeContract contract)
                    {
                        lastError ??= "constructed instance is not IShapeContract";
                        continue;
                    }

                    int[]? perSample = TryArchitectureInputShape(candidate);
                    if (perSample is null || perSample.Length == 0 || perSample.Any(d => d <= 0))
                    {
                        lastError ??= "no concrete declared input shape";
                        continue;
                    }

                    var shape = new int[perSample.Length + 1];
                    shape[0] = 1;
                    for (int i = 0; i < perSample.Length; i++) shape[i + 1] = Math.Min(perSample[i], extent);

                    BoundInputContract? candidateContract = null;
                    if (candidate is NeuralNetworkBase<double> network)
                    {
                        shape = InputContractShapeResolver.Conform(
                            shape,
                            network.GetInputShapeConstraint());
                        candidateContract = network.BindInputContract(shape);
                        candidateContract.RequireReady();
                    }

                    int[]? prediction = ShapeInference.InferOutputShape(contract, shape);
                    if (model is null || prediction is not null)
                    {
                        (model as IDisposable)?.Dispose();
                        model = candidate;
                        candidate = null;
                        inputShape = shape;
                        predictedShape = prediction;
                        inputContract = candidateContract;
                        if (prediction is not null) break;
                    }
                }
                catch (Exception ex)
                {
                    var cause = Unwrap(ex);
                    lastError ??= $"{cause.GetType().Name} constructing: {FirstLine(cause.Message)}";
                }
                finally
                {
                    (candidate as IDisposable)?.Dispose();
                }
            }

            if (model is null || inputShape is null)
                return new ShapeProbeMeasurement("unconstructable", null, null, null, lastError);

            if (predictedShape is null)
                return new ShapeProbeMeasurement("declined", inputShape, null, null,
                    "the concrete contract did not answer any supported probe rank");

            var (actualShape, failure) = TryPredict(model, inputShape, inputContract);
            if (actualShape is null)
                return new ShapeProbeMeasurement("predict-failed", inputShape, predictedShape, null, failure);

            string status = predictedShape.SequenceEqual(actualShape) ? "agreed" : "disagreed";
            return new ShapeProbeMeasurement(status, inputShape, predictedShape, actualShape, null);
        }
        finally
        {
            (model as IDisposable)?.Dispose();
        }
    }

    private static object? Construct(Type type, InputType inputType, int extent, int classes)
    {
        var ctor = type.GetConstructors(BindingFlags.Public | BindingFlags.Instance).FirstOrDefault(c =>
        {
            var parameters = c.GetParameters();
            return parameters.Length > 0
                && parameters[0].ParameterType == typeof(NeuralNetworkArchitecture<double>)
                && parameters.Skip(1).All(p => p.HasDefaultValue);
        });

        if (ctor is null)
            return type.GetConstructor(Type.EmptyTypes) is not null ? Activator.CreateInstance(type) : null;

        var parameters = ctor.GetParameters();
        var arguments = new object?[parameters.Length];
        arguments[0] = inputType switch
        {
            InputType.OneDimensional => new NeuralNetworkArchitecture<double>(
                InputType.OneDimensional, NeuralNetworkTaskType.Regression,
                inputSize: extent, outputSize: classes),
            InputType.TwoDimensional => new NeuralNetworkArchitecture<double>(
                InputType.TwoDimensional, NeuralNetworkTaskType.Regression,
                inputHeight: extent, inputWidth: extent, outputSize: classes),
            _ => new NeuralNetworkArchitecture<double>(
                InputType.ThreeDimensional, NeuralNetworkTaskType.Regression,
                inputDepth: 3, inputHeight: extent, inputWidth: extent, outputSize: classes),
        };

        for (int i = 1; i < parameters.Length; i++)
        {
            var parameter = parameters[i];
            bool isClassCount = parameter.ParameterType == typeof(int)
                && (parameter.Name?.IndexOf("numClasses", StringComparison.OrdinalIgnoreCase) >= 0
                    || parameter.Name?.IndexOf("classCount", StringComparison.OrdinalIgnoreCase) >= 0);
            arguments[i] = isClassCount ? classes : ResolveProbeArgument(parameter);
        }

        return ctor.Invoke(arguments);
    }

    /// <summary>
    /// Uses an options type's faithful small profile for bounded CI probes when it publishes one.
    /// The convention keeps the worker model-agnostic and prevents production-scale defaults from
    /// turning structural conformance into an allocation benchmark.
    /// </summary>
    private static object? ResolveProbeArgument(ParameterInfo parameter)
    {
        var factory = parameter.ParameterType
            .GetMethods(BindingFlags.Public | BindingFlags.Static)
            .FirstOrDefault(method =>
                method.Name == "TinyForTests"
                && parameter.ParameterType.IsAssignableFrom(method.ReturnType)
                && method.GetParameters().All(p => p.HasDefaultValue));

        if (factory is null) return parameter.DefaultValue;

        object?[] arguments = factory.GetParameters().Select(p => p.DefaultValue).ToArray();
        return factory.Invoke(null, arguments);
    }

    private static int[]? TryArchitectureInputShape(object model)
    {
        try
        {
            dynamic architecture = ((dynamic)model).GetArchitecture();
            return (int[])architecture.GetInputShape();
        }
        catch
        {
            return null;
        }
    }

    private static (int[]? Shape, string? Failure) TryPredict(
        object model,
        int[] shape,
        BoundInputContract? contract)
    {
        try
        {
            Tensor<double> probe;
            if (contract is null)
            {
                probe = new Tensor<double>(shape);
                for (int i = 0; i < probe.Length; i++) probe[i] = (i * 7) % 13;
            }
            else
            {
                contract.RequireReady();
                probe = InputContractTensorFactory.CreateValid<double>(contract, new Random(1789));
                contract.Validate(probe);

                if (contract.PrimaryInput.ValueDomain.Kind !=
                    AiDotNet.NeuralNetworks.Layers.LayerInputDomainKind.Continuous)
                {
                    var invalid = InputContractTensorFactory.CreateInvalid<double>(
                        contract.PrimaryInput.Shape.ToArray(),
                        contract.PrimaryInput.ValueDomain);
                    try
                    {
                        contract.Validate(invalid);
                        return (null,
                            "the bound input contract accepted its generated negative example");
                    }
                    catch (InputContractViolationException)
                    {
                        // This is the negative-sweep success condition.
                    }
                }
            }

            var result = (Tensor<double>?)((dynamic)model).Predict(probe);
            return result is null ? (null, "Predict returned null") : (result.Shape.ToArray(), null);
        }
        catch (Exception ex)
        {
            var cause = Unwrap(ex);
            return (null, $"{cause.GetType().Name}: {FirstLine(cause.Message)}");
        }
    }

    private static Exception Unwrap(Exception ex) =>
        ex is TargetInvocationException { InnerException: not null } tie ? tie.InnerException : ex;

    private static string FirstLine(string message)
    {
        string line = message.Split('\n')[0].Trim();
        return line.Length > 240 ? line.Substring(0, 240) + "..." : line;
    }
}

internal sealed record ShapeProbeMeasurement(
    string Status,
    int[]? InputShape,
    int[]? PredictedShape,
    int[]? ActualShape,
    string? Error);

using System.Diagnostics;
using System.Reflection;
using System.Text.Json;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Builds one model under a list of construction profiles and records the output shape Predict
/// returns for a planned set of input shapes. Used by the model shape-law and shape-discovery sweeps.
/// </summary>
/// <remarks>
/// <para>
/// Those sweeps construct hundreds of models by reflection. Run in the xUnit process, one model with
/// a paper-scale default configuration, a pathological constructor, or a forward that never returns
/// held the whole sweep hostage: the law sweep spent its entire 30-minute budget inside its first
/// family and, once xUnit abandoned it, the still-running work surfaced as an unhandled exception in
/// Task.RunContinuations. Here each model gets its own process, a bounded managed heap, and a deadline
/// imposed by the caller, exactly as the contract-conformance sweep already does.
/// </para>
/// <para>
/// The worker only OBSERVES. It never interprets a shape; the calling test owns the analysis, so the
/// measurement is identical to the previous in-process one.
/// </para>
/// </remarks>
internal static class ShapeObservationWorker
{
    public static int Run(string[] args)
    {
        ShapeObservationResult result;
        try
        {
            if (args.Length != 2)
                throw new ArgumentException("Usage: observe <type> <plan-json>.");

            var plan = JsonSerializer.Deserialize<ShapeObservationPlan>(args[1],
                           new JsonSerializerOptions { PropertyNameCaseInsensitive = true })
                       ?? throw new ArgumentException("The observation plan is empty.");
            result = Observe(args[0], plan);
        }
        catch (Exception ex)
        {
            var cause = ex.GetBaseException();
            result = new ShapeObservationResult("error", $"{cause.GetType().Name}: {cause.Message}", []);
        }

        Console.WriteLine(JsonSerializer.Serialize(result));
        return result.Status == "error" ? 2 : 0;
    }

    private static ShapeObservationResult Observe(string assemblyQualifiedType, ShapeObservationPlan plan)
    {
        var type = Type.GetType(assemblyQualifiedType, throwOnError: true)!;
        var profiles = new List<ProfileObservation>(plan.Profiles.Length);

        foreach (var profile in plan.Profiles)
        {
            var clock = Stopwatch.StartNew();
            object? model = null;
            try
            {
                try { model = Construct(type, profile); }
                catch (Exception ex)
                {
                    profiles.Add(new ProfileObservation(
                        $"{Unwrap(ex).GetType().Name} constructing", null, [], clock.ElapsedMilliseconds));
                    continue;
                }

                if (model is null)
                {
                    profiles.Add(new ProfileObservation("no usable constructor", null, [], clock.ElapsedMilliseconds));
                    continue;
                }

                int[]? perSample = TryArchitectureInputShape(model);
                if (perSample is null || perSample.Length == 0 || perSample.Any(d => d <= 0))
                {
                    profiles.Add(new ProfileObservation(
                        "no concrete declared input shape", perSample, [], clock.ElapsedMilliseconds));
                    continue;
                }

                var observations = new List<ShapeObservation>();
                foreach (int[] shape in PlannedShapes(perSample, profile))
                {
                    var (output, failure) = TryPredict(model, shape);
                    observations.Add(new ShapeObservation(shape, output, failure));
                    if (output is null && profile.StopAtFirstFailure) break;
                }

                profiles.Add(new ProfileObservation(null, perSample, observations.ToArray(), clock.ElapsedMilliseconds));
            }
            finally
            {
                (model as IDisposable)?.Dispose();
            }
        }

        return new ShapeObservationResult("observed", null, profiles.ToArray());
    }

    /// <summary>
    /// The shapes a profile asks about: each planned batch at the declared per-sample shape with
    /// every axis capped at <see cref="ObservationProfile.AxisCap"/>, then (when requested) each
    /// per-sample axis moved on its own to <see cref="ObservationProfile.AltExtent"/>.
    /// </summary>
    private static IEnumerable<int[]> PlannedShapes(int[] perSample, ObservationProfile profile)
    {
        int[] BaseShape(int batch)
        {
            var shape = new int[perSample.Length + 1];
            shape[0] = batch;
            for (int i = 0; i < perSample.Length; i++) shape[i + 1] = Math.Min(perSample[i], profile.AxisCap);
            return shape;
        }

        foreach (int batch in profile.Batches) yield return BaseShape(batch);

        if (profile.AltExtent <= 0) yield break;
        for (int axis = 0; axis < perSample.Length; axis++)
        {
            var shape = BaseShape(1);
            if (shape[axis + 1] >= profile.AltExtent) continue;
            shape[axis + 1] = profile.AltExtent;
            yield return shape;
        }
    }

    private static object? Construct(Type type, ObservationProfile profile)
    {
        if (profile.UseDefaultConstructor)
            return type.GetConstructor(Type.EmptyTypes) is not null ? Activator.CreateInstance(type) : null;

        var ctor = type.GetConstructors(BindingFlags.Public | BindingFlags.Instance).FirstOrDefault(c =>
        {
            var parameters = c.GetParameters();
            return parameters.Length > 0
                && parameters[0].ParameterType == typeof(NeuralNetworkArchitecture<double>)
                && parameters.Skip(1).All(p => p.HasDefaultValue);
        });

        if (ctor is null)
        {
            return profile.FallBackToDefaultConstructor && type.GetConstructor(Type.EmptyTypes) is not null
                ? Activator.CreateInstance(type)
                : null;
        }

        var parameters = ctor.GetParameters();
        var arguments = new object?[parameters.Length];
        arguments[0] = profile.InputType switch
        {
            nameof(InputType.OneDimensional) => new NeuralNetworkArchitecture<double>(
                InputType.OneDimensional, NeuralNetworkTaskType.Regression,
                inputSize: profile.InputSize, outputSize: profile.Classes),
            nameof(InputType.ThreeDimensional) => new NeuralNetworkArchitecture<double>(
                InputType.ThreeDimensional, NeuralNetworkTaskType.Regression,
                inputDepth: profile.InputDepth, inputHeight: profile.InputSize, inputWidth: profile.InputSize,
                outputSize: profile.Classes),
            _ => throw new ArgumentException($"Unsupported observation input type '{profile.InputType}'."),
        };

        for (int i = 1; i < parameters.Length; i++)
        {
            var parameter = parameters[i];
            bool isClassCount = profile.OverrideClassCountParameters
                && parameter.ParameterType == typeof(int)
                && (parameter.Name?.IndexOf("numClasses", StringComparison.OrdinalIgnoreCase) >= 0
                    || parameter.Name?.IndexOf("classCount", StringComparison.OrdinalIgnoreCase) >= 0);
            arguments[i] = isClassCount ? profile.Classes : ResolveProbeArgument(parameter);
        }

        return ctor.Invoke(arguments);
    }

    /// <summary>
    /// Same convention as the conformance worker: an options type that publishes a faithful small
    /// profile (<c>TinyForTests</c>) is built at that profile, so a structural probe does not become an
    /// allocation benchmark of a production-scale default.
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

    private static (int[]? Shape, string? Failure) TryPredict(object model, int[] shape)
    {
        try
        {
            // Whole numbers, as the in-process sweeps used: valid token indices for every vocabulary
            // in the inventory and equally valid continuous features. Shape does not depend on values.
            var probe = new Tensor<double>(shape);
            for (int i = 0; i < probe.Length; i++) probe[i] = (i * 7) % 13;
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
        return line.Length > 140 ? line.Substring(0, 140) + "..." : line;
    }
}

internal sealed record ShapeObservationPlan(ObservationProfile[] Profiles);

internal sealed record ObservationProfile(
    string Name,
    string InputType,
    int InputSize,
    int InputDepth,
    int Classes,
    int[] Batches,
    int AxisCap,
    int AltExtent,
    bool UseDefaultConstructor,
    bool FallBackToDefaultConstructor,
    bool OverrideClassCountParameters,
    bool StopAtFirstFailure);

internal sealed record ShapeObservation(int[] Input, int[]? Output, string? Failure);

internal sealed record ProfileObservation(
    string? Failure,
    int[]? PerSampleInput,
    ShapeObservation[] Observations,
    long ElapsedMilliseconds);

internal sealed record ShapeObservationResult(string Status, string? Error, ProfileObservation[] Profiles);

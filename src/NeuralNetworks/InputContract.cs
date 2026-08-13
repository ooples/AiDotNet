using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks;

/// <summary>Whether a bound input contract can safely execute.</summary>
public enum InputContractReadiness
{
    Ready,
    Deferred,
    Invalid
}

/// <summary>
/// Adapts a caller's preferred probe geometry to a declared input constraint. The resolver only
/// removes leading unit axes, so satisfying an exact rank can never silently discard data.
/// </summary>
public static class InputContractShapeResolver
{
    public static int[] Conform(
        IReadOnlyList<int> requestedShape,
        ModelInputShapeConstraint constraint)
    {
        if (requestedShape is null) throw new ArgumentNullException(nameof(requestedShape));
        if (requestedShape.Count == 0 || requestedShape.Any(axis => axis <= 0))
            throw new InputContractBindingException(
                $"Cannot conform input shape [{string.Join(",", requestedShape)}]: every axis must be positive.");

        ValidateConstraint(constraint);
        var shape = requestedShape.ToArray();
        int declaredAxisCount = Math.Max(
            constraint.MinimumAxisSizes?.Count ?? 0,
            constraint.AxisDivisors?.Count ?? 0);
        int requiredRank = constraint.ExactRank > 0
            ? constraint.ExactRank
            : Math.Max(constraint.MinimumRank, declaredAxisCount);

        if (constraint.ExactRank > 0 && declaredAxisCount > constraint.ExactRank)
            throw new InputContractBindingException(
                $"The input constraint declares {declaredAxisCount} axis rules but exact rank is "
                + $"{constraint.ExactRank}.");

        int maximumRank = constraint.ExactRank > 0
            ? constraint.ExactRank
            : constraint.MaximumRank;
        int targetRank = Math.Max(requiredRank, shape.Length);
        if (maximumRank > 0) targetRank = Math.Min(targetRank, maximumRank);

        if (shape.Length > targetRank)
        {
            int remove = shape.Length - targetRank;
            if (shape.Take(remove).Any(axis => axis != 1))
                throw new InputContractBindingException(
                    $"Input shape [{string.Join(",", shape)}] cannot be reduced to rank {targetRank} "
                    + "without discarding a non-unit leading axis.");
            shape = shape.Skip(remove).ToArray();
        }
        else if (shape.Length < targetRank)
        {
            shape = Enumerable.Repeat(1, targetRank - shape.Length).Concat(shape).ToArray();
        }

        ApplyAxisRules(shape, constraint.MinimumAxisSizes, constraint.AxisDivisors);

        if (constraint.MinimumElementCount > 0)
        {
            long elementsBeforeLast = 1;
            for (int axis = 0; axis < shape.Length - 1; axis++)
                elementsBeforeLast = checked(elementsBeforeLast * shape[axis]);
            long requiredLast = (constraint.MinimumElementCount + elementsBeforeLast - 1)
                                / elementsBeforeLast;
            if (requiredLast > int.MaxValue)
                throw new InputContractBindingException(
                    $"Minimum element count {constraint.MinimumElementCount} cannot be represented "
                    + $"by input shape [{string.Join(",", shape)}].");
            shape[shape.Length - 1] = Math.Max(shape[shape.Length - 1], (int)requiredLast);
            RoundAxisToDivisor(shape, shape.Length - 1, constraint.AxisDivisors);
        }

        var reasons = new List<string>();
        InputContractManifest.ValidateModelShape(shape, constraint, reasons);
        if (reasons.Count > 0)
            throw new InputContractBindingException(
                $"Input shape [{string.Join(",", shape)}] could not satisfy its contract: "
                + string.Join("; ", reasons) + ".");
        return shape;
    }

    private static void ValidateConstraint(ModelInputShapeConstraint constraint)
    {
        if (constraint.ExactRank < 0 || constraint.MinimumRank < 0
            || constraint.MaximumRank < 0 || constraint.MinimumElementCount < 0)
            throw new InputContractBindingException("Input shape constraints cannot be negative.");
        if (constraint.ExactRank > 0
            && (constraint.MinimumRank > constraint.ExactRank
                || constraint.MaximumRank > 0 && constraint.MaximumRank < constraint.ExactRank))
            throw new InputContractBindingException(
                "Exact input rank conflicts with the declared minimum or maximum rank.");
        if (constraint.MaximumRank > 0 && constraint.MinimumRank > constraint.MaximumRank)
            throw new InputContractBindingException(
                "Minimum input rank cannot be greater than maximum input rank.");
    }

    private static void ApplyAxisRules(
        int[] shape,
        IReadOnlyList<int>? minima,
        IReadOnlyList<int>? divisors)
    {
        for (int axis = 0; axis < shape.Length; axis++)
        {
            if (minima is not null && axis < minima.Count && minima[axis] > 0)
                shape[axis] = Math.Max(shape[axis], minima[axis]);
            RoundAxisToDivisor(shape, axis, divisors);
        }
    }

    private static void RoundAxisToDivisor(
        int[] shape,
        int axis,
        IReadOnlyList<int>? divisors)
    {
        if (divisors is null || axis >= divisors.Count || divisors[axis] <= 1) return;
        int divisor = divisors[axis];
        shape[axis] = checked(((shape[axis] + divisor - 1) / divisor) * divisor);
    }
}

/// <summary>A named set of mutually compatible input ports.</summary>
public sealed class InputContractVariant
{
    public InputContractVariant(string name, IReadOnlyList<LayerPort> ports)
    {
        Name = string.IsNullOrWhiteSpace(name) ? "default" : name;
        Ports = ports ?? throw new ArgumentNullException(nameof(ports));
    }

    public string Name { get; }
    public IReadOnlyList<LayerPort> Ports { get; }
}

/// <summary>
/// Canonical input schema for a layer or model. Generated tensor-port declarations, runtime
/// validation and test-data synthesis all consume this single representation.
/// </summary>
public sealed class InputContractManifest
{
    public InputContractManifest(
        string ownerName,
        IReadOnlyList<LayerPort> inputPorts,
        IReadOnlyList<LayerPort>? outputPorts = null,
        ModelInputShapeConstraint shapeConstraint = default)
    {
        OwnerName = string.IsNullOrWhiteSpace(ownerName) ? "component" : ownerName;
        InputPorts = inputPorts ?? throw new ArgumentNullException(nameof(inputPorts));
        OutputPorts = outputPorts ?? Array.Empty<LayerPort>();
        ShapeConstraint = shapeConstraint;

        Variants = InputPorts
            .GroupBy(port => port.Variant, StringComparer.Ordinal)
            .Select(group => new InputContractVariant(group.Key, group.ToArray()))
            .ToArray();
    }

    public string OwnerName { get; }
    public IReadOnlyList<LayerPort> InputPorts { get; }
    public IReadOnlyList<LayerPort> OutputPorts { get; }
    public IReadOnlyList<InputContractVariant> Variants { get; }
    public ModelInputShapeConstraint ShapeConstraint { get; }

    /// <summary>Selects the one declared variant satisfied by the supplied named ports.</summary>
    public string ResolveVariant(IEnumerable<string> suppliedPortNames)
    {
        var supplied = new HashSet<string>(suppliedPortNames, StringComparer.Ordinal);
        var matches = Variants.Where(variant =>
        {
            var visible = variant.Ports
                .Where(port => port.Source is TensorPortSource.External or TensorPortSource.Defaulted)
                .ToArray();
            return visible.Where(port => port.Required).All(port => supplied.Contains(port.Name))
                   && supplied.All(name => visible.Any(port =>
                       string.Equals(port.Name, name, StringComparison.Ordinal)));
        }).ToArray();

        if (matches.Length == 1) return matches[0].Name;
        if (matches.Length == 0)
            throw new InputContractViolationException(
                $"{OwnerName} inputs [{string.Join(", ", supplied)}] do not match any declared "
                + $"variant. Available variants: {string.Join(", ", Variants.Select(item => item.Name))}.");
        throw new InputContractViolationException(
            $"{OwnerName} inputs match more than one contract variant ({string.Join(", ", matches.Select(item => item.Name))}). "
            + "Make the alternative signatures structurally distinct.");
    }

    /// <summary>Binds the default variant to the caller's concrete primary-input shape.</summary>
    public BoundInputContract Bind(
        int[] primaryInputShape,
        LayerInputDomain? resolvedPrimaryDomain = null,
        string variant = "default")
    {
        if (primaryInputShape is null) throw new ArgumentNullException(nameof(primaryInputShape));

        var selected = Variants.FirstOrDefault(item =>
            string.Equals(item.Name, variant, StringComparison.Ordinal));
        if (selected is null)
        {
            return BoundInputContract.Invalid(
                this,
                variant,
                $"Input variant '{variant}' does not exist. Available variants: "
                + string.Join(", ", Variants.Select(item => item.Name)) + ".");
        }

        var primary = selected.Ports.FirstOrDefault(port => port.Source == TensorPortSource.External);
        if (primary is null)
            return BoundInputContract.Invalid(this, variant,
                "the selected input variant has no external input port");

        var shapes = new Dictionary<string, int[]>(StringComparer.Ordinal)
        {
            [primary.Name] = (int[])primaryInputShape.Clone()
        };
        return BindCore(selected, shapes, resolvedPrimaryDomain);
    }

    /// <summary>Binds every supplied named input to its own concrete shape.</summary>
    public BoundInputContract Bind(
        IReadOnlyDictionary<string, int[]> inputShapes,
        string variant = "default")
    {
        if (inputShapes is null) throw new ArgumentNullException(nameof(inputShapes));

        var selected = Variants.FirstOrDefault(item =>
            string.Equals(item.Name, variant, StringComparison.Ordinal));
        if (selected is null)
        {
            return BoundInputContract.Invalid(
                this,
                variant,
                $"Input variant '{variant}' does not exist. Available variants: "
                + string.Join(", ", Variants.Select(item => item.Name)) + ".");
        }

        return BindCore(selected, inputShapes, resolvedPrimaryDomain: null);
    }

    private BoundInputContract BindCore(
        InputContractVariant selected,
        IReadOnlyDictionary<string, int[]> inputShapes,
        LayerInputDomain? resolvedPrimaryDomain)
    {
        var primary = selected.Ports.FirstOrDefault(port => port.Source == TensorPortSource.External);
        if (primary is null)
            return BoundInputContract.Invalid(this, selected.Name,
                "the selected input variant has no external input port");

        int[] primaryInputShape = inputShapes.TryGetValue(primary.Name, out var suppliedPrimaryShape)
            ? (int[])suppliedPrimaryShape.Clone()
            : primary.Shape.ToArray();

        var reasons = new List<string>();
        ValidateModelShape(primaryInputShape, ShapeConstraint, reasons);
        bool hasInvalidInput = primaryInputShape.Length == 0
            || primaryInputShape.Any(axis => axis <= 0);
        bool hasDeferredDeclaration = false;

        var seenStableIds = new HashSet<string>(StringComparer.Ordinal);
        var boundPorts = new List<LayerPort>(selected.Ports.Count);
        bool primaryBound = false;
        foreach (var port in selected.Ports)
        {
            if (!seenStableIds.Add(port.StableId))
                reasons.Add($"stable port id '{port.StableId}' is declared more than once");

            bool isPrimary = !primaryBound && port.Source == TensorPortSource.External;
            if (isPrimary) primaryBound = true;
            var domain = isPrimary && resolvedPrimaryDomain.HasValue
                ? resolvedPrimaryDomain.Value
                : port.ValueDomain;
            var shape = ResolvePortShape(port, selected.Ports, inputShapes, new HashSet<string>(StringComparer.Ordinal));

            if (port.Required && port.Source == TensorPortSource.External && !domain.IsResolved)
            {
                reasons.Add($"port '{port.Name}' is not ready: {domain}");
                hasDeferredDeclaration = true;
            }
            if (port.Required && port.Source == TensorPortSource.External
                && (shape.Length == 0 || shape.Any(axis => axis <= 0)))
            {
                if (inputShapes.ContainsKey(port.Name))
                {
                    reasons.Add(
                        $"port '{port.Name}' has invalid shape [{string.Join(",", shape)}]; "
                        + "every caller-supplied axis must be positive");
                    hasInvalidInput = true;
                }
                else
                {
                    reasons.Add(
                        $"port '{port.Name}' is not ready: shape [{string.Join(",", shape)}] "
                        + "must be concrete before execution");
                    hasDeferredDeclaration = true;
                }
            }

            ValidatePortShape(shape, port.ShapeConstraint, port.Name, reasons);
            boundPorts.Add(new LayerPort(
                port.Name,
                shape,
                port.Required,
                domain,
                port.Role,
                port.StableId,
                port.Source,
                port.Variant,
                port.ShapeConstraint));
        }

        if (!primaryBound)
            reasons.Add("the selected input variant has no external input port");

        InputContractReadiness readiness = reasons.Count == 0
            ? InputContractReadiness.Ready
            : hasDeferredDeclaration && !hasInvalidInput
                ? InputContractReadiness.Deferred
                : InputContractReadiness.Invalid;

        return new BoundInputContract(this, selected.Name, boundPorts, readiness, reasons);
    }

    private static int[] ResolvePortShape(
        LayerPort port,
        IReadOnlyList<LayerPort> ports,
        IReadOnlyDictionary<string, int[]> suppliedShapes,
        ISet<string> resolving)
    {
        if (suppliedShapes.TryGetValue(port.Name, out var supplied))
            return (int[])supplied.Clone();

        string? relatedName = port.ShapeConstraint.SameShapeAs;
        if (string.IsNullOrWhiteSpace(relatedName))
            return port.Shape.ToArray();

        if (!resolving.Add(port.Name))
            return port.Shape.ToArray();

        var related = ports.FirstOrDefault(candidate =>
            string.Equals(candidate.Name, relatedName, StringComparison.Ordinal));
        int[] resolved = related is null
            ? port.Shape.ToArray()
            : ResolvePortShape(related, ports, suppliedShapes, resolving);
        resolving.Remove(port.Name);
        return resolved;
    }

    internal static void ValidateModelShape(
        int[] shape,
        ModelInputShapeConstraint constraint,
        ICollection<string> reasons)
    {
        if (shape.Length == 0 || shape.Any(axis => axis <= 0))
        {
            reasons.Add($"primary input shape [{string.Join(",", shape)}] contains an empty or non-positive axis");
            return;
        }

        if (constraint.ExactRank > 0 && shape.Length != constraint.ExactRank)
            reasons.Add($"primary input rank is {shape.Length}, but the contract requires exactly {constraint.ExactRank}");
        if (constraint.MinimumRank > 0 && shape.Length < constraint.MinimumRank)
            reasons.Add($"primary input rank is {shape.Length}, but the contract requires at least {constraint.MinimumRank}");
        if (constraint.MaximumRank > 0 && shape.Length > constraint.MaximumRank)
            reasons.Add($"primary input rank is {shape.Length}, but the contract permits at most {constraint.MaximumRank}");

        long elements = ElementCount(shape);
        if (constraint.MinimumElementCount > 0 && elements < constraint.MinimumElementCount)
            reasons.Add($"primary input has {elements} elements, but the contract requires at least {constraint.MinimumElementCount}");

        ValidateAxes(shape, constraint.MinimumAxisSizes, constraint.AxisDivisors, "primary input", reasons);
    }

    internal static void ValidatePortShape(
        int[] shape,
        PortShapeConstraint constraint,
        string portName,
        ICollection<string> reasons)
    {
        if (!constraint.IsConstrained) return;

        if (constraint.ExactRank > 0 && shape.Length != constraint.ExactRank)
            reasons.Add($"port '{portName}' has rank {shape.Length}; expected exactly {constraint.ExactRank}");
        if (constraint.MinimumRank > 0 && shape.Length < constraint.MinimumRank)
            reasons.Add($"port '{portName}' has rank {shape.Length}; expected at least {constraint.MinimumRank}");
        if (constraint.MaximumRank > 0 && shape.Length > constraint.MaximumRank)
            reasons.Add($"port '{portName}' has rank {shape.Length}; expected at most {constraint.MaximumRank}");
        if (constraint.MinimumElementCount > 0 && ElementCount(shape) < constraint.MinimumElementCount)
            reasons.Add($"port '{portName}' has fewer than {constraint.MinimumElementCount} elements");

        ValidateAxes(shape, constraint.MinimumAxisSizes, constraint.AxisDivisors, $"port '{portName}'", reasons);
    }

    private static void ValidateAxes(
        int[] shape,
        IReadOnlyList<int>? minima,
        IReadOnlyList<int>? divisors,
        string label,
        ICollection<string> reasons)
    {
        if (minima is not null)
        {
            for (int axis = 0; axis < minima.Count && axis < shape.Length; axis++)
                if (minima[axis] > 0 && shape[axis] < minima[axis])
                    reasons.Add($"{label} axis {axis} is {shape[axis]}; expected at least {minima[axis]}");
        }

        if (divisors is not null)
        {
            for (int axis = 0; axis < divisors.Count && axis < shape.Length; axis++)
                if (divisors[axis] > 1 && shape[axis] % divisors[axis] != 0)
                    reasons.Add($"{label} axis {axis} is {shape[axis]}; it must be divisible by {divisors[axis]}");
        }
    }

    internal static long ElementCount(IReadOnlyList<int> shape)
    {
        long result = 1;
        for (int i = 0; i < shape.Count; i++)
            result = checked(result * shape[i]);
        return result;
    }
}

/// <summary>A static input manifest after instance configuration and concrete shape are resolved.</summary>
public sealed class BoundInputContract
{
    internal BoundInputContract(
        InputContractManifest manifest,
        string variant,
        IReadOnlyList<LayerPort> inputPorts,
        InputContractReadiness readiness,
        IReadOnlyList<string> reasons)
    {
        Manifest = manifest;
        Variant = variant;
        InputPorts = inputPorts;
        Readiness = readiness;
        Reasons = reasons;
    }

    public InputContractManifest Manifest { get; }
    public string Variant { get; }
    public IReadOnlyList<LayerPort> InputPorts { get; }
    public InputContractReadiness Readiness { get; }
    public IReadOnlyList<string> Reasons { get; }

    public LayerPort PrimaryInput => InputPorts.First(port => port.Source == TensorPortSource.External);

    internal static BoundInputContract Invalid(
        InputContractManifest manifest,
        string variant,
        string reason) =>
        new(manifest, variant, Array.Empty<LayerPort>(), InputContractReadiness.Invalid, new[] { reason });

    /// <summary>Throws an actionable error instead of executing an unproved contract.</summary>
    public void RequireReady()
    {
        if (Readiness == InputContractReadiness.Ready) return;

        throw new InputContractBindingException(
            $"{Manifest.OwnerName} input contract '{Variant}' is {Readiness.ToString().ToLowerInvariant()}: "
            + string.Join("; ", Reasons) + ".");
    }

    /// <summary>Validates the primary public input against this bound contract.</summary>
    public void Validate<T>(Tensor<T> input)
    {
        RequireReadyForCaller();
        InputContractValidator.Validate(input, PrimaryInput, Manifest.OwnerName);
    }

    /// <summary>Validates named inputs, including required/defaulted ports and shape relations.</summary>
    public void Validate<T>(IReadOnlyDictionary<string, Tensor<T>> inputs)
    {
        RequireReadyForCaller();
        InputContractValidator.Validate(inputs, InputPorts, Manifest.OwnerName);
    }

    private void RequireReadyForCaller()
    {
        if (Readiness == InputContractReadiness.Invalid)
        {
            throw new InputContractViolationException(
                $"{Manifest.OwnerName} input contract '{Variant}' rejected the supplied input: "
                + string.Join("; ", Reasons) + ".",
                "input");
        }

        RequireReady();
    }
}

/// <summary>Thrown when configuration cannot produce an executable input contract.</summary>
public sealed class InputContractBindingException : InvalidOperationException
{
    public InputContractBindingException(string message) : base(message) { }
    public InputContractBindingException(string message, Exception innerException)
        : base(message, innerException) { }
}

/// <summary>Thrown when a supplied tensor violates a ready input contract.</summary>
public sealed class InputContractViolationException : ArgumentException
{
    public InputContractViolationException(string message, string? parameterName = null)
        : base(message, parameterName) { }
}

/// <summary>Central value/shape validation used by model and layer entry points.</summary>
public static class InputContractValidator
{
    public static void Validate<T>(Tensor<T> input, LayerPort port, string ownerName)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        var shapeReasons = new List<string>();
        InputContractManifest.ValidatePortShape(
            input.Shape.ToArray(), port.ShapeConstraint, port.Name, shapeReasons);
        if (shapeReasons.Count > 0)
            throw new InputContractViolationException(
                $"{ownerName}.{port.Name} violates its shape contract: {string.Join("; ", shapeReasons)}.",
                port.Name);

        ValidateValues(input, port.ValueDomain, ownerName, port.Name);
    }

    public static void Validate<T>(
        IReadOnlyDictionary<string, Tensor<T>> inputs,
        IReadOnlyList<LayerPort> ports,
        string ownerName)
    {
        if (inputs is null) throw new ArgumentNullException(nameof(inputs));

        foreach (var port in ports)
        {
            if (!inputs.TryGetValue(port.Name, out var tensor))
            {
                if (port.Required && port.Source == TensorPortSource.External)
                    throw new InputContractViolationException(
                        $"{ownerName} requires input port '{port.Name}'.", port.Name);
                continue;
            }

            Validate(tensor, port, ownerName);
            if (!string.IsNullOrWhiteSpace(port.ShapeConstraint.SameShapeAs)
                && inputs.TryGetValue(port.ShapeConstraint.SameShapeAs!, out var related)
                && !SameShape(tensor.Shape.ToArray(), related.Shape.ToArray()))
            {
                throw new InputContractViolationException(
                    $"{ownerName}.{port.Name} must have the same shape as "
                    + $"'{port.ShapeConstraint.SameShapeAs}', but received "
                    + $"[{string.Join(",", tensor.Shape)}] and [{string.Join(",", related.Shape)}].",
                    port.Name);
            }
        }
    }

    public static void ValidateValues<T>(
        Tensor<T> input,
        LayerInputDomain domain,
        string ownerName,
        string portName)
    {
        if (!domain.IsResolved)
            throw new InputContractBindingException(
                $"{ownerName}.{portName} cannot execute because its value domain is {domain}.");

        if (domain.Kind == LayerInputDomainKind.Custom)
        {
            InputDomainProviderRegistry.Require(domain.Detail)
                .Validate(input, ownerName, portName);
            return;
        }

        if (domain.Kind == LayerInputDomainKind.Continuous) return;
        EnsureIntegerRangeRepresentable<T>(domain, ownerName, portName);

        var numOps = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < input.Length; i++)
        {
            double value = numOps.ToDouble(input[i]);
            bool valid = domain.Kind switch
            {
                LayerInputDomainKind.IntegerIndices => !double.IsNaN(value) && !double.IsInfinity(value)
                    && value == Math.Truncate(value)
                    && value >= domain.MinInclusive
                    && value < domain.MaxExclusive,
                LayerInputDomainKind.BooleanMask => value is 0.0 or 1.0,
                LayerInputDomainKind.AdditiveMask => !double.IsNaN(value) && value <= 0.0,
                _ => true
            };

            if (valid) continue;

            string requirement = domain.Kind switch
            {
                LayerInputDomainKind.IntegerIndices =>
                    $"requires token indices in [{domain.MinInclusive}, {domain.MaxExclusive})",
                LayerInputDomainKind.BooleanMask => "requires a Boolean mask containing only 0 or 1",
                LayerInputDomainKind.AdditiveMask => "requires an additive mask containing zero or negative values",
                _ => $"requires {domain}"
            };
            throw new InputContractViolationException(
                $"{ownerName}.{portName} {requirement}, but element {i} is {value}.", portName);
        }
    }

    private static void EnsureIntegerRangeRepresentable<T>(
        LayerInputDomain domain,
        string ownerName,
        string portName)
    {
        if (domain.Kind != LayerInputDomainKind.IntegerIndices) return;
        if (typeof(T) == typeof(float) && domain.MaxExclusive > 16_777_216)
        {
            throw new InputContractBindingException(
                $"{ownerName}.{portName} has {domain.MaxExclusive} integer values, but float can "
                + "represent consecutive integers exactly only through 16,777,216. Use double or "
                + "a smaller cardinality so token identity cannot be rounded.");
        }
    }

    private static bool SameShape(IReadOnlyList<int> left, IReadOnlyList<int> right)
    {
        if (left.Count != right.Count) return false;
        for (int i = 0; i < left.Count; i++)
            if (left[i] != right[i]) return false;
        return true;
    }
}

/// <summary>
/// Contract-driven tensor creation for generated tests, examples and custom tooling. It is the
/// inverse of <see cref="InputContractValidator"/>: every generated value is legal by construction.
/// </summary>
public static class InputContractTensorFactory
{
    public static Tensor<T> CreateValid<T>(
        int[] shape,
        LayerInputDomain domain,
        Random random)
    {
        if (shape is null) throw new ArgumentNullException(nameof(shape));
        if (random is null) throw new ArgumentNullException(nameof(random));
        if (!domain.IsResolved)
            throw new InputContractBindingException(
                $"Cannot synthesize an input for {domain}; bind the component configuration first.");

        if (domain.Kind == LayerInputDomainKind.Custom)
        {
            var custom = InputDomainProviderRegistry.Require(domain.Detail)
                .CreateValid<T>(shape, random);
            InputContractValidator.ValidateValues(
                custom, domain, nameof(InputContractTensorFactory), "output");
            return custom;
        }

        var tensor = new Tensor<T>(shape);
        var numOps = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < tensor.Length; i++)
        {
            double value = domain.Kind switch
            {
                LayerInputDomainKind.IntegerIndices =>
                    random.Next(domain.MinInclusive, domain.MaxExclusive),
                LayerInputDomainKind.BooleanMask => random.Next(0, 2),
                LayerInputDomainKind.AdditiveMask => random.Next(0, 2) == 0 ? 0.0 : -10_000.0,
                _ => random.NextDouble()
            };
            tensor[i] = numOps.FromDouble(value);
        }

        InputContractValidator.ValidateValues(tensor, domain, nameof(InputContractTensorFactory), "output");
        return tensor;
    }

    public static Tensor<T> CreateValid<T>(BoundInputContract contract, Random random)
    {
        contract.RequireReady();
        var port = contract.PrimaryInput;
        return CreateValid<T>(port.Shape.ToArray(), port.ValueDomain, random);
    }

    /// <summary>Creates every caller-supplied port in a selected named-input variant.</summary>
    public static IReadOnlyDictionary<string, Tensor<T>> CreateValidInputs<T>(
        BoundInputContract contract,
        Random random,
        bool includeOptional = false)
    {
        if (contract is null) throw new ArgumentNullException(nameof(contract));
        if (random is null) throw new ArgumentNullException(nameof(random));
        contract.RequireReady();

        var inputs = new Dictionary<string, Tensor<T>>(StringComparer.Ordinal);
        foreach (var port in contract.InputPorts)
        {
            if (port.Source != TensorPortSource.External || !port.Required && !includeOptional)
                continue;
            inputs.Add(
                port.Name,
                CreateValid<T>(port.Shape.ToArray(), port.ValueDomain, random));
        }

        contract.Validate(inputs);
        return inputs;
    }

    public static Tensor<T> CreateNearby<T>(
        Tensor<T> input,
        LayerInputDomain domain,
        double epsilon = 1e-6)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (!domain.IsResolved)
            throw new InputContractBindingException($"Cannot mutate an input for {domain}.");

        if (domain.Kind == LayerInputDomainKind.Custom)
        {
            var custom = InputDomainProviderRegistry.Require(domain.Detail)
                .CreateNearby(input, epsilon);
            InputContractValidator.ValidateValues(
                custom, domain, nameof(InputContractTensorFactory), "output");
            return custom;
        }

        var nearby = new Tensor<T>(input.Shape.ToArray());
        var numOps = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < input.Length; i++) nearby[i] = input[i];
        if (nearby.Length == 0) return nearby;

        switch (domain.Kind)
        {
            case LayerInputDomainKind.IntegerIndices:
                int span = domain.MaxExclusive - domain.MinInclusive;
                if (span > 1)
                {
                    int current = (int)numOps.ToDouble(nearby[0]);
                    nearby[0] = numOps.FromDouble(
                        domain.MinInclusive + ((current - domain.MinInclusive + 1) % span));
                }
                break;
            case LayerInputDomainKind.BooleanMask:
                nearby[0] = numOps.FromDouble(numOps.ToDouble(nearby[0]) == 0.0 ? 1.0 : 0.0);
                break;
            case LayerInputDomainKind.AdditiveMask:
                nearby[0] = numOps.FromDouble(numOps.ToDouble(nearby[0]) == 0.0 ? -10_000.0 : 0.0);
                break;
            default:
                T delta = numOps.FromDouble(epsilon);
                for (int i = 0; i < nearby.Length; i++) nearby[i] = numOps.Add(nearby[i], delta);
                break;
        }

        InputContractValidator.ValidateValues(nearby, domain, nameof(InputContractTensorFactory), "output");
        return nearby;
    }

    public static Tensor<T> CreateInvalid<T>(
        int[] shape,
        LayerInputDomain domain)
    {
        if (!domain.IsResolved)
            throw new InputContractBindingException($"Cannot create a negative example for {domain}.");

        if (domain.Kind == LayerInputDomainKind.Custom)
            return InputDomainProviderRegistry.Require(domain.Detail).CreateInvalid<T>(shape);

        var tensor = new Tensor<T>(shape);
        var numOps = MathHelper.GetNumericOperations<T>();
        double invalid = domain.Kind switch
        {
            LayerInputDomainKind.IntegerIndices => domain.MaxExclusive,
            LayerInputDomainKind.BooleanMask => 0.5,
            LayerInputDomainKind.AdditiveMask => 1.0,
            _ => double.NaN
        };
        for (int i = 0; i < tensor.Length; i++) tensor[i] = numOps.FromDouble(invalid);
        return tensor;
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Interfaces;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// Checkpoint properties every optimizer must hold, discovered by reflection so a new optimizer is covered the moment
/// it exists rather than when someone remembers to add it to a list:
/// <list type="bullet">
/// <item><b>Exact round trip:</b> a trained optimizer serialized and restored into a fresh instance re-serializes to
/// the identical bytes.</item>
/// <item><b>Atomic restore:</b> a damaged payload either restores or throws, and when it throws, the target
/// optimizer is left byte-for-byte as it was. A half-applied restore silently resumes training from a mix of the
/// old and the new state.</item>
/// </list>
/// </summary>
public class OptimizerCheckpointAtomicityTests
{
    private readonly ITestOutputHelper _output;
    public OptimizerCheckpointAtomicityTests(ITestOutputHelper output) => _output = output;

    public static IEnumerable<object[]> AllOptimizers() =>
        OptimizerTypes().Select(t => new object[] { t.Name.Split('`')[0] });

    private static IEnumerable<Type> OptimizerTypes() =>
        typeof(OptimizerBase<,,>).Assembly.GetTypes()
            .Where(t => t.IsClass && !t.IsAbstract && t.IsGenericTypeDefinition
                && t.GetGenericArguments().Length == 3
                && DerivesFromGeneric(t, typeof(OptimizerBase<,,>)))
            .OrderBy(t => t.FullName, StringComparer.Ordinal);

    private static bool DerivesFromGeneric(Type type, Type generic)
    {
        for (var current = type.BaseType; current is not null; current = current.BaseType)
            if (current.IsGenericType && current.GetGenericTypeDefinition() == generic) return true;
        return false;
    }

    /// <summary>
    /// Builds the optimizer with default options, closed over the tensor types the tape path uses (or the
    /// matrix/vector types when an optimizer constrains its inputs), and gives it real state: gradient-based
    /// optimizers take tape steps so they carry moments and a step count.
    /// </summary>
    private static object Create(string name, int steps)
    {
        var definition = OptimizerTypes().Single(t => t.Name.Split('`')[0] == name);
        Type closed;
        try { closed = definition.MakeGenericType(typeof(double), typeof(Tensor<double>), typeof(Tensor<double>)); }
        catch (ArgumentException) { closed = definition.MakeGenericType(typeof(double), typeof(Matrix<double>), typeof(Vector<double>)); }

        var ctor = closed.GetConstructors()
            .OrderBy(c => c.GetParameters().Length)
            .FirstOrDefault(c => c.GetParameters().All(p => !p.ParameterType.IsValueType || p.HasDefaultValue))
            ?? throw new InvalidOperationException($"{name} has no constructor callable with defaults.");
        var args = ctor.GetParameters().Select(p => p.HasDefaultValue ? p.DefaultValue : null).ToArray();
        var optimizer = ctor.Invoke(args);

        if (optimizer is IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>> gradient && steps > 0)
        {
            var parameters = new[] { Filled(new[] { 3, 2 }, 0.5), Filled(new[] { 4 }, -0.25) };
            for (int s = 0; s < steps; s++)
            {
                var grads = parameters.ToDictionary(p => p, p => Filled(p.Shape.ToArray(), 0.1 * (s + 1)),
                    global::AiDotNet.Helpers.TensorReferenceComparer<Tensor<double>>.Instance);
                try { gradient.Step(new TapeStepContext<double>(parameters, grads, 0.0)); }
                catch (NotSupportedException)
                {
                    // Second-order methods (Newton, Levenberg-Marquardt) need a context that can re-evaluate the loss;
                    // they are checked in their untrained state.
                    break;
                }
            }
        }
        return optimizer;
    }

    private static Tensor<double> Filled(int[] shape, double start)
    {
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = start + 0.01 * i;
        return t;
    }

    private static byte[] Serialize(object optimizer) =>
        (byte[])optimizer.GetType().GetMethod("Serialize", Type.EmptyTypes)!.Invoke(optimizer, null)!;

    private static void Deserialize(object optimizer, byte[] data)
    {
        try { optimizer.GetType().GetMethod("Deserialize", new[] { typeof(byte[]) })!.Invoke(optimizer, new object[] { data }); }
        catch (TargetInvocationException ex) when (ex.InnerException is not null)
        {
            System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(ex.InnerException).Throw();
        }
    }

    [Fact]
    public void Every_optimizer_is_discovered()
    {
        var names = OptimizerTypes().Select(t => t.Name.Split('`')[0]).ToList();
        _output.WriteLine($"{names.Count} optimizers: {string.Join(", ", names)}");
        Assert.True(names.Count >= 40, $"discovery found only {names.Count} optimizers");
    }

    /// <summary>
    /// The rollback path itself. A subclass hook that reads-and-applies (DeserializeAdditionalData) runs at commit, after
    /// the declared state and options were installed; when it throws, the snapshot must put all of that back. The fuzz
    /// cases above never reach a failing commit, so this forces one.
    /// </summary>
    [Fact]
    public void A_commit_that_fails_is_rolled_back()
    {
        var source = new RestoreFailingAdam();
        var sourceParameters = new[] { Filled(new[] { 3, 2 }, 0.5) };
        for (int s = 0; s < 3; s++)
        {
            var grads = new Dictionary<Tensor<double>, Tensor<double>>(
                global::AiDotNet.Helpers.TensorReferenceComparer<Tensor<double>>.Instance)
            { [sourceParameters[0]] = Filled(new[] { 3, 2 }, 0.3 * (s + 1)) };
            source.Step(new TapeStepContext<double>(sourceParameters, grads, 0.0));
        }
        byte[] payload = source.Serialize();

        var target = new RestoreFailingAdam();
        var targetParameters = new[] { Filled(new[] { 3, 2 }, -0.5) };
        var targetGrads = new Dictionary<Tensor<double>, Tensor<double>>(
            global::AiDotNet.Helpers.TensorReferenceComparer<Tensor<double>>.Instance)
        { [targetParameters[0]] = Filled(new[] { 3, 2 }, -0.2) };
        target.Step(new TapeStepContext<double>(targetParameters, targetGrads, 0.0));
        byte[] before = target.Serialize();
        Assert.NotEqual(payload, before);   // the two states really differ, so a partial install would show

        // The incoming payload fails in the hook; replaying the snapshot through the same hook succeeds.
        target.FailuresRemaining = 1;
        var failure = Assert.Throws<InvalidOperationException>(() => target.Deserialize(payload));
        Assert.Equal(RestoreFailingAdam.FailureMessage, failure.Message);   // the ORIGINAL failure is what surfaces

        Assert.Equal(before, target.Serialize());
    }

    /// <summary>
    /// When the rollback fails too, the optimizer's state is unknown, and the restore must say so rather than surface
    /// the first failure as if the optimizer were intact.
    /// </summary>
    [Fact]
    public void A_failed_rollback_reports_undefined_state()
    {
        var source = new RestoreFailingAdam();
        byte[] payload = source.Serialize();
        var target = new RestoreFailingAdam { FailuresRemaining = 2 };

        var failure = Assert.Throws<InvalidOperationException>(() => target.Deserialize(payload));
        Assert.Contains("state is undefined", failure.Message);
        var both = Assert.IsType<AggregateException>(failure.InnerException);
        Assert.Equal(2, both.InnerExceptions.Count);
        Assert.All(both.InnerExceptions, e => Assert.Equal(RestoreFailingAdam.FailureMessage, e.Message));
    }

    /// <summary>Adam with a DeserializeAdditionalData hook that fails on demand, after the rest has been installed.</summary>
    private sealed class RestoreFailingAdam : AdamOptimizer<double, Tensor<double>, Tensor<double>>
    {
        public const string FailureMessage = "simulated failure in a subclass restore hook";
        public RestoreFailingAdam() : base(null) { }
        public int FailuresRemaining { get; set; }

        protected override void DeserializeAdditionalData(System.IO.BinaryReader reader)
        {
            if (FailuresRemaining <= 0) return;
            FailuresRemaining--;
            throw new InvalidOperationException(FailureMessage);
        }
    }
    /// <summary>
    /// Which optimizers need a rollback snapshot on restore. The snapshot doubles the memory held by optimizer state at
    /// load, so it must be the exception: only commits that restore through another object may need it. Any optimizer
    /// newly appearing here is a regression in load cost and has to be justified in <see cref="SnapshotExpected"/>.
    /// </summary>
    [Fact]
    public void Restores_stay_snapshot_free_unless_a_commit_can_fail()
    {
        var needsSnapshot = new List<string>();
        foreach (var type in OptimizerTypes())
        {
            string name = type.Name.Split('`')[0];
            var optimizer = Create(name, steps: 2);
            byte[] payload = Serialize(optimizer);
            // The optimizer's OWN staging entry point: the most-derived byte[] -> StagedRestore method (Adam8Bit wraps
            // the base payload in its own format, so the base stager cannot read it directly).
            MethodInfo? stage = null;
            for (var t = optimizer.GetType(); t is not null && stage is null; t = t.BaseType)
                stage = t.GetMethods(BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.DeclaredOnly)
                    .FirstOrDefault(m => m.ReturnType.Name == "StagedRestore"
                        && m.GetParameters().Select(p => p.ParameterType).SequenceEqual(new[] { typeof(byte[]) }));
            if (stage is null) throw new InvalidOperationException($"{name} has no staging entry point");
            var staged = stage.Invoke(optimizer, new object[] { payload })
                ?? throw new InvalidOperationException("StageDeserialize returned null");
            bool fallible = (bool)(staged.GetType().GetProperty("HasFallibleCommit")?.GetValue(staged)
                ?? throw new InvalidOperationException("StagedRestore.HasFallibleCommit not found"));
            if (fallible) needsSnapshot.Add(name);
        }

        _output.WriteLine($"need a rollback snapshot: {(needsSnapshot.Count == 0 ? "none" : string.Join(", ", needsSnapshot))}");
        var unexpected = needsSnapshot.Except(SnapshotExpected).ToList();
        Assert.True(unexpected.Count == 0, $"these optimizers now need a rollback snapshot on load: {string.Join(", ", unexpected)}");
    }

    /// <summary>Optimizers whose restore legitimately includes a commit that can fail. Each needs a stated reason.</summary>
    private static readonly string[] SnapshotExpected =
    {
        // Its Gaussian-process surrogate is a declared CHILD model restored through the child's own Deserialize, which
        // can fail after other state is installed. A metaheuristic's state is small, so the snapshot is cheap here.
        "BayesianOptimizer",
    };
    [Theory]
    [MemberData(nameof(AllOptimizers))]
    public void Round_trip_is_byte_exact(string name)
    {
        var trained = Create(name, steps: 3);
        byte[] payload = Serialize(trained);

        var restored = Create(name, steps: 0);
        Deserialize(restored, payload);

        Assert.Equal(payload, Serialize(restored));
    }

    [Theory]
    [MemberData(nameof(AllOptimizers))]
    public void A_restore_that_throws_changes_nothing(string name)
    {
        byte[] source = Serialize(Create(name, steps: 2));
        var target = Create(name, steps: 4);   // its own, different state
        byte[] before = Serialize(target);

        // Truncations across the whole payload, plus single-byte corruptions: every cut and flip either restores or
        // throws, and a throw must leave the target untouched. Positions are deterministic so failures reproduce.
        var damaged = new List<(string What, byte[] Data)>();
        int stride = Math.Max(1, source.Length / 64);
        for (int cut = 1; cut < source.Length; cut += stride)
            damaged.Add(($"truncated at {cut}", source.Take(cut).ToArray()));
        for (int at = 0; at < source.Length; at += stride)
        {
            var flipped = (byte[])source.Clone();
            flipped[at] ^= 0xFF;
            damaged.Add(($"byte {at} flipped", flipped));
        }

        int threw = 0;
        var violations = new List<string>();
        foreach (var (what, data) in damaged)
        {
            var probe = Create(name, steps: 4);
            byte[] probeBefore = Serialize(probe);
            try
            {
                Deserialize(probe, data);
                continue;   // restored: a valid (if different) state is acceptable
            }
            catch (Exception)
            {
                threw++;
            }
            if (!Serialize(probe).SequenceEqual(probeBefore))
                violations.Add(what);
        }

        _output.WriteLine($"{name}: {damaged.Count} damaged payloads, {threw} rejected, {violations.Count} left partial state");
        Assert.True(threw > 0, "no damaged payload was rejected, so the atomicity check never ran");
        Assert.True(violations.Count == 0,
            $"{name}: a failed restore changed the optimizer ({violations.Count} cases), e.g. {string.Join("; ", violations.Take(5))}");
        Assert.Equal(before, Serialize(target));
    }
}
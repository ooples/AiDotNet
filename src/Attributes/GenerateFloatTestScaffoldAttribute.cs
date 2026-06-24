namespace AiDotNet.Attributes;

/// <summary>
/// Marks a model whose auto-generated model-family test scaffold should run in <c>float</c>
/// (single precision) rather than the default <c>double</c>.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why:</b> a handful of large models (#1624 inventory) have a fast single FORWARD but their
/// training / clone invariant tests OOM or time out on the 16 GB CI runner, because <c>double</c>
/// roughly doubles the training footprint (gradients + optimizer state + activations). Running
/// their generated scaffold in <c>float</c> halves that footprint and per-op cost — removing the
/// OOM/timeout — while keeping every code path and the self-relative training invariants intact.
/// </para>
/// <para>
/// <b>This is the going-forward source of truth.</b> When you add a new large/training-perf-bound
/// model, apply this attribute to its class instead of editing the hard-coded
/// <c>TestScaffoldGenerator.Fp32TestClassNames</c> list — the generator reads this attribute during
/// model discovery and floats the scaffold automatically, so the float roster cannot silently fall
/// out of sync with the model roster.
/// </para>
/// <para>
/// <b>Usage:</b>
/// <code>
/// [GenerateFloatTestScaffold]
/// public class MyHugeBackbone&lt;T&gt; : NeuralNetworkBase&lt;T&gt; { }
/// </code>
/// </para>
/// </remarks>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = false)]
public sealed class GenerateFloatTestScaffoldAttribute : Attribute
{
}

using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Guardrails;

/// <summary>
/// Chains multiple guardrails into a single composite guardrail that runs all child guardrails
/// in sequence and aggregates their findings.
/// </summary>
/// <remarks>
/// <para>
/// CompositeGuardrail acts as an orchestrator: it contains a list of child guardrails and
/// executes them in order. If any child guardrail returns a finding with a blocking action,
/// evaluation can optionally short-circuit (fail-fast). The direction of the composite
/// guardrail is determined by the combination of child directions.
/// </para>
/// <para>
/// <b>For Beginners:</b> Sometimes you want to apply many safety checks at once. Instead of
/// calling each guardrail separately, CompositeGuardrail bundles them together. Think of it
/// as a checklist — the text must pass all the checks before it's considered safe.
/// </para>
/// <para>
/// <b>References:</b>
/// - Guardrails AI: Validator chaining (2024)
/// - NeMo Guardrails: Multi-rail pipelines (NVIDIA, 2024)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class CompositeGuardrail<T> : IGuardrail<T>
{
    private readonly List<IGuardrail<T>> _guardrails;
    private readonly bool _failFast;

    /// <inheritdoc />
    public string ModuleName => "CompositeGuardrail";

    /// <inheritdoc />
    public bool IsReady => _guardrails.Count > 0 && _guardrails.TrueForAll(g => g.IsReady);

    /// <inheritdoc />
    public GuardrailDirection Direction { get; }

    /// <summary>
    /// Gets the number of child guardrails in this composite.
    /// </summary>
    public int Count => _guardrails.Count;

    /// <summary>
    /// Initializes a new composite guardrail.
    /// </summary>
    /// <param name="guardrails">The child guardrails to chain. Evaluated in order.</param>
    /// <param name="failFast">
    /// If true, stops evaluation after the first guardrail returns a Block action.
    /// If false (default), all guardrails are always evaluated.
    /// </param>
    /// <param name="direction">
    /// The direction this composite applies to. Default: Both.
    /// If null, the direction is inferred from child guardrails.
    /// </param>
    public CompositeGuardrail(
        IReadOnlyList<IGuardrail<T>> guardrails,
        bool failFast = false,
        GuardrailDirection? direction = null)
    {
        _guardrails = new List<IGuardrail<T>>(guardrails ?? Array.Empty<IGuardrail<T>>());
        _failFast = failFast;
        Direction = direction ?? InferDirection();
    }

    /// <summary>
    /// Adds a guardrail to the end of the chain.
    /// </summary>
    /// <param name="guardrail">The guardrail to add.</param>
    public void Add(IGuardrail<T> guardrail)
    {
        _guardrails.Add(guardrail);
    }

    /// <inheritdoc />
    public IReadOnlyList<SafetyFinding> EvaluateText(string text)
    {
        var findings = new List<SafetyFinding>();

        foreach (var guardrail in _guardrails)
        {
            var childFindings = guardrail.EvaluateText(text);
            findings.AddRange(childFindings);

            if (_failFast && HasBlockAction(childFindings))
            {
                break;
            }
        }

        return findings;
    }

    /// <inheritdoc />
    public IReadOnlyList<SafetyFinding> Evaluate(Vector<T> content)
    {
        var findings = new List<SafetyFinding>();

        foreach (var guardrail in _guardrails)
        {
            var childFindings = guardrail.Evaluate(content);
            findings.AddRange(childFindings);

            if (_failFast && HasBlockAction(childFindings))
            {
                break;
            }
        }

        return findings;
    }

    private GuardrailDirection InferDirection()
    {
        bool hasInput = false, hasOutput = false;
        foreach (var g in _guardrails)
        {
            switch (g.Direction)
            {
                case GuardrailDirection.Input:
                    hasInput = true;
                    break;
                case GuardrailDirection.Output:
                    hasOutput = true;
                    break;
                case GuardrailDirection.Both:
                    hasInput = true;
                    hasOutput = true;
                    break;
            }
        }

        if (hasInput && hasOutput) return GuardrailDirection.Both;
        if (hasInput) return GuardrailDirection.Input;
        if (hasOutput) return GuardrailDirection.Output;
        return GuardrailDirection.Both;
    }

    private static bool HasBlockAction(IReadOnlyList<SafetyFinding> findings)
    {
        foreach (var finding in findings)
        {
            if (finding.RecommendedAction == SafetyAction.Block)
            {
                return true;
            }
        }
        return false;
    }
}

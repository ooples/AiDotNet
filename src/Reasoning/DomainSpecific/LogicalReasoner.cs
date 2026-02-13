using AiDotNet.Interfaces;
using AiDotNet.Reasoning.Components;
using AiDotNet.Reasoning.Models;
using AiDotNet.Reasoning.Strategies;
using AiDotNet.Reasoning.Verification;
using AiDotNet.Validation;

namespace AiDotNet.Reasoning.DomainSpecific;

/// <summary>
/// Specialized reasoner for formal logic and logical reasoning problems.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This reasoner solves logic puzzles and formal reasoning problems.
///
/// **What is LogicalReasoner?**
/// A specialized system for solving problems requiring formal logic, including:
/// - Propositional logic (AND, OR, NOT, IF-THEN)
/// - Predicate logic (All, Some, None)
/// - Logical inference and deduction
/// - Puzzle solving with constraints
///
/// **Example 1 - Propositional Logic:**
/// ```
/// Problem: "If it rains, the ground gets wet. It rained. What can we conclude?"
///
/// Logical reasoning:
/// Premise 1: Rain → Wet ground
/// Premise 2: Rain (happened)
/// Rule: Modus ponens (If P→Q and P, then Q)
/// Conclusion: The ground is wet
/// ```
///
/// **Example 2 - Predicate Logic:**
/// ```
/// Problem: "All cats are mammals. All mammals are animals. Therefore?"
///
/// Logical reasoning:
/// Premise 1: ∀x (Cat(x) → Mammal(x))
/// Premise 2: ∀x (Mammal(x) → Animal(x))
/// Rule: Transitivity (If A→B and B→C, then A→C)
/// Conclusion: ∀x (Cat(x) → Animal(x)) - All cats are animals
/// ```
///
/// **Example 3 - Logic Puzzle:**
/// ```
/// Problem: "There are 3 people: Alice, Bob, and Carol. One always tells truth,
/// one always lies, one alternates. Alice says 'Bob lies.' Bob says 'Carol alternates.'
/// Carol says 'Alice tells truth.' Who is who?"
///
/// Logical reasoning:
/// 1. Assume Alice = truth-teller
///    - Bob lies (from Alice)
///    - If Bob lies, "Carol alternates" is false
///    - So Carol is truth-teller or liar
///    - But Alice is truth-teller, so Carol must be liar
///    - Carol says "Alice tells truth" (true statement from liar) - CONTRADICTION
///
/// 2. Assume Alice = liar
///    - Bob doesn't lie (from Alice lying)
///    - So Bob is truth-teller or alternator
///    - Bob says "Carol alternates"
///    - If Bob is truth-teller, Carol alternates, Alice lies ✓
///    - Carol says "Alice tells truth" (false from alternator in lying mode) ✓
///    - Consistent!
///
/// Conclusion: Alice = liar, Bob = truth-teller, Carol = alternator
/// ```
///
/// **Example 4 - Contrapositive:**
/// ```
/// Problem: "If you study, you pass. You didn't pass. What can we conclude?"
///
/// Logical reasoning:
/// Original: Study → Pass
/// Contrapositive: ¬Pass → ¬Study
/// Given: ¬Pass (didn't pass)
/// Conclusion: ¬Study (didn't study)
/// ```
///
/// **Logical inference rules:**
/// - Modus ponens: P→Q, P ⊢ Q
/// - Modus tollens: P→Q, ¬Q ⊢ ¬P
/// - Hypothetical syllogism: P→Q, Q→R ⊢ P→R
/// - Disjunctive syllogism: P∨Q, ¬P ⊢ Q
/// - Conjunction: P, Q ⊢ P∧Q
/// - Simplification: P∧Q ⊢ P
/// - Addition: P ⊢ P∨Q
///
/// **Types of problems:**
/// - Deductive reasoning (must be true)
/// - Inductive reasoning (probably true)
/// - Abductive reasoning (best explanation)
/// - Puzzle solving (knights and knaves, etc.)
/// - Argument validity assessment
/// - Fallacy detection
///
/// **Common logical fallacies to avoid:**
/// - Affirming the consequent: P→Q, Q ⊬ P
/// - Denying the antecedent: P→Q, ¬P ⊬ ¬Q
/// - False dilemma: Assuming only two options exist
/// - Circular reasoning: Using conclusion as premise
/// - Ad hominem: Attacking person, not argument
///
/// **Usage example:**
/// ```csharp
/// var reasoner = new LogicalReasoner<double>(chatModel);
///
/// // Deductive reasoning
/// var result = await reasoner.SolveAsync(
///     "All programmers are logical. John is a programmer. What can we conclude?",
///     logicType: "deductive"
/// );
///
/// // Logic puzzle
/// result = await reasoner.SolvePuzzleAsync(
///     "In a room are 3 switches and 3 light bulbs in another room.
///      You can flip switches but only enter the bulb room once.
///      How do you determine which switch controls which bulb?"
/// );
/// ```
///
/// **Key features:**
/// - Formal logic notation support
/// - Step-by-step inference
/// - Assumption tracking
/// - Contradiction detection
/// - Proof verification
/// - Multiple solution paths (Tree-of-Thoughts)
/// </para>
/// </remarks>
public class LogicalReasoner<T>
{
    private readonly IChatModel<T> _chatModel;
    private readonly ChainOfThoughtStrategy<T> _cotStrategy;
    private readonly TreeOfThoughtsStrategy<T> _totStrategy;
    private readonly ContradictionDetector<T>? _contradictionDetector;
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the <see cref="LogicalReasoner{T}"/> class.
    /// </summary>
    /// <param name="chatModel">The chat model to use for reasoning.</param>
    /// <param name="enableContradictionDetection">Whether to enable contradiction detection.</param>
    public LogicalReasoner(
        IChatModel<T> chatModel,
        bool enableContradictionDetection = false)
    {
        Guard.NotNull(chatModel);
        _chatModel = chatModel;
        _contradictionDetector = enableContradictionDetection ? new ContradictionDetector<T>(chatModel) : null;
        _numOps = MathHelper.GetNumericOperations<T>();

        _cotStrategy = new ChainOfThoughtStrategy<T>(chatModel);
        _totStrategy = new TreeOfThoughtsStrategy<T>(chatModel);
    }

    /// <summary>
    /// Solves a logical reasoning problem.
    /// </summary>
    /// <param name="problem">The logical problem to solve.</param>
    /// <param name="logicType">Type of logic (deductive, inductive, abductive).</param>
    /// <param name="config">Reasoning configuration.</param>
    /// <param name="useTreeSearch">Whether to explore multiple reasoning paths.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Reasoning result with solution.</returns>
    public async Task<ReasoningResult<T>> SolveAsync(
        string problem,
        string logicType = "deductive",
        ReasoningConfig? config = null,
        bool useTreeSearch = false,
        CancellationToken cancellationToken = default)
    {
        config ??= new ReasoningConfig();

        // Build logic-specific prompt
        string enhancedProblem = BuildLogicalPrompt(problem, logicType);

        // Choose strategy
        IReasoningStrategy<T> strategy = useTreeSearch
            ? _totStrategy
            : _cotStrategy;

        // Solve
        var result = await strategy.ReasonAsync(enhancedProblem, config, cancellationToken);

        // Check for contradictions
        if (result.Success && result.ReasoningChain != null && _contradictionDetector != null)
        {
            var contradictions = await _contradictionDetector.DetectContradictionsAsync(
                result.ReasoningChain,
                cancellationToken
            );

            if (contradictions.Count > 0)
            {
                result.Metadata["contradictions"] = string.Join("; ", contradictions);
                result.Metadata["warning"] = "Logical contradictions detected";
            }
        }

        return result;
    }

    /// <summary>
    /// Solves a logic puzzle with constraints.
    /// </summary>
    public async Task<ReasoningResult<T>> SolvePuzzleAsync(
        string puzzle,
        ReasoningConfig? config = null,
        CancellationToken cancellationToken = default)
    {
        config ??= new ReasoningConfig();
        config.ExplorationDepth = Math.Max(config.ExplorationDepth, 5); // Deeper for puzzles

        string prompt = $@"Logic Puzzle:

{puzzle}

Solve this puzzle systematically:

1. Identify all constraints and rules
2. List what we know for certain
3. Consider possibilities (use process of elimination)
4. Test assumptions
5. Look for contradictions
6. Deduce the solution step by step

Show your reasoning clearly at each step:";

        // Use Tree-of-Thoughts for puzzle solving (explore multiple paths)
        return await _totStrategy.ReasonAsync(prompt, config, cancellationToken);
    }

    /// <summary>
    /// Evaluates the validity of a logical argument.
    /// </summary>
    public async Task<ReasoningResult<T>> EvaluateArgumentAsync(
        string argument,
        List<string> premises,
        string conclusion,
        ReasoningConfig? config = null,
        CancellationToken cancellationToken = default)
    {
        config ??= new ReasoningConfig();

        string premisesStr = string.Join("\n", premises.Select((p, i) => $"Premise {i + 1}: {p}"));

        string prompt = $@"Argument Evaluation:

Argument:
{argument}

{premisesStr}
Conclusion: {conclusion}

Evaluate this argument:

1. Is the argument structure valid?
2. Do the premises support the conclusion?
3. Are there any logical fallacies?
4. What inference rules are used?
5. Is the argument sound (valid + true premises)?

Provide a detailed logical analysis:";

        return await _cotStrategy.ReasonAsync(prompt, config, cancellationToken);
    }

    /// <summary>
    /// Identifies logical fallacies in reasoning.
    /// </summary>
    public async Task<ReasoningResult<T>> DetectFallaciesAsync(
        string argument,
        ReasoningConfig? config = null,
        CancellationToken cancellationToken = default)
    {
        config ??= new ReasoningConfig();

        string prompt = $@"Fallacy Detection:

Argument: {argument}

Identify any logical fallacies in this argument.

Common fallacies to check:
- Ad hominem (attacking person, not argument)
- Straw man (misrepresenting the argument)
- False dilemma (only two options when more exist)
- Slippery slope (assuming chain reaction without evidence)
- Circular reasoning (conclusion in premises)
- Appeal to authority (inappropriate authority)
- Appeal to emotion (using emotions vs. logic)
- Hasty generalization (insufficient evidence)
- Post hoc (false causation)
- Red herring (irrelevant point)

For each fallacy found:
1. Name the fallacy
2. Quote the problematic part
3. Explain why it's fallacious
4. Suggest how to fix it

Provide your analysis:";

        return await _cotStrategy.ReasonAsync(prompt, config, cancellationToken);
    }

    /// <summary>
    /// Constructs a formal proof for a logical statement.
    /// </summary>
    public async Task<ReasoningResult<T>> ProveAsync(
        string statement,
        List<string>? axioms = null,
        ReasoningConfig? config = null,
        CancellationToken cancellationToken = default)
    {
        config ??= new ReasoningConfig();

        string axiomsStr = axioms != null && axioms.Count > 0
            ? "Given axioms:\n" + string.Join("\n", axioms.Select((a, i) => $"{i + 1}. {a}"))
            : "Use standard logical axioms.";

        string prompt = $@"Formal Proof:

Statement to prove: {statement}

{axiomsStr}

Construct a formal proof:

1. State what you're trying to prove
2. List your assumptions/axioms
3. Apply inference rules step by step
4. Justify each step with the rule used
5. Reach the conclusion

Provide a rigorous proof:";

        return await _cotStrategy.ReasonAsync(prompt, config, cancellationToken);
    }

    /// <summary>
    /// Determines the logical relationship between statements.
    /// </summary>
    public async Task<ReasoningResult<T>> AnalyzeRelationshipAsync(
        string statement1,
        string statement2,
        ReasoningConfig? config = null,
        CancellationToken cancellationToken = default)
    {
        config ??= new ReasoningConfig();

        string prompt = $@"Logical Relationship Analysis:

Statement 1: {statement1}
Statement 2: {statement2}

Determine the logical relationship:

Possible relationships:
- Equivalent (logically the same)
- Implies (one follows from the other)
- Contradictory (cannot both be true)
- Contrary (cannot both be true, but both can be false)
- Subcontrary (cannot both be false, but both can be true)
- Independent (no logical connection)

Analyze:
1. What is the relationship?
2. Provide logical justification
3. Give examples if helpful
4. Consider all logical cases

Provide your analysis:";

        return await _cotStrategy.ReasonAsync(prompt, config, cancellationToken);
    }

    private string BuildLogicalPrompt(string problem, string logicType)
    {
        string typeGuidance = logicType.ToLowerInvariant() switch
        {
            "deductive" => @"
Use deductive reasoning:
- Start with general premises
- Apply logical rules
- Reach certain conclusions
- Ensure conclusions necessarily follow from premises",

            "inductive" => @"
Use inductive reasoning:
- Observe specific cases
- Identify patterns
- Generalize to broader principle
- Note: Conclusion is probable, not certain",

            "abductive" => @"
Use abductive reasoning:
- Consider the observed facts
- Generate possible explanations
- Choose the most likely explanation
- Apply Occam's Razor (simplest explanation)",

            _ => @"
Use rigorous logical reasoning:
- State premises clearly
- Apply valid inference rules
- Avoid logical fallacies
- Reach justified conclusions"
        };

        return $@"Logical Reasoning Problem ({logicType}):

{problem}

{typeGuidance}

Steps to follow:
1. Identify premises and conclusion
2. Convert to logical form if helpful
3. Apply appropriate inference rules
4. Check for logical validity
5. State your conclusion clearly

Solve this problem logically:";
    }
}

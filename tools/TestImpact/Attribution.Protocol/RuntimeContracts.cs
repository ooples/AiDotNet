namespace AiDotNet.TestImpact;

public enum RuntimeContractId { XunitBooleanAssertion293 }
public enum RuntimeContractStatus { Unknown, ReviewedConditional }
public enum RuntimeContractEffect { ReadScalarArguments, ThrowOnFailure }
public enum AssertionFailurePropagation { Unresolved, LeavesMethod, ForwardsToTaskBuilder }
public enum AsyncOwnerBinding { Unresolved, ReturnsStateMachineTask }
public enum OwnedFactoryCallPath { Unresolved, NullGuardPrecedesOwnedChange }
public enum RuntimeContractRequirement
{
    ExactBinaryAndRuntimeBinding,
    InitializationEffectsProven,
    SuccessfulAssertionCall,
    FailurePropagatesToObservedOwner,
    SuccessfulOwnerExecution
}

// A review identifies semantics under explicit requirements. It is not evidence
// that those requirements held, nor permission to close a dependency boundary.
// Keep this diagnostic type out of ExecutionReuse's authorization inputs.
public sealed record RuntimeContractAssessment(RuntimeContractStatus Status, RuntimeContractId? Contract,
    string Method, string AssemblyHash, string RuntimeHash,
    RuntimeContractEffect[] Effects, RuntimeContractRequirement[] Requirements);

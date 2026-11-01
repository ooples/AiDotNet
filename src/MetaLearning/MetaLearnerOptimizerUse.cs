namespace AiDotNet.MetaLearning;

/// <summary>
/// Which of the supplied optimizers a meta-learning algorithm's updates go through.
/// </summary>
/// <remarks>
/// <para>An algorithm declares this so that an optimizer it has no step for is rejected when the
/// meta-learner is built, instead of being accepted and silently never used.</para>
/// <para><b>For Beginners:</b> Most meta-learners have an outer loop (learning the starting point
/// shared by all tasks) and some also have an inner loop (adapting to one task). An optimizer can
/// only be plugged into a loop the algorithm actually runs with gradient steps.</para>
/// </remarks>
[Flags]
public enum MetaLearnerOptimizerUse
{
    /// <summary>Neither loop takes plain gradient steps; the algorithm's own update rule is part of the method.</summary>
    None = 0,

    /// <summary>The outer (meta-parameter) step goes through the meta-optimizer.</summary>
    Outer = 1,

    /// <summary>The inner (task adaptation) step goes through the inner optimizer.</summary>
    Inner = 2,

    /// <summary>Both loops go through their optimizers.</summary>
    InnerAndOuter = Outer | Inner,
}

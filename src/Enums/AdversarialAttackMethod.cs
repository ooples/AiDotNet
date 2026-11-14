namespace AiDotNet.Enums;

/// <summary>
/// An adversarial attack that robustness testing can run against a trained model.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Each value is a different way of searching for a small change to an input that
/// makes the model answer wrongly. Testing against several of them gives a more honest picture of robustness.</para>
/// </remarks>
public enum AdversarialAttackMethod
{
    /// <summary>
    /// Fast Gradient Sign Method (Goodfellow et al. 2015): one step of size epsilon along the gradient sign.
    /// </summary>
    Fgsm,

    /// <summary>
    /// Projected Gradient Descent (Madry et al. 2018): iterated gradient steps projected back into the epsilon ball.
    /// </summary>
    Pgd,

    /// <summary>
    /// The Carlini-Wagner L2 attack (Carlini and Wagner 2017): an optimization for the smallest misclassifying change.
    /// </summary>
    CarliniWagner,

    /// <summary>
    /// AutoAttack (Croce and Hein 2020): an ensemble that keeps the strongest result of its component attacks.
    /// </summary>
    AutoAttack
}

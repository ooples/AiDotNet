namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for adversarial attack algorithms that generate adversarial examples.
/// </summary>
/// <remarks>
/// An adversarial attack crafts inputs that cause machine learning models to make mistakes,
/// used for robustness testing and improving model security.
///
/// <b>For Beginners:</b> Think of an adversarial attack as a "stress test" for your AI model.
/// Just like testing if a building can withstand an earthquake, these attacks test if your model
/// can handle tricky inputs that are designed to fool it.
///
/// Common examples of adversarial attacks include:
/// - FGSM (Fast Gradient Sign Method): Quick attacks using gradient information
/// - PGD (Projected Gradient Descent): More powerful iterative attacks
/// - C&amp;W (Carlini &amp; Wagner): Sophisticated optimization-based attacks
///
/// Why adversarial attacks matter:
/// - They reveal vulnerabilities in models before deployment
/// - They help create more robust models through adversarial training
/// - They're essential for safety-critical applications (self-driving cars, medical diagnosis)
/// - They demonstrate potential security risks
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public interface IAdversarialAttack<T> : IModelSerializer
{
    /// <summary>
    /// Generates adversarial examples from clean input data.
    /// </summary>
    /// <remarks>
    /// This method takes normal inputs and perturbs them slightly to create adversarial examples
    /// that fool the target model while appearing similar to the original inputs.
    ///
    /// <b>For Beginners:</b> This is like creating optical illusions for AI. You make tiny changes
    /// to an image or input that a human wouldn't notice, but these changes trick the AI into
    /// making wrong predictions.
    ///
    /// The process typically involves:
    /// 1. Taking a clean input (e.g., an image of a cat)
    /// 2. Calculating how to modify it to fool the model
    /// 3. Creating a modified version (adversarial example)
    /// 4. The model might now think the cat is a dog, even though it looks the same to humans
    /// </remarks>
    /// <param name="input">The clean input data to be perturbed.</param>
    /// <param name="trueLabel">The correct label for the input.</param>
    /// <param name="targetModel">A function representing the model to attack.</param>
    /// <returns>The generated adversarial example.</returns>
    T[] GenerateAdversarialExample(T[] input, int trueLabel, Func<T[], T[]> targetModel);

    /// <summary>
    /// Generates a batch of adversarial examples from multiple clean inputs.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is the same as GenerateAdversarialExample, but it processes
    /// multiple inputs at once for efficiency. It's like batch processing - instead of attacking
    /// one image at a time, you attack many images together.
    /// </remarks>
    /// <param name="inputs">The batch of clean input data.</param>
    /// <param name="trueLabels">The correct labels for each input.</param>
    /// <param name="targetModel">A function representing the model to attack.</param>
    /// <returns>The batch of generated adversarial examples.</returns>
    T[][] GenerateAdversarialBatch(T[][] inputs, int[] trueLabels, Func<T[], T[]> targetModel);

    /// <summary>
    /// Calculates the perturbation added to create an adversarial example.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This shows you what changes were made to fool the model.
    /// By comparing the original input with the adversarial example, you can see exactly
    /// what the attack changed. This helps understand how the attack works.
    /// </remarks>
    /// <param name="original">The original clean input.</param>
    /// <param name="adversarial">The generated adversarial example.</param>
    /// <returns>The perturbation vector (difference between adversarial and original).</returns>
    T[] CalculatePerturbation(T[] original, T[] adversarial);

    /// <summary>
    /// Gets the configuration options for the adversarial attack.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> These are the "settings" for the attack, like:
    /// - How strong the attack should be (perturbation budget)
    /// - How many steps to take when crafting the adversarial example
    /// - What type of perturbation to use (L2, L-infinity, etc.)
    /// </remarks>
    /// <returns>The configuration options for the attack.</returns>
    AdversarialAttackOptions<T> GetOptions();

    /// <summary>
    /// Resets the attack state to prepare for a fresh attack run.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This clears any saved state from previous attacks,
    /// ensuring each new attack starts fresh without being influenced by previous runs.
    /// </remarks>
    void Reset();
}

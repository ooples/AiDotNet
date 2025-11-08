namespace AiDotNet.RetrievalAugmentedGeneration.QueryProcessors
{
    /// <summary>
    /// Provides shared utility methods for query processors.
    /// </summary>
    internal static class QueryProcessorHelpers
    {
        /// <summary>
        /// Preserves the case of the original string when applying a transformation.
        /// If the original string starts with an uppercase letter, the transformed string
        /// will also start with an uppercase letter.
        /// </summary>
        /// <param name="original">The original string whose case pattern should be preserved.</param>
        /// <param name="transformed">The transformed string to apply the case pattern to.</param>
        /// <returns>The transformed string with preserved case from the original.</returns>
        internal static string PreserveCase(string original, string transformed)
        {
            if (string.IsNullOrEmpty(original) || string.IsNullOrEmpty(transformed))
                return transformed;

            if (char.IsUpper(original[0]))
            {
                return char.ToUpper(transformed[0]) + transformed.Substring(1);
            }

            return transformed;
        }
    }
}

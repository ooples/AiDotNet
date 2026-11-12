namespace AiDotNet.NER.SpanBased;

/// <summary>An entity a span-based NER model found: tokens <paramref name="Start"/> to <paramref name="End"/> inclusive,
/// of category <paramref name="Label"/>, with the probability the model gave it.</summary>
/// <param name="Start">The first token.</param>
/// <param name="End">The last token (inclusive).</param>
/// <param name="Label">The entity category (never the non-entity class 0).</param>
/// <param name="Probability">The model's probability of the category for this span.</param>
public sealed record NamedEntitySpan(int Start, int End, int Label, double Probability);

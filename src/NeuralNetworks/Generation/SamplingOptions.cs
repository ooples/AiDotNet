// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.NeuralNetworks.Generation;

/// <summary>
/// Decoding/sampling configuration for autoregressive text generation (#1632 / #95).
/// Centralises the knobs that were previously reimplemented per multimodal model
/// (GPT4Vision/Blip/Flamingo each rolled their own temperature + softmax + sample loop).
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> these control how the model picks the next token.
/// <list type="bullet">
/// <item><b>Temperature</b> — randomness. 0 (or very small) = always pick the single most likely
/// token (greedy, deterministic). 1.0 = sample from the model's raw distribution. &gt;1 = flatter /
/// more random; &lt;1 = sharper / more focused.</item>
/// <item><b>TopK</b> — only consider the K most-likely tokens (0 = no limit).</item>
/// <item><b>TopP</b> (nucleus) — only consider the smallest set of tokens whose probabilities sum to
/// at least P (0 or ≥1 = no limit).</item>
/// <item><b>Seed</b> — set for reproducible sampling; null uses the shared thread-safe RNG.</item>
/// </list>
/// </remarks>
public sealed class SamplingOptions
{
    /// <summary>Softmax temperature. ≤ <see cref="GreedyTemperatureEpsilon"/> ⇒ greedy (argmax). Default 1.0.</summary>
    public double Temperature { get; init; } = 1.0;

    /// <summary>Keep only the K highest-logit tokens before sampling. 0 ⇒ disabled. Default 0.</summary>
    public int TopK { get; init; }

    /// <summary>Nucleus threshold: keep the smallest token set whose cumulative probability ≥ TopP.
    /// 0 or ≥ 1 ⇒ disabled. Default 0.</summary>
    public double TopP { get; init; }

    /// <summary>Seed for reproducible sampling; null ⇒ shared thread-safe RNG.</summary>
    public int? Seed { get; init; }

    /// <summary>At/below this temperature, sampling collapses to greedy argmax (avoids divide-by-~0).</summary>
    public const double GreedyTemperatureEpsilon = 1e-6;

    /// <summary>Greedy (argmax) decoding — deterministic.</summary>
    public static SamplingOptions Greedy { get; } = new() { Temperature = 0.0 };

    /// <summary>Default sampling (temperature 1.0, no top-k/top-p).</summary>
    public static SamplingOptions Default { get; } = new();

    /// <summary>True when these options request deterministic greedy decoding.</summary>
    public bool IsGreedy => Temperature <= GreedyTemperatureEpsilon;
}

using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Serving.Models;
using AiDotNet.Serving.Services;

namespace AiDotNet.Serving.Controllers;

/// <summary>
/// Resolves the generic model names that coding CLIs send by default (Claude Code sends <c>claude-*</c>,
/// OpenCode and OpenAI SDKs send <c>gpt-*</c>, <c>o1-*</c>, <c>o3-*</c> or <c>default</c>) to the single
/// loaded generative model. Shared by the OpenAI and Anthropic controllers so the two cannot diverge.
/// </summary>
internal static class CliModelAliasResolver
{
    /// <summary>True when <paramref name="modelName"/> is a generic CLI default rather than a served model name.</summary>
    public static bool IsStandardCliAlias(string modelName)
    {
        return modelName.Equals("default", StringComparison.OrdinalIgnoreCase)
            || modelName.StartsWith("gpt-", StringComparison.OrdinalIgnoreCase)
            || modelName.StartsWith("o1-", StringComparison.OrdinalIgnoreCase)
            || modelName.StartsWith("o3-", StringComparison.OrdinalIgnoreCase)
            || modelName.StartsWith("claude-", StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Looks up <paramref name="modelName"/>; when it is not loaded but is a CLI alias, falls back to the
    /// loaded generative model only if exactly one model supports generation and has a tokenizer. With
    /// several candidates the alias is ambiguous (repository enumeration order is unspecified), so it stays
    /// unresolved and the caller returns its not-found error rather than serving an arbitrary model.
    /// </summary>
    /// <returns>The model info, or null when the name is unknown or the alias is ambiguous.</returns>
    public static ModelInfo? Resolve(
        string modelName,
        IModelRepository modelRepository,
        ITextGenerationService textGeneration,
        ITokenizerRegistry tokenizers,
        out string resolvedName)
    {
        resolvedName = modelName;
        var info = modelRepository.GetModelInfo(modelName);
        if (info != null || !IsStandardCliAlias(modelName))
        {
            return info;
        }

        List<ModelInfo> candidates = modelRepository.GetAllModelInfo()
            .Where(m => textGeneration.SupportsGeneration(m.Name, m.NumericType) && tokenizers.TryGet(m.Name, out _))
            .Take(2)
            .ToList();
        if (candidates.Count != 1)
        {
            return null;
        }

        resolvedName = candidates[0].Name;
        return candidates[0];
    }
}

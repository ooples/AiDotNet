using System;
using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Rag.Benchmarks
{
    /// <summary>
    /// A fully deterministic, dependency-free "bag-of-words" embedder used for the
    /// end-to-end retrieval benchmark.
    /// </summary>
    /// <remarks>
    /// Each lowercased whitespace token is hashed (FNV-1a) into a fixed-width vector
    /// and its slot incremented; the vector is then L2-normalized. Documents and
    /// queries that share distinctive vocabulary land close under cosine similarity.
    /// This is intentionally NOT a learned model — it removes any dependency on a live
    /// LLM/embedding service so the retrieval numbers are reproducible.
    /// </remarks>
    internal sealed class StubEmbedder
    {
        private readonly int _dim;

        internal StubEmbedder(int dim = 128)
        {
            _dim = dim;
        }

        internal int Dimension => _dim;

        internal Vector<double> Embed(string text)
        {
            var data = new double[_dim];
            foreach (var token in Tokenize(text))
            {
                uint h = Fnv1a(token);
                int slot = (int)(h % (uint)_dim);
                data[slot] += 1.0;
            }

            // L2 normalize so cosine similarity is well-conditioned.
            double norm = 0.0;
            for (int i = 0; i < _dim; i++)
                norm += data[i] * data[i];
            norm = Math.Sqrt(norm);
            if (norm > 0.0)
            {
                for (int i = 0; i < _dim; i++)
                    data[i] /= norm;
            }

            return new Vector<double>(data);
        }

        private static IEnumerable<string> Tokenize(string text)
        {
            foreach (var raw in text.Split(
                new[] { ' ', '\t', '\n', '\r', ',', '.', ';', ':', '!', '?', '(', ')', '"', '\'' },
                StringSplitOptions.RemoveEmptyEntries))
            {
                yield return raw.ToLowerInvariant();
            }
        }

        private static uint Fnv1a(string s)
        {
            const uint offset = 2166136261;
            const uint prime = 16777619;
            uint hash = offset;
            foreach (char c in s)
            {
                hash ^= c;
                hash *= prime;
            }
            return hash;
        }
    }

    /// <summary>
    /// A small, fixed, hand-labeled corpus with a known query -&gt; relevant-doc mapping,
    /// used to measure end-to-end IR quality (recall@k, MRR, nDCG@10).
    /// </summary>
    internal static class LabeledCorpus
    {
        internal sealed class Query
        {
            public required string Text { get; init; }
            public required HashSet<string> RelevantIds { get; init; }
        }

        /// <summary>Returns (docId -&gt; content) for the corpus.</summary>
        internal static (string Id, string Content)[] Documents() => new (string, string)[]
        {
            // --- Topic: Python programming ---
            ("py-1",  "Python is a high level programming language used for scripting and automation"),
            ("py-2",  "In Python you define a function with the def keyword and indentation matters"),
            ("py-3",  "Python list comprehensions provide a concise way to build lists from iterables"),
            ("py-4",  "The Python interpreter runs scripts line by line and manages memory automatically"),
            // --- Topic: cooking / recipes ---
            ("cook-1", "To bake sourdough bread you need flour water salt and an active starter culture"),
            ("cook-2", "A good tomato sauce simmers with garlic olive oil basil and a pinch of salt"),
            ("cook-3", "Roasting vegetables in the oven caramelizes their natural sugars for deeper flavor"),
            ("cook-4", "Whisk eggs sugar and butter together before folding in flour to make a cake batter"),
            // --- Topic: astronomy / space ---
            ("space-1", "A black hole is a region of spacetime where gravity is so strong light cannot escape"),
            ("space-2", "The planet Mars is a cold desert world and a target for future crewed space missions"),
            ("space-3", "Galaxies contain billions of stars bound together by gravity across the universe"),
            ("space-4", "A telescope collects light from distant stars and planets to study the cosmos"),
            // --- Topic: personal finance / investing ---
            ("fin-1", "Diversifying your investment portfolio across stocks and bonds reduces overall risk"),
            ("fin-2", "Compound interest grows your savings faster as returns are reinvested over time"),
            ("fin-3", "An index fund tracks a market benchmark and typically charges low management fees"),
            ("fin-4", "A budget helps you track monthly income and expenses so you can save money"),
            // --- Topic: gardening ---
            ("garden-1", "Tomato plants need full sunlight regular watering and well drained fertile soil"),
            ("garden-2", "Composting kitchen scraps creates nutrient rich soil for your vegetable garden"),
            ("garden-3", "Pruning roses in early spring encourages healthy new growth and more blooms"),
            ("garden-4", "Mulch around plants retains soil moisture and suppresses competing weeds"),
        };

        /// <summary>Returns the labeled query set with known relevant document ids.</summary>
        internal static Query[] Queries() => new[]
        {
            new Query
            {
                Text = "how do I define a function in the python programming language",
                RelevantIds = new HashSet<string> { "py-1", "py-2", "py-3", "py-4" },
            },
            new Query
            {
                Text = "recipe for baking bread and making tomato sauce with flour and salt",
                RelevantIds = new HashSet<string> { "cook-1", "cook-2", "cook-3", "cook-4" },
            },
            new Query
            {
                Text = "stars planets black hole and telescopes in space and the universe",
                RelevantIds = new HashSet<string> { "space-1", "space-2", "space-3", "space-4" },
            },
            new Query
            {
                Text = "how to invest savings in stocks bonds and an index fund to reduce risk",
                RelevantIds = new HashSet<string> { "fin-1", "fin-2", "fin-3", "fin-4" },
            },
            new Query
            {
                Text = "growing tomato plants and vegetables in fertile garden soil with compost and mulch",
                RelevantIds = new HashSet<string> { "garden-1", "garden-2", "garden-3", "garden-4" },
            },
        };
    }
}

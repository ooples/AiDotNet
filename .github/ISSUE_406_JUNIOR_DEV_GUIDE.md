# Junior Developer Implementation Guide: Issue #406
## Tokenization Infrastructure (BPE, WordPiece, SentencePiece, Code Tokenizers)

**Issue:** [#406 - Tokenization Infrastructure](https://github.com/ooples/AiDotNet/issues/406)

**Estimated Complexity:** Advanced (Critical Infrastructure)

**Time Estimate:** 40-50 hours

**CRITICAL:** This is blocking infrastructure for ALL NLP work. High priority.

---

## Table of Contents
1. [Understanding the Problem](#understanding-the-problem)
2. [Background Concepts](#background-concepts)
3. [Architecture Overview](#architecture-overview)
4. [Implementation Steps](#implementation-steps)
5. [Testing Strategy](#testing-strategy)
6. [Common Pitfalls](#common-pitfalls)
7. [Resources](#resources)

---

## Understanding the Problem

### What is Tokenization?

Tokenization is the process of breaking down text into smaller units (tokens) that machine learning models can understand and process.

**Real-world analogy:**
- **For humans**: We read words and sentences
- **For ML models**: They process numerical tokens (like "running" → [7894])

### Why Do We Need Specialized Tokenizers?

Different tokenization strategies solve different problems:

1. **Word-level tokenization**: "Hello world" → ["Hello", "world"]
   - Problem: Massive vocabulary size (millions of words)
   - Problem: Can't handle unknown words ("supercalifragilisticexpialidocious")

2. **Character-level tokenization**: "Hello" → ["H", "e", "l", "l", "o"]
   - Problem: Very long sequences (slow training)
   - Problem: Loses word structure and meaning

3. **Subword tokenization** (BPE, WordPiece, SentencePiece): "running" → ["run", "##ning"]
   - Solution: Balance between word and character level
   - Solution: Fixed vocabulary size, handles unknown words
   - Solution: Captures morphology (word structure)

### What This Issue Implements

This issue creates the **foundation for ALL NLP models** in AiDotNet:

1. **BPE (Byte Pair Encoding)**: Used by GPT models
2. **WordPiece**: Used by BERT models
3. **SentencePiece**: Language-agnostic, used by T5 and multilingual models
4. **Code Tokenizers**: Special handling for programming languages

---

## Background Concepts

### 1. BPE (Byte Pair Encoding)

**What it is:** A greedy algorithm that iteratively merges the most frequent pair of tokens.

**How it works:**

**Step 1: Initialize with character vocabulary**
```
Text: "low lower lowest"
Initial tokens: ["l", "o", "w", "space", "l", "o", "w", "e", "r", ...]
Vocabulary: {l, o, w, e, r, s, t, space}
```

**Step 2: Count pair frequencies**
```
Pair frequencies:
("l", "o"): 3 times
("o", "w"): 3 times
("w", "space"): 1 time
("w", "e"): 2 times
("e", "r"): 2 times
```

**Step 3: Merge most frequent pair**
```
Most frequent: ("l", "o") → "lo"
Updated text: "lo w lo w e r lo w e s t"
New vocabulary: {l, o, w, e, r, s, t, space, lo}
```

**Step 4: Repeat until desired vocabulary size**
```
Iteration 2: ("lo", "w") → "low"
Iteration 3: ("low", "e") → "lowe"
Iteration 4: ("lowe", "r") → "lower"
...continue until 50k tokens (typical)
```

**Final tokenization:**
```
Original: "lowest"
Tokens: ["low", "est"]
IDs: [1024, 2048]  (example IDs)
```

**Mathematical Formula:**
```
Given vocabulary V and text corpus C:

1. Initialize V = set of all characters
2. While |V| < target_vocab_size:
   a. Count all adjacent pairs in C: freq(x, y)
   b. Find most frequent pair: (x*, y*) = argmax freq(x, y)
   c. Create new token: xy = merge(x*, y*)
   d. Replace all occurrences of (x*, y*) with xy
   e. Add xy to V, record merge rule: (x*, y*) → xy
3. Return V and merge rules
```

**Real Implementation Details:**

```csharp
public class BpeTokenizer
{
    private Dictionary<string, int> _vocabulary;
    private List<(string, string, string)> _mergeRules; // (token1, token2, merged)

    public void Train(IEnumerable<string> corpus, int vocabSize)
    {
        // Step 1: Pre-tokenize (split on whitespace, punctuation)
        var words = PreTokenize(corpus);

        // Step 2: Initialize with character vocabulary
        var vocabulary = InitializeCharVocabulary(words);

        // Step 3: Count word frequencies for efficiency
        var wordFreqs = words.GroupBy(w => w)
                             .ToDictionary(g => g.Key, g => g.Count());

        // Step 4: Iteratively merge
        while (vocabulary.Count < vocabSize)
        {
            var pairFreqs = CountPairFrequencies(wordFreqs);
            if (pairFreqs.Count == 0) break;

            var mostFrequent = pairFreqs.OrderByDescending(p => p.Value).First();
            var (token1, token2) = mostFrequent.Key;
            var merged = token1 + token2;

            // Update word representations
            UpdateWordFrequencies(wordFreqs, token1, token2, merged);

            // Record merge rule
            _mergeRules.Add((token1, token2, merged));
            vocabulary[merged] = vocabulary.Count;
        }

        _vocabulary = vocabulary;
    }

    private Dictionary<string, int> CountPairFrequencies(
        Dictionary<string, int> wordFreqs)
    {
        var pairFreqs = new Dictionary<(string, string), int>();

        foreach (var (word, freq) in wordFreqs)
        {
            var symbols = word.Split(' '); // Words are space-separated tokens
            for (int i = 0; i < symbols.Length - 1; i++)
            {
                var pair = (symbols[i], symbols[i + 1]);
                pairFreqs[pair] = pairFreqs.GetValueOrDefault(pair, 0) + freq;
            }
        }

        return pairFreqs;
    }

    public int[] Encode(string text)
    {
        // Step 1: Pre-tokenize
        var words = PreTokenize(new[] { text });

        // Step 2: Apply merge rules in order
        foreach (var (token1, token2, merged) in _mergeRules)
        {
            words = ApplyMerge(words, token1, token2, merged);
        }

        // Step 3: Convert to IDs
        return words.SelectMany(w => w.Split(' '))
                    .Select(token => _vocabulary[token])
                    .ToArray();
    }
}
```

**Worked Example:**

```
Corpus: "low low low lowest lower low"

Step 1: Pre-tokenize and add end-of-word marker
["l o w </w>", "l o w </w>", "l o w </w>",
 "l o w e s t </w>", "l o w e r </w>", "l o w </w>"]

Word frequencies:
"l o w </w>": 4
"l o w e s t </w>": 1
"l o w e r </w>": 1

Step 2: Count pairs
("l", "o"): 4 + 1 + 1 = 6
("o", "w"): 6
("w", "</w>"): 4
("w", "e"): 2
("e", "s"): 1
("s", "t"): 1
("t", "</w>"): 1
("e", "r"): 1
("r", "</w>"): 1

Step 3: Merge most frequent: ("l", "o") → "lo"
["lo w </w>": 4, "lo w e s t </w>": 1, "lo w e r </w>": 1]

Step 4: Count pairs again
("lo", "w"): 6
("w", "</w>"): 4
("w", "e"): 2
...

Step 5: Merge ("lo", "w") → "low"
["low </w>": 4, "low e s t </w>": 1, "low e r </w>": 1]

...continue until vocabulary size reached...

Final merges:
1. l + o → lo
2. lo + w → low
3. e + r → er
4. low + er → lower
5. e + s → es
6. es + t → est
7. low + est → lowest

Final vocabulary (simplified):
{l, o, w, e, r, s, t, </w>, lo, low, er, lower, es, est, lowest}

Encoding "lowest": [lowest, </w>] → [vocabulary["lowest"], vocabulary["</w>"]]
```

### 2. WordPiece

**What it is:** Similar to BPE but uses likelihood-based scoring instead of frequency.

**Key difference from BPE:**
- BPE: Merges most frequent pair
- WordPiece: Merges pair that maximizes likelihood of training data

**How it works:**

**Likelihood formula:**
```
score(x, y) = log P(xy) - log P(x) - log P(y)
            = log(freq(xy) / total_pairs)
              - log(freq(x) / total_tokens)
              - log(freq(y) / total_tokens)

Simplified:
score(x, y) = freq(xy) / (freq(x) * freq(y))
```

**Why likelihood matters:**

Consider two pairs with same frequency:
```
Pair A: ("th", "e") appears 1000 times
  - "th" appears 1500 times total
  - "e" appears 5000 times total
  - score = 1000 / (1500 * 5000) = 0.000133

Pair B: ("qu", "e") appears 1000 times
  - "qu" appears 1200 times total (almost always followed by "e")
  - "e" appears 5000 times total
  - score = 1000 / (1200 * 5000) = 0.000167
```

**Result:** WordPiece prefers ("qu", "e") because "qu" is almost always followed by "e", making it a more meaningful unit.

**Implementation:**

```csharp
public class WordPieceTokenizer
{
    private Dictionary<string, int> _vocabulary;
    private string _unkToken = "[UNK]";
    private string _continuingSubwordPrefix = "##";

    public void Train(IEnumerable<string> corpus, int vocabSize)
    {
        // Step 1: Pre-tokenize by whitespace
        var words = PreTokenize(corpus);

        // Step 2: Initialize with character vocabulary + special tokens
        var vocabulary = new HashSet<string>
        {
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"
        };

        // Add all characters
        foreach (var word in words)
        {
            foreach (var c in word)
            {
                vocabulary.Add(c.ToString());
            }
        }

        // Step 3: Count token frequencies
        var tokenFreqs = CountTokenFrequencies(words);

        // Step 4: Iteratively add subwords based on likelihood
        while (vocabulary.Count < vocabSize)
        {
            var candidates = GenerateCandidateSubwords(words, vocabulary);
            if (candidates.Count == 0) break;

            var bestCandidate = candidates.OrderByDescending(c =>
                CalculateLikelihood(c, tokenFreqs)).First();

            vocabulary.Add(bestCandidate);
            UpdateTokenFrequencies(tokenFreqs, bestCandidate);
        }

        _vocabulary = vocabulary
            .Select((token, idx) => (token, idx))
            .ToDictionary(x => x.token, x => x.idx);
    }

    private double CalculateLikelihood(
        string candidate,
        Dictionary<string, int> tokenFreqs)
    {
        // Find all ways to split candidate into known tokens
        var splits = FindAllValidSplits(candidate);
        if (splits.Count == 0) return double.MinValue;

        // Calculate likelihood as log(P(xy)) - log(P(x)) - log(P(y))
        int freqCandidate = tokenFreqs.GetValueOrDefault(candidate, 0);
        if (freqCandidate == 0) return double.MinValue;

        double totalTokens = tokenFreqs.Values.Sum();
        double logProbCandidate = Math.Log(freqCandidate / totalTokens);

        // For each split, calculate alternative probability
        double maxAlternative = splits.Max(split =>
        {
            double logProbSplit = split.Sum(token =>
                Math.Log(tokenFreqs[token] / totalTokens));
            return logProbSplit;
        });

        return logProbCandidate - maxAlternative;
    }

    public int[] Encode(string text)
    {
        var tokens = new List<string>();
        var words = text.Split(' ');

        foreach (var word in words)
        {
            // Greedy longest-match-first tokenization
            int start = 0;
            var subTokens = new List<string>();

            while (start < word.Length)
            {
                int end = word.Length;
                string foundToken = null;

                // Find longest matching subword
                while (start < end)
                {
                    string substr = word.Substring(start, end - start);

                    // Add ## prefix for continuing subwords
                    if (start > 0)
                        substr = _continuingSubwordPrefix + substr;

                    if (_vocabulary.ContainsKey(substr))
                    {
                        foundToken = substr;
                        break;
                    }
                    end--;
                }

                if (foundToken == null)
                {
                    // Unknown token
                    subTokens.Add(_unkToken);
                    break;
                }

                subTokens.Add(foundToken);
                start = end;
            }

            tokens.AddRange(subTokens);
        }

        return tokens.Select(t => _vocabulary[t]).ToArray();
    }
}
```

**Worked Example:**

```
Corpus: "playing played player"

Step 1: Initialize
vocabulary = {p, l, a, y, i, n, g, e, d, r, [UNK], [PAD], ...}

Step 2: Count unigrams, bigrams, trigrams
Unigrams: p:3, l:3, a:3, y:3, i:1, n:1, g:1, e:2, d:1, r:1
Bigrams: pl:3, la:3, ay:3, yi:1, in:1, ng:1, ye:2, ed:1, er:1
Trigrams: pla:3, lay:3, ayi:1, yin:1, ing:1, aye:2, yed:1, yer:1

Step 3: Calculate likelihood scores
"play": freq=3, constituent freq (p:3, l:3, a:3, y:3)
score = log(3) - log(3) - log(3) - log(3) - log(3) = log(3) - 4*log(3) = -3*log(3)
(This is simplified; actual calculation considers token pairs)

Better approach - score for merges:
"pl": freq=3, P(pl)=3/total, P(p)=3/total, P(l)=3/total
score = 3 / (3 * 3) = 0.33

"##ing": freq=1, P(ing)=1/total, P(i)=1/total, P(n)=1/total, P(g)=1/total
score = 1 / (1 * 1 * 1) = 1.0  (very high - always appears together)

Step 4: Add highest scoring subwords
Add "##ing" (score 1.0)
Add "play" (high frequency, meaningful)
Add "##ed" (suffix pattern)
Add "##er" (suffix pattern)

Final vocabulary includes:
{p, l, a, y, ..., play, ##ing, ##ed, ##er, [UNK], ...}

Encoding "playing":
- Start: "playing"
- Longest match: "play" (in vocab)
- Remaining: "ing"
- Add prefix: "##ing" (in vocab)
- Result: ["play", "##ing"]
```

### 3. SentencePiece

**What it is:** A language-agnostic tokenizer that treats input as raw byte stream, not pre-tokenized words.

**Key innovations:**
1. **No pre-tokenization**: Works on raw text (handles Chinese, Japanese, etc.)
2. **Reversible**: Can perfectly reconstruct original text including spaces
3. **Subword regularization**: Multiple segmentations for robustness

**How it works:**

**Unigram Language Model approach:**

```
Goal: Find segmentation that maximizes probability of sentence

P(sentence) = Π P(token_i)

Example:
"hello" can be segmented as:
1. ["h", "e", "l", "l", "o"]
2. ["he", "ll", "o"]
3. ["hel", "lo"]
4. ["hello"]

Choose segmentation with highest probability:
P1 = P(h) * P(e) * P(l) * P(l) * P(o)
P2 = P(he) * P(ll) * P(o)
P3 = P(hel) * P(lo)
P4 = P(hello)

Use Viterbi algorithm to find best path
```

**Training algorithm:**

```
1. Start with large seed vocabulary (all characters + all substrings)
2. Estimate probability of each token from corpus
3. Remove tokens that minimally affect loss
4. Repeat until desired vocabulary size
```

**Mathematical formulation:**

```
Given sentence S and vocabulary V:

1. All possible segmentations: Seg(S) = {x_1, x_2, ..., x_n | x_i ∈ V, concat(x) = S}

2. Best segmentation: x* = argmax P(x | S)
   where P(x | S) = Π P(x_i)

3. Viterbi decoding:
   Let best[i] = probability of best segmentation of S[0:i]

   best[0] = 1.0
   best[i] = max(best[j] * P(S[j:i])) for all j < i where S[j:i] ∈ V
```

**Implementation:**

```csharp
public class SentencePieceTokenizer
{
    private Dictionary<string, double> _tokenProbabilities;
    private Dictionary<string, int> _vocabulary;
    private string _unkToken = "<unk>";
    private string _bosToken = "<s>";
    private string _eosToken = "</s>";

    public void Train(IEnumerable<string> corpus, int vocabSize)
    {
        // Step 1: Build seed vocabulary (all substrings up to length 10)
        var seedVocab = BuildSeedVocabulary(corpus, maxLength: 10);

        // Step 2: Initialize probabilities using EM algorithm
        EstimateInitialProbabilities(seedVocab, corpus);

        // Step 3: Prune vocabulary
        while (seedVocab.Count > vocabSize)
        {
            // Remove token that minimally increases loss
            var tokenToRemove = FindLeastImpactfulToken(seedVocab, corpus);
            seedVocab.Remove(tokenToRemove);

            // Re-estimate probabilities
            EstimateInitialProbabilities(seedVocab, corpus);
        }

        _tokenProbabilities = seedVocab.ToDictionary(
            kvp => kvp.Key,
            kvp => kvp.Value);

        _vocabulary = seedVocab.Keys
            .Select((token, idx) => (token, idx))
            .ToDictionary(x => x.token, x => x.idx);
    }

    private void EstimateInitialProbabilities(
        Dictionary<string, double> vocab,
        IEnumerable<string> corpus)
    {
        // Use forward-backward algorithm (EM) to estimate probabilities
        var tokenCounts = new Dictionary<string, double>();

        foreach (var sentence in corpus)
        {
            // E-step: Get expected counts using current probabilities
            var expectedCounts = ComputeExpectedCounts(sentence, vocab);

            foreach (var (token, count) in expectedCounts)
            {
                tokenCounts[token] = tokenCounts.GetValueOrDefault(token, 0) + count;
            }
        }

        // M-step: Update probabilities
        double total = tokenCounts.Values.Sum();
        foreach (var token in vocab.Keys.ToList())
        {
            vocab[token] = tokenCounts.GetValueOrDefault(token, 1e-10) / total;
        }
    }

    private Dictionary<string, double> ComputeExpectedCounts(
        string sentence,
        Dictionary<string, double> vocab)
    {
        int n = sentence.Length;

        // Forward probabilities: forward[i] = P(sentence[0:i])
        var forward = new double[n + 1];
        var backpointers = new List<(int, string)>[n + 1];

        forward[0] = 1.0;
        for (int i = 0; i <= n; i++)
            backpointers[i] = new List<(int, string)>();

        // Fill forward table
        for (int i = 1; i <= n; i++)
        {
            for (int j = 0; j < i; j++)
            {
                string token = sentence.Substring(j, i - j);
                if (vocab.TryGetValue(token, out double prob))
                {
                    double score = forward[j] * prob;
                    forward[i] += score;
                    backpointers[i].Add((j, token));
                }
            }
        }

        // Backward pass to get expected counts
        var counts = new Dictionary<string, double>();
        var backward = new double[n + 1];
        backward[n] = 1.0;

        for (int i = n - 1; i >= 0; i--)
        {
            foreach (var (j, token) in backpointers[i])
            {
                double prob = vocab[token];
                double posterior = (forward[j] * prob * backward[i]) / forward[n];
                counts[token] = counts.GetValueOrDefault(token, 0) + posterior;
            }
        }

        return counts;
    }

    public int[] Encode(string text)
    {
        // Viterbi decoding: find best segmentation
        int n = text.Length;
        var best = new double[n + 1];
        var backpointers = new int[n + 1];
        var tokens = new string[n + 1];

        best[0] = 0.0; // Use log probabilities

        for (int i = 1; i <= n; i++)
        {
            best[i] = double.NegativeInfinity;

            for (int j = 0; j < i; j++)
            {
                string token = text.Substring(j, i - j);

                if (_tokenProbabilities.TryGetValue(token, out double prob))
                {
                    double score = best[j] + Math.Log(prob);

                    if (score > best[i])
                    {
                        best[i] = score;
                        backpointers[i] = j;
                        tokens[i] = token;
                    }
                }
            }

            // Handle unknown characters
            if (best[i] == double.NegativeInfinity)
            {
                best[i] = best[i - 1] + Math.Log(1e-10);
                backpointers[i] = i - 1;
                tokens[i] = _unkToken;
            }
        }

        // Backtrack to get segmentation
        var result = new List<string>();
        int pos = n;
        while (pos > 0)
        {
            result.Add(tokens[pos]);
            pos = backpointers[pos];
        }

        result.Reverse();
        return result.Select(t => _vocabulary[t]).ToArray();
    }

    // Subword regularization: sample from multiple segmentations
    public int[] EncodeWithRegularization(string text, double alpha = 0.1)
    {
        // Use forward-filtering backward-sampling
        int n = text.Length;
        var forward = ComputeForwardProbabilities(text);

        // Sample path from end to start
        var result = new List<string>();
        int pos = n;

        while (pos > 0)
        {
            // Get all possible previous positions
            var candidates = new List<(int, string, double)>();

            for (int j = 0; j < pos; j++)
            {
                string token = text.Substring(j, pos - j);
                if (_tokenProbabilities.TryGetValue(token, out double prob))
                {
                    double score = forward[j] * Math.Pow(prob, alpha);
                    candidates.Add((j, token, score));
                }
            }

            // Sample proportional to scores
            var chosen = SampleFromCandidates(candidates);
            result.Add(chosen.token);
            pos = chosen.prevPos;
        }

        result.Reverse();
        return result.Select(t => _vocabulary[t]).ToArray();
    }
}
```

**Worked Example:**

```
Corpus: "▁hello ▁world"  (▁ represents space)

Step 1: Build seed vocabulary (all substrings)
{▁, h, e, l, o, w, r, d, ▁h, he, el, ll, lo, o▁, ▁w, wo, or, rl, ld,
 ▁he, hel, ell, llo, lo▁, ▁wo, wor, orl, rld,
 ▁hel, hell, ello, llo▁, ▁wor, worl, orld,
 ▁hell, hello, ello▁, ▁worl, world,
 ▁hello, hello▁, ▁world}

Step 2: Count frequencies
▁hello: 1
▁world: 1
hello: 1
world: 1
...

Step 3: Initial probabilities (uniform over frequencies)
P(▁hello) = 1/total
P(▁world) = 1/total
P(▁) = 2/total (appears before both words)
P(l) = 3/total (appears 3 times in hello)
...

Step 4: EM iterations
Iteration 1:
  E-step: Find all segmentations and their probabilities
    "▁hello" → [▁hello] with P = P(▁hello)
    "▁hello" → [▁h, el, lo] with P = P(▁h)*P(el)*P(lo)
    "▁hello" → [▁, hel, lo] with P = P(▁)*P(hel)*P(lo)
    ...

  M-step: Update probabilities based on expected usage
    If [▁hello] is most probable → increase P(▁hello)
    If [▁, hel, lo] is competitive → increase P(▁), P(hel), P(lo)

Step 5: Pruning
  Remove tokens with lowest probabilities
  Prefer: ▁hello, ▁world, common subwords (▁, ##lo, ##ld)
  Remove: rare subwords (▁hel, ▁wor, etc.)

Final vocabulary (vocabSize=20):
{▁, h, e, l, o, w, r, d, ▁h, ▁w, he, ll, or, ld, ▁he, ▁wo, ell, orl, ▁hello, ▁world}

Encoding "▁hello" using Viterbi:
Position 0: ""
  best[0] = 0 (log prob)

Position 1: "▁"
  From 0 with "▁": score = 0 + log(P(▁))
  best[1] = log(P(▁))

Position 2: "▁h"
  From 0 with "▁h": score = 0 + log(P(▁h))
  From 1 with "h": score = best[1] + log(P(h))
  best[2] = max(log(P(▁h)), log(P(▁)) + log(P(h)))

...continue...

Position 6: "▁hello"
  From 0 with "▁hello": score = 0 + log(P(▁hello))  [best path!]
  From 1 with "hello": score = best[1] + log(P(hello))
  From 2 with "ello": ...
  best[6] = log(P(▁hello))

Result: ["▁hello"]
```

### 4. Special Tokens

**What they are:** Reserved tokens with special meanings in NLP models.

**Common special tokens:**

```csharp
public class SpecialTokens
{
    // Padding: Used to make sequences equal length in batches
    public const string PAD = "[PAD]";
    public const int PAD_ID = 0;

    // Unknown: Represents out-of-vocabulary words
    public const string UNK = "[UNK]";
    public const int UNK_ID = 1;

    // Classification: Marks start of sequence (BERT)
    public const string CLS = "[CLS]";
    public const int CLS_ID = 2;

    // Separator: Separates segments (BERT)
    public const string SEP = "[SEP]";
    public const int SEP_ID = 3;

    // Mask: Used in masked language modeling (BERT)
    public const string MASK = "[MASK]";
    public const int MASK_ID = 4;

    // Beginning/End of sentence (GPT, SentencePiece)
    public const string BOS = "<s>";
    public const string EOS = "</s>";

    // Custom markers for code
    public const string CODE_START = "<code>";
    public const string CODE_END = "</code>";
    public const string COMMENT_START = "<comment>";
    public const string COMMENT_END = "</comment>";
}
```

**Usage examples:**

```
BERT input format:
[CLS] what is the capital of france [SEP] paris [SEP]
→ Used for question answering

GPT input format:
<s> Once upon a time </s>
→ Used for text generation

Masked Language Model:
[CLS] the cat sat on the [MASK] [SEP]
→ Model predicts [MASK] = "mat"

Code tokenization:
<code> def hello(): <comment> # greet user </comment> print("hi") </code>
→ Preserves code structure
```

### 5. Code Tokenization

**What it is:** Specialized tokenization for programming languages using AST (Abstract Syntax Tree) awareness.

**Challenges:**
1. **Whitespace matters**: Indentation in Python, formatting in all languages
2. **Identifiers**: Variable names should be split meaningfully (`getUserName` → `get`, `User`, `Name`)
3. **Comments**: Natural language mixed with code
4. **Literals**: Strings, numbers should be preserved

**AST-based approach:**

```csharp
public class CodeTokenizer
{
    private ITokenizer _baseTokenizer; // BPE or WordPiece
    private Dictionary<string, string[]> _keywords;

    public CodeTokenizer(ITokenizer baseTokenizer)
    {
        _baseTokenizer = baseTokenizer;

        // Language keywords (never split)
        _keywords = new Dictionary<string, string[]>
        {
            ["python"] = new[] { "def", "class", "if", "for", "while", "import", ... },
            ["csharp"] = new[] { "public", "private", "class", "void", "int", ... },
            ["javascript"] = new[] { "function", "const", "let", "var", "class", ... }
        };
    }

    public CodeToken[] Tokenize(string code, string language)
    {
        // Step 1: Parse code into AST
        var ast = ParseAST(code, language);

        // Step 2: Walk AST and tokenize each node
        var tokens = new List<CodeToken>();

        WalkAST(ast, node =>
        {
            switch (node.Type)
            {
                case ASTNodeType.Keyword:
                    tokens.Add(new CodeToken
                    {
                        Text = node.Text,
                        Type = TokenType.Keyword,
                        Ids = new[] { GetKeywordId(node.Text) }
                    });
                    break;

                case ASTNodeType.Identifier:
                    // Split camelCase/snake_case
                    var parts = SplitIdentifier(node.Text);
                    var ids = _baseTokenizer.Encode(string.Join(" ", parts));
                    tokens.Add(new CodeToken
                    {
                        Text = node.Text,
                        Type = TokenType.Identifier,
                        Ids = ids,
                        Parts = parts
                    });
                    break;

                case ASTNodeType.String:
                    tokens.Add(new CodeToken
                    {
                        Text = node.Text,
                        Type = TokenType.StringLiteral,
                        Ids = EncodeStringLiteral(node.Text)
                    });
                    break;

                case ASTNodeType.Number:
                    tokens.Add(new CodeToken
                    {
                        Text = node.Text,
                        Type = TokenType.NumberLiteral,
                        Ids = new[] { GetNumberId(node.Text) }
                    });
                    break;

                case ASTNodeType.Comment:
                    // Tokenize as natural language
                    var commentText = StripCommentMarkers(node.Text);
                    var commentIds = _baseTokenizer.Encode(commentText);
                    tokens.Add(new CodeToken
                    {
                        Text = node.Text,
                        Type = TokenType.Comment,
                        Ids = commentIds
                    });
                    break;

                case ASTNodeType.Whitespace:
                    // Preserve indentation
                    if (IsIndentation(node.Text))
                    {
                        tokens.Add(new CodeToken
                        {
                            Text = node.Text,
                            Type = TokenType.Indentation,
                            Ids = new[] { GetIndentationId(node.Text.Length / 4) }
                        });
                    }
                    break;
            }
        });

        return tokens.ToArray();
    }

    private string[] SplitIdentifier(string identifier)
    {
        // Handle camelCase
        if (IsCamelCase(identifier))
        {
            // "getUserName" → ["get", "User", "Name"]
            return SplitCamelCase(identifier);
        }

        // Handle snake_case
        if (IsSnakeCase(identifier))
        {
            // "get_user_name" → ["get", "user", "name"]
            return identifier.Split('_');
        }

        // Handle kebab-case
        if (IsKebabCase(identifier))
        {
            // "get-user-name" → ["get", "user", "name"]
            return identifier.Split('-');
        }

        // Single word or unknown pattern
        return new[] { identifier };
    }

    private string[] SplitCamelCase(string text)
    {
        var parts = new List<string>();
        var current = new StringBuilder();

        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];

            if (char.IsUpper(c) && current.Length > 0)
            {
                // Start new word on uppercase
                parts.Add(current.ToString());
                current.Clear();
            }

            current.Append(c);
        }

        if (current.Length > 0)
            parts.Add(current.ToString());

        return parts.ToArray();
    }
}
```

**Worked Example:**

```python
# Input code
def get_user_name(user_id):
    """Get username from user ID"""
    return database.query(f"SELECT name FROM users WHERE id={user_id}")

# Step 1: AST Parse
AST:
- FunctionDef
  - name: "get_user_name"
  - args: ["user_id"]
  - body:
    - Docstring: "Get username from user ID"
    - Return
      - Call: database.query
        - Arg: f"SELECT name FROM users WHERE id={user_id}"

# Step 2: Tokenization
Keyword: "def" → [KW_DEF]
Identifier: "get_user_name" → split to ["get", "user", "name"]
  → encode ["get", "user", "name"] → [1234, 5678, 9012]
Operator: "(" → [OP_LPAREN]
Identifier: "user_id" → ["user", "id"] → [5678, 3456]
Operator: ")" → [OP_RPAREN]
Operator: ":" → [OP_COLON]
Indentation: 4 spaces → [INDENT_1]
Comment: "Get username from user ID"
  → encode as text → [get, username, from, user, id]
  → [7890, 2345, 6789, 5678, 3456]
Keyword: "return" → [KW_RETURN]
Identifier: "database" → [8901]
Operator: "." → [OP_DOT]
Identifier: "query" → [4567]
...

# Final token sequence
[KW_DEF, 1234, 5678, 9012, OP_LPAREN, 5678, 3456, OP_RPAREN,
 OP_COLON, NEWLINE, INDENT_1, COMMENT_START, 7890, 2345, 6789,
 5678, 3456, COMMENT_END, NEWLINE, INDENT_1, KW_RETURN, ...]
```

**Comment handling strategies:**

```csharp
public class CommentHandler
{
    public string[] TokenizeComment(string comment, string language)
    {
        // Step 1: Strip comment markers
        comment = StripMarkers(comment, language);
        // "# This is a comment" → "This is a comment"
        // "// This is a comment" → "This is a comment"
        // "/* Multi-line */" → "Multi-line"

        // Step 2: Check if it's a docstring vs inline comment
        if (IsDocstring(comment))
        {
            // Parse structured documentation
            return TokenizeDocstring(comment);
        }
        else
        {
            // Treat as natural language
            return _baseTokenizer.Encode(comment);
        }
    }

    private string[] TokenizeDocstring(string docstring)
    {
        // Example: Python docstring
        // """
        // Get user by ID.
        //
        // Args:
        //     user_id: The user's ID
        //
        // Returns:
        //     User object
        // """

        var tokens = new List<string>();
        tokens.Add("[DOCSTRING_START]");

        // Parse sections
        var sections = ParseDocstringSections(docstring);

        foreach (var (sectionType, content) in sections)
        {
            tokens.Add($"[DOC_{sectionType.ToUpper()}]");
            tokens.AddRange(_baseTokenizer.Encode(content));
        }

        tokens.Add("[DOCSTRING_END]");
        return tokens.ToArray();
    }
}
```

### 6. Vocabulary Serialization

**What it is:** Saving and loading tokenizer vocabulary and merge rules.

**File formats:**

**1. JSON format (simple, human-readable):**

```json
{
  "version": "1.0",
  "type": "BPE",
  "vocab_size": 50000,
  "special_tokens": {
    "[PAD]": 0,
    "[UNK]": 1,
    "[CLS]": 2,
    "[SEP]": 3,
    "[MASK]": 4
  },
  "vocabulary": {
    "a": 5,
    "b": 6,
    "the": 100,
    "##ing": 500,
    ...
  },
  "merges": [
    ["l", "o", "lo"],
    ["lo", "w", "low"],
    ["e", "r", "er"],
    ...
  ]
}
```

**2. Binary format (fast loading, compact):**

```
File structure:
[Header: 16 bytes]
  - Magic number: "AIDNTKN\0" (8 bytes)
  - Version: uint32 (4 bytes)
  - Type: uint32 (4 bytes) // 0=BPE, 1=WordPiece, 2=SentencePiece

[Vocabulary section]
  - Count: uint32 (4 bytes)
  - For each token:
    - Length: uint32 (4 bytes)
    - Token: UTF-8 bytes (variable)
    - ID: uint32 (4 bytes)

[Merge rules section] (BPE only)
  - Count: uint32 (4 bytes)
  - For each merge:
    - Token1 length: uint32
    - Token1: UTF-8 bytes
    - Token2 length: uint32
    - Token2: UTF-8 bytes
    - Merged length: uint32
    - Merged: UTF-8 bytes

[Probabilities section] (SentencePiece only)
  - Count: uint32 (4 bytes)
  - For each token:
    - Token ID: uint32
    - Probability: double (8 bytes)
```

**Implementation:**

```csharp
public interface ITokenizerSerializer
{
    void Save(ITokenizer tokenizer, string path);
    ITokenizer Load(string path);
}

public class JsonTokenizerSerializer : ITokenizerSerializer
{
    public void Save(ITokenizer tokenizer, string path)
    {
        var data = new
        {
            version = "1.0",
            type = tokenizer.GetType().Name,
            vocab_size = tokenizer.VocabularySize,
            special_tokens = tokenizer.SpecialTokens,
            vocabulary = tokenizer.Vocabulary,
            merges = (tokenizer as BpeTokenizer)?.MergeRules,
            probabilities = (tokenizer as SentencePieceTokenizer)?.TokenProbabilities
        };

        var json = JsonSerializer.Serialize(data, new JsonSerializerOptions
        {
            WriteIndented = true,
            Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping
        });

        File.WriteAllText(path, json);
    }

    public ITokenizer Load(string path)
    {
        var json = File.ReadAllText(path);
        var data = JsonSerializer.Deserialize<TokenizerData>(json);

        return data.type switch
        {
            "BpeTokenizer" => LoadBpe(data),
            "WordPieceTokenizer" => LoadWordPiece(data),
            "SentencePieceTokenizer" => LoadSentencePiece(data),
            _ => throw new NotSupportedException($"Unknown tokenizer type: {data.type}")
        };
    }
}

public class BinaryTokenizerSerializer : ITokenizerSerializer
{
    private const ulong MAGIC_NUMBER = 0x004E4B544E444941; // "AIDNTKN\0"

    public void Save(ITokenizer tokenizer, string path)
    {
        using var fs = new FileStream(path, FileMode.Create);
        using var writer = new BinaryWriter(fs);

        // Header
        writer.Write(MAGIC_NUMBER);
        writer.Write((uint)1); // Version
        writer.Write((uint)GetTokenizerType(tokenizer));

        // Vocabulary
        writer.Write((uint)tokenizer.Vocabulary.Count);
        foreach (var (token, id) in tokenizer.Vocabulary)
        {
            var bytes = Encoding.UTF8.GetBytes(token);
            writer.Write((uint)bytes.Length);
            writer.Write(bytes);
            writer.Write((uint)id);
        }

        // Type-specific data
        if (tokenizer is BpeTokenizer bpe)
        {
            writer.Write((uint)bpe.MergeRules.Count);
            foreach (var (token1, token2, merged) in bpe.MergeRules)
            {
                WriteString(writer, token1);
                WriteString(writer, token2);
                WriteString(writer, merged);
            }
        }
        else if (tokenizer is SentencePieceTokenizer sp)
        {
            writer.Write((uint)sp.TokenProbabilities.Count);
            foreach (var (token, prob) in sp.TokenProbabilities)
            {
                writer.Write((uint)sp.Vocabulary[token]);
                writer.Write(prob);
            }
        }
    }

    public ITokenizer Load(string path)
    {
        using var fs = new FileStream(path, FileMode.Open);
        using var reader = new BinaryReader(fs);

        // Read header
        ulong magic = reader.ReadUInt64();
        if (magic != MAGIC_NUMBER)
            throw new InvalidDataException("Invalid tokenizer file");

        uint version = reader.ReadUInt32();
        uint type = reader.ReadUInt32();

        // Read vocabulary
        uint vocabCount = reader.ReadUInt32();
        var vocabulary = new Dictionary<string, int>();

        for (int i = 0; i < vocabCount; i++)
        {
            string token = ReadString(reader);
            int id = (int)reader.ReadUInt32();
            vocabulary[token] = id;
        }

        // Read type-specific data
        return type switch
        {
            0 => LoadBpeBinary(reader, vocabulary),
            1 => LoadWordPieceBinary(reader, vocabulary),
            2 => LoadSentencePieceBinary(reader, vocabulary),
            _ => throw new NotSupportedException($"Unknown tokenizer type: {type}")
        };
    }

    private void WriteString(BinaryWriter writer, string str)
    {
        var bytes = Encoding.UTF8.GetBytes(str);
        writer.Write((uint)bytes.Length);
        writer.Write(bytes);
    }

    private string ReadString(BinaryReader reader)
    {
        uint length = reader.ReadUInt32();
        var bytes = reader.ReadBytes((int)length);
        return Encoding.UTF8.GetString(bytes);
    }
}
```

### 7. Fast C# Implementation Strategies

**Performance considerations:**

1. **Use spans and memory for zero-copy operations**
2. **Avoid string allocations in hot paths**
3. **Use value types for tokens**
4. **Parallel processing for batch encoding**
5. **Memory-mapped files for large vocabularies**

**Optimized implementation:**

```csharp
public class OptimizedBpeTokenizer
{
    // Use arrays instead of dictionaries for O(1) lookup
    private int[] _vocabIds; // Maps token hash to ID
    private string[] _vocabTokens; // Maps ID to token
    private (int, int, int)[] _mergeRules; // (token1_id, token2_id, merged_id)

    // Use Span<char> to avoid allocations
    public int[] EncodeOptimized(ReadOnlySpan<char> text)
    {
        // Step 1: Pre-tokenize without allocating strings
        Span<Range> wordRanges = stackalloc Range[1024]; // Most texts < 1024 words
        int wordCount = SplitIntoWords(text, wordRanges);

        // Step 2: Encode each word
        var result = new List<int>(text.Length); // Pre-allocate

        for (int i = 0; i < wordCount; i++)
        {
            var word = text[wordRanges[i]];
            EncodeWord(word, result);
        }

        return result.ToArray();
    }

    private void EncodeWord(ReadOnlySpan<char> word, List<int> output)
    {
        // Use stack allocation for small words
        Span<int> symbols = word.Length <= 256
            ? stackalloc int[word.Length]
            : new int[word.Length];

        // Initialize with character IDs
        for (int i = 0; i < word.Length; i++)
        {
            symbols[i] = GetCharId(word[i]);
        }

        int length = word.Length;

        // Apply merge rules
        foreach (var (token1, token2, merged) in _mergeRules)
        {
            for (int i = 0; i < length - 1; i++)
            {
                if (symbols[i] == token1 && symbols[i + 1] == token2)
                {
                    // Merge in-place
                    symbols[i] = merged;

                    // Shift remaining symbols
                    symbols[(i + 1)..length].CopyTo(symbols[i..]);
                    length--;
                }
            }
        }

        // Copy to output
        for (int i = 0; i < length; i++)
        {
            output.Add(symbols[i]);
        }
    }

    // Parallel batch encoding
    public int[][] EncodeBatch(string[] texts)
    {
        return texts.AsParallel()
                   .AsOrdered()
                   .Select(text => EncodeOptimized(text.AsSpan()))
                   .ToArray();
    }

    // Memory-mapped vocabulary for huge vocabs (100k+ tokens)
    public class MemoryMappedVocabulary
    {
        private MemoryMappedFile _mmf;
        private MemoryMappedViewAccessor _accessor;

        public MemoryMappedVocabulary(string vocabFilePath)
        {
            _mmf = MemoryMappedFile.CreateFromFile(vocabFilePath);
            _accessor = _mmf.CreateViewAccessor();
        }

        public int GetTokenId(ReadOnlySpan<char> token)
        {
            // Hash-based lookup in memory-mapped file
            ulong hash = ComputeHash(token);
            long offset = (long)(hash % (ulong)_accessor.Capacity);

            // Read from memory-mapped file (no deserialization overhead)
            return _accessor.ReadInt32(offset);
        }

        private ulong ComputeHash(ReadOnlySpan<char> text)
        {
            // Fast hash function (xxHash, MurmurHash, etc.)
            ulong hash = 0xcbf29ce484222325;
            foreach (char c in text)
            {
                hash ^= c;
                hash *= 0x100000001b3;
            }
            return hash;
        }
    }

    // SIMD optimizations for character classification
    public unsafe bool IsWhitespace(ReadOnlySpan<char> text)
    {
        // Use Vector<T> for SIMD operations
        if (Vector.IsHardwareAccelerated && text.Length >= Vector<ushort>.Count)
        {
            var spaceVector = new Vector<ushort>(' ');
            var tabVector = new Vector<ushort>('\t');
            var newlineVector = new Vector<ushort>('\n');

            fixed (char* ptr = text)
            {
                for (int i = 0; i <= text.Length - Vector<ushort>.Count; i += Vector<ushort>.Count)
                {
                    var chunk = Vector.Load((ushort*)(ptr + i));

                    if (Vector.EqualsAny(chunk, spaceVector) ||
                        Vector.EqualsAny(chunk, tabVector) ||
                        Vector.EqualsAny(chunk, newlineVector))
                    {
                        return true;
                    }
                }
            }
        }

        // Fallback for remainder
        foreach (char c in text)
        {
            if (char.IsWhiteSpace(c))
                return true;
        }

        return false;
    }
}
```

---

## Architecture Overview

### File Organization

```
AiDotNet/
├── src/
│   ├── Interfaces/
│   │   ├── ITokenizer.cs              // Base tokenizer interface
│   │   ├── ITokenizerTrainer.cs       // Training interface
│   │   ├── ITokenizerSerializer.cs    // Serialization interface
│   │   └── IVocabulary.cs             // Vocabulary interface
│   │
│   ├── NLP/
│   │   ├── Tokenization/
│   │   │   ├── Base/
│   │   │   │   ├── TokenizerBase.cs           // Common functionality
│   │   │   │   ├── VocabularyBase.cs          // Vocabulary management
│   │   │   │   └── SpecialTokens.cs           // Special token definitions
│   │   │   │
│   │   │   ├── BPE/
│   │   │   │   ├── BpeTokenizer.cs            // BPE implementation
│   │   │   │   ├── BpeTrainer.cs              // BPE training
│   │   │   │   └── BpeMergeRule.cs            // Merge rule struct
│   │   │   │
│   │   │   ├── WordPiece/
│   │   │   │   ├── WordPieceTokenizer.cs      // WordPiece implementation
│   │   │   │   ├── WordPieceTrainer.cs        // WordPiece training
│   │   │   │   └── WordPieceVocabulary.cs     // WordPiece vocabulary
│   │   │   │
│   │   │   ├── SentencePiece/
│   │   │   │   ├── SentencePieceTokenizer.cs  // SentencePiece implementation
│   │   │   │   ├── SentencePieceTrainer.cs    // Unigram LM training
│   │   │   │   └── UnigramLanguageModel.cs    // Unigram LM
│   │   │   │
│   │   │   ├── Code/
│   │   │   │   ├── CodeTokenizer.cs           // Code tokenization
│   │   │   │   ├── ASTParser.cs               // AST parsing wrapper
│   │   │   │   ├── IdentifierSplitter.cs      // camelCase/snake_case
│   │   │   │   └── CommentHandler.cs          // Comment processing
│   │   │   │
│   │   │   ├── Serialization/
│   │   │   │   ├── JsonTokenizerSerializer.cs // JSON format
│   │   │   │   ├── BinaryTokenizerSerializer.cs // Binary format
│   │   │   │   └── TokenizerFormat.cs         // Format definitions
│   │   │   │
│   │   │   └── Utilities/
│   │   │       ├── UnicodeNormalizer.cs       // Unicode normalization
│   │   │       ├── PreTokenizer.cs            // Pre-tokenization
│   │   │       └── LevenshteinDistance.cs     // Edit distance
│   │
│   └── Models/
│       └── Tokenization/
│           ├── TokenizerConfig.cs             // Configuration
│           ├── Token.cs                       // Token data structure
│           ├── TokenizedSequence.cs           // Tokenized output
│           └── VocabularyEntry.cs             // Vocabulary entry
│
└── tests/
    └── AiDotNet.Tests/
        └── NLP/
            └── Tokenization/
                ├── BpeTokenizerTests.cs       // BPE tests
                ├── WordPieceTokenizerTests.cs // WordPiece tests
                ├── SentencePieceTokenizerTests.cs // SentencePiece tests
                └── CodeTokenizerTests.cs      // Code tokenization tests
```

### Core Interfaces

```csharp
public interface ITokenizer
{
    // Vocabulary
    int VocabularySize { get; }
    IReadOnlyDictionary<string, int> Vocabulary { get; }
    IReadOnlyDictionary<string, int> SpecialTokens { get; }

    // Encoding
    int[] Encode(string text);
    int[] Encode(ReadOnlySpan<char> text);
    int[][] EncodeBatch(IEnumerable<string> texts);

    // Decoding
    string Decode(int[] ids);
    string Decode(ReadOnlySpan<int> ids);
    string[] DecodeBatch(int[][] ids);

    // Token operations
    string IdToToken(int id);
    int TokenToId(string token);
    bool ContainsToken(string token);
}

public interface ITokenizerTrainer
{
    void Train(IEnumerable<string> corpus, TokenizerConfig config);
    void Train(string corpusPath, TokenizerConfig config);
    TrainingStats GetTrainingStats();
}

public interface IVocabulary
{
    int Size { get; }
    void Add(string token, int? id = null);
    void Remove(string token);
    bool Contains(string token);
    int GetId(string token);
    string GetToken(int id);
    void Save(string path);
    void Load(string path);
}

public class TokenizerConfig
{
    public int VocabularySize { get; set; } = 32000;
    public int MinFrequency { get; set; } = 2;
    public bool LowercaseInput { get; set; } = false;
    public bool StripAccents { get; set; } = false;
    public Dictionary<string, int> SpecialTokens { get; set; } = new();
    public string UnknownToken { get; set; } = "[UNK]";
    public string PaddingToken { get; set; } = "[PAD]";
    public bool AddPrefixSpace { get; set; } = false;
    public string ContinuingSubwordPrefix { get; set; } = "##";
}
```

---

## Implementation Steps

### Step 1: Set up base infrastructure (3-4 hours)

```csharp
// 1.1: Create ITokenizer interface
namespace AiDotNet.Interfaces
{
    public interface ITokenizer
    {
        int VocabularySize { get; }
        IReadOnlyDictionary<string, int> Vocabulary { get; }

        int[] Encode(string text);
        string Decode(int[] ids);
    }
}

// 1.2: Create TokenizerBase
namespace AiDotNet.NLP.Tokenization.Base
{
    public abstract class TokenizerBase : ITokenizer
    {
        protected Dictionary<string, int> _vocabulary;
        protected Dictionary<int, string> _reverseVocabulary;
        protected Dictionary<string, int> _specialTokens;

        public int VocabularySize => _vocabulary.Count;
        public IReadOnlyDictionary<string, int> Vocabulary => _vocabulary;

        protected TokenizerBase()
        {
            _vocabulary = new Dictionary<string, int>();
            _reverseVocabulary = new Dictionary<int, string>();
            _specialTokens = new Dictionary<string, int>();

            InitializeSpecialTokens();
        }

        protected virtual void InitializeSpecialTokens()
        {
            AddSpecialToken("[PAD]", 0);
            AddSpecialToken("[UNK]", 1);
            AddSpecialToken("[CLS]", 2);
            AddSpecialToken("[SEP]", 3);
            AddSpecialToken("[MASK]", 4);
        }

        protected void AddSpecialToken(string token, int id)
        {
            _vocabulary[token] = id;
            _reverseVocabulary[id] = token;
            _specialTokens[token] = id;
        }

        public abstract int[] Encode(string text);
        public abstract string Decode(int[] ids);

        public string IdToToken(int id)
        {
            return _reverseVocabulary.TryGetValue(id, out var token)
                ? token
                : _specialTokens.First(kvp => kvp.Value == 1).Key; // [UNK]
        }

        public int TokenToId(string token)
        {
            return _vocabulary.TryGetValue(token, out var id)
                ? id
                : _specialTokens["[UNK]"];
        }
    }
}
```

### Step 2: Implement BPE (8-10 hours)

```csharp
// 2.1: BpeMergeRule structure
public readonly struct BpeMergeRule
{
    public string Token1 { get; }
    public string Token2 { get; }
    public string Merged { get; }
    public int Priority { get; }

    public BpeMergeRule(string token1, string token2, string merged, int priority)
    {
        Token1 = token1;
        Token2 = token2;
        Merged = merged;
        Priority = priority;
    }
}

// 2.2: BpeTokenizer implementation
public class BpeTokenizer : TokenizerBase
{
    private List<BpeMergeRule> _mergeRules;
    private string _endOfWordToken = "</w>";

    public BpeTokenizer()
    {
        _mergeRules = new List<BpeMergeRule>();
    }

    public override int[] Encode(string text)
    {
        var words = PreTokenize(text);
        var result = new List<int>();

        foreach (var word in words)
        {
            var tokens = EncodeWord(word);
            result.AddRange(tokens.Select(TokenToId));
        }

        return result.ToArray();
    }

    private string[] EncodeWord(string word)
    {
        // Add end-of-word marker
        var symbols = word.Select(c => c.ToString()).ToList();
        symbols.Add(_endOfWordToken);

        // Apply merge rules in order
        foreach (var rule in _mergeRules)
        {
            for (int i = 0; i < symbols.Count - 1; i++)
            {
                if (symbols[i] == rule.Token1 && symbols[i + 1] == rule.Token2)
                {
                    symbols[i] = rule.Merged;
                    symbols.RemoveAt(i + 1);
                    i--; // Re-check current position
                }
            }
        }

        return symbols.ToArray();
    }

    public override string Decode(int[] ids)
    {
        var tokens = ids.Select(IdToToken).ToArray();
        var text = string.Join("", tokens);

        // Remove end-of-word markers
        text = text.Replace(_endOfWordToken, " ");

        return text.Trim();
    }

    private string[] PreTokenize(string text)
    {
        // Simple whitespace tokenization
        // In production, use regex for better handling
        return text.Split(new[] { ' ', '\t', '\n', '\r' },
            StringSplitOptions.RemoveEmptyEntries);
    }
}

// 2.3: BpeTrainer implementation
public class BpeTrainer : ITokenizerTrainer
{
    public void Train(IEnumerable<string> corpus, TokenizerConfig config)
    {
        // Step 1: Count word frequencies
        var wordFreqs = new Dictionary<string, int>();
        foreach (var line in corpus)
        {
            var words = line.Split(' ');
            foreach (var word in words)
            {
                wordFreqs[word] = wordFreqs.GetValueOrDefault(word, 0) + 1;
            }
        }

        // Step 2: Initialize with character vocabulary
        var vocabulary = new HashSet<string>();
        foreach (var word in wordFreqs.Keys)
        {
            foreach (var c in word)
            {
                vocabulary.Add(c.ToString());
            }
        }
        vocabulary.Add("</w>"); // End-of-word marker

        // Step 3: Represent words as character sequences
        var wordReprs = wordFreqs.ToDictionary(
            kvp => kvp.Key,
            kvp => string.Join(" ", kvp.Key.Select(c => c.ToString())) + " </w>"
        );

        var mergeRules = new List<BpeMergeRule>();
        int priority = 0;

        // Step 4: Iteratively merge
        while (vocabulary.Count < config.VocabularySize)
        {
            var pairFreqs = CountPairFrequencies(wordReprs, wordFreqs);
            if (pairFreqs.Count == 0) break;

            var mostFrequent = pairFreqs.OrderByDescending(p => p.Value).First();
            var (token1, token2) = mostFrequent.Key;
            var merged = token1 + token2;

            // Update word representations
            foreach (var word in wordReprs.Keys.ToList())
            {
                wordReprs[word] = wordReprs[word].Replace(
                    $"{token1} {token2}",
                    merged);
            }

            vocabulary.Add(merged);
            mergeRules.Add(new BpeMergeRule(token1, token2, merged, priority++));
        }

        // Save to tokenizer
        // (Implementation continues with saving vocabulary and merge rules)
    }

    private Dictionary<(string, string), int> CountPairFrequencies(
        Dictionary<string, string> wordReprs,
        Dictionary<string, int> wordFreqs)
    {
        var pairFreqs = new Dictionary<(string, string), int>();

        foreach (var (word, repr) in wordReprs)
        {
            var symbols = repr.Split(' ');
            var freq = wordFreqs[word];

            for (int i = 0; i < symbols.Length - 1; i++)
            {
                var pair = (symbols[i], symbols[i + 1]);
                pairFreqs[pair] = pairFreqs.GetValueOrDefault(pair, 0) + freq;
            }
        }

        return pairFreqs;
    }
}
```

### Step 3: Implement WordPiece (8-10 hours)

See implementation in Background Concepts section above.

### Step 4: Implement SentencePiece (10-12 hours)

See implementation in Background Concepts section above.

### Step 5: Implement Code Tokenizer (6-8 hours)

See implementation in Background Concepts section above.

### Step 6: Implement Serialization (4-5 hours)

See implementation in Background Concepts section above.

### Step 7: Performance Optimization (4-6 hours)

See Fast C# Implementation Strategies in Background Concepts section above.

---

## Testing Strategy

### Unit Tests

```csharp
[TestClass]
public class BpeTokenizerTests
{
    [TestMethod]
    public void Encode_SimpleText_ReturnsCorrectTokens()
    {
        // Arrange
        var tokenizer = new BpeTokenizer();
        TrainSimpleBpe(tokenizer);

        // Act
        var result = tokenizer.Encode("low lower");

        // Assert
        Assert.IsNotNull(result);
        Assert.IsTrue(result.Length > 0);

        // Verify specific tokens
        var decoded = tokenizer.Decode(result);
        Assert.AreEqual("low lower", decoded.Trim());
    }

    [TestMethod]
    public void Train_WithSmallCorpus_CreatesVocabulary()
    {
        // Arrange
        var corpus = new[] { "low low low", "lower lower", "lowest" };
        var config = new TokenizerConfig { VocabularySize = 20 };
        var trainer = new BpeTrainer();

        // Act
        trainer.Train(corpus, config);

        // Assert
        // Verify merge rules were created
        // Verify vocabulary size is correct
    }

    [TestMethod]
    public void Encode_UnknownWord_ReturnsSubwordTokens()
    {
        // Arrange
        var tokenizer = new BpeTokenizer();
        TrainSimpleBpe(tokenizer);

        // Act
        var result = tokenizer.Encode("supercalifragilisticexpialidocious");

        // Assert
        Assert.IsNotNull(result);
        Assert.IsTrue(result.Length > 1); // Should be split into subwords
        Assert.IsFalse(result.Contains(tokenizer.TokenToId("[UNK]"))); // No UNK if trained on characters
    }

    [TestMethod]
    public void EncodeDecode_RoundTrip_PreservesText()
    {
        // Arrange
        var tokenizer = new BpeTokenizer();
        TrainSimpleBpe(tokenizer);
        var original = "the quick brown fox jumps over the lazy dog";

        // Act
        var encoded = tokenizer.Encode(original);
        var decoded = tokenizer.Decode(encoded);

        // Assert
        Assert.AreEqual(original, decoded.Trim());
    }
}

[TestClass]
public class WordPieceTokenizerTests
{
    [TestMethod]
    public void Encode_WithPrefixMarkers_AddsDoubleHash()
    {
        // Arrange
        var tokenizer = new WordPieceTokenizer();
        TrainSimpleWordPiece(tokenizer);

        // Act
        var result = tokenizer.Encode("playing");
        var tokens = result.Select(tokenizer.IdToToken).ToArray();

        // Assert
        Assert.IsTrue(tokens.Any(t => t.StartsWith("##")));
    }

    [TestMethod]
    public void Encode_SpecialTokens_PreservesTokens()
    {
        // Arrange
        var tokenizer = new WordPieceTokenizer();
        var text = "[CLS] hello world [SEP]";

        // Act
        var result = tokenizer.Encode(text);
        var tokens = result.Select(tokenizer.IdToToken).ToArray();

        // Assert
        Assert.AreEqual("[CLS]", tokens[0]);
        Assert.AreEqual("[SEP]", tokens[tokens.Length - 1]);
    }
}

[TestClass]
public class SentencePieceTokenizerTests
{
    [TestMethod]
    public void Encode_JapaneseText_WorksWithoutPreTokenization()
    {
        // Arrange
        var tokenizer = new SentencePieceTokenizer();
        var text = "こんにちは世界";

        // Act
        var result = tokenizer.Encode(text);

        // Assert
        Assert.IsNotNull(result);
        Assert.IsTrue(result.Length > 0);
    }

    [TestMethod]
    public void EncodeWithRegularization_SameText_DifferentSegmentations()
    {
        // Arrange
        var tokenizer = new SentencePieceTokenizer();
        var text = "hello world";

        // Act
        var result1 = tokenizer.EncodeWithRegularization(text);
        var result2 = tokenizer.EncodeWithRegularization(text);

        // Assert
        // With regularization, segmentations may differ
        // (or be the same, but probabilities are sampled)
    }
}

[TestClass]
public class CodeTokenizerTests
{
    [TestMethod]
    public void Tokenize_PythonFunction_PreservesStructure()
    {
        // Arrange
        var tokenizer = new CodeTokenizer(new BpeTokenizer());
        var code = @"
def hello_world():
    print('Hello, World!')
";

        // Act
        var tokens = tokenizer.Tokenize(code, "python");

        // Assert
        Assert.IsTrue(tokens.Any(t => t.Type == TokenType.Keyword && t.Text == "def"));
        Assert.IsTrue(tokens.Any(t => t.Type == TokenType.Identifier));
        Assert.IsTrue(tokens.Any(t => t.Type == TokenType.StringLiteral));
    }

    [TestMethod]
    public void SplitIdentifier_CamelCase_SplitsCorrectly()
    {
        // Arrange
        var tokenizer = new CodeTokenizer(new BpeTokenizer());

        // Act
        var result = tokenizer.SplitIdentifier("getUserName");

        // Assert
        CollectionAssert.AreEqual(new[] { "get", "User", "Name" }, result);
    }

    [TestMethod]
    public void SplitIdentifier_SnakeCase_SplitsCorrectly()
    {
        // Arrange
        var tokenizer = new CodeTokenizer(new BpeTokenizer());

        // Act
        var result = tokenizer.SplitIdentifier("get_user_name");

        // Assert
        CollectionAssert.AreEqual(new[] { "get", "user", "name" }, result);
    }
}
```

### Integration Tests

```csharp
[TestClass]
public class TokenizationIntegrationTests
{
    [TestMethod]
    public void TrainAndSave_ThenLoad_ProducesSameResults()
    {
        // Arrange
        var corpus = LoadLargeCorpus();
        var config = new TokenizerConfig { VocabularySize = 32000 };
        var trainer = new BpeTrainer();
        var originalTokenizer = new BpeTokenizer();

        // Act - Train and save
        trainer.Train(corpus, config);
        var serializer = new BinaryTokenizerSerializer();
        serializer.Save(originalTokenizer, "test_tokenizer.bin");

        // Act - Load
        var loadedTokenizer = serializer.Load("test_tokenizer.bin");

        // Assert - Same results
        var testText = "the quick brown fox";
        var originalResult = originalTokenizer.Encode(testText);
        var loadedResult = loadedTokenizer.Encode(testText);

        CollectionAssert.AreEqual(originalResult, loadedResult);
    }

    [TestMethod]
    public void EncodeBatch_LargeDataset_CompletesInReasonableTime()
    {
        // Arrange
        var tokenizer = new BpeTokenizer();
        var texts = Enumerable.Range(0, 10000)
            .Select(i => $"This is test sentence number {i}")
            .ToArray();

        // Act
        var stopwatch = Stopwatch.StartNew();
        var results = tokenizer.EncodeBatch(texts);
        stopwatch.Stop();

        // Assert
        Assert.AreEqual(10000, results.Length);
        Assert.IsTrue(stopwatch.ElapsedMilliseconds < 5000); // Should be under 5 seconds
    }
}
```

### Benchmark Tests

```csharp
[TestClass]
public class TokenizationBenchmarks
{
    [TestMethod]
    public void Benchmark_BpeEncoding_Speed()
    {
        var tokenizer = new BpeTokenizer();
        var text = string.Join(" ", Enumerable.Repeat("hello world", 1000));

        var stopwatch = Stopwatch.StartNew();
        for (int i = 0; i < 100; i++)
        {
            tokenizer.Encode(text);
        }
        stopwatch.Stop();

        Console.WriteLine($"BPE encoding: {stopwatch.ElapsedMilliseconds}ms for 100 iterations");
        Assert.IsTrue(stopwatch.ElapsedMilliseconds < 1000); // Should be under 1 second
    }

    [TestMethod]
    public void Benchmark_CompareBpeVsWordPiece()
    {
        var bpe = new BpeTokenizer();
        var wordPiece = new WordPieceTokenizer();
        var text = LoadLargeText();

        var bpeTime = Measure(() => bpe.Encode(text));
        var wordPieceTime = Measure(() => wordPiece.Encode(text));

        Console.WriteLine($"BPE: {bpeTime}ms, WordPiece: {wordPieceTime}ms");
    }
}
```

---

## Common Pitfalls

### 1. Unicode Handling

**Problem:** Not handling Unicode characters correctly (emojis, accents, multi-byte characters).

**Solution:**
```csharp
// Use StringInfo for correct grapheme handling
using System.Globalization;

public string[] SplitGraphemes(string text)
{
    var enumerator = StringInfo.GetTextElementEnumerator(text);
    var result = new List<string>();

    while (enumerator.MoveNext())
    {
        result.Add(enumerator.GetTextElement());
    }

    return result.ToArray();
}

// Example: "👨‍👩‍👧‍👦" is one grapheme (family emoji), not 7 characters
```

### 2. Out-of-Memory Errors During Training

**Problem:** Training on large corpora causes memory issues.

**Solution:**
```csharp
// Stream corpus instead of loading all at once
public void Train(string corpusPath, TokenizerConfig config)
{
    // Don't do this:
    // var corpus = File.ReadAllLines(corpusPath); // Loads entire file!

    // Do this:
    using var reader = new StreamReader(corpusPath);
    var wordFreqs = new Dictionary<string, int>();

    string line;
    while ((line = reader.ReadLine()) != null)
    {
        ProcessLine(line, wordFreqs);

        // Periodically checkpoint to disk
        if (wordFreqs.Count > 10_000_000)
        {
            FlushToDisk(wordFreqs);
            wordFreqs.Clear();
        }
    }
}
```

### 3. Incorrect Merge Order

**Problem:** Applying BPE merges in wrong order produces incorrect tokenization.

**Solution:**
```csharp
// Merge rules MUST be applied in the order they were learned
public class BpeTokenizer
{
    private List<BpeMergeRule> _mergeRules; // Ordered list

    private string[] EncodeWord(string word)
    {
        var symbols = word.Select(c => c.ToString()).ToList();

        // MUST iterate in order of learning (priority)
        foreach (var rule in _mergeRules.OrderBy(r => r.Priority))
        {
            ApplyMerge(symbols, rule);
        }

        return symbols.ToArray();
    }
}
```

### 4. Special Tokens Not Reserved

**Problem:** Special tokens like [CLS], [SEP] get split during tokenization.

**Solution:**
```csharp
public int[] Encode(string text)
{
    // Extract special tokens BEFORE tokenization
    var (cleanedText, specialTokenPositions) = ExtractSpecialTokens(text);

    // Tokenize normal text
    var tokens = EncodeInternal(cleanedText);

    // Re-insert special tokens at correct positions
    return InsertSpecialTokens(tokens, specialTokenPositions);
}

private (string, List<(int, string)>) ExtractSpecialTokens(string text)
{
    var positions = new List<(int, string)>();
    var pattern = @"\[(CLS|SEP|MASK|PAD|UNK)\]";

    var matches = Regex.Matches(text, pattern);
    foreach (Match match in matches)
    {
        positions.Add((match.Index, match.Value));
    }

    var cleaned = Regex.Replace(text, pattern, "");
    return (cleaned, positions);
}
```

### 5. Not Handling Empty Strings

**Problem:** Edge cases with empty input cause crashes.

**Solution:**
```csharp
public int[] Encode(string text)
{
    // Handle null/empty
    if (string.IsNullOrEmpty(text))
        return Array.Empty<int>();

    // Handle whitespace-only
    if (string.IsNullOrWhiteSpace(text))
        return new[] { _specialTokens["[PAD]"] };

    // Continue with normal encoding...
}
```

### 6. Vocabulary Size Not Matching Config

**Problem:** Final vocabulary size differs from requested size.

**Solution:**
```csharp
public void Train(IEnumerable<string> corpus, TokenizerConfig config)
{
    // Account for special tokens
    int targetSize = config.VocabularySize - _specialTokens.Count;

    while (_vocabulary.Count < targetSize)
    {
        // Training logic...

        // Stop if no more pairs to merge
        if (pairFreqs.Count == 0)
        {
            Console.WriteLine($"Warning: Reached {_vocabulary.Count} tokens, " +
                            $"target was {config.VocabularySize}");
            break;
        }
    }
}
```

---

## Resources

### Academic Papers

1. **BPE**: [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) (Sennrich et al., 2016)

2. **WordPiece**: [Japanese and Korean Voice Search](https://research.google/pubs/pub37842/) (Schuster & Nakajima, 2012)

3. **SentencePiece**: [SentencePiece: A simple and language independent approach to subword tokenization](https://arxiv.org/abs/1808.06226) (Kudo & Richardson, 2018)

4. **Code Tokenization**: [Unified Pre-training for Program Understanding and Generation](https://arxiv.org/abs/2103.06333) (Ahmad et al., 2021)

### Open Source Implementations

1. **Hugging Face Tokenizers**: [https://github.com/huggingface/tokenizers](https://github.com/huggingface/tokenizers)
   - Fast Rust implementation with Python bindings
   - Reference for optimization techniques

2. **SentencePiece**: [https://github.com/google/sentencepiece](https://github.com/google/sentencepiece)
   - Google's official implementation
   - C++ with Python bindings

3. **tiktoken**: [https://github.com/openai/tiktoken](https://github.com/openai/tiktoken)
   - OpenAI's BPE tokenizer (used in GPT models)
   - Rust core with Python bindings

### Existing AiDotNet Patterns

Look at these existing implementations for code style guidance:

1. **Optimizer implementations** (`src/Optimizers/`) for base class patterns
2. **Layer implementations** (`src/Layers/`) for interface design
3. **Loss function implementations** (`src/Losses/`) for validation patterns
4. **Test patterns** (`tests/AiDotNet.Tests/`) for unit test structure

### Tools

1. **Regex Tester**: [https://regex101.com/](https://regex101.com/) for pre-tokenization patterns
2. **Unicode Explorer**: [https://unicode-explorer.com/](https://unicode-explorer.com/) for Unicode handling
3. **BenchmarkDotNet**: For performance testing (already used in AiDotNet)

### Debugging Tips

1. **Visualize tokenization**:
```csharp
public void DebugTokenization(string text)
{
    var ids = Encode(text);
    var tokens = ids.Select(IdToToken).ToArray();

    Console.WriteLine($"Input: {text}");
    Console.WriteLine($"Tokens: [{string.Join(", ", tokens)}]");
    Console.WriteLine($"IDs: [{string.Join(", ", ids)}]");
    Console.WriteLine($"Decoded: {Decode(ids)}");
}
```

2. **Log merge operations**:
```csharp
private void ApplyMerge(List<string> symbols, BpeMergeRule rule)
{
    #if DEBUG
    var before = string.Join(" ", symbols);
    #endif

    // Apply merge...

    #if DEBUG
    var after = string.Join(" ", symbols);
    Console.WriteLine($"Merge: {rule.Token1} + {rule.Token2} → {rule.Merged}");
    Console.WriteLine($"Before: {before}");
    Console.WriteLine($"After: {after}");
    #endif
}
```

3. **Validate round-trip**:
```csharp
[Conditional("DEBUG")]
private void ValidateRoundTrip(string original, int[] encoded, string decoded)
{
    if (original != decoded)
    {
        Console.WriteLine($"Round-trip failed!");
        Console.WriteLine($"Original: {original}");
        Console.WriteLine($"Decoded:  {decoded}");
        Console.WriteLine($"IDs: [{string.Join(", ", encoded)}]");
        Debug.Assert(false);
    }
}
```

---

## Summary

This tokenization infrastructure is **critical** for all NLP work in AiDotNet. Take your time to:

1. Understand the algorithms (BPE, WordPiece, SentencePiece)
2. Implement carefully with proper testing
3. Optimize for performance (this will be used heavily)
4. Document thoroughly (others will build on this)

**Estimated timeline:**
- Week 1: Base infrastructure + BPE (12-15 hours)
- Week 2: WordPiece + SentencePiece (18-22 hours)
- Week 3: Code tokenization + serialization (10-13 hours)
- Week 4: Testing + optimization (8-10 hours)

**Total: 48-60 hours (6-8 weeks part-time)**

Good luck! This is foundational work that will enable all transformer models, language models, and NLP tasks in AiDotNet.

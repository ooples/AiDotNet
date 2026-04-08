using AiDotNet.Preprocessing.TextVectorizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

public class TextVectorizersDeepMathIntegrationTests
{
    private const double Tolerance = 1e-6;
    private const double LooseTolerance = 1e-3;

    // =====================================================================
    // CountVectorizer: bag-of-words (term counts per document)
    // =====================================================================

    [Fact]
    public void CountVectorizer_BasicCounts_HandComputed()
    {
        var docs = new[] { "cat dog cat", "dog bird", "cat bird bird" };
        var vectorizer = new CountVectorizer<double>();
        var result = vectorizer.FitTransform(docs);

        Assert.NotNull(vectorizer.Vocabulary);
        // Vocabulary should contain: bird, cat, dog (alphabetically sorted)
        Assert.Equal(3, vectorizer.Vocabulary.Count);
        Assert.True(vectorizer.Vocabulary.ContainsKey("bird"));
        Assert.True(vectorizer.Vocabulary.ContainsKey("cat"));
        Assert.True(vectorizer.Vocabulary.ContainsKey("dog"));

        int birdIdx = vectorizer.Vocabulary["bird"];
        int catIdx = vectorizer.Vocabulary["cat"];
        int dogIdx = vectorizer.Vocabulary["dog"];

        // Doc 0: "cat dog cat" → cat:2, dog:1, bird:0
        Assert.Equal(0.0, result[0, birdIdx], Tolerance);
        Assert.Equal(2.0, result[0, catIdx], Tolerance);
        Assert.Equal(1.0, result[0, dogIdx], Tolerance);

        // Doc 1: "dog bird" → cat:0, dog:1, bird:1
        Assert.Equal(1.0, result[1, birdIdx], Tolerance);
        Assert.Equal(0.0, result[1, catIdx], Tolerance);
        Assert.Equal(1.0, result[1, dogIdx], Tolerance);

        // Doc 2: "cat bird bird" → cat:1, dog:0, bird:2
        Assert.Equal(2.0, result[2, birdIdx], Tolerance);
        Assert.Equal(1.0, result[2, catIdx], Tolerance);
        Assert.Equal(0.0, result[2, dogIdx], Tolerance);
    }

    [Fact]
    public void CountVectorizer_Binary_CapsAtOne()
    {
        var docs = new[] { "cat cat cat dog", "bird bird" };
        var vectorizer = new CountVectorizer<double>(binary: true);
        var result = vectorizer.FitTransform(docs);

        Assert.NotNull(vectorizer.Vocabulary);
        int catIdx = vectorizer.Vocabulary["cat"];
        int dogIdx = vectorizer.Vocabulary["dog"];
        int birdIdx = vectorizer.Vocabulary["bird"];

        // Binary mode: 1 if present, 0 if absent
        Assert.Equal(1.0, result[0, catIdx], Tolerance); // cat appears 3 times but binary = 1
        Assert.Equal(1.0, result[0, dogIdx], Tolerance);
        Assert.Equal(0.0, result[0, birdIdx], Tolerance);
        Assert.Equal(0.0, result[1, catIdx], Tolerance);
        Assert.Equal(1.0, result[1, birdIdx], Tolerance);
    }

    [Fact]
    public void CountVectorizer_Lowercase_MergesCase()
    {
        var docs = new[] { "Cat CAT cat", "DOG Dog" };
        var vectorizer = new CountVectorizer<double>(lowercase: true);
        var result = vectorizer.FitTransform(docs);

        Assert.NotNull(vectorizer.Vocabulary);
        // All should be lowercased
        Assert.True(vectorizer.Vocabulary.ContainsKey("cat"));
        Assert.True(vectorizer.Vocabulary.ContainsKey("dog"));
        Assert.False(vectorizer.Vocabulary.ContainsKey("Cat"));
        Assert.False(vectorizer.Vocabulary.ContainsKey("CAT"));

        int catIdx = vectorizer.Vocabulary["cat"];
        Assert.Equal(3.0, result[0, catIdx], Tolerance); // Cat + CAT + cat = 3
    }

    [Fact]
    public void CountVectorizer_StopWords_Filtered()
    {
        var stopWords = new HashSet<string> { "the", "a", "is" };
        var docs = new[] { "the cat is a cat", "a dog is the dog" };
        var vectorizer = new CountVectorizer<double>(stopWords: stopWords);
        var result = vectorizer.FitTransform(docs);

        Assert.NotNull(vectorizer.Vocabulary);
        // "the", "a", "is" should be filtered out
        Assert.False(vectorizer.Vocabulary.ContainsKey("the"));
        Assert.False(vectorizer.Vocabulary.ContainsKey("a"));
        Assert.False(vectorizer.Vocabulary.ContainsKey("is"));
        Assert.True(vectorizer.Vocabulary.ContainsKey("cat"));
        Assert.True(vectorizer.Vocabulary.ContainsKey("dog"));
    }

    [Fact]
    public void CountVectorizer_NGrams_BigramsGenerated()
    {
        var docs = new[] { "cat dog bird", "dog bird fish" };
        var vectorizer = new CountVectorizer<double>(nGramRange: (1, 2));
        var result = vectorizer.FitTransform(docs);

        Assert.NotNull(vectorizer.Vocabulary);
        // Should have unigrams and bigrams
        Assert.True(vectorizer.Vocabulary.ContainsKey("cat"));
        Assert.True(vectorizer.Vocabulary.ContainsKey("dog"));
        Assert.True(vectorizer.Vocabulary.ContainsKey("bird"));
        Assert.True(vectorizer.Vocabulary.ContainsKey("cat dog"));
        Assert.True(vectorizer.Vocabulary.ContainsKey("dog bird"));
    }

    [Fact]
    public void CountVectorizer_MaxFeatures_LimitsVocabulary()
    {
        var docs = new[] { "alpha beta gamma delta", "alpha beta gamma", "alpha beta" };
        var vectorizer = new CountVectorizer<double>(maxFeatures: 2);
        vectorizer.FitTransform(docs);

        Assert.NotNull(vectorizer.Vocabulary);
        Assert.Equal(2, vectorizer.Vocabulary.Count);
    }

    [Fact]
    public void CountVectorizer_MinDf_FiltersRareTerms()
    {
        var docs = new[] { "cat dog", "cat bird", "cat fish" };
        // minDf=2 means term must appear in at least 2 documents
        var vectorizer = new CountVectorizer<double>(minDf: 2);
        vectorizer.FitTransform(docs);

        Assert.NotNull(vectorizer.Vocabulary);
        // "cat" appears in all 3 docs, "dog", "bird", "fish" each in 1
        Assert.True(vectorizer.Vocabulary.ContainsKey("cat"));
        Assert.False(vectorizer.Vocabulary.ContainsKey("dog"));
        Assert.False(vectorizer.Vocabulary.ContainsKey("bird"));
        Assert.False(vectorizer.Vocabulary.ContainsKey("fish"));
    }

    [Fact]
    public void CountVectorizer_OutputDimensions_CorrectShape()
    {
        var docs = new[] { "a b c", "d e f", "a d" };
        var vectorizer = new CountVectorizer<double>();
        var result = vectorizer.FitTransform(docs);

        Assert.Equal(3, result.Rows);    // 3 documents
        Assert.Equal(6, result.Columns); // 6 unique words: a, b, c, d, e, f
    }

    // =====================================================================
    // TF-IDF Vectorizer
    // TF = count of term in doc (or 1+log(tf) if sublinearTf)
    // IDF smooth = log((N+1)/(df+1)) + 1
    // IDF standard = log(N/df) + 1
    // =====================================================================

    [Fact]
    public void Tfidf_SmoothIDF_HandComputed()
    {
        // 3 docs: term "cat" in all 3, "dog" in 2, "bird" in 1
        var docs = new[] { "cat dog", "cat dog bird", "cat" };
        var vectorizer = new TfidfVectorizer<double>(
            norm: TfidfNorm.None, useIdf: true, smoothIdf: true);
        var result = vectorizer.FitTransform(docs);

        Assert.NotNull(vectorizer.IdfWeights);
        Assert.NotNull(vectorizer.Vocabulary);

        // Smooth IDF formula: log((N+1)/(df+1)) + 1
        // N=3
        // cat: df=3, IDF = log(4/4) + 1 = 0 + 1 = 1.0
        // dog: df=2, IDF = log(4/3) + 1 ≈ 0.2877 + 1 = 1.2877
        // bird: df=1, IDF = log(4/2) + 1 = 0.6931 + 1 = 1.6931

        int catIdx = vectorizer.Vocabulary["cat"];
        int dogIdx = vectorizer.Vocabulary["dog"];
        int birdIdx = vectorizer.Vocabulary["bird"];

        double expectedCatIdf = Math.Log(4.0 / 4.0) + 1.0;
        double expectedDogIdf = Math.Log(4.0 / 3.0) + 1.0;
        double expectedBirdIdf = Math.Log(4.0 / 2.0) + 1.0;

        Assert.Equal(expectedCatIdf, vectorizer.IdfWeights[catIdx], Tolerance);
        Assert.Equal(expectedDogIdf, vectorizer.IdfWeights[dogIdx], Tolerance);
        Assert.Equal(expectedBirdIdf, vectorizer.IdfWeights[birdIdx], Tolerance);

        // TF-IDF (no norm) = TF * IDF
        // Doc 0 "cat dog": cat TF=1, dog TF=1, bird TF=0
        Assert.Equal(1.0 * expectedCatIdf, result[0, catIdx], Tolerance);
        Assert.Equal(1.0 * expectedDogIdf, result[0, dogIdx], Tolerance);
        Assert.Equal(0.0, result[0, birdIdx], Tolerance);
    }

    [Fact]
    public void Tfidf_StandardIDF_HandComputed()
    {
        var docs = new[] { "cat dog", "cat bird", "cat" };
        var vectorizer = new TfidfVectorizer<double>(
            norm: TfidfNorm.None, useIdf: true, smoothIdf: false);
        var result = vectorizer.FitTransform(docs);

        Assert.NotNull(vectorizer.IdfWeights);
        Assert.NotNull(vectorizer.Vocabulary);

        // Standard IDF: log(N/df) + 1
        // cat: df=3, IDF = log(3/3) + 1 = 0 + 1 = 1.0
        // dog: df=1, IDF = log(3/1) + 1 = 1.0986 + 1 = 2.0986
        // bird: df=1, IDF = log(3/1) + 1 = 2.0986

        int catIdx = vectorizer.Vocabulary["cat"];
        double expectedCatIdf = Math.Log(3.0 / 3.0) + 1.0;
        Assert.Equal(expectedCatIdf, vectorizer.IdfWeights[catIdx], Tolerance);
    }

    [Fact]
    public void Tfidf_SublinearTf_AppliesLogToTf()
    {
        // SublinearTf: TF = 1 + log(raw_tf) when raw_tf > 0
        var docs = new[] { "cat cat cat dog", "bird" };
        var vectorizer = new TfidfVectorizer<double>(
            norm: TfidfNorm.None, useIdf: false, sublinearTf: true);
        var result = vectorizer.FitTransform(docs);

        Assert.NotNull(vectorizer.Vocabulary);
        int catIdx = vectorizer.Vocabulary["cat"];
        int dogIdx = vectorizer.Vocabulary["dog"];

        // Doc 0: cat raw_tf=3, sublinear TF = 1 + log(3) ≈ 2.0986
        // Doc 0: dog raw_tf=1, sublinear TF = 1 + log(1) = 1.0
        double expectedCatTf = 1 + Math.Log(3);
        Assert.Equal(expectedCatTf, result[0, catIdx], Tolerance);
        Assert.Equal(1.0, result[0, dogIdx], Tolerance);
    }

    [Fact]
    public void Tfidf_L2Norm_UnitVectors()
    {
        var docs = new[] { "cat dog bird", "cat cat", "dog dog dog" };
        var vectorizer = new TfidfVectorizer<double>(
            norm: TfidfNorm.L2, useIdf: true, smoothIdf: true);
        var result = vectorizer.FitTransform(docs);

        // Each row should have L2 norm = 1
        for (int i = 0; i < result.Rows; i++)
        {
            double l2Norm = 0;
            for (int j = 0; j < result.Columns; j++)
            {
                l2Norm += result[i, j] * result[i, j];
            }
            l2Norm = Math.Sqrt(l2Norm);
            Assert.Equal(1.0, l2Norm, Tolerance);
        }
    }

    [Fact]
    public void Tfidf_L1Norm_SumsToOne()
    {
        var docs = new[] { "cat dog bird", "cat cat", "dog dog dog" };
        var vectorizer = new TfidfVectorizer<double>(
            norm: TfidfNorm.L1, useIdf: true, smoothIdf: true);
        var result = vectorizer.FitTransform(docs);

        for (int i = 0; i < result.Rows; i++)
        {
            double l1Norm = 0;
            for (int j = 0; j < result.Columns; j++)
            {
                l1Norm += Math.Abs(result[i, j]);
            }
            Assert.Equal(1.0, l1Norm, Tolerance);
        }
    }

    [Fact]
    public void Tfidf_NoIdf_ReducesToTfOnly()
    {
        // Without IDF, TF-IDF = TF (raw term counts)
        var docs = new[] { "cat dog", "cat cat" };
        var tfidf = new TfidfVectorizer<double>(
            norm: TfidfNorm.None, useIdf: false, sublinearTf: false);
        var tfidfResult = tfidf.FitTransform(docs);

        var count = new CountVectorizer<double>();
        var countResult = count.FitTransform(docs);

        // Without IDF and norm, TF-IDF output should equal CountVectorizer output
        Assert.Equal(countResult.Rows, tfidfResult.Rows);
        Assert.Equal(countResult.Columns, tfidfResult.Columns);

        for (int i = 0; i < countResult.Rows; i++)
        {
            for (int j = 0; j < countResult.Columns; j++)
            {
                Assert.Equal(countResult[i, j], tfidfResult[i, j], Tolerance);
            }
        }
    }

    [Fact]
    public void Tfidf_RareTermHasHigherIdf()
    {
        // A rare term should get higher IDF than a common term
        var docs = new[] { "cat", "cat", "cat", "cat dog", "cat" };
        var vectorizer = new TfidfVectorizer<double>(
            norm: TfidfNorm.None, useIdf: true, smoothIdf: true);
        vectorizer.Fit(docs);

        Assert.NotNull(vectorizer.IdfWeights);
        Assert.NotNull(vectorizer.Vocabulary);

        int catIdx = vectorizer.Vocabulary["cat"];
        int dogIdx = vectorizer.Vocabulary["dog"];

        // cat df=5, dog df=1
        Assert.True(vectorizer.IdfWeights[dogIdx] > vectorizer.IdfWeights[catIdx],
            $"Rare term 'dog' IDF ({vectorizer.IdfWeights[dogIdx]}) should be > common term 'cat' IDF ({vectorizer.IdfWeights[catIdx]})");
    }

    // =====================================================================
    // BM25 Vectorizer
    // BM25 IDF: log(1 + (N - df + 0.5) / (df + 0.5))
    // BM25 term weight: (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * |D|/avgdl)) + delta
    // BM25 score = IDF * term weight
    // =====================================================================

    [Fact]
    public void BM25_IDF_HandComputed()
    {
        // 3 docs: "cat" in all 3, "dog" in 1
        var docs = new[] { "cat dog", "cat", "cat" };
        var vectorizer = new BM25Vectorizer<double>(k1: 1.5, b: 0.75);
        vectorizer.Fit(docs);

        Assert.NotNull(vectorizer.IdfWeights);
        Assert.NotNull(vectorizer.Vocabulary);

        // BM25 IDF: log(1 + (N - df + 0.5) / (df + 0.5))
        // cat: df=3, IDF = log(1 + (3-3+0.5)/(3+0.5)) = log(1 + 0.5/3.5) = log(1 + 0.1429) ≈ 0.1335
        // dog: df=1, IDF = log(1 + (3-1+0.5)/(1+0.5)) = log(1 + 2.5/1.5) = log(1 + 1.6667) ≈ 0.9808

        int catIdx = vectorizer.Vocabulary["cat"];
        int dogIdx = vectorizer.Vocabulary["dog"];

        double expectedCatIdf = Math.Log(1 + (3 - 3 + 0.5) / (3 + 0.5));
        double expectedDogIdf = Math.Log(1 + (3 - 1 + 0.5) / (1 + 0.5));

        Assert.Equal(expectedCatIdf, vectorizer.IdfWeights[catIdx], Tolerance);
        Assert.Equal(expectedDogIdf, vectorizer.IdfWeights[dogIdx], Tolerance);
    }

    [Fact]
    public void BM25_TermWeight_HandComputed()
    {
        // Single doc "cat cat dog", k1=1.5, b=0.75
        var docs = new[] { "cat cat dog", "cat", "dog" };
        var vectorizer = new BM25Vectorizer<double>(k1: 1.5, b: 0.75, norm: BM25Norm.None);
        var result = vectorizer.FitTransform(docs);

        Assert.NotNull(vectorizer.Vocabulary);

        // avgDocLength: doc0=3, doc1=1, doc2=1 → avg=(3+1+1)/3 = 5/3 ≈ 1.6667
        double avgdl = vectorizer.AverageDocumentLength;
        Assert.Equal(5.0 / 3.0, avgdl, Tolerance);

        int catIdx = vectorizer.Vocabulary["cat"];
        int dogIdx = vectorizer.Vocabulary["dog"];

        // For doc 0 ("cat cat dog"): docLength=3
        // cat: tf=2, term_weight = (2 * (1.5+1)) / (2 + 1.5*(1 - 0.75 + 0.75*3/1.6667))
        //                        = (2 * 2.5) / (2 + 1.5*(1 - 0.75 + 0.75*1.8))
        //                        = 5 / (2 + 1.5*(0.25 + 1.35))
        //                        = 5 / (2 + 1.5*1.6)
        //                        = 5 / (2 + 2.4)
        //                        = 5 / 4.4
        //                        ≈ 1.1364
        double k1 = 1.5;
        double b = 0.75;
        double docLen = 3;
        double catTf = 2;
        double catTermWeight = (catTf * (k1 + 1)) / (catTf + k1 * (1 - b + b * docLen / avgdl));

        double catIdf = Math.Log(1 + (3 - 2 + 0.5) / (2 + 0.5)); // cat df=2
        double expectedCatScore = catIdf * catTermWeight;

        Assert.Equal(expectedCatScore, result[0, catIdx], Tolerance);
    }

    [Fact]
    public void BM25_TermFrequencySaturation()
    {
        // BM25 saturates: increasing TF beyond a point gives diminishing returns
        // A document with tf=10 should not score 10x a document with tf=1
        var docs = new[]
        {
            "rare common common common common common common common common common common",
            "rare common",
            "other"
        };
        var vectorizer = new BM25Vectorizer<double>(k1: 1.5, b: 0.75, norm: BM25Norm.None);
        var result = vectorizer.FitTransform(docs);

        Assert.NotNull(vectorizer.Vocabulary);
        int commonIdx = vectorizer.Vocabulary["common"];

        // Doc 0 has common appearing 10 times, doc 1 has it appearing 1 time
        // Ratio should be much less than 10 due to BM25 saturation
        double score0 = result[0, commonIdx];
        double score1 = result[1, commonIdx];

        // score0 should be positive and score1 should be positive
        Assert.True(score0 > 0, $"Score for doc0 should be positive, got {score0}");
        Assert.True(score1 > 0, $"Score for doc1 should be positive, got {score1}");

        // The ratio should be well below 10 (demonstrating saturation)
        double ratio = score0 / score1;
        Assert.True(ratio < 5.0,
            $"BM25 saturation: score ratio should be much less than 10x, got {ratio:F2}x");
    }

    [Fact]
    public void BM25_DocumentLengthNormalization()
    {
        // With b=0.75, longer documents are penalized
        // Same word, once in short doc vs once in long doc
        var docs = new[]
        {
            "target",
            "target padding padding padding padding padding padding padding padding padding",
            "other"
        };
        var vectorizer = new BM25Vectorizer<double>(k1: 1.5, b: 0.75, norm: BM25Norm.None);
        var result = vectorizer.FitTransform(docs);

        Assert.NotNull(vectorizer.Vocabulary);
        int targetIdx = vectorizer.Vocabulary["target"];

        // "target" appears once in both doc 0 and doc 1, but doc 1 is longer
        // Doc 0 should have higher BM25 score for "target" due to length normalization
        Assert.True(result[0, targetIdx] > result[1, targetIdx],
            $"Short doc BM25 score ({result[0, targetIdx]}) should be > long doc ({result[1, targetIdx]})");
    }

    [Fact]
    public void BM25_NoLengthNorm_WhenBIsZero()
    {
        // With b=0, document length has no effect
        var docs = new[]
        {
            "target",
            "target padding padding padding padding",
            "other"
        };
        var vectorizer = new BM25Vectorizer<double>(k1: 1.5, b: 0.0, norm: BM25Norm.None);
        var result = vectorizer.FitTransform(docs);

        Assert.NotNull(vectorizer.Vocabulary);
        int targetIdx = vectorizer.Vocabulary["target"];

        // With b=0, "target" should have same score in both docs (same tf=1)
        // term_weight = (tf * (k1+1)) / (tf + k1*(1-0+0)) = (1*2.5)/(1+1.5) = 1.0
        Assert.Equal(result[0, targetIdx], result[1, targetIdx], Tolerance);
    }

    [Fact]
    public void BM25_Delta_AddedToTermWeight()
    {
        // BM25+ adds delta to term weight
        var docs = new[] { "cat dog", "cat" };

        var vectorizerNoDelta = new BM25Vectorizer<double>(
            k1: 1.5, b: 0.75, delta: 0, norm: BM25Norm.None);
        var resultNoDelta = vectorizerNoDelta.FitTransform(docs);

        var vectorizerDelta = new BM25Vectorizer<double>(
            k1: 1.5, b: 0.75, delta: 1.0, norm: BM25Norm.None);
        var resultDelta = vectorizerDelta.FitTransform(docs);

        Assert.NotNull(vectorizerNoDelta.Vocabulary);
        int catIdx = vectorizerNoDelta.Vocabulary["cat"];

        // Score with delta should be higher (delta adds to term weight before IDF multiplication)
        Assert.True(resultDelta[0, catIdx] > resultNoDelta[0, catIdx],
            "BM25+ with delta should give higher scores");
    }

    [Fact]
    public void BM25_AvgDocLength_HandComputed()
    {
        var docs = new[] { "a b c", "d", "e f" };
        var vectorizer = new BM25Vectorizer<double>();
        vectorizer.Fit(docs);

        // Doc lengths: 3, 1, 2 → avg = 6/3 = 2.0
        Assert.Equal(2.0, vectorizer.AverageDocumentLength, Tolerance);
    }

    [Fact]
    public void BM25_L2Norm_UnitVectors()
    {
        var docs = new[] { "cat dog bird", "cat cat", "dog dog dog" };
        var vectorizer = new BM25Vectorizer<double>(norm: BM25Norm.L2);
        var result = vectorizer.FitTransform(docs);

        for (int i = 0; i < result.Rows; i++)
        {
            double l2Norm = 0;
            for (int j = 0; j < result.Columns; j++)
            {
                l2Norm += result[i, j] * result[i, j];
            }
            l2Norm = Math.Sqrt(l2Norm);
            Assert.Equal(1.0, l2Norm, Tolerance);
        }
    }

    [Fact]
    public void BM25_ValidationRejectsInvalidParams()
    {
        Assert.Throws<ArgumentException>(() =>
            new BM25Vectorizer<double>(k1: -1));
        Assert.Throws<ArgumentException>(() =>
            new BM25Vectorizer<double>(b: 1.5));
        Assert.Throws<ArgumentException>(() =>
            new BM25Vectorizer<double>(b: -0.1));
    }

    // =====================================================================
    // Cross-vectorizer consistency tests
    // =====================================================================

    [Fact]
    public void AllVectorizers_SameVocabulary_SameDocumentCount()
    {
        var docs = new[] { "hello world", "hello cat", "world cat" };

        var count = new CountVectorizer<double>();
        count.Fit(docs);

        var tfidf = new TfidfVectorizer<double>();
        tfidf.Fit(docs);

        var bm25 = new BM25Vectorizer<double>();
        bm25.Fit(docs);

        // All should have same vocabulary size (same docs, same tokenization)
        Assert.NotNull(count.Vocabulary);
        Assert.NotNull(tfidf.Vocabulary);
        Assert.NotNull(bm25.Vocabulary);

        Assert.Equal(count.Vocabulary.Count, tfidf.Vocabulary.Count);
        Assert.Equal(count.Vocabulary.Count, bm25.Vocabulary.Count);
    }

    [Fact]
    public void Tfidf_TransformUnseen_UnknownWordsIgnored()
    {
        var trainDocs = new[] { "cat dog", "cat bird" };
        var vectorizer = new TfidfVectorizer<double>(norm: TfidfNorm.None, useIdf: false);
        vectorizer.Fit(trainDocs);

        var testDocs = new[] { "cat elephant" }; // "elephant" not in vocab
        var result = vectorizer.Transform(testDocs);

        Assert.NotNull(vectorizer.Vocabulary);
        int catIdx = vectorizer.Vocabulary["cat"];

        // "cat" should have count 1, "elephant" should be ignored
        Assert.Equal(1.0, result[0, catIdx], Tolerance);
        // All other entries should be 0
        for (int j = 0; j < result.Columns; j++)
        {
            if (j != catIdx)
            {
                Assert.Equal(0.0, result[0, j], Tolerance);
            }
        }
    }

    [Fact]
    public void CountVectorizer_GetFeatureNames_MatchesVocabulary()
    {
        var docs = new[] { "alpha beta gamma" };
        var vectorizer = new CountVectorizer<double>();
        vectorizer.Fit(docs);

        var names = vectorizer.GetFeatureNamesOut();
        Assert.NotNull(vectorizer.Vocabulary);

        Assert.Equal(vectorizer.Vocabulary.Count, names.Length);
        // Feature names should be alphabetically sorted
        for (int i = 1; i < names.Length; i++)
        {
            Assert.True(string.Compare(names[i], names[i - 1], StringComparison.Ordinal) >= 0,
                $"Feature names should be sorted: '{names[i]}' < '{names[i - 1]}'");
        }
    }

    [Fact]
    public void BM25_RareTermHigherScoreThanCommonTerm()
    {
        // BM25 should give higher score to rare terms (similar to TF-IDF)
        var docs = new[]
        {
            "common common common rare",
            "common common",
            "common common",
            "common"
        };
        var vectorizer = new BM25Vectorizer<double>(norm: BM25Norm.None);
        var result = vectorizer.FitTransform(docs);

        Assert.NotNull(vectorizer.Vocabulary);
        Assert.NotNull(vectorizer.IdfWeights);

        int commonIdx = vectorizer.Vocabulary["common"];
        int rareIdx = vectorizer.Vocabulary["rare"];

        // "rare" (df=1) should have higher IDF than "common" (df=4)
        Assert.True(vectorizer.IdfWeights[rareIdx] > vectorizer.IdfWeights[commonIdx],
            $"Rare term IDF ({vectorizer.IdfWeights[rareIdx]}) should be > common ({vectorizer.IdfWeights[commonIdx]})");
    }
}

package net.jllama.llama.cpp.java.bindings.runner;

import java.io.Closeable;
import java.time.LocalDateTime;
import java.time.temporal.ChronoField;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import net.jllama.api.Context;
import net.jllama.api.Llama;
import net.jllama.api.Model;
import net.jllama.core.LlamaContext;
import net.jllama.core.LlamaContext.LlamaBatch;
import net.jllama.core.LlamaCpp;
import net.jllama.core.LlamaTokenDataArray;
import net.jllama.core.exceptions.LlamaCppException;

public class Evaluator implements Closeable {

  private final Llama llamaApi;
  private final int eosToken;
  private final LlamaContext llamaContext;
  private final Model model;
  private final int contextSize;
  private final Context context;
  private final LlamaBatch batch;
  private final int systemPromptLength;
  private int nextSeqId;

  Evaluator(final String initialPrompt) {
    final String modelPath = System.getProperty("modelpath");
    llamaApi = Llama.library();

    model = llamaApi.newModel()
        .withDefaults()
        .path(modelPath)
        .load();
    final int threads = Runtime.getRuntime().availableProcessors() / 2 - 1;
    contextSize = 3500;
    context = model.newContext()
        .withDefaults()
        .nThreads(threads)
        .nThreadsBatch(threads)
        .nCtx(contextSize)
        .seed(ThreadLocalRandom.current().nextInt())
        .create();
    llamaContext = context.getLlamaContext();
    eosToken = model.tokens().eos();
    nextSeqId = 0;
    batch = context.getLlamaContext().llamaBatchInit(1000, 0, 1);
    final int[] initialTokens = model.tokens().tokenize(initialPrompt, false, true);
    final int seqId = nextSeqId++;
    int pos = 0;
    batch.nTokens = initialTokens.length;
    for (; pos < initialTokens.length; pos++) {
      batch.token[pos] = initialTokens[pos];
      batch.pos[pos] = pos;
      batch.nSeqId[pos] = 1;
      batch.seqId[pos][0] = seqId;
      batch.logits[pos] = 0;
    }
    int ret = llamaContext.llamaDecode(batch);
    if (ret != 0) {
      throw new LlamaCppException("Initial decode failed with ret=" + ret);
    }
    systemPromptLength = pos;
  }

  public void evaluate(final String prompt) {
    final List<Long> evaluationTimings = new ArrayList<>();
    final int[] tokens = model.tokens().tokenize(prompt);

    // initial prompt
    System.out.print(model.tokens().detokenize(toList(tokens)));

    final int seqId = nextSeqId++;
    llamaContext.llamaKvCacheSeqCp(0, seqId, 0, systemPromptLength - 1);
    int pos = systemPromptLength;
    batch.nTokens = tokens.length;
    for (int i = 0; i < tokens.length; i++) {
      batch.token[i] = tokens[i];
      batch.pos[i] = pos;
      batch.nSeqId[i] = 1;
      batch.seqId[i][0] = seqId;
      batch.logits[i] = 0;
      pos++;
    }
    batch.logits[tokens.length - 1] = 1;
    long timeStamp1 = System.currentTimeMillis();
    decode();
    long timeStamp2 = System.currentTimeMillis();
    evaluationTimings.add(timeStamp2 - timeStamp1);
    int previousToken = sample(llamaContext.llamaGetLogitsIth(tokens.length - 1), tokens.length);

    System.out.print(model.tokens().detokenize(previousToken));

    final List<Integer> previousTokens = new ArrayList<>();
    previousTokens.add(previousToken);

    batch.nTokens = 1;
    batch.nSeqId[0] = 1;
    batch.seqId[0][0] = seqId;
    batch.logits[0] = 1;
    for (int i = tokens.length + 1; previousToken != eosToken && i < contextSize; i++) {
      batch.token[0] = previousToken;
      batch.pos[0] = pos;
      timeStamp1 = System.currentTimeMillis();
      decode();
      timeStamp2 = System.currentTimeMillis();
      evaluationTimings.add(timeStamp2 - timeStamp1);
      previousToken = sample(llamaContext.llamaGetLogitsIth(0), 1);
      previousTokens.add(previousToken);
      System.out.print(model.tokens().detokenize(previousToken));
      pos += 1;
    }
    System.out.printf("%navgEvalTime=%.2f ms%n",  evaluationTimings.stream().mapToDouble(Long::doubleValue).average().getAsDouble());
  }

  private void decode() {
    final int decodeResult = llamaContext.llamaDecode(batch);
    if (decodeResult != 0) {
      throw new LlamaCppException("decode failed with ret=" + decodeResult);
    }
  }

  private int sample(final float[] logits, final int tokenCount) {
    LlamaTokenDataArray candidates = LlamaTokenDataArray.logitsToTokenDataArray(logits);
      llamaContext.llamaSampleTopK(candidates, 50, 1);
//      llamaContext.llamaSampleTopP(candidates, 0.9f, 1);
//      llamaContext.llamaSampleSoftmax(candidates);
    llamaContext.llamaSampleTemp(candidates, 1.1f);
//    llamaContext.llamaSampleTypical(candidates, 0.1f, 5);
//    llamaContext.llamaSampleTailFree(candidates, 0.1f, 5);
//    llamaContext.llamaSampleMinP(candidates, 0.1f, 5);
    llamaContext.llamaSampleRepetitionPenalties(candidates, batch.token, tokenCount, 1f, 1.1f, 1.5f);

    return llamaContext.llamaSampleToken(candidates);
  }

  private static List<Integer> toList(int[] tokens) {
    return Arrays.stream(tokens).boxed().collect(Collectors.toList());
  }

  @Override
  public void close() {
    batch.llamaBatchFree();
    llamaContext.close();
    model.close();
    LlamaCpp.llamaBackendFree();
    LlamaCpp.closeLibrary();
  }
}

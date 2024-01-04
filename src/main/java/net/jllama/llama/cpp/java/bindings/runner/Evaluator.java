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
import net.jllama.api.Context.SequenceType;
import net.jllama.api.Llama;
import net.jllama.api.Model;
import net.jllama.api.Sequence;
import net.jllama.api.Sequence.SequenceId;
import net.jllama.api.batch.Batch;
import net.jllama.core.LlamaContext;
import net.jllama.core.LlamaContext.LlamaBatch;
import net.jllama.core.LlamaCpp;
import net.jllama.core.LlamaTokenDataArray;
import net.jllama.core.exceptions.LlamaCppException;

public class Evaluator implements Closeable {

  private final Llama llamaApi;
  private final int eosToken;
  private final Model model;
  private final int contextSize;
  private final Context context;
  private int nextSeqId;

  Evaluator() {
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
        .evaluationThreads(threads)
        .batchEvaluationThreads(threads)
        .contextLength(contextSize)
        .seed(ThreadLocalRandom.current().nextInt())
        .create();
    eosToken = model.tokens().eos();
    nextSeqId = 0;
    final Batch batch = context.batch()
        .type(SequenceType.TOKEN)
        .configure()
        .batchSize(1500)
        .get();
//    final int[] initialTokens = model.tokens().tokenize(initialPrompt, false, true);
//    final int rawSeqId = nextSeqId++;
//    final Sequence sequence = Sequence.sequence(new int[]{rawSeqId}, SequenceType.TOKEN);
//    batch.stage(sequence.piece(initialTokens, new int[]{}));
//    context.evaluate(batch);
  }

  public void evaluate(final String prompt) {
    final List<Long> evaluationTimings = new ArrayList<>();
    final int[] tokens = model.tokens().tokenize(prompt);

    // initial prompt
    System.out.print(model.tokens().detokenize(toList(tokens)));

    final int seqId = nextSeqId++;
//    llamaContext.llamaKvCacheSeqCp(0, seqId, 0, systemPromptLength - 1);
    final Batch batch = context.batch()
        .type(SequenceType.TOKEN)
        .get();
    final Sequence sequence = Sequence.sequence(new int[]{seqId}, SequenceType.TOKEN);
    batch.stage(sequence.piece(tokens, new int[]{tokens.length - 1}));

    long timeStamp1 = System.currentTimeMillis();
    context.evaluate(batch);
    long timeStamp2 = System.currentTimeMillis();
    evaluationTimings.add(timeStamp2 - timeStamp1);
    int previousToken = sample(context.getLogitsAtIndex(sequence, tokens.length - 1), tokens.length);

    System.out.print(model.tokens().detokenize(previousToken));

    for (int i = tokens.length + 1; previousToken != eosToken && i < contextSize; i++) {
      batch.stage(sequence.piece(new int[]{previousToken}, new int[]{0}));
      timeStamp1 = System.currentTimeMillis();
      context.evaluate(batch);
      timeStamp2 = System.currentTimeMillis();
      evaluationTimings.add(timeStamp2 - timeStamp1);
      previousToken = sample(context.getLogitsAtIndex(sequence, 0), 1);
      System.out.print(model.tokens().detokenize(previousToken));
    }
    System.out.printf("%navgEvalTime=%.2f ms%n", evaluationTimings.stream().mapToDouble(Long::doubleValue).average().getAsDouble());
  }

  private int sample(final float[] logits, final int tokenCount) {
    final LlamaContext llamaContext = context.getLlamaContext();
    LlamaTokenDataArray candidates = LlamaTokenDataArray.logitsToTokenDataArray(logits);
    llamaContext.llamaSampleTopK(candidates, 50, 1);
//      llamaContext.llamaSampleTopP(candidates, 0.9f, 1);
//      llamaContext.llamaSampleSoftmax(candidates);
    llamaContext.llamaSampleTemp(candidates, 1.1f);
//    llamaContext.llamaSampleTypical(candidates, 0.1f, 5);
//    llamaContext.llamaSampleTailFree(candidates, 0.1f, 5);
//    llamaContext.llamaSampleMinP(candidates, 0.1f, 5);
//    llamaContext.llamaSampleRepetitionPenalties(candidates, batch.token, tokenCount, 1f, 1.1f, 1.5f);

    return llamaContext.llamaSampleToken(candidates);
  }

  private static List<Integer> toList(int[] tokens) {
    return Arrays.stream(tokens).boxed().collect(Collectors.toList());
  }

  @Override
  public void close() {
    context.batch()
        .type(SequenceType.TOKEN)
        .get()
        .close();
    model.close();
    LlamaCpp.llamaBackendFree();
    LlamaCpp.closeLibrary();
  }
}

package net.jllama.llama.cpp.java.bindings.runner;

import java.io.Closeable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import net.jllama.api.Context;
import net.jllama.api.Context.SequenceType;
import net.jllama.api.Llama;
import net.jllama.api.Model;
import net.jllama.api.Sequence;
import net.jllama.api.batch.Batch;

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
    List<Integer> previousTokens = new ArrayList<>();
    int previousToken = sample(context.getLogitsAtIndex(sequence, tokens.length - 1), previousTokens);

    System.out.print(model.tokens().detokenize(previousToken));

    for (int i = tokens.length + 1; previousToken != eosToken && i < contextSize; i++) {
      batch.stage(sequence.piece(new int[]{previousToken}, new int[]{0}));
      timeStamp1 = System.currentTimeMillis();
      context.evaluate(batch);
      timeStamp2 = System.currentTimeMillis();
      evaluationTimings.add(timeStamp2 - timeStamp1);
      previousToken = sample(context.getLogitsAtIndex(sequence, 0), previousTokens);
      System.out.print(model.tokens().detokenize(previousToken));
    }
    System.out.printf("%navgEvalTime=%.2f ms%n", evaluationTimings.stream().mapToDouble(Long::doubleValue).average().getAsDouble());
  }

  private int sample(final float[] logits, final List<Integer> previousTokensList) {
    final int[] previousTokens = previousTokensList.stream()
        .mapToInt(Integer::valueOf)
        .toArray();
    return context.sampler(logits)
//        .keepTopK(50)
//        .applyTemperature(1.1f)
//        .keepMinP(0.4f)
//        .keepTopP(0.9f)
//        .applySoftmax()
//        .applyLocallyTypical(0.1f)
//        .applyTailFree(0.1f)
        .applyRepetitionPenalties(previousTokens, 1f, 1.1f, 1.5f)
        .sample();

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
    llamaApi.close();
  }
}

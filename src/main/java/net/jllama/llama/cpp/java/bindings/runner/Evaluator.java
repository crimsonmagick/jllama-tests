package net.jllama.llama.cpp.java.bindings.runner;

import java.io.Closeable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import net.jllama.api.Batch;
import net.jllama.api.Context;
import net.jllama.api.Context.SequenceType;
import net.jllama.api.Llama;
import net.jllama.api.Model;
import net.jllama.api.Sequence;

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
    context.batch()
        .type(SequenceType.TOKEN)
        .configure()
        .batchSize(1500);
  }

  public void evaluate(final String prompt) {
    final List<Long> evaluationTimings = new ArrayList<>();
    final List<Integer> tokens = model.tokens().tokenize(prompt);

    // initial prompt
    System.out.print(model.tokens().detokenize(tokens));

    final int seqId = nextSeqId++;
    final Batch batch = context.batch()
        .type(SequenceType.TOKEN)
        .get();
    final Sequence<Integer> sequence = Sequence.tokenSequence(seqId);
    batch.stage(sequence.piece(tokens));

    long timeStamp1 = System.currentTimeMillis();
    context.evaluate(batch);
    long timeStamp2 = System.currentTimeMillis();
    evaluationTimings.add(timeStamp2 - timeStamp1);
    List<Integer> previousTokens = new ArrayList<>();
    int previousToken = sample(context.getLogits(sequence), previousTokens);

    System.out.print(model.tokens().detokenize(previousToken));

    for (int i = tokens.size() + 1; previousToken != eosToken && i < contextSize; i++) {
      batch.stage(sequence.piece(Collections.singletonList(previousToken)));
      timeStamp1 = System.currentTimeMillis();
      context.evaluate(batch);
      timeStamp2 = System.currentTimeMillis();
      evaluationTimings.add(timeStamp2 - timeStamp1);
      previousToken = sample(context.getLogits(sequence), previousTokens);
      System.out.print(model.tokens().detokenize(previousToken));
    }
    //noinspection OptionalGetWithoutIsPresent
    System.out.printf("%navgEvalTime=%.2f ms%n", evaluationTimings.stream().mapToDouble(Long::doubleValue).average().getAsDouble());
  }

  private int sample(final List<Float> logits, final List<Integer> previousTokens) {
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

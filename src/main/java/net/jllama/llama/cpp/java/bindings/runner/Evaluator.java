package net.jllama.llama.cpp.java.bindings.runner;

import java.io.Closeable;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import net.jllama.api.Context;
import net.jllama.api.Llama;
import net.jllama.api.Model;
import net.jllama.core.LlamaContext;
import net.jllama.core.LlamaContext.LlamaBatch;
import net.jllama.core.LlamaCpp;
import net.jllama.core.LlamaModel;
import net.jllama.core.LlamaTokenDataArray;
import net.jllama.core.exceptions.LlamaCppException;

public class Evaluator implements Closeable {

  private final Llama llamaApi;
  private final Detokenizer detokenizer;
  private final int eosToken;
  private final LlamaContext llamaContext;
  private final LlamaModel llamaModel;
  private final Model model;
  private final int contextSize;
  private final Context context;
  private final LlamaBatch batch;
  private int nextSeqId;

  Evaluator() {
    final String modelPath = System.getProperty("modelpath");
    llamaApi = Llama.library();
    detokenizer = new Detokenizer();

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
        .create();
    llamaContext = context.getLlamaContext();
    llamaModel = model.getLlamaModel();
    eosToken = llamaModel.llamaTokenEos();
    nextSeqId = 0;
    batch = context.getLlamaContext().llamaBatchInit(1000, 0, 1);
  }

  public void evaluate(final String prompt) {
    final int[] tokens = tokenize(llamaModel, prompt, true);


    System.out.print(detokenizer.detokenize(toList(tokens), llamaModel));

    final int seqId = nextSeqId++;

    int pos = 0;
    batch.nTokens = tokens.length;
    for (; pos < tokens.length; pos++) {
      batch.token[pos] = tokens[pos];
      batch.pos[pos] = pos;
      batch.nSeqId[pos] = 1;
      batch.seqId[pos][0] = seqId;
      batch.logits[pos] = 0;
    }
    batch.logits[tokens.length - 1] = 1;
    final int initialRet = llamaContext.llamaDecode(batch);
    if (initialRet != 0) {
      throw new LlamaCppException("decode failed with ret=" + initialRet);
    }
    float[] logits = llamaContext.llamaGetLogitsIth(tokens.length - 1);
    LlamaTokenDataArray candidates = LlamaTokenDataArray.logitsToTokenDataArray(logits);
    llamaContext.llamaSampleTopK(candidates, 40, 1);
    llamaContext.llamaSampleTopP(candidates, 0.9f, 1);
    llamaContext.llamaSampleTemperature(candidates, 0.4f);
    int previousToken = llamaContext.llamaSampleToken(candidates);

    System.out.print(detokenizer.detokenize(previousToken, llamaModel));

    final List<Integer> previousTokenList = new ArrayList<>();
    previousTokenList.add(previousToken);

    batch.nTokens = 1;
    batch.nSeqId[0] = 1;
    batch.seqId[0][0] = seqId;
    batch.logits[0] = 1;
    for (int i = tokens.length + 1; previousToken != eosToken && i < contextSize; i++) {
      batch.token[0] = previousToken;
      batch.pos[0] = pos++;
      final int ret = llamaContext.llamaDecode(batch);
      if (ret != 0) {
        throw new LlamaCppException("decode failed with ret=" + ret);
      }
      logits = llamaContext.llamaGetLogitsIth(0);
      candidates = LlamaTokenDataArray.logitsToTokenDataArray(logits);
      llamaContext.llamaSampleTopK(candidates, 40, 1);
      llamaContext.llamaSampleTopP(candidates, 0.9f, 1);
      llamaContext.llamaSampleTemperature(candidates, 0.1f);
      previousToken = llamaContext.llamaSampleToken(candidates);
      previousTokenList.add(previousToken);
      System.out.print(detokenizer.detokenize(previousToken, llamaModel));
    }

    System.out.println("\ngenTokenCount=" + previousTokenList.size());
  }

  private static int[] tokenize(final LlamaModel llamaModel, final String text, boolean addBos) {
    final int maxLength = text.length();
    final int[] temp = new int[maxLength];
    int length = llamaModel.tokenize(text.getBytes(StandardCharsets.UTF_8), temp, maxLength, addBos);
    final int[] ret = new int[length];
    System.arraycopy(temp, 0, ret, 0, length);
    return ret;
  }

  private static List<Integer> toList(int[] tokens) {
    return Arrays.stream(tokens).boxed().collect(Collectors.toList());
  }

  private static int[] toArray(List<Integer> tokenList) {
    int[] tokens = new int[tokenList.size()];
    Arrays.setAll(tokens, tokenList::get);
    return tokens;
  }

  @Override
  public void close() {
    batch.llamaBatchFree();
    llamaContext.close();
    llamaModel.close();
    LlamaCpp.llamaBackendFree();
    LlamaCpp.closeLibrary();
  }
}

package net.jllama.llama.cpp.java.bindings.runner;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import net.jllama.llama.cpp.java.bindings.LlamaContext.LlamaBatch;
import net.jllama.llama.cpp.java.bindings.LlamaContextParams;
import net.jllama.llama.cpp.java.bindings.LlamaCpp;
import net.jllama.llama.cpp.java.bindings.LlamaLogLevel;
import net.jllama.llama.cpp.java.bindings.LlamaModelParams;
import net.jllama.llama.cpp.java.bindings.LlamaContext;
import net.jllama.llama.cpp.java.bindings.LlamaModel;
import net.jllama.llama.cpp.java.bindings.LlamaTokenDataArray;
import net.jllama.llama.cpp.java.bindings.Sequence;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.lang.management.ManagementFactory;
import java.nio.charset.StandardCharsets;

public class Main {

  private static final String CHAT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant, "
    + "specializing in code generation. Always answer as helpfully as possible. "
    + "If you don't know something, answer that you do not know.";

  private static final String CHAT_PROMPT = "Write \"Hello World\" in C.";
  private static final String COMPLETION_PROMPT = "I love my Cat Winnie, he is such a great cat! Let me tell you more about ";
  private static final String INSTRUCT_SYSTEM_PROMPT = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n%s\n\n### Response:";
  private static final String INSTRUCT_PROMPT = "What is attention mechanism of a transformer model? \n"
    + " Write a python code to illustrate how attention works within a transformer model using numpy library. Do not use pytorch or tensorflow.";

  private static final String B_INST = "<s>[INST]";
  private static final String E_INST = "[/INST]";
  private static final String B_SYS = "<<SYS>>\n";
  private static final String E_SYS = "\n<</SYS>>\n\n";

  static {
    final String jvmName = ManagementFactory.getRuntimeMXBean().getName();
    final String pid = jvmName.split("@")[0];
    System.out.printf("pid=%s%n", pid);
  }
  private static volatile String appLogLevel = System.getProperty("loglevel");
  private static LlamaModel llamaModel;
  private static LlamaContext llamaContext;

  public static void main(final String[] args) {
    try {
      final Detokenizer detokenizer = new Detokenizer();
      final String modelPath = System.getProperty("modelpath");
      LlamaCpp.loadLibrary();
      LlamaCpp.llamaBackendInit(true);
      LlamaCpp.llamaLogSet((logLevel, message) -> {
        final Logger log = LogManager.getLogger(LlamaCpp.class);
        final String messageText = new String(message, StandardCharsets.UTF_8);
        if ("OFF".equalsIgnoreCase(appLogLevel)) {
          return;
        }
        if (logLevel == LlamaLogLevel.INFO && "INFO".equalsIgnoreCase(appLogLevel)) {
          log.info(messageText);
        } else if (logLevel == LlamaLogLevel.WARN) {
          log.warn(messageText);
        } else {
          log.error(messageText);
        }
      });
      long timestamp1 = LlamaCpp.llamaTimeUs();

      final LlamaContextParams llamaContextParams = LlamaContext.llamaContextDefaultParams();
      final int threads = Runtime.getRuntime().availableProcessors() / 2 - 1;
      llamaContextParams.setnThreads(threads);
      llamaContextParams.setnThreadsBatch(threads);
      llamaContextParams.setnCtx(500);

      final LlamaModelParams llamaModelParams = LlamaModel.llamaModelDefaultParams();
      llamaModel = LlamaCpp.loadModel(modelPath.getBytes(StandardCharsets.UTF_8), llamaModelParams);
      llamaContext = llamaModel.createContext(llamaContextParams);
      final int eosToken = llamaModel.llamaTokenEos();

      long timestamp2 = LlamaCpp.llamaTimeUs();

      System.out.printf("timestamp1=%s, timestamp2=%s, initialization time=%s%n", timestamp1, timestamp2, timestamp2 - timestamp1);

      // chat prompt
//      final String prompt = B_INST + B_SYS + CHAT_SYSTEM_PROMPT + E_SYS + CHAT_PROMPT + E_INST;
      final String prompt = String.format(INSTRUCT_SYSTEM_PROMPT, INSTRUCT_PROMPT);
      final int[] tokens = tokenize(prompt, true);

      final LlamaBatch batch = llamaContext.createBatch(1000);
      final Sequence sequence = batch.submitSequence(tokens);

      System.out.print(detokenizer.detokenize(toList(tokens), llamaModel));

      llamaContext.evaluate(batch);
      float[] logits = llamaContext.getLogits(sequence);
      LlamaTokenDataArray candidates = LlamaTokenDataArray.logitsToTokenDataArray(logits);
      llamaContext.llamaSampleTopK(candidates, 40, 1);
      llamaContext.llamaSampleTopP(candidates, 0.9f, 1);
      llamaContext.llamaSampleTemperature(candidates, 0.4f);      int previousToken = llamaContext.llamaSampleToken(candidates);

      System.out.print(detokenizer.detokenize(previousToken, llamaModel));

      final List<Integer> previousTokenList = new ArrayList<>();
      previousTokenList.add(previousToken);


      for (int i = tokens.length + 1; previousToken != eosToken && i < llamaContextParams.getnCtx(); i++) {
        batch.appendToSequence(new int[]{previousToken}, sequence);
        llamaContext.evaluate(batch);
        logits = llamaContext.getLogits(sequence);
        candidates = LlamaTokenDataArray.logitsToTokenDataArray(logits);
        llamaContext.llamaSampleTopK(candidates, 40, 1);
        llamaContext.llamaSampleTopP(candidates, 0.9f, 1);
        llamaContext.llamaSampleTemperature(candidates, 0.1f);
        previousToken = llamaContext.llamaSampleToken(candidates);
        previousTokenList.add(previousToken);
        System.out.print(detokenizer.detokenize(previousToken, llamaModel));
      }

      batch.close();
      llamaContext.close();
      llamaModel.close();
      LlamaCpp.llamaBackendFree();
      LlamaCpp.closeLibrary();
      System.out.println("\ngenTokenCount=" + previousTokenList.size());
    } catch (Error e) {
      System.out.println("Fatal error occurred, errorMessage=" + e.getMessage());
    }
  }

  private static int[] tokenize(final String text, boolean addBos) {
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

}

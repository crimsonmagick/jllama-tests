package net.jllama.llama.cpp.java.bindings.runner;

import net.jllama.core.LlamaCpp;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.stream.Collectors;
import net.jllama.core.LlamaModel;

public class Detokenizer {

  public String detokenize(List<Integer> tokens, LlamaModel llamaModel) {
    return tokens.stream()
        .map(token -> detokenize(token, llamaModel))
        .collect(Collectors.joining());
  }

  public String detokenize(int token, LlamaModel llamaModel) {
    byte[] buf = new byte[8];
    int length = llamaModel.detokenize(token, buf);
    if (length < 0) {
      final int size = Math.abs(length);
      buf = new byte[size];
      length = llamaModel.detokenize(token, buf);
    }
    if (length < 0) {
      throw new RuntimeException("Unable to allocate a large enough buffer for detokenized string length.");
    }
    return new String(buf, 0, length, StandardCharsets.UTF_8);
  }

}

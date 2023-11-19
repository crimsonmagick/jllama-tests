package net.jllama.llama.cpp.java.bindings.runner;

import net.jllama.llama.cpp.java.bindings.LlamaCpp;
import net.jllama.llama.cpp.java.bindings.LlamaOpaqueContext;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.stream.Collectors;
import net.jllama.llama.cpp.java.bindings.LlamaOpaqueModel;

public class Detokenizer {

  public Detokenizer(final LlamaCpp llamaCpp) {
    this.llamaManager = llamaCpp;
  }

  private final LlamaCpp llamaManager;

  public String detokenize(List<Integer> tokens, LlamaOpaqueModel llamaOpaqueModel) {
    return tokens.stream()
        .map(token -> detokenize(token, llamaOpaqueModel))
        .collect(Collectors.joining());
  }

  public String detokenize(int token, LlamaOpaqueModel llamaOpaqueModel) {
    byte[] buf = new byte[8];
    int length = llamaManager.llamaTokenToPiece(llamaOpaqueModel, token, buf);
    if (length < 0) {
      final int size = Math.abs(length);
      buf = new byte[size];
      length = llamaManager.llamaTokenToPiece(llamaOpaqueModel, token, buf);
    }
    if (length < 0) {
      throw new RuntimeException("Unable to allocate a large enough buffer for detokenization length.");
    }
    return new String(buf, 0, length, StandardCharsets.UTF_8);
  }

}

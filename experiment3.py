import torch
import torch.nn as nn
import transformers
from typing import Optional, List, Tuple


class ResidualStreamMemory:
    """
    Captures and injects actual residual stream values (not layer outputs)
    """
    
    def __init__(self, model_name='gpt2-medium'):
        self.model = transformers.GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Buffer for residual stream values
        self.residual_buffer = []
        
        # For capturing residuals during forward pass
        self.captured_residuals = []
        self.capture_enabled = False
        
        # For injection
        self.injection_value = None
        
        # Store original forward methods
        self._original_forwards = {}
        
    def create_capturing_forward(self, block_idx, original_block):
        """
        Modified forward that captures residual stream values
        """
        def forward_with_capture(
            hidden_states,
            layer_past=None,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            output_attentions=False,
        ):
            # THIS is the residual stream we want to capture
            residual = hidden_states
            
            # Capture if enabled and this is the last layer
            if self.capture_enabled and block_idx == len(self.model.transformer.h) - 1:
                # Store the last position's residual
                # residual is [batch, seq_len, hidden_size]
                # We take only the last token: [batch, hidden_size]
                last_token_residual = residual[:, -1, :]
                self.captured_residuals.append(last_token_residual.squeeze(0).detach().cpu())
            
            # Inject into first layer if we have a value
            if self.injection_value is not None and block_idx == 0:
                residual = residual + self.injection_value
            
            # Normal transformer operations
            hidden_states = original_block.ln_1(hidden_states)
            attn_outputs = original_block.attn(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            attn_output = attn_outputs[0]
            outputs = attn_outputs[1:]
            
            # First residual add
            hidden_states = attn_output + residual
            
            # MLP part
            residual = hidden_states  # New residual for second connection
            hidden_states = original_block.ln_2(hidden_states)
            feed_forward_hidden_states = original_block.mlp(hidden_states)
            hidden_states = feed_forward_hidden_states + residual
            
            if use_cache:
                outputs = (hidden_states,) + outputs
            else:
                outputs = (hidden_states,) + outputs[1:]
                
            return outputs
            
        return forward_with_capture
    
    def patch_model(self):
        """
        Patch all transformer blocks to enable capture/injection
        """
        for idx, block in enumerate(self.model.transformer.h):
            self._original_forwards[idx] = block.forward
            block.forward = self.create_capturing_forward(idx, block)
    
    def unpatch_model(self):
        """
        Restore original forward methods
        """
        for idx, block in enumerate(self.model.transformer.h):
            if idx in self._original_forwards:
                block.forward = self._original_forwards[idx]
        self._original_forwards.clear()
    
    def prepare_injection(self, buffer_size=3, buffer_weight=0.1):
        """
        Prepare injection value from buffer
        """
        if not self.residual_buffer:
            self.injection_value = None
            return
            
        # Use only recent residuals
        recent_residuals = self.residual_buffer[-buffer_size:]
        
        # Exponential decay weights
        decay_rate = 0.7
        weights = torch.tensor([decay_rate**i for i in range(len(recent_residuals))])
        weights = weights / weights.sum()
        
        # Compute weighted sum
        buffer_sum = torch.zeros_like(recent_residuals[0])
        for w, residual in zip(weights, recent_residuals):
            buffer_sum += w * residual
        
        # Debug: Check dimensions
        # buffer_sum should be [hidden_size] since we captured residual[:, -1, :]
        # We need [1, seq_len, hidden_size] for injection
        
        # For now, let's make it [1, 1, hidden_size] to add to all positions
        if len(buffer_sum.shape) == 1:
            buffer_sum = buffer_sum.unsqueeze(0).unsqueeze(0)
        elif len(buffer_sum.shape) == 2:
            buffer_sum = buffer_sum.unsqueeze(0)
            
        self.injection_value = buffer_weight * buffer_sum.to(self.model.device)
    
    def generate_with_residual_memory(self, prompt: str, max_length: int = 50,
                                     buffer_size: int = 3, use_buffer: bool = True,
                                     buffer_weight: float = 0.1):
        """
        Generate text while capturing/injecting residual streams
        """
        self.model.eval()
        
        # Prepare injection from previous residuals
        if use_buffer:
            self.prepare_injection(buffer_size, buffer_weight)
        else:
            self.injection_value = None
        
        # Patch model
        self.patch_model()
        
        try:
            # Clear captured residuals
            self.captured_residuals = []
            
            # Encode prompt
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
            generated = input_ids
            
            # Generate token by token
            for _ in range(max_length - len(input_ids[0])):
                with torch.no_grad():
                    # Enable capture for this forward pass
                    self.capture_enabled = True
                    
                    outputs = self.model(generated, output_hidden_states=True)
                    logits = outputs.logits[:, -1, :]
                    
                    # Disable capture
                    self.capture_enabled = False
                    
                    # Get next token
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)
                    
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
            
            # Store the final captured residual in buffer
            if use_buffer and self.captured_residuals:
                # Take the last captured residual
                final_residual = self.captured_residuals[-1]
                self.residual_buffer.append(final_residual)
                
                # Maintain buffer size
                if len(self.residual_buffer) > buffer_size:
                    self.residual_buffer = self.residual_buffer[-buffer_size:]
            
            return self.tokenizer.decode(generated[0], skip_special_tokens=True)
            
        finally:
            self.unpatch_model()
            self.captured_residuals = []
    
    def clear_buffer(self):
        """Clear residual buffer"""
        self.residual_buffer = []
        self.injection_value = None


def run_residual_experiment():
    """
    Run experiment with true residual stream capture/injection
    """
    torch.manual_seed(42)
    
    print("=== Residual Stream Memory Experiment ===\n")
    
    model = ResidualStreamMemory('gpt2-medium')
    
    prompts = [
        "The nature of consciousness is",
        "Interestingly,",                      # Super minimal
        "Moreover,",                           # Just a transition
        "Therefore,"                           # Pure continuation
    ]
    
    BUFFER_SIZE = 1
    BUFFER_WEIGHT = 0.0028
    
    print(f"Buffer size: {BUFFER_SIZE}")
    print(f"Buffer weight: {BUFFER_WEIGHT}")
    print(f"Mode: Deterministic (temperature=0)\n")
    
    # Run with residual memory
    print("--- WITH Residual Stream Memory ---")
    with_memory_responses = []
    for i, prompt in enumerate(prompts, 1):
        print(f"\nPrompt {i}: {prompt}")
        response = model.generate_with_residual_memory(
            prompt,
            max_length=50,
            buffer_size=BUFFER_SIZE,
            use_buffer=True,
            buffer_weight=BUFFER_WEIGHT
        )
        with_memory_responses.append(response)
        print(f"Response: {response}")
        print(f"Buffer contains {len(model.residual_buffer)} residual(s)")
    
    # Clear and run without
    model.clear_buffer()
    
    print("\n\n--- WITHOUT Residual Stream Memory ---")
    without_memory_responses = []
    for i, prompt in enumerate(prompts, 1):
        print(f"\nPrompt {i}: {prompt}")
        response = model.generate_with_residual_memory(
            prompt,
            max_length=50,
            use_buffer=False
        )
        without_memory_responses.append(response)
        print(f"Response: {response}")
    
    # Compare results
    print("\n\n--- Comparison ---")
    for i, prompt in enumerate(prompts):
        if with_memory_responses[i] == without_memory_responses[i]:
            print(f"Prompt {i+1}: IDENTICAL")
        else:
            print(f"Prompt {i+1}: DIFFERENT")


if __name__ == "__main__":
    run_residual_experiment()

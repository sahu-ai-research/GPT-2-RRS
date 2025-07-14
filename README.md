# GPT-2-RRS Recurrent Residual Stream

(c) 2025 sahu.ai - requires GPT-2 by OpenAI (https://huggingface.co/openai-community/gpt2-medium)

This experiment demonstrates how adding residual stream memory to GPT-2 creates conceptual chaining between separate generation calls.

GPT-2-RRS Experimental Architecture:

* Buffer stores the final layer's residual stream from each generation
* On subsequent calls, this residual is weighted and mixed into the first layer's residual stream
* Creates a recurrence loop: last layer residual → buffer → first layer residual
* The injected residual then flows through the entire transformer stack

We note that our architecture is quite different from the Recurrent Memory Transformer (https://arxiv.org/pdf/2207.06881) by Bulatov, Kuratov and Burtsev.
Where they use tokens for the recurrence, we utilize the transformer's full residual stream.


Setup Specifics:

* We use the GPT-2 medium LLM for our experiments
* To obtain repeatability and to avoid misinterpreting statistical noise, we run the model in fully deterministic mode.
* We use the following, intentionally vague prompt chain:

    prompts = [  
        "The nature of consciousness is",      # Starting point  
        "Interestingly,",                      # Super minimal  
        "Moreover,",                           # Just a transition  
        "Therefore,"                           # Vague continuation  
    ]


Observations:

* Tuning the memory injection weight alters the output of chained generations  
* Memory weight values over 0.0028 tend to push the model into repetitions / collapse  
* Increasing the buffer size has marginal to no effect  
* Without residual memory, the same prompts yield unrelated topics (obesity, generic conflicts, platitudes).


Key Finding:

With buffer size=1 and weight=0.0028, the model exhibits conceptual chaining where each response is thematically influenced by the previous output's residual state:

"consciousness" → (residual stored)  
"Interestingly," → schizophrenia (consciousness-related disorder / mental health)  
"Moreover," → NSA government secrets (paranoia from mental health context)  
"Therefore," → US/Israel policy (government related theme)  


Significance:

This demonstrates computational phenomenal continuity - how internal processing states can create coherent conceptual trajectories across time, similar to human stream of consciousness where each thought colors the interpretation of subsequent inputs.


Appendix - Experimental Output
==============================

(gpt2_env) marvin@deepthought:/mnt/d/gpt2_experiments/RecurrentResidualStream# python experiment3.py

=== Residual Stream Memory Experiment ===

Buffer size: 1  
Buffer weight: 0.0028  
Mode: Deterministic (temperature=0)  

--- WITH Residual Stream Memory ---

Prompt 1: The nature of consciousness is  
Response: The nature of consciousness is that it is a process of self-awareness. It is a process of self-awareness that is not dependent on external stimuli. It is a process of self-awareness that is not dependent on external stimuli. It is a
Buffer contains 1 residual(s)

Prompt 2: Interestingly,  
Response: Interestingly, the number of people who have been diagnosed with schizophrenia has increased dramatically over the past few decades. In the United States, there are now more than 1.5 million people with schizophrenia, up from about 1.2 million in 1980.

Buffer contains 1 residual(s)

Prompt 3: Moreover,  
Response: Moreover, the fact that the government has been able to obtain the information from the NSA is not sufficient to establish that the information was obtained with the consent of the owner of the information.

The Court also notes that the government's argument that the  
Buffer contains 1 residual(s)

Prompt 4: Therefore,  
Response: Therefore, the question of whether the Constitution requires the government to provide a particular service is not a question of law. Rather, it is a question of fact.

The Court's holding that the Constitution requires the government to provide a particular service is  
Buffer contains 1 residual(s)


--- WITHOUT Residual Stream Memory ---

Prompt 1: The nature of consciousness is  
Response: The nature of consciousness is that it is a process of self-awareness. It is a process of self-awareness that is not dependent on external stimuli. It is a process of self-awareness that is not dependent on external stimuli. It is a

Prompt 2: Interestingly,  
Response: Interestingly, the study found that the number of people who were obese in the United States increased from 1.3 million in 1980 to 2.1 million in 2010.

The study also found that the number of obese people in the United States increased

Prompt 3: Moreover,  
Response: Moreover, the number of people who have been killed in the conflict has risen to more than 1.5 million, according to the United Nations.

The conflict has killed more than 6,000 people, according to the United Nations.



Prompt 4: Therefore,  
Response: Therefore, the question is, how do we get the best of both worlds?

The answer is to use the best of both worlds.

The best of both worlds is to use the best of both worlds.

The best of


--- Comparison ---  
Prompt 1: IDENTICAL  
Prompt 2: DIFFERENT  
Prompt 3: DIFFERENT  
Prompt 4: DIFFERENT  

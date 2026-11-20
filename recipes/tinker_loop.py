"""The smallest RL loop on a NexCloud RL run: sample, score, train one step, sample again. It uses the tinker SDK and,
for the chat template and stop sequences only, tinker-cookbook's renderer (pip install tinker tinker-cookbook).

    eval "$(nextron-cloud rl env rl_...)"      # TINKER_BASE_URL, TINKER_API_KEY
    python recipes/tinker_loop.py
    nextron-cloud rl stop rl_... --wait        # if the loop did not reach its end

The reward here is a toy (a reply that is a prime number scores 1). Everything in this file runs on your machine;
the RL run samples and trains.
"""
import tinker
from tinker_cookbook import renderers

MODEL = "Qwen/Qwen3.8-27B"
PRIMES = {"2", "3", "5", "7", "11", "13", "17", "19"}


def sample(sampler, n):
    """n replies to the prompt: (tokens, the sampler's logprob of each token, reward)."""
    replies = []
    for sequence in sampler.sample(prompt, num_samples=n, sampling_params=params).result().sequences:
        text = renderer.parse_response(sequence.tokens)[0]["content"].strip()
        replies.append((sequence.tokens, sequence.logprobs, 1.0 if text in PRIMES else 0.0))
    return replies


def datum(tokens, logprobs, advantage):
    """One sampled sequence as Tinker's RL datum: inputs without the last token, targets without the first, and a
    sampler logprob and an advantage per target (zero on the prompt)."""
    context = prompt.to_ints()
    sequence = context + list(tokens)
    padding = [0.0] * (len(context) - 1)
    return tinker.Datum(model_input=tinker.ModelInput.from_ints(sequence[:-1]),
                        loss_fn_inputs={"target_tokens": sequence[1:], "logprobs": padding + list(logprobs),
                                        "advantages": padding + [advantage] * len(tokens)})


served = tinker.RestClient().get_server_capabilities().result().supported_models[0].model_name
if served != MODEL:
    raise SystemExit(f"This RL run trains {served}. The chat template and tokenizer of this loop are {MODEL}'s: "
                     f"start a run with `nextron-cloud rl start --model qwen3.8-27b`.")

with tinker.ServiceClient() as service:   # leaving the block ends the session: the run stops now, not 3 minutes later
    training = service.create_lora_training_client(base_model=MODEL).result()   # on NexCloud all weights are trained; rank is ignored
    tokenizer = training.get_tokenizer()
    renderer = renderers.get_renderer("qwen3_8_disable_thinking", tokenizer)
    prompt = renderer.build_generation_prompt([{"role": "user", "content": "Name one number below 20. Reply with the number only."}])
    params = tinker.SamplingParams(max_tokens=16, temperature=1.0, stop=renderer.get_stop_sequences())

    sampler = training.save_weights_and_get_sampling_client().result()
    replies = sample(sampler, 16)
    mean = sum(reward for _, _, reward in replies) / len(replies)
    print(f"before: {mean:.2f} of the replies are prime")

    data = [datum(tokens, logprobs, reward - mean) for tokens, logprobs, reward in replies]
    # forward_backward only queues the data; the step runs with optim_step, so submit both before waiting.
    queued = training.forward_backward(data, "importance_sampling")
    step = training.optim_step(tinker.AdamParams(learning_rate=2e-6))
    queued.result()
    print("trained one step:", step.result().metrics)

    sampler = training.save_weights_and_get_sampling_client().result()   # the run's vLLM, now with the new weights
    after = sample(sampler, 16)
    print(f"after:  {sum(reward for _, _, reward in after) / len(after):.2f} of the replies are prime")

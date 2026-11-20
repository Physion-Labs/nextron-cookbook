# Competitive programming with tests as the reward (Serverless RL)

The model writes a Python solution to a competitive programming problem. Your code runs it against the problem's tests
in a sandbox and, when a test fails, tells the model the verdict, the failing input and its program's output; the model
gets one more attempt. An episode scores 1 when its last program passes every test, minus 0.1 when the reply has no
code block.

The problems, tests and grader are tinker-cookbook's (DeepCoder: 24,287 training problems from PrimeIntellect, TACO and
LiveCodeBench; 687 test problems from Codeforces and LiveCodeBench), the loop is NexCloud's own calls.

```bash
pip install nextron-cloud "nextron-cookbook @ git+https://github.com/Physion-Labs/nextron-cookbook"
docker run -d -p 127.0.0.1:8080:8080 \
  -v "$(python -c 'import tinker_cookbook, os; print(os.path.dirname(tinker_cookbook.__file__))')/recipes/code_rl/sandbox_config/local.yaml:/root/sandbox/sandbox/configs/local.yaml" \
  volcengine/sandbox-fusion:server-20250609
export NEXTRON_API_KEY=nk_live_... SANDBOX_URL=http://127.0.0.1:8080/run_code SANDBOX_MAX_CONCURRENCY=16
nextron-cloud rl start --model qwen3.8-27b --max-hours 2 --yes
nextron-cloud rl wait rl_...
python recipes/code_rl/train.py rl_... --eval-every 6     # 6 steps; the 687 test problems before and after
nextron-cloud rl stop rl_... --wait
```

## The sandbox

Model-written code never runs in the loop's process: every program goes to
[SandboxFusion](https://bytedance.github.io/SandboxFusion/) over HTTP (`SANDBOX_URL`) and runs inside that container.
The image is 37 GB on disk and built for x86-64 (on an Apple Silicon Mac it runs under emulation). The loop first sends
a one-line program and stops if the sandbox does not run it, so a missing sandbox is not scored as failed solutions.

## How it maps to the API

- Every attempt is one `sample` call for all the episodes still open: the problem, and for a second attempt the first
  reply and what failed.
- An episode is `Episode(samples=[its one or two replies], reward, group=...)`, compared with the other episodes on
  the same problem (8 by default).
- The model is not shown a failing test's expected output: most problems have few tests, and a program that prints
  them would pass.

## Settings

- 32 problems x 8 episodes a step, up to 4,096 new tokens a reply, learning rate 2e-6, temperature 1.
- The time your sandbox takes is part of every step (`client_s`) and is billed like the rest: raise
  `SANDBOX_MAX_CONCURRENCY` to the cores you have.

## Results

Qwen3.8-27B, one node of 8 H100, learning rate 2e-6, the sandbox on a laptop (16 programs at a time):

| | |
|---|---|
| all tests pass on the 687 test problems, one episode each | 44.3% -> 54.1% in 6 steps |
| attempts an episode on them | 1.74 -> 1.60 |
| all tests pass in the training episodes, steps 1-6 | 61%, 61%, 52%, 60%, 77%, 60% (each step has other problems) |
| a step (256 episodes, 1.0-1.6M tokens) | 240-509 s: sampling 138-177 s, training 57-86 s at 2,420-2,490 tokens/s/GPU, 31-301 s in the sandbox and the loop |
| an evaluation of the 687 problems | 605-678 s |

The first training step takes about 100 s longer (warm-up). The time in the sandbox depends on the programs: a training
problem has about 100 tests, and one step with many slow programs spent 301 s there. The run was ready for 62 minutes
and cost $29.10 (3,742 s at $28 an hour, the model's load not billed): two evaluations, six steps, and about seven
minutes of setup (loading the problems, a first check of the loop).

`nextron-cookbook run code` ([docs/recipes.md](../../docs/recipes.md#code-multi-turn-with-a-tool)) trains on the same
problems with tinker-cookbook's loop, where the model may call a tool that runs its code. Its 20.7% -> 37.8% is a
different measurement: here every failed first attempt gets a second one.

"""nextron-cookbook: run tinker-cookbook recipes on a NexTron RL run with tested settings.

    nextron-cookbook list
    nextron-cookbook show math
    nextron-cookbook check
    nextron-cookbook run math max_steps=10 learning_rate=1e-6
"""
import argparse
import os
import shlex
import subprocess
import sys
import time

from nextron_cookbook.recipes import MODEL, RECIPES, arguments


def run_env():
    missing = [name for name in ("TINKER_BASE_URL", "TINKER_API_KEY") if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"{' and '.join(missing)} not set. Start an RL run first: "
                         f"`nextron-cloud rl start`, then `eval \"$(nextron-cloud rl env <id>)\"`.")
    if "thinkingmachines" in os.environ["TINKER_BASE_URL"]:
        raise SystemExit("TINKER_BASE_URL points at Tinker, not at a NexTron RL run.")


def command(name, overrides):
    args = arguments(name, overrides)
    if not any(a.startswith("log_path=") for a in args):
        args.append(f"log_path=runs/{name}-{time.strftime('%m%d-%H%M%S')}")
    return [sys.executable, "-m", RECIPES[name].module, *args]


def check_model(cmd):
    """Stop before a recipe drives a run of another model: it would train on token ids of its own model's tokenizer."""
    import tinker

    asked = next(a for a in cmd if a.startswith("model_name=")).removeprefix("model_name=")
    served = tinker.RestClient().get_server_capabilities().result().supported_models[0].model_name
    if served != asked:
        raise SystemExit(f"This RL run trains {served}; the recipe's tokenizer and chat template are {asked}'s. "
                         f"The recipes are tested on {MODEL}: `nextron-cloud rl start --model qwen3.8-27b`.")


def cmd_list(_args):
    for name, recipe in RECIPES.items():
        print(f"{name:8} {recipe.summary}")
        if recipe.needs:
            print(f"{'':8} needs {recipe.needs}")


def cmd_show(args):
    recipe = RECIPES[args.recipe]
    print(recipe.summary)
    print(f"RL run: max context >= {recipe.min_context}" + (f"; needs {recipe.needs}" if recipe.needs else ""))
    if recipe.extra:
        print(f"install: nextron-cookbook[{recipe.extra}]")
    print("command (every key=value is the cookbook's own argument; add yours to `run` to replace one):")
    print("  " + shlex.join(["python", "-m", *command(args.recipe, args.overrides)[2:]]))


def cmd_run(args):
    run_env()
    cmd = command(args.recipe, args.overrides)
    check_model(cmd)
    print("+ " + shlex.join(cmd), flush=True)
    code = subprocess.call(cmd)
    parts = os.environ["TINKER_BASE_URL"].rstrip("/").split("/")   # .../sessions/<id>/tinker
    if parts[-3:-2] == ["sessions"]:
        print(f"\nThe recipe ended (exit {code}). An RL run holds one training loop: stop it now, which also stops "
              f"billing if the recipe did not end cleanly:\n  nextron-cloud rl stop {parts[-2]} --wait", flush=True)
    sys.exit(code)


def cmd_check(_args):
    """The RL run answers, which model it trains, its context length, and one sampled reply."""
    run_env()
    import tinker
    from tinker_cookbook import tokenizer_utils

    models = tinker.RestClient().get_server_capabilities().result().supported_models
    for model in models:
        print(f"model {model.model_name}  max context {model.max_context_length}")
    name = models[0].model_name
    try:
        tokenizer = tokenizer_utils.get_tokenizer(name)
    except OSError as error:
        raise SystemExit(f"The run answers, but its tokenizer could not be loaded, so no reply is sampled: {error}\n"
                         f"The recipes here are set up for {MODEL}; nextron-cloud's own calls (`client.rl.connect`) "
                         f"need no tokenizer.") from None
    sampler = tinker.ServiceClient().create_sampling_client(base_model=name).result()
    prompt = tinker.ModelInput.from_ints(tokenizer.encode("The capital of France is"))
    started = time.time()
    reply = sampler.sample(prompt, num_samples=1, sampling_params=tinker.SamplingParams(max_tokens=8)).result()
    print(f"sample {tokenizer.decode(reply.sequences[0].tokens)!r} in {time.time() - started:.1f} s")
    print("ok")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="nextron-cookbook", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="recipes with tested settings").set_defaults(run=cmd_list)
    sub.add_parser("check", help="check the RL run in TINKER_BASE_URL").set_defaults(run=cmd_check)
    for name, run, text in (("show", cmd_show, "print a recipe's command"), ("run", cmd_run, "run a recipe")):
        p = sub.add_parser(name, help=text)
        p.add_argument("recipe", choices=sorted(RECIPES))
        p.add_argument("overrides", nargs="*", metavar="key=value", help="cookbook arguments that replace ours")
        p.set_defaults(run=run)
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()

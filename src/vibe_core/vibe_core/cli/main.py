# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import sys

from vibe_core.cli.logging import set_log_level, setup_logging

from .helper import set_auto_confirm
from .logging import log
from .workflow import main as workflow_main


def _resolve_and_apply_profile():
    """Resolve the active deployment profile and inject it into config.

    This MUST run before parsers/local/remote are imported: those modules call
    load_config() at import time and bake the values into argparse defaults.
    Chicken-and-egg is broken by pre-scanning argv, same as --verbose.

    Returns the profile name on success, or None if profile handling failed
    (error already printed).
    """
    from .config import FarmVibesConfig, load_config
    from .profiles import (
        ProfileError,
        load_profile,
        resolve_profile_name,
        validate_profile_keys,
    )

    name = resolve_profile_name(sys.argv[1:])
    try:
        overrides, source = load_profile(name)
        validate_profile_keys(overrides, set(FarmVibesConfig.__fields__), source)
    except ProfileError as e:
        print(f"Profile error: {e}", file=sys.stderr)
        return None

    # Re-wire load_config so every subsequent caller (parsers.py import,
    # local.py import, dispatch validation) picks up profile overrides.
    import vibe_core.cli.config as config_mod
    config_mod.load_config = lambda: load_config(profile_overrides=overrides)

    return name


def main():
    # Handle 'workflow' command separately (doesn't need cluster type or profiles)
    if len(sys.argv) > 1 and sys.argv[1] == "workflow":
        sys.exit(workflow_main(sys.argv[2:]))

    # Resolve profile BEFORE importing parsers — config is baked at import time.
    profile_name = _resolve_and_apply_profile()
    if profile_name is None:
        sys.exit(1)

    # Now safe to import modules that read config at import time.
    from .local import dispatch as dispatch_local
    from .parsers import LocalCliParser, RemoteCliParser
    from .remote import dispatch as dispatch_remote

    parser = argparse.ArgumentParser(description="FarmVibes.AI cluster deployment tool")
    parser.add_argument(
        "cluster_type",
        choices=["remote", "local", "workflow"],
        help="Cluster type to manage, or 'workflow' for workflow tools",
    )
    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action="store_true")
    parser.add_argument(
        "--auto-confirm", required=False, help="Answer every question as yes", action="store_true"
    )

    verbose_requested, help_requested = False, False
    for help in ["-h", "--help"]:
        if help in sys.argv:
            help_requested = True
            sys.argv.remove(help)
    for verbose in ["-v", "--verbose"]:
        if verbose in sys.argv:
            verbose_requested = True
            sys.argv.remove(verbose)

    args, unknown_args = parser.parse_known_args()
    if help_requested:
        unknown_args += ["-h"]
    if args.auto_confirm:
        unknown_args += ["--auto-confirm"]

    if args.auto_confirm:
        set_auto_confirm()

    # Determine the type of cluster we have
    # Given that, build the subparsers for that cluster type
    if args.cluster_type == "remote":
        parser = RemoteCliParser("remote")
        dispatcher = dispatch_remote
    elif args.cluster_type == "local":
        parser = LocalCliParser("local")
        dispatcher = dispatch_local
    else:
        raise RuntimeError(f"Unknown cluster type: {args.cluster_type}")

    logfile = setup_logging(parser.name)
    if args.verbose or verbose_requested:
        set_log_level("DEBUG")

    args = parser.parse(unknown_args)
    log(f"Active deployment profile: {profile_name}")
    try:
        result = dispatcher(args)
        if isinstance(result, int):
            sys.exit(result)
        elif result is False:
            sys.exit(1)
    except Exception as e:
        log(
            f"farmvibes-ai {parser.name} failed ({e}). Please see the above error descriptions. "
            "If you think this is an error with the program, please file an issue at "
            "https://github.com/microsoft/farmvibes-ai/issues. "
            f"Please also include the contents of {logfile} in your issue.",
            level="error",
        )
        sys.exit(1)
    except KeyboardInterrupt:
        log(f"farmvibes-ai {parser.name} interrupted by user. Goodbye.", level="error")
        sys.exit(1)


if __name__ == "__main__":
    main()

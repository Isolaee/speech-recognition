import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice Agent")
    parser.add_argument("--skill", metavar="NAME", help="Override the active skill")
    parser.add_argument(
        "--no-voice",
        action="store_true",
        help="Text-only mode: read from stdin, print to stdout",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--list-skills",
        action="store_true",
        help="Print available skill names and exit",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    args = parser.parse_args()

    from utils.logger import configure_logging
    configure_logging("DEBUG" if args.debug else None)

    from config import load_config

    config = load_config(args.config)

    if args.list_skills:
        from skills.loader import SkillLoader

        loader = SkillLoader("skills/definitions")
        for name in sorted(loader.list_skills()):
            print(name)
        return

    from agent.orchestrator import Orchestrator

    orchestrator = Orchestrator(config, no_voice=args.no_voice)

    if args.skill:
        orchestrator.set_skill(args.skill)

    orchestrator.run()


if __name__ == "__main__":
    main()

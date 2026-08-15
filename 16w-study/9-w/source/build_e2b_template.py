"""Build the resource-bounded E2B template used by ``sandbox.py``.

This command changes managed E2B state and consumes account resources, so it
is never called by tests.  Run it explicitly after setting ``E2B_API_KEY``.
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="week9-secure-1c-512m")
    args = parser.parse_args()
    if not os.getenv("E2B_API_KEY"):
        raise SystemExit("E2B_API_KEY is required")

    from e2b import Template

    template = (
        Template()
        .from_python_image("3.12")
        .make_dir("/home/user/work", mode=0o700, user="user")
        .set_workdir("/home/user/work")
        .set_user("user")
    )
    info = Template.build(template, args.name, cpu_count=1, memory_mb=512)
    print(f"template={info.template_id} build={info.build_id} name={args.name}")


if __name__ == "__main__":
    main()

"""Apply MSVC Blackwell patches to .deps after FetchContent populates them."""
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DEPS_DIR = os.path.join(ROOT_DIR, ".deps")

PATCHES = [
    ("cutlass-src", "cutlass-msvc-blackwell.patch"),
    ("qutlass-src", "qutlass-msvc-blackwell.patch"),
    ("vllm-flash-attn-src", "vllm-flash-attn-msvc-blackwell.patch"),
]


def apply():
    for dep_dir, patch_file in PATCHES:
        dep_path = os.path.join(DEPS_DIR, dep_dir)
        patch_path = os.path.join(SCRIPT_DIR, patch_file)

        if not os.path.isdir(dep_path):
            print(f"  skip {dep_dir} (not found)")
            continue
        if not os.path.isfile(patch_path):
            print(f"  skip {patch_file} (not found)")
            continue

        print(f"  patching {dep_dir}...")
        result = subprocess.run(
            ["git", "apply", "--check", patch_path],
            cwd=dep_path, capture_output=True
        )
        if result.returncode != 0:
            print(f"  already applied or conflict in {dep_dir}, skipping")
            continue

        subprocess.run(
            ["git", "apply", patch_path],
            cwd=dep_path, check=True
        )
        print(f"  applied {patch_file}")


if __name__ == "__main__":
    print("Applying MSVC Blackwell patches to .deps:")
    apply()
    print("Done.")

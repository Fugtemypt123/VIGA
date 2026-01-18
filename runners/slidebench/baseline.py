"""AutoPresent Baseline Runner.

Runs baseline slide generation using LLM-generated Python code.
"""
import os
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple

example_dict = {
    "sufficient": {
        0: "runners/slidebench/prompt/example_full_pptx.txt",
        1: "runners/slidebench/prompt/example_full_wlib.txt",
    },
    "visual": {
        0: "runners/slidebench/prompt/example_noimg_pptx.txt",
        1: "runners/slidebench/prompt/example_noimg_wlib.txt",
    },
    "creative": {
        0: "runners/slidebench/prompt/example_hl_pptx.txt",
        1: "runners/slidebench/prompt/example_hl_wlib.txt",
    },
}

def run_slide_command(slide_command: list, pptx_path: str) -> Tuple[bool, str]:
    """Run a slide generation command.

    Args:
        slide_command: Command list to execute.
        pptx_path: Path to the output PPTX file.

    Returns:
        Tuple of (success, error_message).
    """
    try:
        subprocess.run(slide_command, check=True)
        return True, ""
    except subprocess.CalledProcessError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def main() -> None:
    """Run the baseline slide generation pipeline."""
    command = []
    if args.setting != "sufficient":
        command.append("--no_image")
    if args.setting == "visual":
        command.extend(["--instruction_name", "instruction_no_image.txt"])
    elif args.setting == "creative":
        command.extend(["--instruction_name", "instruction_high_level.txt"])
        
    if args.use_library:
        if args.setting == "sufficient":
            command.extend(["--library_path", "runners/slidebench/library/library_basic.txt"])
        else:
            command.extend(["--library_path", "runners/slidebench/library/library.txt"])
    
    # Remove example path for seed-prompt test
    # command.extend(["--example_path", example_dict[args.setting][int(args.use_library)]])

    if args.slide_deck == 'all':
        slide_dirs = ['art_photos', 'business', 'design', 'entrepreneur', 'environment', 'food', 'marketing', 'social_media', 'technology']
    else:
        slide_dirs = [args.slide_deck]

    tasks = []

    for slide_name in slide_dirs:
        slide_dir = os.path.join("data/autopresent/examples", slide_name)
        page_dirs = [d for d in os.listdir(slide_dir) if d.startswith("slide_")]
        page_dirs = sorted(page_dirs, key=lambda x: int(x.split("_")[1]))
        for page_dir in page_dirs:
            output_path = os.path.join(
                "data/autopresent/examples", slide_name, page_dir, f"baseline/{args.model_name.replace('-', '_')}.py"
            )
            pptx_path = output_path.replace(".py", ".pptx")
            if os.path.exists(pptx_path) and not args.cover_all:
                print(f"Slide deck already exists: {pptx_path}")
                continue
            else:
                print(f"Creating slide deck: {pptx_path}")

            slide_command = [
                "python", "runners/slidebench/create_slide.py",
                "--example_dir", f"data/autopresent/examples/{slide_name}/{page_dir}",
                "--model_name", args.model_name
            ] + command

            tasks.append({
                "slide_name": slide_name,
                "page_dir": page_dir,
                "pptx_path": pptx_path,
                "command": slide_command,
            })

    if not tasks:
        print("All slide decks already exist. Nothing to do.")
        return

    print(f"Starting parallel slide generation with max {args.max_workers} workers...")
    successful = 0
    failed = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_task = {
            executor.submit(run_slide_command, task["command"], task["pptx_path"]): task
            for task in tasks
        }

        for future in as_completed(future_to_task):
            task = future_to_task[future]
            slide_label = f"{task['slide_name']}/{task['page_dir']}"
            try:
                success, error_msg = future.result()
            except Exception as exc:
                success = False
                error_msg = str(exc)

            if success:
                successful += 1
                print(f"Created slide deck for {task['pptx_path']}")
            else:
                failed.append({
                    "slide": slide_label,
                    "pptx_path": task['pptx_path'],
                    "error": error_msg,
                })
                print(f"Failed to create slide deck for {task['pptx_path']}: {error_msg}")

    print("\nExecution summary")
    print("=================")
    print(f"Total slides attempted: {len(tasks)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed slide details:")
        for item in failed:
            print(f"- {item['slide']}: {item['error']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide_deck", type=str, default='all', 
                        help="Path to the slide deck")
    parser.add_argument("--setting", type=str, default="sufficient", 
                        choices=["sufficient", "visual", "creative"],
                        help="Experimental setting.")
    parser.add_argument("--use_library", action="store_true",
                        help="Use the library to create the slide deck.")
    parser.add_argument("--model_name", type=str, default="gpt-4o", 
                        help="Model name to use.")
    parser.add_argument("--max-workers", type=int, default=8,
                        help="Maximum number of slide generation workers to run in parallel.")
    parser.add_argument("--cover_all", action="store_true",
                        help="Cover all slides in the slide deck.")
    args = parser.parse_args()

    main()

"""SlideBench evaluation runner for reference-based and reference-free metrics."""
import os
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def evaluate_single_slide(slide_dirs: str, index: int) -> int:
    """
    Evaluate a single slide index: run ref-based and ref-free evaluations if needed.
    Returns the slide index for bookkeeping.
    """
    print(f"Start evaluating {slide_dirs}/slide_{index} ...")

    # ref-based evaluation
    # find the last executable pptx path
    slide_name = slide_dirs.split("/")[-1]
    refine_path = Path(slide_dirs) / f"slide_{index}"
    try:
        files = os.listdir(refine_path)  # files: 0-9
        files = [file for file in files if file.isdigit()]
    except Exception:
        print(f"No refine directory found for slide {index}: {refine_path}")
        print(f"Finish evaluating slide {index} !")
        return index

    files = sorted(files, key=lambda x: int(x), reverse=True)  # sorted in descending order
    
    for file in files:
        pptx_path = os.path.join(refine_path, f"{file}/refine.pptx")
        if os.path.exists(pptx_path):
            print(f"Using pptx: {pptx_path}")
        else:
            continue
            
        # Run ref-based evaluation if needed
        ref_eval_path = os.path.join(os.path.dirname(pptx_path), "ref_based.txt")
        if os.path.exists(ref_eval_path):
            print(f"Ref-based evaluation already exists: {ref_eval_path}")
        else:
            print(f"Running ref-based evaluation: {ref_eval_path}")
            command = [
                "python", "evaluators/slidebench/page_eval.py",
                "--reference_pptx", f"data/slidebench/examples/{slide_name}/{slide_name}.pptx",
                "--generated_pptx", pptx_path,
                "--reference_page", str(index),
                "--output_path", ref_eval_path,
            ]
            process = subprocess.Popen(command)
            process.wait()
            print(f"Finished ref-based evaluation: {ref_eval_path}")

        # ref-free evaluation
        jpg_path = pptx_path.replace(".pptx", ".jpg")
        ref_free_eval_path = os.path.join(os.path.dirname(pptx_path), "ref_free.json")
        if os.path.exists(jpg_path):
            print(f"Using image: {jpg_path}")
        else:
            print(f"No image found for slide {index}: {jpg_path}")
            continue
        
        if os.path.exists(ref_free_eval_path):
            print(f"Ref-free evaluation already exists: {ref_free_eval_path}")
        else:
            print(f"Running ref-free evaluation: {ref_free_eval_path}")
            command = [
                "python", "evaluators/slidebench/reference_free_eval.py",
                "--image_path", jpg_path,
                "--response_path", ref_free_eval_path,
            ]
            process = subprocess.Popen(command)
            process.wait()
            print(f"Finished ref-free evaluation: {ref_free_eval_path}")

    print(f"Finish evaluating slide {index} !")
    return index

def main() -> None:
    """Run evaluation for all slides in the test output directory."""
    if args.slide_name == 'all':
        slides_list = ['art_photos', 'business', 'design', 'entrepreneur', 'environment', 'food', 'marketing', 'social_media', 'technology']
    else:
        slides_list = [args.slide_name]

    for slide_name in slides_list:
        slide_dirs = f"output/slidebench/{args.test_id}/{slide_name}"
        slides_dirs = os.listdir(slide_dirs)
        # remove the non-directory files
        slides_dirs = [slide_dir for slide_dir in slides_dirs if '.' not in slide_dir]
        index_list = [slide_dir.split("_")[1] for slide_dir in slides_dirs]
        index_list = sorted([int(index) for index in index_list])

        max_workers = args.max_workers if args.max_workers else min(8, (os.cpu_count() or 4))
        print(f"Running evaluations in parallel with max_workers={max_workers}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(evaluate_single_slide, slide_dirs, index) for index in index_list]
            for future in as_completed(futures):
                try:
                    _ = future.result()
                except Exception as exc:
                    print(f"A slide evaluation failed with exception: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('test_id', type=str, help='Test ID (e.g., 20250815_150016)')
    parser.add_argument("--slide_name", type=str, default='all', choices=['all', 'art_photos', 'business', 'design', 'entrepreneur', 'environment', 'food', 'marketing', 'social_media', 'technology'])
    parser.add_argument("--max_workers", type=int, default=None, help="Max number of parallel workers (default: min(8, cpu_count))")

    args = parser.parse_args()

    main()

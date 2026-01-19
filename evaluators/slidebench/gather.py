"""Gather and aggregate SlideBench evaluation results."""
import json
import os
import argparse
from pptx import Presentation


def gather_results(test_id: str, slide_name: str = 'all'):
    """
    Gather evaluation results from all slides and calculate overall metrics.
    """
    if slide_name == 'all':
        names = ['art_photos', 'business', 'design', 'entrepreneur', 'environment', 'food', 'marketing', 'social_media', 'technology']
    else:
        names = [slide_name]

    ref_eval_result = {'match': 0, 'text': 0, 'color': 0, 'position': 0}
    ref_free_eval_result = {'text': 0, 'image': 0, 'layout': 0, 'color': 0}
    
    # Round 1 specific statistics
    round1_ref_eval_result = {'match': 0, 'text': 0, 'color': 0, 'position': 0}
    round1_ref_free_eval_result = {'text': 0, 'image': 0, 'layout': 0, 'color': 0}
    round1_count = 0

    last_round_ref_eval_result = {'match': 0, 'text': 0, 'color': 0, 'position': 0}
    last_round_ref_free_eval_result = {'text': 0, 'image': 0, 'layout': 0, 'color': 0}
    last_round_count = 0
    
    fail_num = 0
    total_num = 0

    for name in names:
        slide_path = os.path.join(f"data/slidebench/examples", name, f"{name}.pptx")
        if not os.path.exists(slide_path):
            print(f"Warning: {slide_path} not found, skipping {name}")
            continue
            
        prs = Presentation(slide_path)
        pages_num = len(prs.slides)
        
        for slide_num in range(1, pages_num + 1):
            slide_dir = os.path.join(f"output/slidebench/{test_id}", name, f"slide_{slide_num}")
            
            # Find the best round (highest average score) for this slide
            best_round_score = -1
            best_round_ref_eval = None
            best_round_ref_free_eval = None
            best_round_num = None
            
            # Track last available round statistics
            last_round_ref_eval = None
            last_round_ref_free_eval = None
            last_round_num = None

            # Check rounds from 1 to 10
            for round_num in range(1, 11):
                round_dir = os.path.join(slide_dir, str(round_num))
                if not os.path.exists(round_dir):
                    continue
                    
                # Check for refined results in this round
                ref_eval_path = os.path.join(round_dir, "ref_based.txt")
                ref_free_eval_path = os.path.join(round_dir, "ref_free.json")
                
                if os.path.exists(ref_eval_path) and os.path.exists(ref_free_eval_path):
                    # Calculate score for this round
                    round_ref_eval = {'match': 0, 'text': 0, 'color': 0, 'position': 0}
                    round_ref_free_eval = {'text': 0, 'image': 0, 'layout': 0, 'color': 0}
                    
                    # Read ref-based evaluation
                    with open(ref_eval_path, 'r') as f:
                        current_ref_eval_result = f.read()
                        current_ref_eval_result = current_ref_eval_result.split('\n')
                        for result in current_ref_eval_result:
                            if 'match' in result:
                                round_ref_eval['match'] = float(result.split(': ')[1])
                            elif 'text' in result:
                                round_ref_eval['text'] = float(result.split(': ')[1])
                            elif 'color' in result:
                                round_ref_eval['color'] = float(result.split(': ')[1])
                            elif 'position' in result:
                                round_ref_eval['position'] = float(result.split(': ')[1])
                    
                    # Read ref-free evaluation
                    with open(ref_free_eval_path, 'r') as f:
                        current_ref_free_eval_result = json.load(f)
                        round_ref_free_eval['text'] = current_ref_free_eval_result['text']['score'] * 20
                        round_ref_free_eval['image'] = current_ref_free_eval_result['image']['score'] * 20
                        round_ref_free_eval['layout'] = current_ref_free_eval_result['layout']['score'] * 20
                        round_ref_free_eval['color'] = current_ref_free_eval_result['color']['score'] * 20

                    # Update last round statistics
                    last_round_num = round_num
                    last_round_ref_eval = round_ref_eval.copy()
                    last_round_ref_free_eval = round_ref_free_eval.copy()
                    
                    # Collect Round 1 statistics
                    if round_num == 1:
                        for key in round1_ref_eval_result:
                            round1_ref_eval_result[key] += round_ref_eval[key]
                        for key in round1_ref_free_eval_result:
                            round1_ref_free_eval_result[key] += round_ref_free_eval[key]
                        round1_count += 1
                        print(f"Collected Round 1 data for {name} slide_{slide_num}")
                    
                    # Calculate average score for this round
                    round_score = (round_ref_eval['match'] + round_ref_eval['text'] + 
                                 round_ref_eval['color'] + round_ref_eval['position'] + 
                                 round_ref_free_eval['text'] + round_ref_free_eval['image'] + 
                                 round_ref_free_eval['layout'] + round_ref_free_eval['color']) / 8
                    
                    # Update best round if this round has higher score
                    if round_score > best_round_score:
                        best_round_score = round_score
                        best_round_ref_eval = round_ref_eval.copy()
                        best_round_ref_free_eval = round_ref_free_eval.copy()
                        best_round_num = round_num
            
            # Use the best round results if found
            if best_round_num is not None:
                print(f"Using round {best_round_num} for {name} slide_{slide_num} (score: {best_round_score:.4f})")

                # Add to overall results
                for key in ref_eval_result:
                    ref_eval_result[key] += best_round_ref_eval[key]
                for key in ref_free_eval_result:
                    ref_free_eval_result[key] += best_round_ref_free_eval[key]

            # Collect last round statistics if available
            if last_round_num is not None and last_round_ref_eval is not None and last_round_ref_free_eval is not None:
                print(f"Using last available round {last_round_num} for {name} slide_{slide_num}")
                for key in last_round_ref_eval_result:
                    last_round_ref_eval_result[key] += last_round_ref_eval[key]
                for key in last_round_ref_free_eval_result:
                    last_round_ref_free_eval_result[key] += last_round_ref_free_eval[key]
                last_round_count += 1

            if best_round_num is None and last_round_num is None:
                fail_num += 1
                print(f"Warning: No evaluation results found for {name} slide_{slide_num}")
                    
            total_num += 1

    # Calculate averages
    if total_num - fail_num > 0:
        for key in ref_eval_result.keys():
            ref_eval_result[key] = ref_eval_result[key] / (total_num - fail_num)
        for key in ref_free_eval_result.keys():
            ref_free_eval_result[key] = ref_free_eval_result[key] / (total_num - fail_num)

    # Calculate Round 1 averages
    if round1_count > 0:
        for key in round1_ref_eval_result.keys():
            round1_ref_eval_result[key] = round1_ref_eval_result[key] / round1_count
        for key in round1_ref_free_eval_result.keys():
            round1_ref_free_eval_result[key] = round1_ref_free_eval_result[key] / round1_count

    # Calculate last round averages
    if last_round_count > 0:
        for key in last_round_ref_eval_result.keys():
            last_round_ref_eval_result[key] = last_round_ref_eval_result[key] / last_round_count
        for key in last_round_ref_free_eval_result.keys():
            last_round_ref_free_eval_result[key] = last_round_ref_free_eval_result[key] / last_round_count

    # Print results
    print(f"Test ID: {test_id}")
    print(f"Success rate: {(total_num - fail_num) / total_num:.4f}")
    print(f"Total slides processed: {total_num}")
    print(f"Failed slides: {fail_num}")
    print(f"Ref-based evaluation results: {ref_eval_result}")
    print(f"Ref-free evaluation results: {ref_free_eval_result}")
    
    overall_score = (ref_eval_result['match'] + ref_eval_result['text'] + 
                    ref_eval_result['color'] + ref_eval_result['position'] + 
                    ref_free_eval_result['text'] + ref_free_eval_result['image'] + 
                    ref_free_eval_result['layout'] + ref_free_eval_result['color']) / 8 * ((total_num - fail_num) / total_num)
    print(f"Overall score: {overall_score:.4f}")
    
    # Print Round 1 specific results
    print(f"\n=== Round 1 Statistics ===")
    print(f"Round 1 slides count: {round1_count}")
    if total_num > 0:
        print(f"Round 1 Success rate: {round1_count / total_num:.4f}")
    else:
        print("Round 1 Success rate: N/A (no slides processed)")
    if round1_count > 0:
        print(f"Round 1 ref-based evaluation results: {round1_ref_eval_result}")
        print(f"Round 1 ref-free evaluation results: {round1_ref_free_eval_result}")
        
        round1_overall_score = (round1_ref_eval_result['match'] + round1_ref_eval_result['text'] + 
                              round1_ref_eval_result['color'] + round1_ref_eval_result['position'] + 
                              round1_ref_free_eval_result['text'] + round1_ref_free_eval_result['image'] + 
                              round1_ref_free_eval_result['layout'] + round1_ref_free_eval_result['color']) / 8 * (round1_count / total_num)
        print(f"Round 1 overall score: {round1_overall_score:.4f}")
        
        # Compare Round 1 vs Best Round performance
        score_improvement = overall_score - round1_overall_score
        print(f"Score improvement from Round 1 to Best Round: {score_improvement:.4f}")
        if score_improvement > 0:
            print(f"Performance improved by {score_improvement:.4f} points")
        elif score_improvement < 0:
            print(f"Performance decreased by {abs(score_improvement):.4f} points")
        else:
            print("Performance remained the same")
    else:
        print("No Round 1 data available")

    print(f"\n=== Last Round Statistics ===")
    print(f"Last round slides count: {last_round_count}")
    if total_num > 0:
        print(f"Last round success rate: {last_round_count / total_num:.4f}")
    else:
        print("Last round success rate: N/A (no slides processed)")
    if last_round_count > 0:
        print(f"Last round ref-based evaluation results: {last_round_ref_eval_result}")
        print(f"Last round ref-free evaluation results: {last_round_ref_free_eval_result}")

        last_round_overall_score = (last_round_ref_eval_result['match'] + last_round_ref_eval_result['text'] + 
                                    last_round_ref_eval_result['color'] + last_round_ref_eval_result['position'] + 
                                    last_round_ref_free_eval_result['text'] + last_round_ref_free_eval_result['image'] + 
                                    last_round_ref_free_eval_result['layout'] + last_round_ref_free_eval_result['color']) / 8 * (last_round_count / total_num)
        print(f"Last round overall score: {last_round_overall_score:.4f}")
    else:
        print("No last round data available")
    
    return ref_eval_result, ref_free_eval_result, overall_score


def main():
    if args.slide_name == 'all':
        slides_list = ['art_photos', 'business', 'design', 'entrepreneur', 'environment', 'food', 'marketing', 'social_media', 'technology']
    else:
        slides_list = [args.slide_name]

    gather_results(args.test_id, args.slide_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('test_id', type=str, help='Test ID (e.g., 20250815_150016)')
    parser.add_argument("--slide_name", type=str, default='all', 
                       choices=['all', 'art_photos', 'business', 'design', 'entrepreneur', 'environment', 'food', 'marketing', 'social_media', 'technology'])

    args = parser.parse_args()

    main()
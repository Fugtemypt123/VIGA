import os
import argparse
import subprocess
import sys

# add runners directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.common import get_image_base64, extract_code_pieces, build_client

SYSTEM_MESSAGE = """You are an expert presentation slides designer who creates modern, fashionable, and stylish slides using Python code. Your job is to generate the required PPTX slide by writing and executing a Python script. Make sure to follow the guidelines below and do not skip any of them:
1. Ensure your code can successfully execute. If needed, you can also write tests to verify your code.
2. Maintain proper spacing and arrangements of elements in the slide: make sure to keep sufficient spacing between different elements; do not make elements overlap or overflow to the slide page.
3. Carefully select the colors of text, shapes, and backgrounds, to ensure all contents are readable.
4. The slides should not look empty or incomplete. When filling the content in the slides, maintain good design and layout."""

INSTRUCTION = """Follow the instruction below to create the slide. 
If the instruction is long and specific, follow the instruction carefully and add all elements as required; 
if it is short and concise, you will need to create some content (text, image, layout) and implement it into the slide.
{}

Finally, your code should save the pptx file to path "{}"."""

IMAGE_INSTRUCTION_DICT = {
    "image": "If you need to use the provided images, refer to the image file names in the instructions.",
    "no_image": "If you need to add images, you will need to generate or search for images yourself.",
}

def main():
    # system message
    messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]}]

    # example-specific instructions
    pptx_path = os.path.join(args.output_dir, f"{args.model_name.replace('-', '_').replace('.', '_')}.pptx")
    image_instruction = IMAGE_INSTRUCTION_DICT["no_image"] if args.no_image else IMAGE_INSTRUCTION_DICT["image"]
    instruction = INSTRUCTION.format(image_instruction, pptx_path)

    if args.library_path is not None:
        library_content = open(args.library_path).read()
        instruction += "\n\n" + library_content
    messages += [{"role": "user", "content": [{"type": "text", "text": instruction}]}]

    if args.example_path is not None:
        example_content = open(args.example_path).read()
        messages += [{"role": "user", "content": [{"type": "text", "text": example_content}]}]
    
    # image input messages
    if args.no_image == False:
        media_dir = os.path.join(args.example_dir, "media")
        image_paths = [os.path.join(media_dir, f) for f in os.listdir(media_dir) if f.endswith(".jpg")]
        for ip in image_paths:
            image_url = get_image_base64(ip)
            messages += [{
                "role": "user",
                "content": [
                    {"type": "text", "text": ip},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }]

    instruction_path = os.path.join(args.example_dir, args.instruction_name)
    instruction = open(instruction_path).read()
    messages.append({"role": "user", "content": [{"type": "text", "text": "## Instruction\n" + instruction}]})

    response = client.chat.completions.create(
        model=args.model_name,
        messages=messages,
        # max_tokens=args.max_tokens,
        # n=args.num_samples,
    )
    response_list = [c.message.content for c in response.choices]


    def save_response(response: str):
        md_output_path = os.path.join(args.output_dir, f"{args.output_name}.md")
        with open(md_output_path, 'w') as fw:
            fw.write(response)

        code = extract_code_pieces(response, concat=True)
        code_output_path = os.path.join(args.output_dir, f"{args.output_name}.py")
        with open(code_output_path, 'w') as fw: fw.write(code)
        
        env = os.environ.copy()
        env['PYTHONPATH'] = f"runners/autopresent_baseline/:{env.get('PYTHONPATH', '')}"

        script_name = code_output_path.rstrip(".py").replace('/', '.')
        try:
            result = subprocess.run(["python", "-m", script_name], capture_output=True, text=True, check=True, env=env)
            print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr
            print("Error: ", error_msg)
            return False


    for r in response_list:
        success = save_response(r)
        if success: 
            subprocess.run(["/usr/bin/python3", "/usr/bin/unoconv", "-f", "jpg", pptx_path], check=True)
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create slide')
    parser.add_argument("--example_dir", type=str, required=True, 
                        help="Path to the example directory.")
    parser.add_argument("--instruction_name", type=str, 
                        default="instruction.txt", 
                        help="Instruction file name.")
    parser.add_argument("--model_name", type=str, default="gpt-4o", 
                        help="Model name to use.")
    parser.add_argument("--max_tokens", type=int, default=4096, 
                        help="Max tokens to generate.")
    parser.add_argument("--num_samples", type=int, default=3, 
                        help="Number of samples to generate.")
    parser.add_argument("--output_name", type=str, default=None, 
                        help="File name to save the output.")
    parser.add_argument("--library_path", type=str, default=None, 
                        help="Path of documentation for expert-designed library functions.")
    parser.add_argument("--example_path", type=str, default=None,)
    parser.add_argument("--no_image", action="store_true", 
                        help="Do not provide image to the model.")
    args = parser.parse_args()

    args.output_dir = args.example_dir + "/baseline"
    os.makedirs(args.output_dir, exist_ok=True)
    
    client = build_client(args.model_name)

    if args.output_name is None:
        args.output_name = args.model_name.replace('-', '_').replace('.', '_')
    
    if args.no_image and args.instruction_name not in [
        "instruction_no_image.txt", "instruction_high_level.txt",
    ]:
        print(
            f"Recommend using instruction_no_image.txt for no_image=True.",
            f"Currently using {args.instruction_name}."
        )

    output_media_dir = os.path.join(args.output_dir, "media")
    if not os.path.exists(output_media_dir):
        process = subprocess.Popen([
            "cp", "-r", 
            os.path.join(args.example_dir, "media"), 
            output_media_dir,
        ])
        process.wait()
        
    main()

# SlidesBench

This repository contains slides generation examples (NL instruction and target PPTX slide), as well as scripts to process the examples.

## Installation

```bash
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## Slides Collection

We collect 310 slide decks from different domains from slideshare.com, each slide deck contains 23.3 slides on average. Specifically, 
- train set: 300 slide decks ([link](https://www.slideshare.net/saved/27575466/autopresent_train))
- test set: 10 slide decks ([link](https://www.slideshare.net/saved/27572898/slides))

We provide the examples created from test slides in `./examples/`. Links to the original slides are: [art_photos](https://www.slideshare.net/slideshow/art-appreciation-subject-and-content-kinds-and-sources-of-subjects/266874796), [business](https://www.slideshare.net/slideshow/airbnb-business-casepptx/262263937), [career](https://www.slideshare.net/slideshow/are-top-grades-enough-ppt/260709991), [design](https://www.slideshare.net/slideshow/graphic-designpptx/251990658), [entrepreneur](https://www.slideshare.net/slideshow/about-entrepreneur-elon-musk-pptx/267322126), [environment](https://www.slideshare.net/slideshow/natural-environment-251014908/251014908), [food](https://www.slideshare.net/slideshow/friends-_-joey-doesn-t-share-food-b1-pptx/270323420), [marketing](https://www.slideshare.net/slideshow/market-around-us-pptx/261987901), [social_media](https://www.slideshare.net/slideshow/contemporary-world-global-media-culturespptx/265794192), [technology](https://www.slideshare.net/slideshow/blockchain-technology1pptx/259106386).

## Example Creation: Single Slide Generation

Given a slide deck, e.g., `examples/art_photos/art_photos.pptx`, we can parse the media and generate NL instructions for each slide.

First, run `parse_media.py` to collect all the images in the given slide.

```bash
python parse_media.py --slides_path examples/art_photos/art_photos.pptx --output_dir examples/art_photos
```

Second, to generate instructions, we start with manually writing three instructions for the first three slides, and save them in `examples/art_photos/slide_{n}/instruction_human.txt`.

Then, first run

```bash
unoconv -f jpg ${your-pptx-file}
```

to convert the pptx file into jpg files, to input to GPT api calls.

Then, run `seed_instruction.py` to generate the NL instruction for all slides, saved as `instruction_model.txt`.

```bash
python seed_instruction.py \
--pptx_path {path-to-your-pptx-file} \  # e.g., "examples/art_photos/art_photos.pptx"
--output_path {folder-path-to-output-the-instructions}  # e.g. "examples/art_photos"
```

## Canonical Program Generation

For each slide, we provide two versions of canonical program to create the slide:

- `code_pptx.py` that uses the `python-pptx` library
- `code_library.py` that uses the functions we designed in the `mylib` library

```bash
python reproduce_code.py --slides_path examples/art_photos
```

This would produce `code_library.py` under each `slide_{n}` directory under `examples/art_photos`.

### To Check the Reference Code Quality

1. Go into the one specific slide directory

```bash
cd examples/art_photos/slide_1
```

2. Run the canonical script to generate the slide

```bash
python code_library.py
```

3. Open the `output.pptx` or transform it to `output.jpg` by

```bash
unoconv -f jpg output.pptx
```

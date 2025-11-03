from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import requests
import re
class VQA:
    def __init__(self, gpu_number=0):
        use_load_8bit= False
        from transformers import AutoProcessor, InstructBlipForConditionalGeneration, InstructBlipProcessor

        
        self.model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b", device_map="auto")
        self.processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

        self.model.eval()
        self.qa_prompt =  "Question: {} Short answer:"
        self.caption_prompt = "\n<image>\na photo of"
        self.max_words = 50
    
    def pre_question(self, question):
        # from LAVIS blip_processors
        question = re.sub(
            r"([.!\"()*#:;~])",
            "",
            question.lower(),
        )
        question = question.rstrip(" ")

        # truncate question
        question_words = question.split(" ")
        if len(question_words) > self.max_words:
            question = " ".join(question_words[: self.max_words])

        return question

    def qa(self, image_path, question):
        image = Image.open(image_path)
        question = self.pre_question(question)
        inputs = self.processor(images=image, text=question, return_tensors="pt", padding="longest").to(self.model.device)
        generated_ids = self.model.generate(**inputs, length_penalty=-1, num_beams=5, max_length=30, min_length=1,
                                            do_sample=False, top_p=0.9, repetition_penalty=1.0,
                                            num_return_sequences=1, temperature=1)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return generated_text[0]

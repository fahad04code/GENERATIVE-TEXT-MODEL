## GENERATIVE TEXT MODEL


*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: HASHMI SYED FAHAD

*INTERN ID*: CT08DL815

*DOMAIN*: ARTIFICIAL INTELLIGENCE

*DURATION*: 8 WEEEKS

*MENTOR*: NEELA SANTOSH



## Description :

## Project Overview :-


I’m a student passionate about natural language processing, and gpt2-text-generator is my latest project—a Python script that uses GPT-2 from Hugging Face to generate creative text from prompts. After struggling with LSTM models that produced nonsense like “musensiblet” and navigating memory limits on my laptop (~1025 MB), I wanted to explore pre-trained models for better results. This script takes prompts like “In a quiet village, there lived” and generates coherent continuations, saving them to a file. It’s a key piece for my AI internship portfolio, showcasing my growth in NLP, coding, and optimization, building on my previous projects like BART summarization and neural style transfer.


## What It Does

This script leverages GPT-2 to generate text from user-defined prompts. It uses the transformers library to load the GPT-2 model and tokenizer, ensuring proper padding and attention masks for stable generation. The script includes three example prompts, generating up to 200 tokens per prompt with parameters like temperature=0.6 for coherence, top_k=40, and top_p=0.95 for diversity, while preventing repetition with no_repeat_ngram_size=2. Outputs are printed to the console and saved to generated_text_gpt2.txt, making it a versatile tool for creative writing, story generation, or learning about language models.


## Lesson Learned :-

My LSTM experience reached a milestone through this project because my previous training efforts produced models which generated non-sensical results. The pre-trained nature of the GPT-2 model provided  a revolutionary solution which achieved superior results through minimal training requirements. The Hugging Face transformers library became my  learning platform to handle generation challenges through pad token configuration because it let me discover this essential step which I  overlooked in previous work. The process of modifying generation parameters enabled me to develop the ability to create content  which maintained both creativity and logical consistency which I lacked in my BART summarization script.

Memory problems persisted as a major issue throughout the project development. I needed to exercise caution with GPT-2  which consumes 500 MB of memory because it fits within my system's 1025 MB space while needing  more optimization compared to my VGG19 style transfer model which demands 1.5 GB of space.  Through several test runs I developed the ability to monitor memory usage which helped maintain system stability. The implementation  of model loading and generation error handling in my script provided robustness that I learned from my speech-to-text  project because API failures surprised me.

The act of writing this README file and creating meaningful code comments required me to simplify technical ideas which  I am developing for my internship opportunities. Writing different prompts for NLP analysis revealed how the technology can generate  creative outputs that simulate storytelling elements for games. The GPT-2 model showed more unpredictability in its  output than my BART project and required specific temperature and top-k modifications to improve results. The project  intensified my fascination with NLP and provided me with future work suggestions including GPT-2 fine-tuning  and chatbot creation that I plan to study further.



## Requirements:-

- torch>=1.9.0 
- transformers>=4.20.0


## Setup and Usage :-

1. Install Dependencies: Install required libraries using requirements.txt:

```
pip install -r requirements.txt

```


2. Run:

```
python text_generator.py

```

3. Customize: Edit the prompts list in text_generator.py with your own prompts, or import the generation logic:

```
from transformers import GPT2LMHeadModel, GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
inputs = tokenizer("Your prompt", return_tensors="pt", padding=True)
outputs = model.generate(inputs["input_ids"], max_length=200)
print(tokenizer.decode(outputs[0]))

```


## Notes :-


- Memory: GPT-2 uses ~500 MB; suitable for low-memory systems.



- Performance: ~5-10 seconds per prompt on CPU.



- Future Plans: Add CLI for prompts, fine-tune on custom data.


## OUTPUT:-

![Image](https://github.com/user-attachments/assets/b3a1172e-ecd8-406a-a2c5-dc0166b5b83e)

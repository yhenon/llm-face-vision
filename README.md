# LLM Face Vision benchmarks


This repository provides a framework for comparing capabilities of Vision-Language Models (VLMs) to dedicated face recognition systems. 

# Recognition

![lfw example](./assets/illustrate2.png)

Results

![LLM face recognition](assets/recognition_metrics.png?raw=true "Title")


Benchmarked models:

	Commercial API VLMs:
	- Anthropic Haiku
	- OpenAI GPT-4o-mini
	- Grok-2 vision
	- Gemini-2-flash-lite

	Open source VLMs:
	- LLava Next (https://github.com/LLaVA-VL/LLaVA-NeXT)
	- Gemma-3-27B
	- Mistral-small-3.1-24b-instruct

	Face recognition systems:
	- Insightface arface-resnet-100 (https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)

Datasets:

	AgeDB-30
	LFW
	CALFW
	CPLFW


# Counting 

We evaluate the capacity of VLMs to count faces in a scene. Counting is performed on Wideface validation data.

Benchmarked models:

	Commercial API VLMs:
	- Anthropic Haiku
	- OpenAI GPT-4o-mini
	- Grok-2 vision
	- Gemini-2-flash-lite

Since models sometimes refuse a query

![LLM face counting](assets/counting_metrics.png?raw=true "Title")

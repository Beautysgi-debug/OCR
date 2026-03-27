# OCR
This paper presents a multimodal automated system
for assessing spoken English responses in educational settings.
The system integrates three key technologies: Optical Charac-
ter Recognition (OCR) for student identity verification from
ID cards, automatic speech recognition (ASR) using OpenAI’s
Whisper model for transcribing spoken responses, and Large
Language Models (LLMs) for evaluating the quality of tran-
scribed answers. The pipeline first verifies student identity by
comparing an OCR-extracted ID number with a spoken ID
number, then proceeds to transcribe and evaluate spoken English
answers against predefined assessment criteria. We evaluate
the system using DeepSeek-Chat as the LLM examiner across
five oral English questions with varying answer qualities. Ex-
perimental results demonstrate that the system can reliably
distinguish between high-quality, grammatically poor, and off-
topic responses, achieving consistent scoring aligned with human
expectations. The system offers a scalable solution for large-
cohort classroom assessment while acknowledging limitations
in evaluating pronunciation and fluency. We discuss system
architecture, evaluation methodology, model behavior analysis,
and directions for future improvement.

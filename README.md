Status: In Progress

Language identification has been of great importance for most NLP models today. However, as NLP models continue to improve, the underperformance of multilingual LLMs on non-English texts is still persistent. In particular, the existence of local slang phrases that are used orally instead of being formally written or taught remains largely underrepresente in the training and testing datasets. This leads to the issue of language inclusion, especially for languages whose dialects or informal phrases are rich in their everyday use. 

Thus, we aim to present a model that correctly identifies the language on the input of foreign conventional phrases. We trained both Naive Bayes and BERT models on Spanish, French, and Italian language subtitles from various movies and TV shows. This approach allowed us to focus on real-world language use and practice text classification with a greater emphasis on linguistic diversity and reduced bias.

My task for this project is to build up the BERT-based model. While the initial traiing process showed promising results, I wanted to see what other ways I could build upon and improve BERT's performance, especially considering the fact that the movie dialouge does not represent the full scope of informal diction within a given langauge. This led to implementing an ensamble model inspired by: Abarna, S., Sheeba, J.I., & Devaneyan, S.P. (2022). An ensemble model for idioms and literal text classification using knowledge-enabled BERT in deep learning. Measurement: Sensors, 24, 100500. https://doi.org/10.1016/j.measen.2022.100500.

What has been completed: 
- Completed training of baseline BERT model
- Implemented a Naive Bayes model (by Ms. Kharel)
- Designed preprocessing pipeline for informal subtitle data

Next steps: 
- Research different BERT transformers that are well equipped for language identification
- Building a stacking ensamble model following the methodology of Abarna et al.
- Attempt to combine the stacking model with the Naive Bayes model for a potential hybrid system

Repo status: This repository is currently private while active development continues.

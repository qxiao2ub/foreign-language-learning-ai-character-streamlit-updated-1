# Foreign Language Learning AI Character App

A Streamlit version of the **Foreign Language Learning AI Character App** notebook prototype by Isabella Fu, mentored by Qingyang Xiao.

The app lets learners practice a target language with an AI-style character tutor named Luna. It includes conversation practice, tutor feedback, points, mini-games, learner clustering, and adaptive difficulty.

## Features

- AI character conversation in the selected target language
- Tutor feedback after each learner message
- Reward points and learning motivation
- Vocabulary quiz, role-play, and sentence expansion mini-games
- Learner profile clustering with scikit-learn
- Reinforcement-learning-style adaptive difficulty selection
- Practice-history CSV download
- Optional Hugging Face model mode using `google/flan-t5-small`
- English-practice language guard: when English is selected and the learner enters Chinese text, Luna prompts them to practice in English

## Project credits

- Author: Isabella Fu
- Mentor: Qingyang Xiao

## Repository files

```text
foreign-language-learning-ai-character-streamlit/
├── streamlit_app.py
├── requirements.txt
├── requirements-full.txt
├── README.md
├── LICENSE
├── .gitignore
├── .streamlit/
│   └── config.toml
└── notebooks/
    └── original_foreign_language_learning_ai_app.ipynb
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The default app uses a lightweight rule-based fallback response engine, so it starts quickly and does not require a model download.

## Optional local Hugging Face LLM mode

To enable the sidebar option **Use local Hugging Face LLM**, install the full requirements:

```bash
pip install -r requirements-full.txt
streamlit run streamlit_app.py
```

Then enable **Use local Hugging Face LLM** in the app sidebar. The model used is `google/flan-t5-small`.

## Deploy on Streamlit Community Cloud

1. Upload these files to a GitHub repository.
2. Go to Streamlit Community Cloud.
3. Create a new app from the GitHub repository.
4. Set the main file path to:

```text
streamlit_app.py
```

5. Deploy.

For the fastest deployment, keep `requirements.txt` as-is. Use `requirements-full.txt` only if you want the optional local Hugging Face model mode and your deployment environment has enough memory.

## Notes

This project is a research and product prototype. For a production app, consider adding user login, database storage, stronger grammar correction, speech input, text-to-speech, and a dedicated backend API.

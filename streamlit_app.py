"""Streamlit version of the Foreign Language Learning AI Character App.

This app was converted from the original Gradio/Jupyter notebook prototype.
It can run with a lightweight rule-based fallback by default, and optionally
use a local Hugging Face text2text-generation model when dependencies are
installed and the sidebar option is enabled.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

APP_NAME = "Foreign Language Learning AI Character App"
AUTHOR = "Isabella Fu"
MENTOR = "Qingyang Xiao"
ADVISOR = MENTOR
MODEL_NAME = "google/flan-t5-small"

SUPPORTED_LANGUAGES = [
    "English",
    "Spanish",
    "French",
    "German",
    "Italian",
    "Portuguese",
    "Chinese",
    "Japanese",
    "Korean",
    "Arabic",
]

DIFFICULTY_LEVELS = ["Beginner", "Intermediate", "Advanced"]

CHARACTER_PERSONA = """
You are Luna, a friendly AI language-learning character.
You help the learner practice a target foreign language.
You must respond in the selected target language.
You should be warm, simple, encouraging, and educational.
Ask one short follow-up question to continue the conversation.
"""

VOCAB_BANK = {
    "Spanish": [("hello", "hola"), ("thank you", "gracias"), ("water", "agua"), ("friend", "amigo"), ("school", "escuela")],
    "French": [("hello", "bonjour"), ("thank you", "merci"), ("water", "eau"), ("friend", "ami"), ("school", "école")],
    "Chinese": [("hello", "你好"), ("thank you", "谢谢"), ("water", "水"), ("friend", "朋友"), ("school", "学校")],
    "Japanese": [("hello", "こんにちは"), ("thank you", "ありがとう"), ("water", "水"), ("friend", "友達"), ("school", "学校")],
    "Korean": [("hello", "안녕하세요"), ("thank you", "감사합니다"), ("water", "물"), ("friend", "친구"), ("school", "학교")],
    "English": [("hola", "hello"), ("gracias", "thank you"), ("agua", "water"), ("amigo", "friend"), ("escuela", "school")],
}

ROLEPLAY_SCENARIOS = [
    "ordering food at a small restaurant",
    "introducing yourself to a new classmate",
    "asking for directions in a city",
    "talking about your favorite hobby",
    "checking in at a hotel",
    "buying a train ticket",
]

FEATURE_COLUMNS = ["turns", "avg_words", "mistakes_per_turn", "points_per_turn", "mini_game_wins"]


@dataclass
class LearnerState:
    """Keeps one user's learning progress for the current Streamlit session."""

    username: str = "Guest Learner"
    target_language: str = "Spanish"
    difficulty: str = "Beginner"
    total_points: int = 0
    conversation_turns: int = 0
    total_words: int = 0
    estimated_mistakes: int = 0
    mini_game_wins: int = 0
    history: List[Dict] = field(default_factory=list)


class DifficultyBandit:
    """Simple epsilon-greedy bandit for reinforcement-learning-style difficulty selection."""

    def __init__(self, difficulties: Optional[List[str]] = None, epsilon: float = 0.15):
        self.difficulties = difficulties or DIFFICULTY_LEVELS
        self.epsilon = epsilon
        self.counts = {difficulty: 0 for difficulty in self.difficulties}
        self.values = {difficulty: 10.0 for difficulty in self.difficulties}

    def choose_difficulty(self) -> str:
        if random.random() < self.epsilon:
            return random.choice(self.difficulties)
        return max(self.values, key=self.values.get)

    def update(self, difficulty: str, reward: float) -> None:
        self.counts[difficulty] += 1
        n = self.counts[difficulty]
        self.values[difficulty] += (reward - self.values[difficulty]) / n

    def summary(self) -> Dict:
        return {
            "counts": self.counts,
            "estimated_values": {key: round(value, 2) for key, value in self.values.items()},
        }


def build_chat_prompt(user_message: str, target_language: str, difficulty: str) -> str:
    return f"""
{CHARACTER_PERSONA}
Target language: {target_language}
Learner difficulty level: {difficulty}
Learner message: {user_message}
Task: Reply naturally in {target_language}, match {difficulty} level, keep it concise, and ask one follow-up question.
"""


def build_feedback_prompt(user_message: str, target_language: str, difficulty: str) -> str:
    return f"""
You are a language tutor.
Target language: {target_language}
Learner difficulty: {difficulty}
Learner sentence: {user_message}
Give brief feedback in English: corrected version if needed, grammar issue, better phrase, and encouragement.
"""


def fallback_character_reply(user_message: str, target_language: str, difficulty: str) -> str:
    templates = {
        "English": "Great! I understand. Can you tell me a little more about that?",
        "Spanish": "¡Muy bien! Entiendo. ¿Puedes contarme un poco más sobre eso?",
        "French": "Très bien ! Je comprends. Peux-tu m'en dire un peu plus ?",
        "German": "Sehr gut! Ich verstehe. Kannst du mir ein bisschen mehr darüber erzählen?",
        "Italian": "Molto bene! Capisco. Puoi dirmi qualcosa di più?",
        "Portuguese": "Muito bem! Eu entendo. Você pode me contar um pouco mais?",
        "Chinese": "很好！我明白你的意思。你可以再多说一点吗？",
        "Japanese": "いいですね！わかりました。もう少し詳しく教えてくれますか？",
        "Korean": "좋아요! 이해했어요. 조금 더 말해 줄 수 있나요?",
        "Arabic": "جيد جدًا! أفهم. هل يمكنك أن تخبرني بالمزيد؟",
    }
    return templates.get(target_language, templates["English"])


def fallback_feedback(user_message: str, target_language: str, difficulty: str) -> str:
    if len(user_message.split()) < 3:
        return "Good start. Try to write a fuller sentence with a subject, verb, and one extra detail."
    if user_message and user_message[0].islower():
        return "Nice practice. One small improvement: begin the sentence with a capital letter when the language uses capitalization."
    return "Nice sentence. For better fluency, add one connector word, one descriptive phrase, or one follow-up question."


@st.cache_resource(show_spinner=False)
def load_huggingface_generator(model_name: str):
    """Load an optional local Hugging Face generator.

    The app works without this dependency. If transformers/torch are unavailable,
    the app keeps running with rule-based fallback responses.
    """

    try:
        from transformers import pipeline

        generator = pipeline("text2text-generation", model=model_name, max_new_tokens=160)
        return generator, None
    except Exception as exc:  # pragma: no cover - UI fallback path
        return None, str(exc)


def generate_with_llm(prompt: str, fallback: str, use_llm: bool = False):
    if not use_llm:
        return fallback, None

    generator, error = load_huggingface_generator(MODEL_NAME)
    if generator is None:
        return fallback, error

    try:
        output = generator(prompt)
        text = output[0].get("generated_text", "").strip()
        return text if text else fallback, None
    except Exception as exc:  # pragma: no cover - UI fallback path
        return fallback, str(exc)


def contains_chinese_characters(text: str) -> bool:
    """Return True when the learner input contains Chinese CJK characters."""

    return bool(re.search(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]", text))


def validate_target_language_input(user_message: str, target_language: str) -> Optional[Dict]:
    """Create a tutor response when the learner uses the wrong practice language."""

    if target_language == "English" and contains_chinese_characters(user_message):
        return {
            "ai_reply": "Please speak and practice in English.",
            "feedback": "Your target practice language is English. Try rewriting your message in English so Luna can help you improve.",
            "points": 0,
            "llm_error": None,
        }
    return None


def estimate_mistakes_simple(text: str) -> int:
    mistakes = 0
    if len(text.strip()) == 0:
        mistakes += 1
    if len(text.split()) < 3:
        mistakes += 1
    if text and text[0].islower():
        mistakes += 1
    mistakes += len(re.findall(r"\s{2,}", text))
    return mistakes


def calculate_reward(user_message: str, feedback: str, difficulty: str) -> int:
    points = 10
    words = len(user_message.split())
    mistakes = estimate_mistakes_simple(user_message)
    if words >= 8:
        points += 5
    if mistakes == 0:
        points += 5
    if difficulty == "Intermediate":
        points += 3
    if difficulty == "Advanced":
        points += 6
    return max(points - 2 * mistakes, 1)


def update_learner_state(state: LearnerState, user_message: str, ai_reply: str, feedback: str, points: int) -> LearnerState:
    mistakes = estimate_mistakes_simple(user_message)
    words = len(user_message.split())
    state.total_points += points
    state.conversation_turns += 1
    state.total_words += words
    state.estimated_mistakes += mistakes
    state.history.append(
        {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_language": state.target_language,
            "difficulty": state.difficulty,
            "user_message": user_message,
            "ai_reply": ai_reply,
            "feedback": feedback,
            "points": points,
            "words": words,
            "estimated_mistakes": mistakes,
        }
    )
    return state


def generate_synthetic_training_profiles(n: int = 120, random_seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    rows = []
    for _ in range(n):
        profile_type = rng.choice(["beginner", "steady", "advanced", "needs_support"])
        if profile_type == "beginner":
            turns, avg_words, mistakes, points, wins = (
                rng.integers(1, 15),
                rng.normal(4, 1),
                rng.normal(1.8, 0.5),
                rng.normal(8, 2),
                rng.integers(0, 3),
            )
        elif profile_type == "steady":
            turns, avg_words, mistakes, points, wins = (
                rng.integers(10, 60),
                rng.normal(9, 2),
                rng.normal(0.8, 0.3),
                rng.normal(14, 3),
                rng.integers(1, 8),
            )
        elif profile_type == "advanced":
            turns, avg_words, mistakes, points, wins = (
                rng.integers(30, 120),
                rng.normal(18, 4),
                rng.normal(0.4, 0.2),
                rng.normal(20, 4),
                rng.integers(5, 20),
            )
        else:
            turns, avg_words, mistakes, points, wins = (
                rng.integers(5, 40),
                rng.normal(7, 2),
                rng.normal(2.5, 0.6),
                rng.normal(7, 2),
                rng.integers(0, 5),
            )
        rows.append(
            {
                "turns": max(float(turns), 1.0),
                "avg_words": max(float(avg_words), 1.0),
                "mistakes_per_turn": max(float(mistakes), 0.0),
                "points_per_turn": max(float(points), 1.0),
                "mini_game_wins": max(float(wins), 0.0),
                "true_profile": profile_type,
            }
        )
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner=False)
def build_profile_model():
    profile_df = generate_synthetic_training_profiles()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(profile_df[FEATURE_COLUMNS])
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    profile_df["cluster"] = kmeans.fit_predict(scaled_features)
    cluster_summary = profile_df.groupby("cluster")[FEATURE_COLUMNS].mean().round(2)
    cluster_labels = {cluster_id: interpret_cluster(cluster_id, cluster_summary) for cluster_id in cluster_summary.index}
    return scaler, kmeans, cluster_summary, cluster_labels


def interpret_cluster(cluster_id: int, summary_df: pd.DataFrame) -> str:
    row = summary_df.loc[cluster_id]
    if row["avg_words"] >= 14 and row["mistakes_per_turn"] < 1:
        return "Advanced conversational learner"
    if row["mistakes_per_turn"] >= 1.8:
        return "Learner needing grammar and sentence-structure support"
    if row["turns"] < 20:
        return "Early-stage beginner learner"
    return "Consistent conversational learner"


def learner_features_from_state(state: LearnerState) -> pd.DataFrame:
    turns = max(state.conversation_turns, 1)
    feature_row = {
        "turns": state.conversation_turns,
        "avg_words": state.total_words / turns,
        "mistakes_per_turn": state.estimated_mistakes / turns,
        "points_per_turn": state.total_points / turns,
        "mini_game_wins": state.mini_game_wins,
    }
    return pd.DataFrame([feature_row], columns=FEATURE_COLUMNS)


def predict_learner_profile(state: LearnerState) -> str:
    scaler, kmeans, _cluster_summary, cluster_labels = build_profile_model()
    features = learner_features_from_state(state)
    cluster = int(kmeans.predict(scaler.transform(features))[0])
    return cluster_labels.get(cluster, "General learner profile")


def language_learning_turn(
    state: LearnerState,
    bandit: DifficultyBandit,
    user_message: str,
    target_language: str,
    selected_difficulty: str,
    auto_adapt_difficulty: bool = True,
    use_llm: bool = False,
) -> Tuple[Dict, LearnerState, DifficultyBandit]:
    if not user_message or not user_message.strip():
        result = {
            "ai_reply": "Please enter a message first.",
            "feedback": "No input detected.",
            "points": 0,
            "total_points": state.total_points,
            "profile": predict_learner_profile(state),
            "difficulty": state.difficulty,
            "bandit": bandit.summary(),
            "llm_error": None,
        }
        return result, state, bandit

    state.target_language = target_language
    difficulty = bandit.choose_difficulty() if auto_adapt_difficulty else selected_difficulty
    state.difficulty = difficulty

    validation_result = validate_target_language_input(user_message.strip(), target_language)
    if validation_result:
        learner_profile = predict_learner_profile(state)
        state.history.append(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "target_language": state.target_language,
                "difficulty": state.difficulty,
                "user_message": user_message,
                "ai_reply": validation_result["ai_reply"],
                "feedback": validation_result["feedback"],
                "points": validation_result["points"],
                "words": len(user_message.split()),
                "estimated_mistakes": 0,
            }
        )
        result = {
            "ai_reply": validation_result["ai_reply"],
            "feedback": validation_result["feedback"],
            "points": validation_result["points"],
            "total_points": state.total_points,
            "profile": learner_profile,
            "difficulty": difficulty,
            "bandit": bandit.summary(),
            "llm_error": validation_result["llm_error"],
        }
        return result, state, bandit

    ai_reply, chat_error = generate_with_llm(
        build_chat_prompt(user_message, target_language, difficulty),
        fallback_character_reply(user_message, target_language, difficulty),
        use_llm=use_llm,
    )
    feedback, feedback_error = generate_with_llm(
        build_feedback_prompt(user_message, target_language, difficulty),
        fallback_feedback(user_message, target_language, difficulty),
        use_llm=use_llm,
    )

    points = calculate_reward(user_message, feedback, difficulty)
    state = update_learner_state(state, user_message, ai_reply, feedback, points)
    learner_profile = predict_learner_profile(state)
    bandit.update(difficulty, points - 2 * estimate_mistakes_simple(user_message))

    result = {
        "ai_reply": ai_reply,
        "feedback": feedback,
        "points": points,
        "total_points": state.total_points,
        "profile": learner_profile,
        "difficulty": difficulty,
        "bandit": bandit.summary(),
        "llm_error": chat_error or feedback_error,
    }
    return result, state, bandit


def generate_vocab_question(target_language: str) -> Tuple[str, str]:
    bank = VOCAB_BANK.get(target_language, VOCAB_BANK["Spanish"])
    english, target = random.choice(bank)
    return f"Translate '{english}' into {target_language}.", target


def check_vocab_answer(answer: str, correct_answer: str) -> Tuple[bool, str]:
    is_correct = answer.strip().lower() == correct_answer.strip().lower()
    if is_correct:
        return True, "Correct! +15 points."
    return False, f"Good try. The expected answer is: {correct_answer}"


def generate_roleplay_prompt(target_language: str, difficulty: str) -> str:
    scenario = random.choice(ROLEPLAY_SCENARIOS)
    return f"Role-play in {target_language}: You are {scenario}. Write one sentence at {difficulty} level to continue."


def sentence_expansion_challenge(target_language: str) -> str:
    return f"Sentence challenge in {target_language}: Write a longer sentence using one emotion word, one time word, and one reason."


def initialize_session_state() -> None:
    if "learner_state" not in st.session_state:
        st.session_state.learner_state = LearnerState()
    if "bandit" not in st.session_state:
        st.session_state.bandit = DifficultyBandit()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_vocab_answer" not in st.session_state:
        st.session_state.current_vocab_answer = None
    if "current_vocab_question" not in st.session_state:
        st.session_state.current_vocab_question = "Click the button to generate a vocabulary question."
    if "roleplay_prompt" not in st.session_state:
        st.session_state.roleplay_prompt = "Click the button to generate a role-play scenario."
    if "sentence_prompt" not in st.session_state:
        st.session_state.sentence_prompt = "Click the button to generate a sentence expansion challenge."


def reset_session_state() -> None:
    st.session_state.learner_state = LearnerState()
    st.session_state.bandit = DifficultyBandit()
    st.session_state.messages = []
    st.session_state.current_vocab_answer = None
    st.session_state.current_vocab_question = "Click the button to generate a vocabulary question."
    st.session_state.roleplay_prompt = "Click the button to generate a role-play scenario."
    st.session_state.sentence_prompt = "Click the button to generate a sentence expansion challenge."


def learner_history_dataframe(state: LearnerState) -> pd.DataFrame:
    if not state.history:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "target_language",
                "difficulty",
                "user_message",
                "ai_reply",
                "feedback",
                "points",
                "words",
                "estimated_mistakes",
            ]
        )
    return pd.DataFrame(state.history)


def render_sidebar() -> Tuple[str, str, bool, bool]:
    st.sidebar.header("Practice Settings")
    st.sidebar.markdown(
        f"""
**Project Info**  
**Author:** {AUTHOR}  
**Mentor:** {MENTOR}
"""
    )
    st.sidebar.divider()
    state: LearnerState = st.session_state.learner_state
    state.username = st.sidebar.text_input("Learner name", value=state.username)
    target_language = st.sidebar.selectbox(
        "Target language",
        SUPPORTED_LANGUAGES,
        index=SUPPORTED_LANGUAGES.index(state.target_language) if state.target_language in SUPPORTED_LANGUAGES else 1,
    )
    difficulty = st.sidebar.selectbox("Manual difficulty", DIFFICULTY_LEVELS, index=DIFFICULTY_LEVELS.index(state.difficulty))
    auto_adapt = st.sidebar.checkbox("Use adaptive difficulty", value=True)
    use_llm = st.sidebar.checkbox(
        "Use local Hugging Face LLM",
        value=False,
        help="Default is off so the app runs quickly. Install the optional packages in requirements-full.txt to enable this.",
    )

    st.sidebar.divider()
    if st.sidebar.button("Reset progress"):
        reset_session_state()
        st.rerun()

    st.sidebar.caption("The app runs with a rule-based fallback by default. Optional LLM mode uses google/flan-t5-small.")
    return target_language, difficulty, auto_adapt, use_llm


def render_metrics(state: LearnerState) -> None:
    turns = max(state.conversation_turns, 1)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total points", state.total_points)
    col2.metric("Conversation turns", state.conversation_turns)
    col3.metric("Avg words / turn", round(state.total_words / turns, 1))
    col4.metric("Mini-game wins", state.mini_game_wins)


def render_chat_tab(target_language: str, difficulty: str, auto_adapt: bool, use_llm: bool) -> None:
    st.subheader("AI Character Conversation")
    st.caption("Practice with Luna, an encouraging AI language-learning character.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Write a sentence in your target language...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        result, state, bandit = language_learning_turn(
            st.session_state.learner_state,
            st.session_state.bandit,
            prompt,
            target_language,
            difficulty,
            auto_adapt_difficulty=auto_adapt,
            use_llm=use_llm,
        )
        st.session_state.learner_state = state
        st.session_state.bandit = bandit

        assistant_message = f"""
{result['ai_reply']}

---
**Tutor feedback:** {result['feedback']}

**Points earned:** {result['points']}  
**Total points:** {result['total_points']}  
**Current difficulty:** {result['difficulty']}  
**Learner profile:** {result['profile']}  
**Adaptive difficulty estimates:** `{json.dumps(result['bandit']['estimated_values'])}`
"""
        st.session_state.messages.append({"role": "assistant", "content": assistant_message})
        with st.chat_message("assistant"):
            st.markdown(assistant_message)
            if result.get("llm_error"):
                st.info("LLM mode could not load, so the app used the fallback response.")


def render_mini_games_tab(target_language: str, difficulty: str) -> None:
    st.subheader("Mini-Games")
    vocab_tab, roleplay_tab, expansion_tab = st.tabs(["Vocabulary Quiz", "Role-Play", "Sentence Expansion"])

    with vocab_tab:
        st.write(st.session_state.current_vocab_question)
        if st.button("Generate vocabulary question"):
            question, answer = generate_vocab_question(target_language)
            st.session_state.current_vocab_question = question
            st.session_state.current_vocab_answer = answer
            st.rerun()

        vocab_answer = st.text_input("Your answer", key="vocab_answer_input")
        if st.button("Check answer"):
            correct_answer = st.session_state.current_vocab_answer
            if not correct_answer:
                st.warning("Generate a vocabulary question first.")
            else:
                is_correct, message = check_vocab_answer(vocab_answer, correct_answer)
                if is_correct:
                    st.session_state.learner_state.total_points += 15
                    st.session_state.learner_state.mini_game_wins += 1
                    st.success(f"{message} Total points: {st.session_state.learner_state.total_points}")
                else:
                    st.info(message)

    with roleplay_tab:
        st.write(st.session_state.roleplay_prompt)
        if st.button("Generate role-play scenario"):
            st.session_state.roleplay_prompt = generate_roleplay_prompt(target_language, difficulty)
            st.rerun()

    with expansion_tab:
        st.write(st.session_state.sentence_prompt)
        if st.button("Generate sentence challenge"):
            st.session_state.sentence_prompt = sentence_expansion_challenge(target_language)
            st.rerun()


def render_analytics_tab(state: LearnerState) -> None:
    st.subheader("Learner Analytics")
    profile = predict_learner_profile(state)
    st.write(f"**Current learner profile:** {profile}")

    bandit_summary = st.session_state.bandit.summary()
    st.write("**Adaptive difficulty estimates**")
    st.json(bandit_summary)

    history_df = learner_history_dataframe(state)
    st.write("**Practice history**")
    st.dataframe(history_df, use_container_width=True, hide_index=True)

    csv = history_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download practice history CSV",
        data=csv,
        file_name="language_learning_history.csv",
        mime="text/csv",
        disabled=history_df.empty,
    )

    with st.expander("Synthetic learner-clustering model summary"):
        _scaler, _kmeans, cluster_summary, cluster_labels = build_profile_model()
        st.dataframe(cluster_summary, use_container_width=True)
        st.json({int(k): v for k, v in cluster_labels.items()})


def render_about_tab() -> None:
    st.subheader("About this prototype")
    st.markdown(
        f"""
**{APP_NAME}** is a Streamlit conversion of the original Jupyter/Gradio prototype.

Core features included from the notebook:

- AI character conversation in the selected target language
- Tutor feedback after each user message
- Reward points and motivation loop
- Mini-games for vocabulary, role-play, and sentence expansion
- Learner profile clustering with scikit-learn
- Reinforcement-learning-style adaptive difficulty selection

**Author:** {AUTHOR}  
**Mentor:** {MENTOR}

The app is designed to run from GitHub on Streamlit Community Cloud. By default, it uses a lightweight fallback conversation engine so it starts quickly. Optional local Hugging Face model mode can be enabled in the sidebar after installing the extra dependencies.
"""
    )


def main() -> None:
    st.set_page_config(page_title=APP_NAME, page_icon="🌍", layout="wide")
    initialize_session_state()

    target_language, difficulty, auto_adapt, use_llm = render_sidebar()
    state: LearnerState = st.session_state.learner_state
    state.target_language = target_language

    st.title(APP_NAME)
    st.caption("Practice a foreign language with Luna, a friendly AI character tutor.")
    render_metrics(state)

    chat_tab, mini_games_tab, analytics_tab, about_tab = st.tabs(["Conversation", "Mini-Games", "Learner Analytics", "About"])
    with chat_tab:
        render_chat_tab(target_language, difficulty, auto_adapt, use_llm)
    with mini_games_tab:
        render_mini_games_tab(target_language, difficulty)
    with analytics_tab:
        render_analytics_tab(st.session_state.learner_state)
    with about_tab:
        render_about_tab()


if __name__ == "__main__":
    main()

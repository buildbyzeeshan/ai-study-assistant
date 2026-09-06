import streamlit as st
import numpy as np

from google import genai
from pypdf import PdfReader


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="wide"
)


# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.block-container {
    max-width: 950px;
    padding-top: 2rem;
}

[data-testid="stChatMessage"] {
    border-radius: 15px;
    padding: 10px;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# Gemini Client
# -----------------------------
client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)


# -----------------------------
# Embedding Function
# -----------------------------
def embed(t):

    r = client.models.embed_content(
        model="gemini-embedding-001",
        contents=t
    )

    return np.array(
        r.embeddings[0].values
    )


# -----------------------------
# App UI
# -----------------------------
st.title("📚 AI Study Assistant")

st.write(
    "Apni notes upload karo aur unse sawal pucho!"
)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:

    st.header("📚 Study Assistant")

    st.write("📄 PDF Summary")
    st.write("📝 Quiz Generator")
    st.write("💬 Notes Q&A")


# -----------------------------
# Chat History Setup
# -----------------------------
if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


# -----------------------------
# PDF Upload
# -----------------------------
pdf = st.file_uploader(
    "Notes (PDF)",
    type="pdf"
)


# -----------------------------
# PDF Processing
# -----------------------------
if pdf:

    if "vectors" not in st.session_state:

        reader = PdfReader(pdf)

        st.session_state.text = ""
        st.session_state.chunks = []
        st.session_state.sources = []

        for page_num, page in enumerate(reader.pages):

            page_text = (
                page.extract_text() or ""
            )

            st.session_state.text += page_text

            page_chunks = [
                page_text[i:i+500]
                for i in range(
                    0,
                    len(page_text),
                    500
                )
            ]

            for c in page_chunks:

                st.session_state.chunks.append(c)

                st.session_state.sources.append(
                    f"Page {page_num + 1}"
                )

        st.session_state.vectors = [
            embed(c)
            for c in st.session_state.chunks
        ]

        st.success(
            "✅ Notes ready! Ab sawal poocho."
        )


    # -----------------------------
    # Summary Button
    # -----------------------------
    if st.button("📄 Summarize Notes"):

        summary_prompt = f"""
        In notes ka short aur clear summary
        Roman Urdu mein do.

        Important points bullets mein batao.

        Notes:
        {st.session_state.text}
        """

        summary = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=summary_prompt
        )

        st.write("### 📝 Notes Summary")
        st.write(summary.text)


    # -----------------------------
    # Quiz Button
    # -----------------------------
    if st.button("📝 Generate Quiz"):

        quiz_prompt = f"""
        In notes se 5 test questions banao.

        Har question ke 4 options do:
        A, B, C, D.

        End mein correct answers bhi do.

        Notes:
        {st.session_state.text}
        """

        quiz = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=quiz_prompt
        )

        st.write("### 🧠 Quiz")
        st.write(quiz.text)


    # -----------------------------
    # Old Chat Show
    # -----------------------------
    for message in st.session_state.chat_history:

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["text"]
            )


    # -----------------------------
    # Question Input
    # -----------------------------
    q = st.chat_input(
        "Apna sawal:"
    )


    # -----------------------------
    # RAG Question Answer
    # -----------------------------
    if q:

        # User question save
        st.session_state.chat_history.append({
            "role": "user",
            "text": q
        })

        # Question embedding
        qv = embed(q)

        # Similarity scores
        scores = [
            np.dot(qv, v)
            for v in st.session_state.vectors
        ]

        # Best chunk index
        best_index = np.argmax(scores)

        # Relevant chunk
        context = (
            st.session_state
            .chunks[best_index]
        )

        # Source page
        source = (
            st.session_state
            .sources[best_index]
        )

        # Prompt
        prompt = f"""
        Sirf in notes se Roman Urdu mein
        jawab do.

        Context:
        {context}

        Sawal:
        {q}
        """

        # Gemini answer
        ans = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        # Assistant answer save
        st.session_state.chat_history.append({
            "role": "assistant",
            "text": f"{ans.text}\n\n📌 Source: {source}"
        })

        # Current question show
        with st.chat_message("user"):
            st.write(q)

        # Current answer show
        with st.chat_message("assistant"):
            st.write(ans.text)
            st.write(f"📌 Source: {source}")


    # -----------------------------
    # Clear Chat Button
    # -----------------------------
    if st.button("🧹 Clear Chat"):

        st.session_state.chat_history = []

        st.rerun()
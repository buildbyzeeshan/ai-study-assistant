import streamlit as st
import numpy as np
import time
import hashlib
import io

from google import genai
from pypdf import PdfReader


# ==================================================
# 1. PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="wide"
)


# ==================================================
# 2. CUSTOM CSS
# ==================================================

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


# ==================================================
# 3. GEMINI CLIENT
# ==================================================

client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)


# ==================================================
# 4. SAFE GEMINI FUNCTION
# ==================================================

def ask_gemini(prompt, retries=3):

    for attempt in range(retries):

        try:

            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )

            return response

        except Exception as e:

            error_text = str(e).lower()

            temporary_error = any(
                word in error_text
                for word in [
                    "500",
                    "502",
                    "503",
                    "504",
                    "unavailable",
                    "high demand",
                    "timeout",
                    "temporarily"
                ]
            )

            if temporary_error and attempt < retries - 1:

                time.sleep(2 * (attempt + 1))

            else:

                return None

    return None


# ==================================================
# 5. SAFE EMBEDDING FUNCTION
# ==================================================

def embed(t, retries=3):

    if not t.strip():
        return None

    for attempt in range(retries):

        try:

            r = client.models.embed_content(
                model="gemini-embedding-001",
                contents=t
            )

            vector = np.array(
                r.embeddings[0].values
            )

            norm = np.linalg.norm(vector)

            if norm == 0:
                return None

            # Normalize vector
            return vector / norm

        except Exception:

            if attempt < retries - 1:

                time.sleep(2 * (attempt + 1))

            else:

                return None

    return None


# ==================================================
# 6. RESET OLD PDF DATA
# ==================================================

def reset_pdf_data():

    keys = [
        "text",
        "chunks",
        "sources",
        "vectors",
        "summary",
        "quiz",
        "chat_history"
    ]

    for key in keys:

        st.session_state.pop(
            key,
            None
        )

    st.session_state.chat_history = []
    st.session_state.summary = None
    st.session_state.quiz = None


# ==================================================
# 7. APP UI
# ==================================================

st.title("📚 AI Study Assistant")

st.write(
    "Apni notes upload karo aur unse sawal pucho!"
)


# ==================================================
# 8. SIDEBAR
# ==================================================

with st.sidebar:

    st.header("📚 Study Assistant")

    st.write("📄 PDF Summary")
    st.write("📝 Quiz Generator")
    st.write("💬 Notes Q&A")

    st.divider()

    st.write("🤖 Powered by Zeeshan")


# ==================================================
# 9. SESSION STATE
# ==================================================

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


if "summary" not in st.session_state:

    st.session_state.summary = None


if "quiz" not in st.session_state:

    st.session_state.quiz = None


# ==================================================
# 10. PDF UPLOAD
# ==================================================

pdf = st.file_uploader(
    "Notes (PDF)",
    type="pdf"
)


# ==================================================
# 11. PDF PROCESSING
# ==================================================

if pdf:

    pdf_bytes = pdf.getvalue()


    # ----------------------------------------------
    # Maximum PDF size = 200 MB
    # ----------------------------------------------

    MAX_PDF_SIZE = 200 * 1024 * 1024

    if len(pdf_bytes) > MAX_PDF_SIZE:

        st.error(
            "❌ PDF 200 MB se bari hai. "
            "Please 200 MB ya us se choti PDF upload karein."
        )

        st.stop()


    # ----------------------------------------------
    # Unique PDF ID
    # ----------------------------------------------

    current_pdf_id = hashlib.sha256(
        pdf_bytes
    ).hexdigest()


    # ----------------------------------------------
    # Detect New PDF
    # ----------------------------------------------

    if st.session_state.get(
        "pdf_id"
    ) != current_pdf_id:

        reset_pdf_data()

        st.session_state.pdf_id = (
            current_pdf_id
        )


    # ----------------------------------------------
    # Process PDF only once
    # ----------------------------------------------

    if "vectors" not in st.session_state:

        try:

            reader = PdfReader(
                io.BytesIO(pdf_bytes)
            )

        except Exception:

            st.error(
                "❌ PDF open nahi ho saki. "
                "File corrupt ya invalid ho sakti hai."
            )

            st.stop()


        st.session_state.text = ""
        st.session_state.chunks = []
        st.session_state.sources = []


        # ------------------------------------------
        # Extract PDF Text
        # ------------------------------------------

        with st.spinner(
            "📚 PDF process ho rahi hai..."
        ):

            for page_num, page in enumerate(
                reader.pages
            ):

                try:

                    page_text = (
                        page.extract_text()
                        or ""
                    )

                except Exception:

                    page_text = ""


                page_text = page_text.strip()


                if not page_text:
                    continue


                # Full PDF text
                st.session_state.text += (
                    page_text + "\n"
                )


                # ----------------------------------
                # Create 500-character chunks
                # ----------------------------------

                page_chunks = [

                    page_text[i:i + 500]

                    for i in range(
                        0,
                        len(page_text),
                        500
                    )
                ]


                # ----------------------------------
                # Save chunks + pages
                # ----------------------------------

                for c in page_chunks:

                    c = c.strip()

                    if not c:
                        continue


                    st.session_state.chunks.append(
                        c
                    )

                    st.session_state.sources.append(
                        f"Page {page_num + 1}"
                    )


        # ------------------------------------------
        # No readable text found
        # ------------------------------------------

        if not st.session_state.chunks:

            st.error(
                "❌ Is PDF se readable text nahi mila. "
                "Ho sakta hai PDF scanned images par based ho."
            )

            st.stop()


        # ------------------------------------------
        # Create Embeddings
        # ------------------------------------------

        vectors = []

        embedding_failed = False


        with st.spinner(
            "🧠 Notes samajh raha hoon..."
        ):

            for chunk in (
                st.session_state.chunks
            ):

                vector = embed(chunk)


                if vector is None:

                    embedding_failed = True
                    break


                vectors.append(
                    vector
                )


        # ------------------------------------------
        # Embedding Error
        # ------------------------------------------

        if embedding_failed:

            st.error(
                "⚠️ AI service abhi busy hai. "
                "Notes process nahi ho sake. "
                "Please thori der baad dobara try karein."
            )

            st.stop()


        st.session_state.vectors = (
            vectors
        )


        st.success(
            "✅ Notes ready! Ab sawal poocho."
        )


    # ==================================================
    # 12. SUMMARY
    # ==================================================

    if st.button(
        "📄 Summarize Notes"
    ):

        summary_prompt = f"""
        In notes ka short aur clear summary
        Roman Urdu mein do.

        Important points bullets mein batao.

        Sirf diye gaye notes ki information
        use karo.

        Notes:

        {st.session_state.text}
        """


        with st.spinner(
            "📝 Summary ban rahi hai..."
        ):

            summary = ask_gemini(
                summary_prompt
            )


        if summary is None:

            st.error(
                "⚠️ Gemini service abhi busy hai. "
                "Summary generate nahi ho saki. "
                "Please dobara try karein."
            )

        else:

            st.session_state.summary = (
                summary.text
            )


    # ----------------------------------------------
    # Show Saved Summary
    # ----------------------------------------------

    if st.session_state.summary:

        st.write(
            "### 📝 Notes Summary"
        )

        st.write(
            st.session_state.summary
        )


    # ==================================================
    # 13. QUIZ GENERATOR
    # ==================================================

    if st.button(
        "📝 Generate Quiz"
    ):

        quiz_prompt = f"""
        In notes se 10 test questions banao.

        Har question ke 4 options do:

        A, B, C, D

        End mein correct answers bhi do.

        Sirf in notes ki information
        use karo.

        Notes:

        {st.session_state.text}
        """


        with st.spinner(
            "🧠 Quiz ban raha hai..."
        ):

            quiz = ask_gemini(
                quiz_prompt
            )


        if quiz is None:

            st.error(
                "⚠️ Quiz generate nahi ho saka. "
                "AI service temporarily busy hai. "
                "Please dobara try karein."
            )

        else:

            st.session_state.quiz = (
                quiz.text
            )


    # ----------------------------------------------
    # Show Saved Quiz
    # ----------------------------------------------

    if st.session_state.quiz:

        st.write(
            "### 🧠 Quiz"
        )

        st.write(
            st.session_state.quiz
        )


    # ==================================================
    # 14. OLD CHAT HISTORY
    # ==================================================

    for message in (
        st.session_state.chat_history
    ):

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["text"]
            )


    # ==================================================
    # 15. QUESTION INPUT
    # ==================================================

    q = st.chat_input(
        "PDF se sawal poocho..."
    )


    # ==================================================
    # 16. RAG QUESTION ANSWER
    # ==================================================

    if q:

        # ------------------------------------------
        # Show Current Question
        # ------------------------------------------

        with st.chat_message(
            "user"
        ):

            st.write(q)


        # ------------------------------------------
        # Save User Question
        # ------------------------------------------

        st.session_state.chat_history.append({

            "role": "user",

            "text": q

        })


        # ------------------------------------------
        # Question Embedding
        # ------------------------------------------

        qv = embed(q)


        # ------------------------------------------
        # Embedding Failed
        # ------------------------------------------

        if qv is None:

            error_message = (
                "⚠️ AI service abhi busy hai. "
                "Please sawal dobara try karein."
            )


            with st.chat_message(
                "assistant"
            ):

                st.write(
                    error_message
                )


            st.session_state.chat_history.append({

                "role": "assistant",

                "text": error_message

            })


        else:

            # --------------------------------------
            # Similarity Scores
            # --------------------------------------

            scores = [

                np.dot(qv, v)

                for v in (
                    st.session_state.vectors
                )
            ]


            # --------------------------------------
            # Top 3 Relevant Chunks
            # --------------------------------------

            top_indices = np.argsort(
                scores
            )[-3:][::-1]


            best_score = float(
                scores[top_indices[0]]
            )


            # --------------------------------------
            # Relevance Threshold
            # --------------------------------------

            RELEVANCE_THRESHOLD = 0.40


            # --------------------------------------
            # Question Not Related to PDF
            # --------------------------------------

            if best_score < (
                RELEVANCE_THRESHOLD
            ):

                not_found_message = (
                    "🤔 Mujhe is PDF mein is sawal ka "
                    "jawab nahi mila.\n\n"
                    "Please PDF se related ya "
                    "milta-julta sawal dobara poochain."
                )


                with st.chat_message(
                    "assistant"
                ):

                    st.write(
                        not_found_message
                    )


                st.session_state.chat_history.append({

                    "role": "assistant",

                    "text": not_found_message

                })


            else:

                # ----------------------------------
                # Build Top Context
                # ----------------------------------

                context_parts = []

                selected_sources = []


                for index in top_indices:

                    context_parts.append(
                        f"""
                        [{st.session_state.sources[index]}]

                        {st.session_state.chunks[index]}
                        """
                    )


                    selected_sources.append(
                        st.session_state.sources[index]
                    )


                # ----------------------------------
                # Join Context
                # ----------------------------------

                context = "\n\n".join(
                    context_parts
                )


                # ----------------------------------
                # Remove Duplicate Pages
                # ----------------------------------

                selected_sources = list(
                    dict.fromkeys(
                        selected_sources
                    )
                )


                source = ", ".join(
                    selected_sources
                )


                # ----------------------------------
                # Strict RAG Prompt
                # ----------------------------------

                prompt = f"""
                Tum AI Study Assistant ho.

                Sirf neeche diye gaye PDF context
                se jawab do.

                User ka sawal exact words mein hona
                zaroori nahi hai.

                Agar user ka sawal meaning ke hisaab
                se context se related hai to clear
                Roman Urdu mein jawab do.

                Apni general knowledge se information
                add mat karo.

                Agar jawab context mein available
                nahi hai to sirf:

                NOT_FOUND

                return karo.

                Context:

                {context}

                Sawal:

                {q}
                """


                # ----------------------------------
                # Ask Gemini
                # ----------------------------------

                with st.spinner(
                    "🤖 Answer dhoond raha hoon..."
                ):

                    ans = ask_gemini(
                        prompt
                    )


                # ----------------------------------
                # Gemini Failed
                # ----------------------------------

                if ans is None:

                    error_message = (
                        "⚠️ AI service abhi busy hai. "
                        "Please thori der baad "
                        "sawal dobara try karein."
                    )


                    with st.chat_message(
                        "assistant"
                    ):

                        st.write(
                            error_message
                        )


                    st.session_state.chat_history.append({

                        "role": "assistant",

                        "text": error_message

                    })


                # ----------------------------------
                # Gemini Says Answer Not Found
                # ----------------------------------

                elif (
                    ans.text.strip()
                    .upper()
                    .startswith("NOT_FOUND")
                ):

                    not_found_message = (
                        "🤔 Mujhe is PDF mein is sawal ka "
                        "jawab nahi mila.\n\n"
                        "Please PDF se related ya "
                        "milta-julta sawal dobara poochain."
                    )


                    with st.chat_message(
                        "assistant"
                    ):

                        st.write(
                            not_found_message
                        )


                    st.session_state.chat_history.append({

                        "role": "assistant",

                        "text": not_found_message

                    })


                # ----------------------------------
                # Correct Answer
                # ----------------------------------

                else:

                    final_answer = (
                        f"{ans.text}"
                        f"\n\n📌 Source: {source}"
                    )


                    with st.chat_message(
                        "assistant"
                    ):

                        st.write(
                            ans.text
                        )

                        st.caption(
                            f"📌 Source: {source}"
                        )


                    # Save Answer
                    st.session_state.chat_history.append({

                        "role": "assistant",

                        "text": final_answer

                    })


    # ==================================================
    # 17. CLEAR CHAT
    # ==================================================

    if st.button(
        "🧹 Clear Chat"
    ):

        st.session_state.chat_history = []

        st.rerun()

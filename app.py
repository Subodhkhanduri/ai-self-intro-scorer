import json
import streamlit as st
import pandas as pd
import plotly.express as px
import docx 
import PyPDF2


from logic import analyze_transcript


st.set_page_config(
    page_title="AI Communication Coach",
    page_icon="🗣️",
    layout="wide",
)

# ---------------------- Sidebar ---------------------- #
st.sidebar.title("🧪 Test Settings")

audio_duration = st.sidebar.number_input(
    "Audio duration (seconds)",
    min_value=1.0,
    max_value=600.0,
    value=60.0,
    step=1.0,
    help="Approximate length of the self-introduction audio."
)

show_raw_json = st.sidebar.checkbox("Show raw JSON output", value=False)

# ---------------------- Main layout ---------------------- #
st.title("🗣️ AI Communication Coach – Self-Introduction Scorer")

st.write("### Transcript Input")

sample_text = (
    "Hello everyone, my name is Alex. I am 12 years old and I study in grade 7 at Greenfield School. "
    "I live with my parents and younger sister. In my free time, I love to play football and read mystery books. "
    "My dream is to become a scientist one day. Thank you for listening!"
)

file_uploaded = st.file_uploader(
    "Upload transcript file (PDF, DOCX, or TXT)",
    type=["pdf", "docx", "txt"],
    help="Upload a transcript file to automatically extract text.",
)

transcript_box = st.empty()

extracted_text = ""

if file_uploaded:
    file_type = file_uploaded.name.split(".")[-1].lower()

    if file_type == "txt":
        extracted_text = file_uploaded.read().decode("utf-8")

    elif file_type == "pdf":
        pdf_reader = PyPDF2.PdfReader(file_uploaded)
        for page in pdf_reader.pages:
            extracted_text += page.extract_text() + "\n"

    elif file_type == "docx":
        doc = docx.Document(file_uploaded)
        extracted_text = "\n".join([p.text for p in doc.paragraphs])

    extracted_text = extracted_text.strip()
    st.session_state["transcript_text"] = extracted_text

transcript = transcript_box.text_area(
    "Transcript",
    value=st.session_state.get("transcript_text", ""),
    height=220,
    help="Upload a file or paste your transcript here."
)

col_btn, _ = st.columns([1, 3])
with col_btn:
    score_btn = st.button("🚀 Score my introduction")

if score_btn:
    if not transcript.strip():
        st.error("Please enter a transcript before scoring.")
    else:
        with st.spinner("Analyzing transcript..."):
            result = analyze_transcript(
                transcript_text=transcript,
                audio_duration_seconds=audio_duration,
            )

        overall_score = result.get("overall_score", 0.0)
        criteria = result.get("criteria_scores", {})
        feedback = result.get("feedback", {})

        # --------- Top-level KPIs --------- #
        kpi1, kpi2 = st.columns(2)
        with kpi1:
            st.metric("Overall Score (0–100)", f"{overall_score:.1f}")
        with kpi2:
            wpm = feedback.get("wpm", 0.0)
            st.metric("Words per minute (WPM)", f"{wpm:.1f}")

        st.markdown("---")

        # --------- Per-criterion breakdown --------- #
        # Build a DataFrame for visuals
        rows = []
        for crit_key, crit_val in criteria.items():
            label = crit_key.replace("_", " ").title()
            score = crit_val.get("score", 0)
            max_score = crit_val.get("max_score", 1)
            pct = (score / max_score) * 100 if max_score > 0 else 0
            rows.append(
                {
                    "Criterion": label,
                    "Score": score,
                    "Max Score": max_score,
                    "Percent": pct,
                }
            )

        df_scores = pd.DataFrame(rows)

        st.subheader("📊 Per-Criterion Score Overview")

        c1, c2 = st.columns([2, 1])

        with c1:
            # Bar chart (0–100 normalized)
            fig_bar = px.bar(
                df_scores,
                x="Criterion",
                y="Percent",
                range_y=[0, 100],
                text=df_scores["Percent"].map(lambda x: f"{x:.1f}%"),
                labels={"Percent": "Score (%)"},
                title="Normalized rubric scores by criterion",
            )
            fig_bar.update_traces(textposition="outside")
            st.plotly_chart(fig_bar, use_container_width=True)

        with c2:
            st.dataframe(
                df_scores[["Criterion", "Score", "Max Score"]],
                use_container_width=True,
            )

        # --------- Radar chart for quick profile --------- #
        st.subheader("📈 Skill Profile (Radar)")

        radar_df = df_scores.copy()
        radar_df["Angle"] = radar_df["Criterion"]
        radar_df["Value"] = radar_df["Percent"]

        fig_radar = px.line_polar(
            radar_df,
            r="Value",
            theta="Angle",
            line_close=True,
            range_r=[0, 100],
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("---")

        # --------- Detailed feedback by section --------- #
        st.subheader("🩺 Detailed Rubric Feedback")

        # Content & Structure
        cs = criteria.get("content_structure", {})
        cs_details = cs.get("details", {})

        with st.expander("📚 Content & Structure", expanded=True):
            st.write(f"**Score:** {cs.get('score', 0)}/{cs.get('max_score', 40)}")

            sal = cs_details.get("salutation", {})
            st.markdown(f"- **Salutation**: {sal.get('score', 0)}/5 — level: `{sal.get('level')}`")

            kw = cs_details.get("keywords", {})
            st.markdown(
                f"- **Keyword coverage**: {kw.get('score', 0)}/30  \n"
                f"  - Must-have present: `{', '.join(kw.get('must_have_present', [])) or 'None'}`  \n"
                f"  - Must-have missing: `{', '.join(kw.get('must_have_missing', [])) or 'None'}`  \n"
                f"  - Good-to-have present: `{', '.join(kw.get('good_to_have_present', [])) or 'None'}`  \n"
                f"  - Good-to-have missing: `{', '.join(kw.get('good_to_have_missing', [])) or 'None'}`"
            )

            flow = cs_details.get("flow", {})
            st.markdown(
                f"- **Flow**: {flow.get('score', 0)}/5 — "
                f"order followed: `{flow.get('order_followed')}`"
            )

        # Speech Rate
        sr = criteria.get("speech_rate", {})
        sr_det = sr.get("details", {})
        with st.expander("⏱️ Speech Rate"):
            st.write(f"**Score:** {sr.get('score', 0)}/{sr.get('max_score', 10)}")
            st.write(
                f"- Band: `{sr_det.get('band')}`  "
                f"(estimated WPM: **{sr_det.get('wpm', 0.0):.1f}**)"
            )

        # Language & Grammar
        lg = criteria.get("language_grammar", {})
        lg_det = lg.get("details", {})
        with st.expander("📖 Language & Grammar"):
            st.write(f"**Score:** {lg.get('score', 0)}/{lg.get('max_score', 20)}")
            gram = lg_det.get("grammar", {})
            vocab = lg_det.get("vocabulary", {})
            st.markdown(
                f"- **Grammar**: {gram.get('score', 0)}/10  \n"
                f"  - Quality fraction: `{gram.get('fraction', 0.0):.2f}`  \n"
                f"  - Error count: `{len(gram.get('errors', []))}`"
            )
            st.markdown(
                f"- **Vocabulary richness (TTR)**: {vocab.get('score', 0)}/10  \n"
                f"  - TTR: `{vocab.get('ttr', 0.0):.2f}`"
            )

        # Clarity
        cl = criteria.get("clarity", {})
        cl_det = cl.get("details", {})
        with st.expander("🔍 Clarity (Filler Words)"):
            st.write(f"**Score:** {cl.get('score', 0)}/{cl.get('max_score', 15)}")
            st.markdown(
                f"- Filler rate: `{cl_det.get('rate', 0.0):.2f}%`  \n"
                f"- Filler count: `{cl_det.get('filler_count', 0)}`  \n"
                f"- Filler words found: `{', '.join(cl_det.get('filler_words', [])) or 'None'}`"
            )

        # Engagement
        eng = criteria.get("engagement", {})
        eng_det = eng.get("details", {})
        with st.expander("✨ Engagement (Sentiment)"):
            st.write(f"**Score:** {eng.get('score', 0)}/{eng.get('max_score', 15)}")
            st.markdown(
                f"- Positive sentiment: `{eng_det.get('positive', 0.0):.2f}`  \n"
                f"- Compound score: `{eng_det.get('compound', 0.0):.2f}`"
            )

        # --------- Raw JSON (optional) --------- #
        if show_raw_json:
            st.markdown("---")
            st.subheader("🧾 Raw JSON Output")
            st.code(json.dumps(result, indent=2), language="json")

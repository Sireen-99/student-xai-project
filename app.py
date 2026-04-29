import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os 

@st.cache_resource
def load_model():
    model_path = os.path.join(
        os.path.dirname(__file__),
        'model.pkl'
    )
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def get_db():
    db_path = os.path.join(
        os.path.dirname(__file__),
        'learning_analytics.db'
    )
    return sqlite3.connect(db_path)

model = load_model()

st.set_page_config(
    page_title="Student XAI Portal",
    page_icon="🎓",
    layout="wide"
)

if 'page' not in st.session_state:
    st.session_state.page = 'login'
if 'user_type' not in st.session_state:
    st.session_state.user_type = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'selected_module' not in st.session_state:
    st.session_state.selected_module = None
if 'selected_student' not in st.session_state:
    st.session_state.selected_student = None

def show_login():
    st.title("🎓 Student XAI Portal")
    st.subheader("Early Warning System")
    st.divider()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        user_type = st.selectbox("Login as", ["Student", "Instructor"])
        user_id = st.text_input("ID", placeholder="Enter your ID")

        if st.button("Login", use_container_width=True):
            if user_id:
                conn = get_db()
                try:
                    uid = int(user_id)
                    if user_type == "Student":
                        cursor = conn.execute(
                            f"SELECT * FROM Student WHERE student_id = {uid}"
                        )
                        result = cursor.fetchall()
                        if result:
                            st.session_state.user_type = 'student'
                            st.session_state.user_id = uid
                            st.session_state.page = 'student_home'
                            st.rerun()
                        else:
                            st.error("Student ID not found!")
                    else:
                        cursor = conn.execute(
                            f"SELECT * FROM Instructor WHERE instructor_id = {uid}"
                        )
                        result = cursor.fetchall()
                        if result:
                            st.session_state.user_type = 'instructor'
                            st.session_state.user_id = uid
                            st.session_state.page = 'instructor_home'
                            st.rerun()
                        else:
                            st.error("Instructor ID not found!")
                except ValueError:
                    st.error("Please enter a valid number")
                finally:
                    conn.close()
            else:
                st.warning("Please enter your ID")

def show_student_home():
    conn = get_db()
    student_id = st.session_state.user_id

    st.title(f"🎓 Welcome, Student {student_id}")

    if st.button("← Logout"):
        st.session_state.page = 'login'
        st.rerun()

    st.divider()
    st.subheader("Your Modules")

    cursor = conn.execute(
        f"SELECT DISTINCT module_id, presentation, final_result FROM Supervises WHERE student_id = {student_id}"
    )
    modules = cursor.fetchall()

    if not modules:
        st.info("No modules found.")
        conn.close()
        return

    for row in modules:
        module_id, presentation, final_result = row
        module_name = 'BBB' if module_id == 0 else 'FFF'

        cursor2 = conn.execute(f"""
            SELECT p.risk_probability, p.prediction_result
            FROM Has_Prediction hp
            JOIN Prediction p ON hp.prediction_id = p.prediction_id
            JOIN Window w ON p.window_id = w.window_id
            WHERE hp.student_id = {student_id}
            AND hp.module_id = {module_id}
            ORDER BY w.window_number DESC
            LIMIT 1
        """)
        pred = cursor2.fetchone()

        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.markdown(f"**{module_name}** — {presentation}")
            st.caption(f"Final result: {final_result}")
        with col2:
            if pred:
                risk = pred[0]
                if risk >= 0.7:
                    st.error(f"Risk: {risk:.1%}")
                elif risk >= 0.5:
                    st.warning(f"Risk: {risk:.1%}")
                else:
                    st.success(f"Risk: {risk:.1%}")
        with col3:
            if st.button("View", key=f"view_{module_id}_{presentation}"):
                st.session_state.selected_module = module_id
                st.session_state.page = 'student_module'
                st.rerun()
        st.divider()

    conn.close()

def show_student_module():
    conn = get_db()
    student_id = st.session_state.user_id
    module_id = st.session_state.selected_module
    module_name = 'BBB' if module_id == 0 else 'FFF'

    if st.button("← Back to modules"):
        st.session_state.page = 'student_home'
        st.rerun()

    st.title(f"📚 {module_name}")
    st.divider()

    cursor = conn.execute(f"""
        SELECT p.risk_probability, p.prediction_result, w.window_number
        FROM Has_Prediction hp
        JOIN Prediction p ON hp.prediction_id = p.prediction_id
        JOIN Window w ON p.window_id = w.window_id
        WHERE hp.student_id = {student_id}
        AND hp.module_id = {module_id}
        ORDER BY w.window_number
    """)
    predictions = cursor.fetchall()

    if not predictions:
        st.info("No predictions available.")
        conn.close()
        return

    latest_risk = predictions[-1][0]
    window_nums = [p[2] for p in predictions]
    risk_vals = [p[0] for p in predictions]

    col1, col2, col3 = st.columns(3)
    with col1:
        if latest_risk >= 0.7:
            st.error(f"Current Risk\n{latest_risk:.1%}")
        elif latest_risk >= 0.5:
            st.warning(f"Current Risk\n{latest_risk:.1%}")
        else:
            st.success(f"Current Risk\n{latest_risk:.1%}")

    cursor2 = conn.execute(f"""
        SELECT wp.num_assessments, wp.avg_score
        FROM Has_Prediction hp
        JOIN Prediction p ON hp.prediction_id = p.prediction_id
        JOIN Window_Performance wp ON wp.prediction_id = p.prediction_id
        JOIN Window w ON p.window_id = w.window_id
        WHERE hp.student_id = {student_id}
        AND hp.module_id = {module_id}
        ORDER BY w.window_number DESC
        LIMIT 1
    """)
    perf = cursor2.fetchone()

    if perf:
        with col2:
            st.metric("Assignments", int(perf[0]))
        with col3:
            st.metric("Avg Score", f"{perf[1]:.1f}%")

    st.divider()
    st.subheader("📈 Risk over time")

    fig, ax = plt.subplots(figsize=(10, 3))
    colors = ['#E24B4A' if r >= 0.7 else '#EF9F27' if r >= 0.5 else '#639922'
              for r in risk_vals]
    ax.bar(window_nums, risk_vals, color=colors)
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Window (every 2 weeks)')
    ax.set_ylabel('Risk probability')
    ax.set_ylim(0, 1)
    st.pyplot(fig)
    plt.close()

    st.divider()
    st.subheader("💡 Your recommendations")

    cursor3 = conn.execute(f"""
        SELECT p.prediction_id
        FROM Has_Prediction hp
        JOIN Prediction p ON hp.prediction_id = p.prediction_id
        JOIN Window w ON p.window_id = w.window_id
        WHERE hp.student_id = {student_id}
        AND hp.module_id = {module_id}
        ORDER BY w.window_number DESC
        LIMIT 1
    """)
    last_pred = cursor3.fetchone()

    if last_pred:
        pred_id = last_pred[0]

        cursor4 = conn.execute(
            f"SELECT rec_text FROM AI_Recommendation WHERE prediction_id = {pred_id}"
        )
        ai_recs = cursor4.fetchall()

        if ai_recs:
            st.markdown("**AI Recommendations:**")
            for rec in ai_recs:
                st.info(rec[0])

        cursor5 = conn.execute(f"""
            SELECT dr.rec_text, i.name
            FROM Doctor_Recommendation dr
            JOIN Instructor i ON dr.instructor_id = i.instructor_id
            WHERE dr.prediction_id = {pred_id}
        """)
        doc_recs = cursor5.fetchall()

        if doc_recs:
            st.markdown("**Instructor Recommendations:**")
            for rec in doc_recs:
                st.success(rec[0])
                st.caption(f"From: {rec[1]}")

    conn.close()

def show_instructor_home():
    conn = get_db()
    instructor_id = st.session_state.user_id

    cursor = conn.execute(
        f"SELECT name FROM Instructor WHERE instructor_id = {instructor_id}"
    )
    instructor = cursor.fetchone()
    name = instructor[0] if instructor else f"Instructor {instructor_id}"

    st.title(f"👨‍🏫 {name}")

    if st.button("← Logout"):
        st.session_state.page = 'login'
        st.rerun()

    st.divider()

    cursor2 = conn.execute(
        f"SELECT DISTINCT student_id, module_id, presentation FROM Supervises WHERE instructor_id = {instructor_id}"
    )
    students = cursor2.fetchall()

    risk_data = []
    for row in students:
        sid, mid, pres = row
        cursor3 = conn.execute(f"""
            SELECT p.risk_probability
            FROM Has_Prediction hp
            JOIN Prediction p ON hp.prediction_id = p.prediction_id
            JOIN Window w ON p.window_id = w.window_id
            WHERE hp.student_id = {sid}
            AND hp.module_id = {mid}
            ORDER BY w.window_number DESC
            LIMIT 1
        """)
        pred = cursor3.fetchone()
        if pred:
            risk_data.append({
                'student_id': sid,
                'module_id': mid,
                'presentation': pres,
                'risk': pred[0]
            })

    if not risk_data:
        st.info("No students found.")
        conn.close()
        return

    high   = len([r for r in risk_data if r['risk'] >= 0.7])
    medium = len([r for r in risk_data if 0.5 <= r['risk'] < 0.7])
    safe   = len([r for r in risk_data if r['risk'] < 0.5])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", len(risk_data))
    col2.metric("High Risk", high)
    col3.metric("Medium Risk", medium)
    col4.metric("Safe", safe)

    st.divider()
    st.subheader("Students list")

    filter_opt = st.selectbox(
        "Filter by risk",
        ["All", "High Risk", "Medium Risk", "Safe"]
    )

    if filter_opt == "High Risk":
        filtered = [r for r in risk_data if r['risk'] >= 0.7]
    elif filter_opt == "Medium Risk":
        filtered = [r for r in risk_data if 0.5 <= r['risk'] < 0.7]
    elif filter_opt == "Safe":
        filtered = [r for r in risk_data if r['risk'] < 0.5]
    else:
        filtered = risk_data

    filtered = sorted(filtered, key=lambda x: x['risk'], reverse=True)

    for row in filtered:
        module_name = 'BBB' if row['module_id'] == 0 else 'FFF'
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        with col1:
            st.write(f"Student {row['student_id']}")
        with col2:
            st.caption(f"{module_name} — {row['presentation']}")
        with col3:
            if row['risk'] >= 0.7:
                st.error(f"{row['risk']:.1%}")
            elif row['risk'] >= 0.5:
                st.warning(f"{row['risk']:.1%}")
            else:
                st.success(f"{row['risk']:.1%}")
        with col4:
            if st.button("View", key=f"inst_{row['student_id']}_{row['module_id']}"):
                st.session_state.selected_student = row['student_id']
                st.session_state.selected_module = row['module_id']
                st.session_state.page = 'instructor_student'
                st.rerun()

    conn.close()

def show_instructor_student():
    conn = get_db()
    student_id = st.session_state.selected_student
    module_id  = st.session_state.selected_module
    module_name = 'BBB' if module_id == 0 else 'FFF'

    if st.button("← Back to dashboard"):
        st.session_state.page = 'instructor_home'
        st.rerun()

    st.title(f"Student {student_id} — {module_name}")
    st.divider()

    cursor = conn.execute(f"""
        SELECT p.risk_probability, w.window_number
        FROM Has_Prediction hp
        JOIN Prediction p ON hp.prediction_id = p.prediction_id
        JOIN Window w ON p.window_id = w.window_id
        WHERE hp.student_id = {student_id}
        AND hp.module_id = {module_id}
        ORDER BY w.window_number
    """)
    predictions = cursor.fetchall()

    if not predictions:
        st.info("No predictions available.")
        conn.close()
        return

    latest_risk = predictions[-1][0]
    window_nums = [p[1] for p in predictions]
    risk_vals   = [p[0] for p in predictions]

    col1, col2, col3 = st.columns(3)
    with col1:
        if latest_risk >= 0.7:
            st.error(f"Current Risk\n{latest_risk:.1%}")
        elif latest_risk >= 0.5:
            st.warning(f"Current Risk\n{latest_risk:.1%}")
        else:
            st.success(f"Current Risk\n{latest_risk:.1%}")

    cursor2 = conn.execute(f"""
        SELECT wp.num_assessments, wp.avg_score
        FROM Has_Prediction hp
        JOIN Prediction p ON hp.prediction_id = p.prediction_id
        JOIN Window_Performance wp ON wp.prediction_id = p.prediction_id
        JOIN Window w ON p.window_id = w.window_id
        WHERE hp.student_id = {student_id}
        AND hp.module_id = {module_id}
        ORDER BY w.window_number DESC
        LIMIT 1
    """)
    perf = cursor2.fetchone()

    if perf:
        with col2:
            st.metric("Assignments", int(perf[0]))
        with col3:
            st.metric("Avg Score", f"{perf[1]:.1f}%")

    st.divider()
    st.subheader("📈 Risk trajectory")

    fig, ax = plt.subplots(figsize=(10, 3))
    colors = ['#E24B4A' if r >= 0.7 else '#EF9F27' if r >= 0.5 else '#639922'
              for r in risk_vals]
    ax.bar(window_nums, risk_vals, color=colors)
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Window')
    ax.set_ylabel('Risk')
    ax.set_ylim(0, 1)
    st.pyplot(fig)
    plt.close()

    st.divider()

    cursor3 = conn.execute(f"""
        SELECT p.prediction_id
        FROM Has_Prediction hp
        JOIN Prediction p ON hp.prediction_id = p.prediction_id
        JOIN Window w ON p.window_id = w.window_id
        WHERE hp.student_id = {student_id}
        AND hp.module_id = {module_id}
        ORDER BY w.window_number DESC
        LIMIT 1
    """)
    last_pred = cursor3.fetchone()

    if last_pred:
        pred_id = last_pred[0]

        st.subheader("🤖 AI Recommendations")
        cursor4 = conn.execute(
            f"SELECT rec_text FROM AI_Recommendation WHERE prediction_id = {pred_id}"
        )
        ai_recs = cursor4.fetchall()
        for rec in ai_recs:
            st.info(rec[0])

        st.divider()
        st.subheader("✏️ Add your recommendation")
        note = st.text_area(
            "Write your note",
            placeholder="Enter your recommendation..."
        )

        if st.button("Save recommendation", use_container_width=True):
            if note.strip():
                conn.execute(f"""
                    INSERT INTO Doctor_Recommendation
                    (rec_text, rec_date, prediction_id, instructor_id)
                    VALUES (
                        '{note}',
                        '{pd.Timestamp.now().date()}',
                        {pred_id},
                        {st.session_state.user_id}
                    )
                """)
                conn.commit()
                st.success("Recommendation saved!")
            else:
                st.warning("Please write a recommendation first")

        cursor5 = conn.execute(
            f"SELECT rec_text, rec_date FROM Doctor_Recommendation WHERE prediction_id = {pred_id}"
        )
        doc_recs = cursor5.fetchall()
        if doc_recs:
            st.subheader("Previous notes")
            for rec in doc_recs:
                st.success(rec[0])
                st.caption(f"Date: {rec[1]}")

    conn.close()

if st.session_state.page == 'login':
    show_login()
elif st.session_state.page == 'student_home':
    show_student_home()
elif st.session_state.page == 'student_module':
    show_student_module()
elif st.session_state.page == 'instructor_home':
    show_instructor_home()
elif st.session_state.page == 'instructor_student':
    show_instructor_student()

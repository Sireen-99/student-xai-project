import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def get_db():
    db_path = os.path.join(os.path.dirname(__file__), 'learning_analytics.db')
    return sqlite3.connect(db_path)

model = load_model()

st.set_page_config(page_title="Student XAI Portal", page_icon="🎓", layout="wide")

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

def get_column_name(conn, table, preferred):
    cursor = conn.execute(f"PRAGMA table_info({table})")
    cols = [row[1] for row in cursor.fetchall()]
    if preferred in cols:
        return preferred
    for col in cols:
        if preferred.lower() in col.lower():
            return col
    return cols[0] if cols else preferred

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
                    
                    # نشوف اسم عمود الـ ID في الجدول
                    student_id_col = get_column_name(conn, 'Student', 'student_id')
                    instructor_id_col = get_column_name(conn, 'Instructor', 'instructor_id')
                    
                    if user_type == "Student":
                        cursor = conn.execute(
                            f"SELECT * FROM Student WHERE {student_id_col} = {uid}"
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
                            f"SELECT * FROM Instructor WHERE {instructor_id_col} = {uid}"
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
    
    # نشوف أسماء الأعمدة
    sid_col = get_column_name(conn, 'Student', 'student_id')
    sup_sid = get_column_name(conn, 'Supervises', 'student_id')
    sup_mid = get_column_name(conn, 'Supervises', 'module_id')
    hp_sid  = get_column_name(conn, 'Has_Prediction', 'student_id')
    hp_mid  = get_column_name(conn, 'Has_Prediction', 'module_id')

    st.title(f"🎓 Welcome, Student {student_id}")
    if st.button("← Logout"):
        st.session_state.page = 'login'
        st.rerun()

    st.divider()
    st.subheader("Your Modules")

    cursor = conn.execute(f"""
        SELECT DISTINCT {sup_mid}, presentation, final_result
        FROM Supervises
        WHERE {sup_sid} = {student_id}
    """)
    modules = cursor.fetchall()

    if not modules:
        st.info("No modules found.")
        conn.close()
        return

    for row in modules:
        module_id, presentation, final_result = row
        module_name = 'BBB' if module_id == 0 else 'FFF'

        cursor2 = conn.execute(f"""
            SELECT p.risk_probability
            FROM Has_Prediction hp
            JOIN Prediction p ON hp.prediction_id = p.prediction_id
            JOIN Window w ON p.window_id = w.window_id
            WHERE hp.{hp_sid} = {student_id}
            AND hp.{hp_mid} = {module_id}
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
    module_id  = st.session_state.selected_module
    module_name = 'BBB' if module_id == 0 else 'FFF'

    hp_sid = get_column_name(conn, 'Has_Prediction', 'student_id')
    hp_mid = get_column_name(conn, 'Has_Prediction', 'module_id')

    if st.button("← Back to modules"):
        st.session_state.page = 'student_home'
        st.rerun()

    st.title(f"📚 {module_name}")
    st.divider()

    cursor = conn.execute(f"""
        SELECT p.risk_probability, w.window_number
        FROM Has_Prediction hp
        JOIN Prediction p ON hp.prediction_id = p.prediction_id
        JOIN Window w ON p.window_id = w.window_id
        WHERE hp.{hp_sid} = {student_id}
        AND hp.{hp_mid} = {module_id}
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
        WHERE hp.{hp_sid} = {student_id}
        AND hp.{hp_mid} = {module_id}
        ORDER BY w.window_number DESC
        LIMIT 1
    """)
    perf = cursor2.fetchone()
    if perf:
        with col2:
            st.metric("Assignment

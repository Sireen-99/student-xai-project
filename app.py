import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap

# ─────────────────────────────────
# تحميل الموديل والداتابيس
# ─────────────────────────────────
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

def get_db():
    return sqlite3.connect('learning_analytics.db')

model = load_model()

# ─────────────────────────────────
# إعدادات الصفحة
# ─────────────────────────────────
st.set_page_config(
    page_title="Student XAI Portal",
    page_icon="🎓",
    layout="wide"
)

# ─────────────────────────────────
# Session State للتنقل
# ─────────────────────────────────
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

# ─────────────────────────────────
# صفحة Login
# ─────────────────────────────────
def show_login():
    st.title("🎓 Student XAI Portal")
    st.subheader("Early Warning System")
    st.divider()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        user_type = st.selectbox(
            "Login as",
            ["Student", "Instructor"]
        )
        user_id = st.text_input(
            "ID",
            placeholder="Enter your ID"
        )

        if st.button("Login", use_container_width=True):
            if user_id:
                conn = get_db()
                if user_type == "Student":
                    result = pd.read_sql(
                        "SELECT * FROM Student WHERE student_id = ?",
                        conn, params=[int(user_id)]
                    )
                    if len(result) > 0:
                        st.session_state.user_type = 'student'
                        st.session_state.user_id = int(user_id)
                        st.session_state.page = 'student_home'
                        st.rerun()
                    else:
                        st.error("Student ID not found!")
                else:
                    result = pd.read_sql(
                        "SELECT * FROM Instructor WHERE instructor_id = ?",
                        conn, params=[int(user_id)]
                    )
                    if len(result) > 0:
                        st.session_state.user_type = 'instructor'
                        st.session_state.user_id = int(user_id)
                        st.session_state.page = 'instructor_home'
                        st.rerun()
                    else:
                        st.error("Instructor ID not found!")
                conn.close()
            else:
                st.warning("Please enter your ID")

# ─────────────────────────────────
# صفحة الطالب - قائمة الموادّ
# ─────────────────────────────────
def show_student_home():
    conn = get_db()
    student_id = st.session_state.user_id

    # بيانات الطالب
    student = pd.read_sql(
        "SELECT * FROM Student WHERE student_id = ?",
        conn, params=[student_id]
    ).iloc[0]

    st.title(f"🎓 Welcome, Student {student_id}")

    if st.button("← Logout"):
        st.session_state.page = 'login'
        st.rerun()

    st.divider()
    st.subheader("Your Modules")

    # موادّ الطالب من Supervises
    modules = pd.read_sql('''
        SELECT DISTINCT s.module_id, s.presentation,
               s.final_result
        FROM Supervises s
        WHERE s.student_id = ?
    ''', conn, params=[student_id])

    if len(modules) == 0:
        st.info("No modules found for this student.")
        conn.close()
        return

    for _, row in modules.iterrows():
        module_name = 'BBB' if row['module_id'] == 0 else 'FFF'

        # آخر تنبؤ للطالب في هاد المادة
        latest_pred = pd.read_sql('''
            SELECT p.risk_probability, p.prediction_result
            FROM Has_Prediction hp
            JOIN Prediction p ON hp.prediction_id = p.prediction_id
            JOIN Window w ON p.window_id = w.window_id
            WHERE hp.student_id = ? AND hp.module_id = ?
            ORDER BY w.window_number DESC
            LIMIT 1
        ''', conn, params=[student_id, row['module_id']])

        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.markdown(f"**{module_name}** — {row['presentation']}")
            st.caption(f"Final result: {row['final_result']}")
        with col2:
            if len(latest_pred) > 0:
                risk = latest_pred.iloc[0]['risk_probability']
                result = latest_pred.iloc[0]['prediction_result']
                if risk >= 0.7:
                    st.error(f"Risk: {risk:.1%}")
                elif risk >= 0.5:
                    st.warning(f"Risk: {risk:.1%}")
                else:
                    st.success(f"Risk: {risk:.1%}")
        with col3:
            if st.button(
                "View",
                key=f"view_{row['module_id']}_{row['presentation']}"
            ):
                st.session_state.selected_module = row['module_id']
                st.session_state.selected_presentation = row['presentation']
                st.session_state.page = 'student_module'
                st.rerun()

        st.divider()

    conn.close()

# ─────────────────────────────────
# صفحة الطالب - تفاصيل المادة
# ─────────────────────────────────
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

    # كل التنبؤات للطالب في هاد المادة
    predictions = pd.read_sql('''
        SELECT p.risk_probability, p.prediction_result,
               w.window_number
        FROM Has_Prediction hp
        JOIN Prediction p ON hp.prediction_id = p.prediction_id
        JOIN Window w ON p.window_id = w.window_id
        WHERE hp.student_id = ? AND hp.module_id = ?
        ORDER BY w.window_number
    ''', conn, params=[student_id, module_id])

    if len(predictions) == 0:
        st.info("No predictions available yet.")
        conn.close()
        return

    # آخر تنبؤ
    latest = predictions.iloc[-1]
    risk = latest['risk_probability']

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        if risk >= 0.7:
            st.error(f"Current Risk\n{risk:.1%}")
        elif risk >= 0.5:
            st.warning(f"Current Risk\n{risk:.1%}")
        else:
            st.success(f"Current Risk\n{risk:.1%}")

    # أداء الطالب
    perf = pd.read_sql('''
        SELECT wp.*
        FROM Has_Prediction hp
        JOIN Prediction p ON hp.prediction_id = p.prediction_id
        JOIN Window_Performance wp ON wp.prediction_id = p.prediction_id
        JOIN Window w ON p.window_id = w.window_id
        WHERE hp.student_id = ? AND hp.module_id = ?
        ORDER BY w.window_number DESC
        LIMIT 1
    ''', conn, params=[student_id, module_id])

    if len(perf) > 0:
        with col2:
            st.metric("Assignments", int(perf.iloc[0]['num_assessments']))
        with col3:
            st.metric("Avg Score", f"{perf.iloc[0]['avg_score']:.1f}%")

    st.divider()

    # Rolling Windows Chart
    st.subheader("📈 Risk over time")
    fig, ax = plt.subplots(figsize=(10, 3))
    colors = ['#E24B4A' if r >= 0.7
              else '#EF9F27' if r >= 0.5
              else '#639922'
              for r in predictions['risk_probability']]
    ax.bar(predictions['window_number'],
           predictions['risk_probability'],
           color=colors)
    ax.axhline(y=0.5, color='black',
               linestyle='--', linewidth=1,
               label='Risk threshold')
    ax.set_xlabel('Window (every 2 weeks)')
    ax.set_ylabel('Risk probability')
    ax.set_ylim(0, 1)
    ax.legend()
    st.pyplot(fig)
    plt.close()

    st.divider()

    # التوصيات
    st.subheader("💡 Your recommendations")

    # آخر نافذة
    last_pred_id = pd.read_sql('''
        SELECT p.prediction_id
        FROM Has_Prediction hp
        JOIN Prediction p ON hp.prediction_id = p.prediction_id
        JOIN Window w ON p.window_id = w.window_id
        WHERE hp.student_id = ? AND hp.module_id = ?
        ORDER BY w.window_number DESC
        LIMIT 1
    ''', conn, params=[student_id, module_id])

    if len(last_pred_id) > 0:
        pred_id = last_pred_id.iloc[0]['prediction_id']

        # AI توصيات
        ai_recs = pd.read_sql('''
            SELECT rec_text FROM AI_Recommendation
            WHERE prediction_id = ?
        ''', conn, params=[int(pred_id)])

        if len(ai_recs) > 0:
            st.markdown("**AI Recommendations:**")
            for _, rec in ai_recs.iterrows():
                st.info(rec['rec_text'])

        # توصيات الدكتور
        doc_recs = pd.read_sql('''
            SELECT dr.rec_text, i.name
            FROM Doctor_Recommendation dr
            JOIN Instructor i ON dr.instructor_id = i.instructor_id
            WHERE dr.prediction_id = ?
        ''', conn, params=[int(pred_id)])

        if len(doc_recs) > 0:
            st.markdown("**Instructor Recommendations:**")
            for _, rec in doc_recs.iterrows():
                st.success(f"{rec['rec_text']}")
                st.caption(f"From: {rec['name']}")

    conn.close()

# ─────────────────────────────────
# صفحة الدكتور - لوحة التحكم
# ─────────────────────────────────
def show_instructor_home():
    conn = get_db()
    instructor_id = st.session_state.user_id

    instructor = pd.read_sql(
        "SELECT * FROM Instructor WHERE instructor_id = ?",
        conn, params=[instructor_id]
    ).iloc[0]

    st.title(f"👨‍🏫 {instructor['name']}")

    if st.button("← Logout"):
        st.session_state.page = 'login'
        st.rerun()

    st.divider()

    # كل طلاب الدكتور
    students = pd.read_sql('''
        SELECT DISTINCT s.student_id, s.module_id,
               s.presentation
        FROM Supervises s
        WHERE s.instructor_id = ?
    ''', conn, params=[instructor_id])

    # آخر تنبؤ لكل طالب
    risk_data = []
    for _, row in students.iterrows():
        pred = pd.read_sql('''
            SELECT p.risk_probability, p.prediction_result
            FROM Has_Prediction hp
            JOIN Prediction p ON hp.prediction_id = p.prediction_id
            JOIN Window w ON p.window_id = w.window_id
            WHERE hp.student_id = ? AND hp.module_id = ?
            ORDER BY w.window_number DESC
            LIMIT 1
        ''', conn, params=[row['student_id'], row['module_id']])

        if len(pred) > 0:
            risk_data.append({
                'student_id': row['student_id'],
                'module_id':  row['module_id'],
                'presentation': row['presentation'],
                'risk': pred.iloc[0]['risk_probability'],
                'result': pred.iloc[0]['prediction_result']
            })

    risk_df = pd.DataFrame(risk_data)

    if len(risk_df) == 0:
        st.info("No students found.")
        conn.close()
        return

    # إحصائيات
    high   = len(risk_df[risk_df['risk'] >= 0.7])
    medium = len(risk_df[(risk_df['risk'] >= 0.5) & (risk_df['risk'] < 0.7)])
    safe   = len(risk_df[risk_df['risk'] < 0.5])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", len(risk_df))
    col2.metric("High Risk", high)
    col3.metric("Medium Risk", medium)
    col4.metric("Safe", safe)

    st.divider()

    # فلتر
    st.subheader("Students list")
    filter_opt = st.selectbox(
        "Filter by risk",
        ["All", "High Risk", "Medium Risk", "Safe"]
    )

    if filter_opt == "High Risk":
        filtered = risk_df[risk_df['risk'] >= 0.7]
    elif filter_opt == "Medium Risk":
        filtered = risk_df[(risk_df['risk'] >= 0.5) & (risk_df['risk'] < 0.7)]
    elif filter_opt == "Safe":
        filtered = risk_df[risk_df['risk'] < 0.5]
    else:
        filtered = risk_df

    filtered = filtered.sort_values('risk', ascending=False)

    for _, row in filtered.iterrows():
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

# ─────────────────────────────────
# صفحة الدكتور - تفاصيل طالب
# ─────────────────────────────────
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

    # كل التنبؤات
    predictions = pd.read_sql('''
        SELECT p.risk_probability, w.window_number
        FROM Has_Prediction hp
        JOIN Prediction p ON hp.prediction_id = p.prediction_id
        JOIN Window w ON p.window_id = w.window_id
        WHERE hp.student_id = ? AND hp.module_id = ?
        ORDER BY w.window_number
    ''', conn, params=[student_id, module_id])

    if len(predictions) == 0:
        st.info("No predictions available.")
        conn.close()
        return

    latest_risk = predictions.iloc[-1]['risk_probability']

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        if latest_risk >= 0.7:
            st.error(f"Current Risk\n{latest_risk:.1%}")
        elif latest_risk >= 0.5:
            st.warning(f"Current Risk\n{latest_risk:.1%}")
        else:
            st.success(f"Current Risk\n{latest_risk:.1%}")

    perf = pd.read_sql('''
        SELECT wp.*
        FROM Has_Prediction hp
        JOIN Prediction p ON hp.prediction_id = p.prediction_id
        JOIN Window_Performance wp ON wp.prediction_id = p.prediction_id
        JOIN Window w ON p.window_id = w.window_id
        WHERE hp.student_id = ? AND hp.module_id = ?
        ORDER BY w.window_number DESC
        LIMIT 1
    ''', conn, params=[student_id, module_id])

    if len(perf) > 0:
        with col2:
            st.metric("Assignments", int(perf.iloc[0]['num_assessments']))
        with col3:
            st.metric("Avg Score", f"{perf.iloc[0]['avg_score']:.1f}%")

    st.divider()

    # Rolling Windows
    st.subheader("📈 Risk trajectory")
    fig, ax = plt.subplots(figsize=(10, 3))
    colors = ['#E24B4A' if r >= 0.7
              else '#EF9F27' if r >= 0.5
              else '#639922'
              for r in predictions['risk_probability']]
    ax.bar(predictions['window_number'],
           predictions['risk_probability'],
           color=colors)
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Window')
    ax.set_ylabel('Risk')
    ax.set_ylim(0, 1)
    st.pyplot(fig)
    plt.close()

    st.divider()

    # AI توصيات
    st.subheader("🤖 AI Recommendations")
    last_pred = pd.read_sql('''
        SELECT p.prediction_id
        FROM Has_Prediction hp
        JOIN Prediction p ON hp.prediction_id = p.prediction_id
        JOIN Window w ON p.window_id = w.window_id
        WHERE hp.student_id = ? AND hp.module_id = ?
        ORDER BY w.window_number DESC
        LIMIT 1
    ''', conn, params=[student_id, module_id])

    if len(last_pred) > 0:
        pred_id = last_pred.iloc[0]['prediction_id']
        ai_recs = pd.read_sql(
            "SELECT rec_text FROM AI_Recommendation WHERE prediction_id = ?",
            conn, params=[int(pred_id)]
        )
        for _, rec in ai_recs.iterrows():
            st.info(rec['rec_text'])

    st.divider()

    # إضافة ملاحظة الدكتور
    st.subheader("✏️ Add your recommendation")
    note = st.text_area(
        "Write your note",
        placeholder="Enter your recommendation for this student..."
    )

    if st.button("Save recommendation", use_container_width=True):
        if note.strip():
            # حفظ في Doctor_Recommendation
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO Doctor_Recommendation
                (rec_text, rec_date, prediction_id, instructor_id)
                VALUES (?, ?, ?, ?)
            ''', (note, pd.Timestamp.now().date(),
                  int(pred_id),
                  st.session_state.user_id))
            conn.commit()
            st.success("Recommendation saved!")
        else:
            st.warning("Please write a recommendation first")

    # توصيات الدكتور السابقة
    doc_recs = pd.read_sql('''
        SELECT rec_text, rec_date
        FROM Doctor_Recommendation
        WHERE prediction_id = ?
    ''', conn, params=[int(pred_id)])

    if len(doc_recs) > 0:
        st.subheader("Previous notes")
        for _, rec in doc_recs.iterrows():
            st.success(f"{rec['rec_text']}")
            st.caption(f"Date: {rec['rec_date']}")

    conn.close()

# ─────────────────────────────────
# التنقل بين الصفحات
# ─────────────────────────────────
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
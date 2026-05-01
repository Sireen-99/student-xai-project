import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import shap

# ─── Feature names — Longitudinal Model (28 features) ────────────────────────
FEATURE_NAMES = [
    'code_module', 'gender', 'highest_education', 'age_band',
    'num_of_prev_attempts', 'studied_credits',
    'total_clicks', 'active_days',
    'avg_clicks_per_window', 'std_clicks', 'peak_clicks', 'min_clicks', 'trend_clicks',
    'clicks_early', 'clicks_mid', 'clicks_late',
    'active_days_early', 'active_days_late',
    'avg_score', 'peak_score', 'score_std',
    'num_assessments', 'late_submissions',
    'score_early', 'score_late', 'score_trend', 'score_trend_slope',
    'prev_attempts_x_score', 'prev_attempts_x_clicks',
]

FEATURE_LABELS = {
    'code_module':             'Course Module',
    'gender':                  'Gender',
    'highest_education':       'Education Level',
    'age_band':                'Age Group',
    'num_of_prev_attempts':    'Previous Attempts',
    'studied_credits':         'Credits Studied',
    'total_clicks':            'Total Platform Clicks',
    'active_days':             'Total Active Days',
    'avg_clicks_per_window':   'Avg Clicks per Fortnight',
    'std_clicks':              'Consistency of Clicks',
    'peak_clicks':             'Peak Activity (best window)',
    'min_clicks':              'Minimum Activity (worst window)',
    'trend_clicks':            'Activity Trend (up/down)',
    'clicks_early':            'Early-Stage Activity (wks 1–6)',
    'clicks_mid':              'Mid-Stage Activity (wks 7–28)',
    'clicks_late':             'Late-Stage Activity (wks 29–40)',
    'active_days_early':       'Active Days — Early Stage',
    'active_days_late':        'Active Days — Late Stage',
    'avg_score':               'Average Score',
    'peak_score':              'Highest Score Achieved',
    'score_std':               'Score Consistency (std)',
    'num_assessments':         'Assignments Submitted',
    'late_submissions':        'Late Submissions',
    'score_early':             'Score — Early Stage',
    'score_late':              'Score — Late Stage',
    'score_trend':             'Score Trend (late − early)',
    'score_trend_slope':       'Score Slope (regression)',
    'prev_attempts_x_score':   'Retakes × Avg Score',
    'prev_attempts_x_clicks':  'Retakes × Total Clicks',
}

# ─── Load model + explainer ───────────────────────────────────────────────────
EXPECTED_N_FEATURES = 29   # الموديل الجديد Longitudinal

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    with open(model_path, 'rb') as f:
        mdl = pickle.load(f)

    # ── تحقق تلقائي من الموديل ──────────────────────────────────────────
    n = getattr(mdl, 'n_features_in_', None)
    try:
        actual_feats = list(mdl.feature_names_in_)
    except AttributeError:
        actual_feats = []

    if n == EXPECTED_N_FEATURES or set(actual_feats) == set(FEATURE_NAMES):
        pass   # ✅ موديل صح
    else:
        st.error(
            f"⚠️ **Wrong Model Detected!**\n\n"
            f"الموديل المحمّل عنده **{n} features** لكن التطبيق يحتاج **{EXPECTED_N_FEATURES} features** (Longitudinal Model).\n\n"
            f"**الحل:** ارفع `model_improved.pkl` بدل `model.pkl` على Streamlit Cloud.",
            icon="🚨"
        )
        st.stop()
    return mdl

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

def get_db():
    db_path = os.path.join(os.path.dirname(__file__), 'learning_analytics.db')
    return sqlite3.connect(db_path)

model     = load_model()
explainer = load_explainer(model)

st.set_page_config(page_title="Student XAI Portal", page_icon="🎓", layout="wide")

# ─── Session state ────────────────────────────────────────────────────────────
for key, val in {
    'page': 'login', 'user_type': None, 'user_id': None,
    'selected_module': None, 'selected_student': None,
    'selected_presentation': None, 'selected_course': None
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─── Helpers ──────────────────────────────────────────────────────────────────
def get_col(conn, table, preferred):
    cursor = conn.execute(f"PRAGMA table_info({table})")
    cols = [row[1] for row in cursor.fetchall()]
    if preferred in cols:
        return preferred
    for col in cols:
        if preferred.lower() in col.lower():
            return col
    return cols[0] if cols else preferred

def risk_badge(risk):
    if risk >= 0.7:
        st.error(f"Current Risk: {risk:.1%}")
    elif risk >= 0.5:
        st.warning(f"Current Risk: {risk:.1%}")
    else:
        st.success(f"Current Risk: {risk:.1%}")

# ─── بناء feature vector longitudinal من كل النوافذ ──────────────────────────
def build_feature_vector(conn, student_id, module_id, presentation):
    hp_sid = get_col(conn, 'Has_Prediction', 'student_id')
    hp_mid = get_col(conn, 'Has_Prediction', 'module_id')

    # جيب كل النوافذ مرتبة (مش آخر نافذة بس)
    rows = conn.execute(f"""
        SELECT w.window_number,
               wp.total_clicks, wp.active_days, wp.avg_clicks_per_day,
               wp.avg_score, wp.num_assessments, wp.late_submissions
        FROM Has_Prediction hp
        JOIN Prediction p ON hp.prediction_id = p.prediction_id
        JOIN Window_Performance wp ON wp.prediction_id = p.prediction_id
        JOIN Window w ON p.window_id = w.window_id
        WHERE hp.{hp_sid} = {student_id}
        AND hp.{hp_mid} = {module_id}
        AND hp.presentation = '{presentation}'
        ORDER BY w.window_number
    """).fetchall()

    sid_col = get_col(conn, 'Student', 'student_id')
    st_row = conn.execute(f"""
        SELECT gender, highest_education, age_band,
               num_of_prev_attempts, studied_credits
        FROM Student WHERE {sid_col} = {student_id}
    """).fetchone()

    if not rows or not st_row:
        return None, None

    gender, highest_education, age_band, num_of_prev_attempts, studied_credits = st_row

    df = pd.DataFrame(rows, columns=[
        'window_number', 'total_clicks', 'active_days', 'avg_clicks_per_window',
        'avg_score', 'num_assessments', 'late_submissions'
    ]).fillna(0)

    # تقسيم النوافذ: early (1-3), mid (4-14), late (15-20)
    early = df[df['window_number'] <= 3]
    mid   = df[(df['window_number'] > 3) & (df['window_number'] <= 14)]
    late  = df[df['window_number'] > 14]

    clicks_arr = df['total_clicks'].values.astype(float)
    scores_arr = df['avg_score'].values.astype(float)
    n = len(df)

    # ── Click features ─────────────────────────────────────────────────────
    total_clicks          = float(df['total_clicks'].sum())
    active_days           = float(df['active_days'].sum())
    avg_clicks_per_window = float(df['total_clicks'].mean())
    std_clicks            = float(df['total_clicks'].std()) if n > 1 else 0.0
    peak_clicks           = float(df['total_clicks'].max())
    min_clicks            = float(df['total_clicks'].min())
    trend_clicks          = float(np.polyfit(np.arange(n), clicks_arr, 1)[0]) if n > 1 else 0.0
    clicks_early          = float(early['total_clicks'].sum())
    clicks_mid            = float(mid['total_clicks'].sum())
    clicks_late           = float(late['total_clicks'].sum())
    active_days_early     = float(early['active_days'].sum())
    active_days_late      = float(late['active_days'].sum())

    # ── Score features ─────────────────────────────────────────────────────
    avg_score         = float(df['avg_score'].mean())
    peak_score        = float(df['avg_score'].max())
    score_std         = float(df['avg_score'].std()) if n > 1 else 0.0
    num_assessments   = float(df['num_assessments'].sum())
    late_submissions  = float(df['late_submissions'].sum())
    score_early       = float(early['avg_score'].mean()) if len(early) > 0 else 0.0
    score_late        = float(late['avg_score'].mean())  if len(late)  > 0 else 0.0
    score_trend       = score_late - score_early
    score_trend_slope = float(np.polyfit(np.arange(n), scores_arr, 1)[0]) if n > 1 else 0.0

    # ── Interaction features ───────────────────────────────────────────────
    prev_attempts_x_score  = num_of_prev_attempts * avg_score
    prev_attempts_x_clicks = num_of_prev_attempts * total_clicks

    fv = pd.DataFrame([{
        'code_module':             module_id,
        'gender':                  gender,
        'highest_education':       highest_education,
        'age_band':                age_band,
        'num_of_prev_attempts':    num_of_prev_attempts,
        'studied_credits':         studied_credits,
        'total_clicks':            total_clicks,
        'active_days':             active_days,
        'avg_clicks_per_window':   avg_clicks_per_window,
        'std_clicks':              std_clicks,
        'peak_clicks':             peak_clicks,
        'min_clicks':              min_clicks,
        'trend_clicks':            trend_clicks,
        'clicks_early':            clicks_early,
        'clicks_mid':              clicks_mid,
        'clicks_late':             clicks_late,
        'active_days_early':       active_days_early,
        'active_days_late':        active_days_late,
        'avg_score':               avg_score,
        'peak_score':              peak_score,
        'score_std':               score_std,
        'num_assessments':         num_assessments,
        'late_submissions':        late_submissions,
        'score_early':             score_early,
        'score_late':              score_late,
        'score_trend':             score_trend,
        'score_trend_slope':       score_trend_slope,
        'prev_attempts_x_score':   prev_attempts_x_score,
        'prev_attempts_x_clicks':  prev_attempts_x_clicks,
    }])[FEATURE_NAMES]

    perf_data = {
        'num_assessments': int(num_assessments),
        'avg_score':        avg_score,
        'late_submissions': int(late_submissions),
        'active_days':      int(active_days),
        'score_trend':      score_trend,
        'clicks_late':      clicks_late,
    }
    return fv, perf_data


# ─── SHAP Explanation ─────────────────────────────────────────────────────────
# ── Plain-language templates — bilingual (AR + EN) ────────────────────────────
# Format: (risk_ar, risk_en, protect_ar, protect_en)
SHAP_EXPLAIN = {
    'clicks_late':            ("نشاط منخفض في الأسابيع الأخيرة من المقرر",
                               "Low activity in the final weeks",
                               "تفاعل قوي في الأسابيع الأخيرة",
                               "Strong engagement in the final weeks"),
    'clicks_mid':             ("نشاط منخفض في منتصف الفصل",
                               "Low activity in mid-course",
                               "تفاعل منتظم في منتصف الفصل",
                               "Consistent engagement in mid-course"),
    'clicks_early':           ("نشاط منخفض في بداية المقرر",
                               "Low activity at course start",
                               "بداية قوية في التفاعل مع المنصة",
                               "Good engagement at course start"),
    'active_days_early':      ("أيام نشاط قليلة في المرحلة الأولى",
                               "Few active days in early stage",
                               "أيام نشاط كافية في المرحلة الأولى",
                               "Good active days in early stage"),
    'active_days_late':       ("أيام نشاط قليلة في نهاية الفصل",
                               "Few active days toward course end",
                               "أيام نشاط جيدة في نهاية الفصل",
                               "Good active days toward course end"),
    'active_days':            ("عدد الأيام النشطة منخفض بشكل عام",
                               "Low overall active days",
                               "عدد أيام نشاط مرتفع بشكل عام",
                               "High overall active days"),
    'total_clicks':           ("نشاط كلي منخفض على المنصة طوال الفصل",
                               "Low total platform activity",
                               "نشاط كلي مرتفع على المنصة",
                               "High total platform activity"),
    'avg_clicks_per_window':  ("متوسط نشاط منخفض لكل نافذة زمنية",
                               "Low average activity per window",
                               "متوسط نشاط مرتفع لكل نافذة زمنية",
                               "High average activity per window"),
    'trend_clicks':           ("اتجاه تراجعي في النشاط مع مرور الوقت",
                               "Declining engagement trend",
                               "اتجاه تصاعدي في النشاط مع الوقت",
                               "Increasing engagement trend"),
    'std_clicks':             ("نمط نشاط غير منتظم وغير ثابت",
                               "Inconsistent activity pattern",
                               "نمط نشاط منتظم وثابت",
                               "Consistent activity pattern"),
    'peak_clicks':            ("لم يكن هناك أي نافذة بنشاط مرتفع",
                               "No high-activity window recorded",
                               "كان هناك على الأقل فترة نشاط مرتفعة",
                               "Had at least one very active period"),
    'min_clicks':             ("بعض النوافذ الزمنية كانت بدون أي نشاط",
                               "Some windows had almost no activity",
                               "حافظ على حد أدنى من النشاط في كل النوافذ",
                               "Maintained minimal activity throughout"),
    'avg_score':              ("متوسط درجات منخفض في التقييمات",
                               "Low average assessment score",
                               "متوسط درجات مرتفع في التقييمات",
                               "High average assessment score"),
    'peak_score':             ("أعلى درجة حققها منخفضة",
                               "Low peak score across assessments",
                               "حقق درجة عالية في أحد التقييمات على الأقل",
                               "Achieved a high score at least once"),
    'score_std':              ("درجات غير ثابتة وتتذبذب كثيراً",
                               "Inconsistent assessment scores",
                               "درجات ثابتة ومتسقة عبر الفصل",
                               "Consistent assessment scores"),
    'score_early':            ("أداء ضعيف في التقييمات المبكرة",
                               "Low performance in early assessments",
                               "أداء قوي في التقييمات المبكرة",
                               "Strong performance in early assessments"),
    'score_late':             ("أداء ضعيف في التقييمات المتأخرة",
                               "Low performance in late assessments",
                               "أداء قوي في التقييمات المتأخرة",
                               "Strong performance in late assessments"),
    'score_trend':            ("تراجع الدرجات مقارنة بالبداية",
                               "Declining scores vs early performance",
                               "تحسن الدرجات مقارنة بالبداية",
                               "Improving scores vs early performance"),
    'score_trend_slope':      ("منحنى الدرجات سلبي على مدار الفصل",
                               "Negative score trajectory",
                               "منحنى الدرجات إيجابي على مدار الفصل",
                               "Positive score trajectory"),
    'num_assessments':        ("عدد التقييمات المقدمة منخفض",
                               "Few assessments submitted",
                               "التزام جيد بتقديم التقييمات",
                               "Good assessment submission rate"),
    'late_submissions':       ("عدة تسليمات متأخرة للواجبات",
                               "Multiple late assignment submissions",
                               "التزام بمواعيد تسليم الواجبات",
                               "Assignments submitted on time"),
    'prev_attempts_x_score':  ("محاولات متكررة مع درجات منخفضة",
                               "Repeated attempts + low scores",
                               "خبرة سابقة مع أداء أكاديمي قوي",
                               "Prior experience + strong scores"),
    'prev_attempts_x_clicks': ("محاولات متكررة مع تفاعل منخفض",
                               "Repeated attempts + low engagement",
                               "أعاد التسجيل مع تفاعل مرتفع",
                               "Re-enrolled with high engagement"),
    'num_of_prev_attempts':   ("محاولات سابقة متعددة للمقرر",
                               "Multiple previous course attempts",
                               "يخوض المقرر لأول مرة",
                               "First attempt at this course"),
    'studied_credits':        ("حمل دراسي ثقيل نسبة للأداء",
                               "Heavy credit load vs performance",
                               "الحمل الدراسي مناسب",
                               "Credit load is manageable"),
    'gender':                 ("عامل ديموغرافي — الجنس",
                               "Demographic factor — gender",
                               "عامل ديموغرافي — الجنس",
                               "Demographic factor — gender"),
    'age_band':               ("عامل ديموغرافي — الفئة العمرية",
                               "Demographic factor — age group",
                               "عامل ديموغرافي — الفئة العمرية",
                               "Demographic factor — age group"),
    'highest_education':      ("مستوى تعليمي سابق منخفض",
                               "Lower prior education level",
                               "مستوى تعليمي سابق مرتفع",
                               "Higher prior education level"),
    'code_module':            ("نمط خطر خاص بهذا المقرر",
                               "Module-specific risk pattern",
                               "نمط حماية خاص بهذا المقرر",
                               "Module-specific protective pattern"),
}

def show_shap_explanation(conn, student_id, module_id, presentation):
    fv, _ = build_feature_vector(conn, student_id, module_id, presentation)
    if fv is None:
        st.warning("Not enough data to explain this prediction.")
        return

    shap_vals = explainer.shap_values(fv)

    if isinstance(shap_vals, list):
        sv = np.array(shap_vals[1]).flatten()
    else:
        sv = np.array(shap_vals).flatten()

    sv = sv[:len(FEATURE_NAMES)]

    df_shap = pd.DataFrame({
        'Feature':  [FEATURE_LABELS.get(f, f) for f in FEATURE_NAMES],
        'FeatureID': FEATURE_NAMES,
        'SHAP':     sv,
        'Value':    fv.values[0],
    })
    df_shap['abs'] = df_shap['SHAP'].abs()
    df_top = df_shap.nlargest(10, 'abs').sort_values('SHAP')

    # ── SHAP bar chart ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ['#E24B4A' if v > 0 else '#4CAF50' for v in df_top['SHAP']]
    bars = ax.barh(df_top['Feature'], df_top['SHAP'], color=colors, height=0.6)

    for bar, (_, row) in zip(bars, df_top.iterrows()):
        val = row['Value']
        val_str = f"{val:.1f}" if isinstance(val, float) else f"{int(val)}"
        x = bar.get_width()
        ax.text(
            x + 0.001 if x >= 0 else x - 0.001,
            bar.get_y() + bar.get_height() / 2,
            val_str, va='center',
            ha='left' if x >= 0 else 'right',
            fontsize=8, color='#555'
        )

    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Impact on Risk Score  (→ increases risk   ← decreases risk)', fontsize=9)
    ax.set_title('🔍 Prediction Explanation — Top 10 Influencing Factors', fontsize=10, pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Plain-language explanation cards ────────────────────────────────────
    top_risk    = df_shap[df_shap['SHAP'] > 0].nlargest(3, 'SHAP')
    top_protect = df_shap[df_shap['SHAP'] < 0].nsmallest(3, 'SHAP')

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔴 What is increasing this risk?**")
        if not top_risk.empty:
            for _, row in top_risk.iterrows():
                fid  = row['FeatureID']
                exp  = SHAP_EXPLAIN.get(fid, (row['Feature'], row['Feature'], '', ''))
                ar   = exp[0]; en = exp[1]
                val  = row['Value']
                val_str = f"{val:.1f}" if isinstance(val, float) else f"{int(val)}"
                st.error(
                    f"**{row['Feature']}** = `{val_str}`\n\n"
                    f"🇸🇦 {ar}\n\n"
                    f"🇬🇧 _{en}_"
                )
        else:
            st.info("No significant risk factors detected. / لا توجد عوامل خطر بارزة.")

    with col2:
        st.markdown("**🟢 What is protecting this student?**")
        if not top_protect.empty:
            for _, row in top_protect.iterrows():
                fid  = row['FeatureID']
                exp  = SHAP_EXPLAIN.get(fid, ('', '', row['Feature'], row['Feature']))
                ar   = exp[2]; en = exp[3]
                val  = row['Value']
                val_str = f"{val:.1f}" if isinstance(val, float) else f"{int(val)}"
                st.success(
                    f"**{row['Feature']}** = `{val_str}`\n\n"
                    f"🇸🇦 {ar}\n\n"
                    f"🇬🇧 _{en}_"
                )
        else:
            st.info("No significant protective factors detected. / لا توجد عوامل حماية بارزة.")

# ─── Login ────────────────────────────────────────────────────────────────────
def show_login():
    st.title("Student XAI Portal")
    st.subheader("Early Warning System")
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        user_type = st.selectbox("Login as", ["Student", "Instructor"])
        user_id   = st.text_input("ID", placeholder="Enter your ID")
        if st.button("Login", use_container_width=True):
            if user_id:
                conn = get_db()
                try:
                    uid     = int(user_id)
                    sid_col = get_col(conn, 'Student', 'student_id')
                    iid_col = get_col(conn, 'Instructor', 'instructor_id')
                    if user_type == "Student":
                        result = conn.execute(
                            f"SELECT * FROM Student WHERE {sid_col} = {uid}").fetchall()
                        if result:
                            st.session_state.user_type = 'student'
                            st.session_state.user_id   = uid
                            st.session_state.page      = 'student_home'
                            st.rerun()
                        else:
                            st.error("Student ID not found!")
                    else:
                        result = conn.execute(
                            f"SELECT * FROM Instructor WHERE {iid_col} = {uid}").fetchall()
                        if result:
                            st.session_state.user_type       = 'instructor'
                            st.session_state.user_id         = uid
                            st.session_state.page            = 'instructor_home'
                            st.session_state.selected_course = None
                            st.rerun()
                        else:
                            st.error("Instructor ID not found!")
                except ValueError:
                    st.error("Please enter a valid number")
                finally:
                    conn.close()
            else:
                st.warning("Please enter your ID")

# ─── Student Home ─────────────────────────────────────────────────────────────
def show_student_home():
    conn       = get_db()
    student_id = st.session_state.user_id
    sup_sid    = get_col(conn, 'Supervises', 'student_id')
    sup_mid    = get_col(conn, 'Supervises', 'module_id')
    hp_sid     = get_col(conn, 'Has_Prediction', 'student_id')
    hp_mid     = get_col(conn, 'Has_Prediction', 'module_id')

    st.title(f"Welcome, Student {student_id}")
    if st.button("Logout"):
        st.session_state.page = 'login'
        st.rerun()
    st.divider()
    st.subheader("Your Modules")

    modules = conn.execute(f"""
        SELECT DISTINCT {sup_mid}, presentation, final_result
        FROM Supervises WHERE {sup_sid} = {student_id}
    """).fetchall()

    if not modules:
        st.info("No modules found.")
        conn.close()
        return

    for row in modules:
        module_id, presentation, final_result = row
        module_name = 'BBB' if module_id == 0 else 'FFF'
        pred = conn.execute(f"""
            SELECT p.risk_probability FROM Has_Prediction hp
            JOIN Prediction p ON hp.prediction_id = p.prediction_id
            JOIN Window w ON p.window_id = w.window_id
            WHERE hp.{hp_sid} = {student_id}
            AND hp.{hp_mid} = {module_id}
            AND hp.presentation = '{presentation}'
            ORDER BY w.window_number DESC LIMIT 1
        """).fetchone()

        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.markdown(f"**{module_name}** - {presentation}")
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
            if st.button("View", key=f"v_{module_id}_{presentation}"):
                st.session_state.selected_module       = module_id
                st.session_state.selected_presentation = presentation
                st.session_state.page                  = 'student_module'
                st.rerun()
        st.divider()
    conn.close()

# ─── Student Module ───────────────────────────────────────────────────────────
def show_student_module():
    conn         = get_db()
    student_id   = st.session_state.user_id
    module_id    = st.session_state.selected_module
    presentation = st.session_state.selected_presentation
    module_name  = 'BBB' if module_id == 0 else 'FFF'
    hp_sid       = get_col(conn, 'Has_Prediction', 'student_id')
    hp_mid       = get_col(conn, 'Has_Prediction', 'module_id')

    if st.button("Back to modules"):
        st.session_state.page = 'student_home'
        st.rerun()

    st.title(f"Module: {module_name} - {presentation}")
    st.divider()

    predictions = conn.execute(f"""
        SELECT p.risk_probability, w.window_number FROM Has_Prediction hp
        JOIN Prediction p ON hp.prediction_id = p.prediction_id
        JOIN Window w ON p.window_id = w.window_id
        WHERE hp.{hp_sid} = {student_id}
        AND hp.{hp_mid} = {module_id}
        AND hp.presentation = '{presentation}'
        ORDER BY w.window_number
    """).fetchall()

    if not predictions:
        st.info("No predictions available.")
        conn.close()
        return

    latest_risk = predictions[-1][0]
    window_nums = [p[1] for p in predictions]
    risk_vals   = [p[0] for p in predictions]

    _, perf_data = build_feature_vector(conn, student_id, module_id, presentation)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        risk_badge(latest_risk)
    if perf_data:
        with col2:
            st.metric("Assignments", int(perf_data['num_assessments']))
        with col3:
            st.metric("Avg Score", f"{perf_data['avg_score']:.1f}%")
        with col4:
            trend = perf_data.get('score_trend', 0)
            arrow = "📈" if trend > 0 else ("📉" if trend < 0 else "➡️")
            st.metric("Score Trend", f"{arrow} {trend:+.1f}")

    # ── Risk Chart ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📈 Risk over time (across all windows)")
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

    # ── SHAP Explanation ────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔍 Why this prediction?")
    st.caption("This chart shows which factors influenced your risk score the most.")
    show_shap_explanation(conn, student_id, module_id, presentation)

    # ── Recommendations ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("💡 Your Recommendations")
    last_pred = conn.execute(f"""
        SELECT p.prediction_id FROM Has_Prediction hp
        JOIN Prediction p ON hp.prediction_id = p.prediction_id
        JOIN Window w ON p.window_id = w.window_id
        WHERE hp.{hp_sid} = {student_id}
        AND hp.{hp_mid} = {module_id}
        AND hp.presentation = '{presentation}'
        ORDER BY w.window_number DESC LIMIT 1
    """).fetchone()

    if last_pred:
        pred_id = last_pred[0]
        ai_recs = conn.execute(
            f"SELECT rec_text FROM AI_Recommendation WHERE prediction_id = {pred_id}"
        ).fetchall()
        if ai_recs:
            st.markdown("**AI Recommendations:**")
            for rec in ai_recs:
                st.info(rec[0])

        doc_recs = conn.execute(f"""
            SELECT dr.rec_text, i.name FROM Doctor_Recommendation dr
            JOIN Instructor i ON dr.instructor_id = i.instructor_id
            WHERE dr.prediction_id = {pred_id}
        """).fetchall()
        if doc_recs:
            st.markdown("**Instructor Recommendations:**")
            for rec in doc_recs:
                st.success(rec[0])
                st.caption(f"From: {rec[1]}")

    conn.close()

# ─── Instructor Home ──────────────────────────────────────────────────────────
def show_instructor_home():
    conn          = get_db()
    instructor_id = st.session_state.user_id
    iid_col       = get_col(conn, 'Instructor', 'instructor_id')
    sup_iid       = get_col(conn, 'Supervises', 'instructor_id')
    sup_sid       = get_col(conn, 'Supervises', 'student_id')
    sup_mid       = get_col(conn, 'Supervises', 'module_id')
    hp_sid        = get_col(conn, 'Has_Prediction', 'student_id')
    hp_mid        = get_col(conn, 'Has_Prediction', 'module_id')

    instructor = conn.execute(
        f"SELECT name FROM Instructor WHERE {iid_col} = {instructor_id}"
    ).fetchone()
    name = instructor[0] if instructor else f"Instructor {instructor_id}"

    st.title(f"Dashboard - {name}")
    if st.button("Logout"):
        st.session_state.page            = 'login'
        st.session_state.selected_course = None
        st.rerun()
    st.divider()

    courses = conn.execute(f"""
        SELECT DISTINCT {sup_mid}, presentation
        FROM Supervises WHERE {sup_iid} = {instructor_id}
        ORDER BY presentation
    """).fetchall()

    if not courses:
        st.info("No courses found.")
        conn.close()
        return

    if st.session_state.selected_course is None:
        st.subheader("📚 Select a Course")
        st.write("Click on a course to view its students:")
        st.divider()
        cols = st.columns(len(courses))
        for i, (mid, pres) in enumerate(courses):
            module_name = 'BBB' if mid == 0 else 'FFF'
            count = conn.execute(f"""
                SELECT COUNT(DISTINCT {sup_sid}) FROM Supervises
                WHERE {sup_iid} = {instructor_id}
                AND {sup_mid} = {mid}
                AND presentation = '{pres}'
            """).fetchone()[0]
            with cols[i]:
                st.markdown(f"""
                <div style='text-align:center; padding:10px;
                            border:1px solid #ddd; border-radius:8px;'>
                    <h3>{module_name}</h3><p>{pres}</p>
                    <p><b>{count} students</b></p>
                </div>""", unsafe_allow_html=True)
                st.write("")
                if st.button(f"Open {module_name} - {pres}",
                             key=f"course_{mid}_{pres}",
                             use_container_width=True):
                    st.session_state.selected_course = (mid, pres)
                    st.rerun()
        conn.close()
        return

    sel_mid, sel_pres = st.session_state.selected_course
    module_name = 'BBB' if sel_mid == 0 else 'FFF'

    if st.button("← Back to Courses"):
        st.session_state.selected_course = None
        st.rerun()

    st.subheader(f"📚 {module_name} - {sel_pres}")
    st.divider()

    students = conn.execute(f"""
        SELECT DISTINCT {sup_sid} FROM Supervises
        WHERE {sup_iid} = {instructor_id}
        AND {sup_mid} = {sel_mid}
        AND presentation = '{sel_pres}'
    """).fetchall()

    risk_data = []
    for (sid,) in students:
        pred = conn.execute(f"""
            SELECT p.risk_probability FROM Has_Prediction hp
            JOIN Prediction p ON hp.prediction_id = p.prediction_id
            JOIN Window w ON p.window_id = w.window_id
            WHERE hp.{hp_sid} = {sid}
            AND hp.{hp_mid} = {sel_mid}
            AND hp.presentation = '{sel_pres}'
            ORDER BY w.window_number DESC LIMIT 1
        """).fetchone()
        if pred:
            risk_data.append({
                'student_id':   sid,
                'module_id':    sel_mid,
                'presentation': sel_pres,
                'risk':         pred[0]
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
    col2.metric("🔴 High Risk",   high)
    col3.metric("🟠 Medium Risk", medium)
    col4.metric("🟢 Safe",        safe)
    st.divider()

    filter_opt = st.selectbox("Filter by risk",
                              ["All", "High Risk", "Medium Risk", "Safe"])
    filtered = {
        "High Risk":   [r for r in risk_data if r['risk'] >= 0.7],
        "Medium Risk": [r for r in risk_data if 0.5 <= r['risk'] < 0.7],
        "Safe":        [r for r in risk_data if r['risk'] < 0.5],
    }.get(filter_opt, risk_data)

    filtered = sorted(filtered, key=lambda x: x['risk'], reverse=True)

    for row in filtered:
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            st.write(f"Student {row['student_id']}")
        with col2:
            if row['risk'] >= 0.7:
                st.error(f"{row['risk']:.1%}")
            elif row['risk'] >= 0.5:
                st.warning(f"{row['risk']:.1%}")
            else:
                st.success(f"{row['risk']:.1%}")
        with col3:
            if st.button("View",
                         key=f"i_{row['student_id']}_{row['module_id']}_{row['presentation']}"):
                st.session_state.selected_student      = row['student_id']
                st.session_state.selected_module       = row['module_id']
                st.session_state.selected_presentation = row['presentation']
                st.session_state.page = 'instructor_student'
                st.rerun()

    conn.close()

# ─── Instructor Student ───────────────────────────────────────────────────────
def show_instructor_student():
    conn         = get_db()
    student_id   = st.session_state.selected_student
    module_id    = st.session_state.selected_module
    presentation = st.session_state.selected_presentation
    module_name  = 'BBB' if module_id == 0 else 'FFF'
    hp_sid       = get_col(conn, 'Has_Prediction', 'student_id')
    hp_mid       = get_col(conn, 'Has_Prediction', 'module_id')

    if st.button("Back to dashboard"):
        st.session_state.page = 'instructor_home'
        st.rerun()

    st.title(f"Student {student_id} — {module_name} ({presentation})")
    st.divider()

    predictions = conn.execute(f"""
        SELECT p.risk_probability, w.window_number FROM Has_Prediction hp
        JOIN Prediction p ON hp.prediction_id = p.prediction_id
        JOIN Window w ON p.window_id = w.window_id
        WHERE hp.{hp_sid} = {student_id}
        AND hp.{hp_mid} = {module_id}
        AND hp.presentation = '{presentation}'
        ORDER BY w.window_number
    """).fetchall()

    if not predictions:
        st.info("No predictions available.")
        conn.close()
        return

    latest_risk = predictions[-1][0]
    window_nums = [p[1] for p in predictions]
    risk_vals   = [p[0] for p in predictions]

    _, perf_data = build_feature_vector(conn, student_id, module_id, presentation)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        risk_badge(latest_risk)
    if perf_data:
        with col2:
            st.metric("Assignments", int(perf_data['num_assessments']))
        with col3:
            st.metric("Avg Score", f"{perf_data['avg_score']:.1f}%")
        with col4:
            trend = perf_data.get('score_trend', 0)
            arrow = "📈" if trend > 0 else ("📉" if trend < 0 else "➡️")
            st.metric("Score Trend", f"{arrow} {trend:+.1f}")

    # ── Risk Chart ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📈 Risk Trajectory (across all windows)")
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

    # ── SHAP Explanation ────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔍 Why this prediction?")
    st.caption("Factors that most influenced this student's risk score.")
    show_shap_explanation(conn, student_id, module_id, presentation)

    # ── Recommendations + Doctor Notes ──────────────────────────────────────
    st.divider()
    last_pred = conn.execute(f"""
        SELECT p.prediction_id FROM Has_Prediction hp
        JOIN Prediction p ON hp.prediction_id = p.prediction_id
        JOIN Window w ON p.window_id = w.window_id
        WHERE hp.{hp_sid} = {student_id}
        AND hp.{hp_mid} = {module_id}
        AND hp.presentation = '{presentation}'
        ORDER BY w.window_number DESC LIMIT 1
    """).fetchone()

    if last_pred:
        pred_id = last_pred[0]

        st.subheader("💡 AI Recommendations")
        ai_recs = conn.execute(
            f"SELECT rec_text FROM AI_Recommendation WHERE prediction_id = {pred_id}"
        ).fetchall()
        for rec in ai_recs:
            st.info(rec[0])

        st.divider()
        st.subheader("📝 Add your recommendation")
        note = st.text_area("Write your note",
                            placeholder="Enter your recommendation...")
        if st.button("Save recommendation", use_container_width=True):
            if note.strip():
                note_clean = note.replace("'", "''")
                conn.execute(f"""
                    INSERT INTO Doctor_Recommendation
                    (rec_text, rec_date, prediction_id, instructor_id)
                    VALUES ('{note_clean}', '{pd.Timestamp.now().date()}',
                            {pred_id}, {st.session_state.user_id})
                """)
                conn.commit()
                st.success("Recommendation saved!")
            else:
                st.warning("Please write a recommendation first")

        doc_recs = conn.execute(f"""
            SELECT rec_text, rec_date FROM Doctor_Recommendation
            WHERE prediction_id = {pred_id}
        """).fetchall()
        if doc_recs:
            st.subheader("Previous notes")
            for rec in doc_recs:
                st.success(rec[0])
                st.caption(f"Date: {rec[1]}")

    conn.close()

# ─── Router ───────────────────────────────────────────────────────────────────
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

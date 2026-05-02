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
    'code_module', 'highest_education', 'age_band',
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
EXPECTED_N_FEATURES = 28   # الموديل الجديد Longitudinal (بدون gender)

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

@st.cache_data
def load_features():
    path = os.path.join(os.path.dirname(__file__), 'test_features.csv')
    return pd.read_csv(path)

features_df = load_features()

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

    # جيب features الطالب من CSV — محسوبة صح من raw data
    row = features_df[features_df['student_id'] == student_id]
    if row.empty:
        return None, None

    r  = row.iloc[0]
    fv = pd.DataFrame([{f: r[f] for f in FEATURE_NAMES if f in r.index}])

    perf_data = {
        'num_assessments':  int(r.get('num_assessments', 0)),
        'avg_score':        float(r.get('avg_score', 0.0)),
        'late_submissions': int(r.get('late_submissions', 0)),
        'active_days':      int(r.get('active_days', 0)),
        'score_trend':      float(r.get('score_trend', 0.0)),
        'clicks_late':      float(r.get('clicks_late', 0.0)),
    }
    return fv, perf_data


# ─── قراءة longitudinal_risk من الداتابيس ────────────────────────────────────
def compute_risk(conn, student_id, module_id, presentation):
    """يحسب الخطر من test_features.csv — نتيجة حقيقية من الموديل"""
    row = features_df[features_df['student_id'] == student_id]
    if row.empty:
        return None
    fv = pd.DataFrame([{f: row.iloc[0][f] for f in FEATURE_NAMES if f in row.columns}])
    return float(model.predict_proba(fv)[0][1])

# ─── SHAP Explanation ─────────────────────────────────────────────────────────
# ── Dynamic threshold-based SHAP explanation ─────────────────────────────────
# الحدود المرجعية (من متوسطات الداتا)
FEATURE_THRESHOLDS = {
    'total_clicks':           1000,
    'active_days':            30,
    'avg_clicks_per_window':  50,
    'clicks_early':           200,
    'clicks_mid':             500,
    'clicks_late':            400,
    'active_days_early':      10,
    'active_days_late':       8,
    'avg_score':              60,
    'peak_score':             70,
    'score_early':            60,
    'score_late':             60,
    'score_trend':            0,
    'score_trend_slope':      0,
    'trend_clicks':           0,
    'num_assessments':        3,
    'late_submissions':       1,
    'prev_attempts_x_score':  60,
    'prev_attempts_x_clicks': 500,
    'num_of_prev_attempts':   1,
    'studied_credits':        60,
    'std_clicks':             200,
    'peak_clicks':            300,
    'min_clicks':             50,
    'score_std':              15,
}

# labels: (low_ar, low_en, high_ar, high_en)
SHAP_LABELS = {
    'clicks_late':            ("نشاط منخفض في الأسابيع الأخيرة",       "Low activity in final weeks",
                               "نشاط مرتفع في الأسابيع الأخيرة",       "High activity in final weeks"),
    'clicks_mid':             ("نشاط منخفض في منتصف الفصل",            "Low activity in mid-course",
                               "نشاط مرتفع في منتصف الفصل",            "High activity in mid-course"),
    'clicks_early':           ("نشاط منخفض في بداية الفصل",            "Low activity at course start",
                               "نشاط مرتفع في بداية الفصل",            "High activity at course start"),
    'total_clicks':           ("نشاط كلي منخفض على المنصة",            "Low total platform activity",
                               "نشاط كلي مرتفع على المنصة",            "High total platform activity"),
    'active_days':            ("أيام نشاط قليلة بشكل عام",             "Few overall active days",
                               "أيام نشاط كثيرة بشكل عام",             "Many overall active days"),
    'active_days_early':      ("أيام نشاط قليلة في البداية",           "Few active days early on",
                               "أيام نشاط كافية في البداية",           "Good active days early on"),
    'active_days_late':       ("أيام نشاط قليلة في النهاية",           "Few active days toward end",
                               "أيام نشاط كافية في النهاية",           "Good active days toward end"),
    'avg_clicks_per_window':  ("متوسط نشاط منخفض لكل نافذة",          "Low avg activity per window",
                               "متوسط نشاط مرتفع لكل نافذة",          "High avg activity per window"),
    'trend_clicks':           ("اتجاه تراجعي في النشاط",               "Declining engagement trend",
                               "اتجاه تصاعدي في النشاط",               "Increasing engagement trend"),
    'avg_score':              ("متوسط درجات منخفض",                    "Low average score",
                               "متوسط درجات مرتفع",                    "High average score"),
    'peak_score':             ("أعلى درجة منخفضة",                     "Low peak score",
                               "أعلى درجة مرتفعة",                     "High peak score"),
    'score_early':            ("أداء ضعيف في التقييمات المبكرة",        "Low early assessment scores",
                               "أداء قوي في التقييمات المبكرة",        "High early assessment scores"),
    'score_late':             ("أداء ضعيف في التقييمات المتأخرة",       "Low late assessment scores",
                               "أداء قوي في التقييمات المتأخرة",       "High late assessment scores"),
    'score_trend':            ("تراجع الدرجات مقارنة بالبداية",         "Scores declined vs early",
                               "تحسن الدرجات مقارنة بالبداية",         "Scores improved vs early"),
    'score_trend_slope':      ("منحنى الدرجات سلبي",                   "Negative score trajectory",
                               "منحنى الدرجات إيجابي",                 "Positive score trajectory"),
    'score_std':              ("درجات غير ثابتة ومتذبذبة",             "Inconsistent scores",
                               "درجات ثابتة ومتسقة",                   "Consistent scores"),
    'num_assessments':        ("عدد تقييمات مقدمة منخفض",              "Few assessments submitted",
                               "عدد تقييمات مقدمة مرتفع",              "Many assessments submitted"),
    'late_submissions':       ("تسليمات متأخرة متعددة",                "Multiple late submissions",
                               "التزام بمواعيد التسليم",               "Submissions on time"),
    'prev_attempts_x_score':  ("محاولات متكررة مع درجات منخفضة",        "Retakes + low scores",
                               "محاولات متكررة مع درجات مرتفعة",       "Retakes + strong scores"),
    'prev_attempts_x_clicks': ("محاولات متكررة مع تفاعل منخفض",        "Retakes + low engagement",
                               "محاولات متكررة مع تفاعل مرتفع",        "Retakes + high engagement"),
    'num_of_prev_attempts':   ("محاولات سابقة متعددة",                  "Multiple previous attempts",
                               "أول محاولة للمقرر",                    "First attempt at course"),
    'studied_credits':        ("حمل دراسي ثقيل",                       "Heavy credit load",
                               "حمل دراسي مناسب",                      "Manageable credit load"),
    'std_clicks':             ("نشاط غير منتظم ومتذبذب",               "Very inconsistent activity",
                               "نشاط منتظم وثابت",                     "Consistent activity pattern"),
    'peak_clicks':            ("لا توجد فترة نشاط مرتفع",              "No high-activity window",
                               "فترة نشاط مرتفع موجودة",               "Had high-activity windows"),
    'min_clicks':             ("بعض النوافذ بدون نشاط تقريباً",         "Some windows near-zero activity",
                               "نشاط موجود حتى في أهدأ فترة",          "Activity even in quiet periods"),
    'highest_education':      ("مستوى تعليمي سابق منخفض",              "Lower prior education",
                               "مستوى تعليمي سابق مرتفع",              "Higher prior education"),
    'code_module':            ("نمط خطر خاص بهذا المقرر",              "Module-specific risk pattern",
                               "نمط حماية خاص بهذا المقرر",            "Module-specific protection"),
    'age_band':               ("عامل ديموغرافي — الفئة العمرية",        "Demographic — age group",
                               "عامل ديموغرافي — الفئة العمرية",        "Demographic — age group"),
}

def get_shap_text(feature_id, value):
    """يرجع التفسير الصحيح بناءً على القيمة الفعلية مقارنة بالحد المرجعي"""
    labels = SHAP_LABELS.get(feature_id)
    if not labels:
        return feature_id, feature_id
    threshold = FEATURE_THRESHOLDS.get(feature_id, 0)
    # late_submissions: أعلى = أسوأ
    reverse_features = {'late_submissions', 'num_of_prev_attempts', 'std_clicks', 'score_std'}
    if feature_id in reverse_features:
        is_high = value > threshold
    else:
        is_high = value >= threshold
    if is_high:
        return labels[2], labels[3]   # high_ar, high_en
    else:
        return labels[0], labels[1]   # low_ar, low_en


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

    # نحذف الـ demographic features من العرض (موجودة في الموديل بس مش تعليمية)
    EXCLUDE_FROM_DISPLAY = {'gender', 'age_band', 'code_module'}

    df_shap = pd.DataFrame({
        'Feature':  [FEATURE_LABELS.get(f, f) for f in FEATURE_NAMES],
        'FeatureID': FEATURE_NAMES,
        'SHAP':     sv,
        'Value':    fv.values[0],
    })
    df_shap = df_shap[~df_shap['FeatureID'].isin(EXCLUDE_FROM_DISPLAY)]
    df_shap['abs'] = df_shap['SHAP'].abs()
    df_top = df_shap.nlargest(10, 'abs').sort_values('SHAP')

    # ── SHAP bar chart — محاور ثابتة لكل الطلاب ────────────────────────────
    # ترتيب ثابت للـ features (نفسه لكل الطلاب)
    FIXED_FEATURES_ORDER = [
        'Late-Stage Activity (wks 29–40)',
        'Mid-Stage Activity (wks 7–28)',
        'Early-Stage Activity (wks 1–6)',
        'Active Days — Late Stage',
        'Active Days — Early Stage',
        'Activity Trend (up/down)',
        'Average Score',
        'Highest Score Achieved',
        'Score Trend (late − early)',
        'Score Slope (regression)',
        'Score Consistency (std)',
        'Assignments Submitted',
        'Late Submissions',
        'Retakes × Total Clicks',
        'Retakes × Avg Score',
        'Previous Attempts',
        'Credits Studied',
        'Education Level',
        'Course Module',
    ]

    # بني df كامل لكل الـ features بالترتيب الثابت
    shap_dict = dict(zip(df_shap['Feature'], df_shap['SHAP']))
    val_dict  = dict(zip(df_shap['Feature'], df_shap['Value']))

    fixed_features = [f for f in FIXED_FEATURES_ORDER if f in shap_dict]
    fixed_shap     = [shap_dict[f] for f in fixed_features]
    fixed_vals     = [val_dict[f]  for f in fixed_features]
    colors         = ['#E24B4A' if v > 0 else '#4CAF50' for v in fixed_shap]

    fig, ax = plt.subplots(figsize=(9, len(fixed_features) * 0.45 + 1))
    bars = ax.barh(fixed_features, fixed_shap, color=colors, height=0.6)

    # حد X ثابت لكل الطلاب
    max_abs = max(abs(s) for s in fixed_shap) if fixed_shap else 0.05
    x_lim   = max(max_abs * 1.4, 0.025)
    ax.set_xlim(-x_lim, x_lim)

    for bar, val in zip(bars, fixed_vals):
        val_str = f"{val:.1f}" if isinstance(val, float) else f"{int(val)}"
        x = bar.get_width()
        ax.text(
            x + x_lim * 0.02 if x >= 0 else x - x_lim * 0.02,
            bar.get_y() + bar.get_height() / 2,
            val_str, va='center',
            ha='left' if x >= 0 else 'right',
            fontsize=8, color='#555'
        )

    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Impact on Risk Score  (→ increases risk   ← decreases risk)', fontsize=9)
    ax.set_title('🔍 Prediction Explanation — Feature Impact', fontsize=10, pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Plain-language explanation cards ────────────────────────────────────
    top_risk    = df_shap[df_shap['SHAP'] > 0].nlargest(3, 'SHAP')
    top_protect = df_shap[df_shap['SHAP'] < 0].nsmallest(3, 'SHAP')

    # ── تحديد عوامل الخطر والحماية بناءً على القيمة الفعلية ──────────────────
    def is_truly_good(feature_id, value):
        """هل القيمة فعلاً جيدة بغض النظر عن إشارة SHAP؟"""
        threshold = FEATURE_THRESHOLDS.get(feature_id, 0)
        bad_if_low = {'late_submissions', 'num_of_prev_attempts', 'std_clicks', 'score_std'}
        if feature_id in bad_if_low:
            return value <= threshold
        return value >= threshold

    # نصنف الـ top 10 features بناءً على القيمة الفعلية
    truly_bad      = []
    truly_good     = []
    contradictory  = []

    for _, row in df_top.iterrows():
        fid  = row['FeatureID']
        val  = row['Value']
        shap = row['SHAP']
        good = is_truly_good(fid, val)

        if shap > 0 and not good:
            truly_bad.append(row)       # SHAP يقول خطر + القيمة سيئة ✓
        elif shap < 0 and good:
            truly_good.append(row)      # SHAP يقول حماية + القيمة جيدة ✓
        elif shap > 0 and good:
            contradictory.append(row)   # SHAP يقول خطر لكن القيمة جيدة
        else:
            contradictory.append(row)   # SHAP يقول حماية لكن القيمة سيئة

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔴 What is increasing this risk?**")
        shown = 0
        for row in truly_bad[:3]:
            fid     = row['FeatureID']
            val     = row['Value']
            val_str = f"{val:.1f}" if isinstance(val, float) else f"{int(val)}"
            ar, en  = get_shap_text(fid, val)
            st.error(
                f"**{row['Feature']}** = `{val_str}`\n\n"
                f"🇸🇦 {ar}\n\n"
                f"🇬🇧 _{en}_"
            )
            shown += 1
        # أضف من contradictory لو ما في كافي
        for row in contradictory[:max(0, 3-shown)]:
            fid     = row['FeatureID']
            val     = row['Value']
            val_str = f"{val:.1f}" if isinstance(val, float) else f"{int(val)}"
            ar, en  = get_shap_text(fid, val)
            st.error(
                f"**{row['Feature']}** = `{val_str}`\n\n"
                f"🇸🇦 {ar}\n\n"
                f"🇬🇧 _{en}_"
            )
        if not truly_bad and not contradictory:
            st.info("No significant risk factors. / لا توجد عوامل خطر بارزة.")

    with col2:
        st.markdown("**🟢 What is protecting this student?**")
        shown = 0
        for row in truly_good[:3]:
            fid     = row['FeatureID']
            val     = row['Value']
            val_str = f"{val:.1f}" if isinstance(val, float) else f"{int(val)}"
            ar, en  = get_shap_text(fid, val)
            st.success(
                f"**{row['Feature']}** = `{val_str}`\n\n"
                f"🇸🇦 {ar}\n\n"
                f"🇬🇧 _{en}_"
            )
            shown += 1
        if shown == 0:
            st.warning(
                "⚠️ لا توجد عوامل حماية واضحة لهذا الطالب\n\n"
                "No clear protective factors for this student."
            )

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
        risk = compute_risk(conn, student_id, module_id, presentation)

        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.markdown(f"**{module_name}** - {presentation}")
            st.caption(f"Final result: {final_result}")
        with col2:
            if risk is not None:
                if risk >= 0.7:
                    st.error(f"🔴 Overall Risk: {risk:.1%}")
                elif risk >= 0.5:
                    st.warning(f"🟠 Overall Risk: {risk:.1%}")
                else:
                    st.success(f"🟢 Overall Risk: {risk:.1%}")
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

    window_nums = [p[1] for p in predictions]
    risk_vals   = [p[0] for p in predictions]

    # الخطر الكلي من الموديل الجديد (longitudinal)
    overall_risk = compute_risk(conn, student_id, module_id, presentation)
    if overall_risk is None:
        overall_risk = predictions[-1][0]

    _, perf_data = build_feature_vector(conn, student_id, module_id, presentation)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        risk_badge(overall_risk)
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
    st.caption(
        "ℹ️ This chart shows per-window screening from the initial model. "
        "The **Overall Risk** score above is computed using full longitudinal analysis across all windows."
    )
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
                SELECT COUNT(DISTINCT hp.student_id)
                FROM Has_Prediction hp
                JOIN Supervises s ON hp.student_id = s.{sup_sid}
                WHERE s.{sup_iid} = {instructor_id}
                AND hp.module_id = {mid}
                AND hp.presentation = '{pres}'
                AND hp.longitudinal_risk IS NOT NULL
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
        risk = compute_risk(conn, sid, sel_mid, sel_pres)
        if risk is not None:
            risk_data.append({
                'student_id':   sid,
                'module_id':    sel_mid,
                'presentation': sel_pres,
                'risk':         risk
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

    window_nums = [p[1] for p in predictions]
    risk_vals   = [p[0] for p in predictions]

    # الخطر الكلي من الموديل الجديد (longitudinal)
    overall_risk = compute_risk(conn, student_id, module_id, presentation)
    if overall_risk is None:
        overall_risk = predictions[-1][0]

    _, perf_data = build_feature_vector(conn, student_id, module_id, presentation)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        risk_badge(overall_risk)
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
    st.caption(
        "ℹ️ This chart shows per-window screening from the initial model. "
        "The **Overall Risk** score above is computed using full longitudinal analysis across all windows."
    )
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

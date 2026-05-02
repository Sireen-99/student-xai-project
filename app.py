import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import os
import shap
import json

# ─── Feature names ────────────────────────────────────────────────────────────
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
    'peak_clicks':             'Peak Activity',
    'min_clicks':              'Minimum Activity',
    'trend_clicks':            'Activity Trend',
    'clicks_early':            'Early-Stage Activity (wks 1-6)',
    'clicks_mid':              'Mid-Stage Activity (wks 7-28)',
    'clicks_late':             'Late-Stage Activity (wks 29-40)',
    'active_days_early':       'Active Days - Early Stage',
    'active_days_late':        'Active Days - Late Stage',
    'avg_score':               'Average Score',
    'peak_score':              'Highest Score Achieved',
    'score_std':               'Score Consistency',
    'num_assessments':         'Assignments Submitted',
    'late_submissions':        'Late Submissions',
    'score_early':             'Score - Early Stage',
    'score_late':              'Score - Late Stage',
    'score_trend':             'Score Trend (late - early)',
    'score_trend_slope':       'Score Slope',
    'prev_attempts_x_score':   'Retakes x Avg Score',
    'prev_attempts_x_clicks':  'Retakes x Total Clicks',
}

# ─── Load model + explainer ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    with open(model_path, 'rb') as f:
        return pickle.load(f)

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

# ─── بناء feature vector من Has_Prediction ───────────────────────────────────
def build_feature_vector(conn, student_id, module_id, presentation):
    hp_sid = get_col(conn, 'Has_Prediction', 'student_id')
    hp_mid = get_col(conn, 'Has_Prediction', 'module_id')
    sid_col = get_col(conn, 'Student', 'student_id')

    # نجيب window_clicks و window_scores من Has_Prediction
    row = conn.execute(f"""
        SELECT window_clicks, window_scores, longitudinal_risk
        FROM Has_Prediction
        WHERE {hp_sid} = {student_id}
        AND {hp_mid} = {module_id}
        AND presentation = '{presentation}'
        AND window_clicks IS NOT NULL
        LIMIT 1
    """).fetchone()

    if not row:
        return None, None

    wc_raw, ws_raw, risk = row
    wc = json.loads(wc_raw) if wc_raw else [0]*20
    ws_raw_list = json.loads(ws_raw) if ws_raw else [0.0]*20
    # نعالج NaN
    ws = [float(v) if v is not None and str(v) != 'NaN' and not (isinstance(v, float) and np.isnan(v)) else 0.0
          for v in ws_raw_list]

    # Student info
    st_row = conn.execute(f"""
        SELECT highest_education, age_band,
               num_of_prev_attempts, studied_credits
        FROM Student WHERE {sid_col} = {student_id}
    """).fetchone()

    if not st_row:
        return None, None

    highest_education, age_band, prev_att, studied_credits = st_row

    wc_arr = np.array(wc, dtype=float)
    ws_arr = np.array(ws, dtype=float)
    n = len(wc_arr)

    EARLY_END  = 3   # نافذة 1-3 = أول 42 يوم
    MID_END    = 14  # نافذة 4-14
    LATE_START = 14  # نافذة 15-20

    clicks_total = int(wc_arr.sum())
    clicks_early = int(wc_arr[:EARLY_END].sum())
    clicks_mid   = int(wc_arr[EARLY_END:MID_END].sum())
    clicks_late  = int(wc_arr[LATE_START:].sum())
    active_total = int(np.count_nonzero(wc_arr))
    active_early = int(np.count_nonzero(wc_arr[:EARLY_END]))
    active_late  = int(np.count_nonzero(wc_arr[LATE_START:]))
    peak_clicks  = int(wc_arr.max())
    min_clicks   = int(wc_arr.min())
    avg_clicks   = float(wc_arr.mean())
    std_clicks   = float(wc_arr.std()) if n > 1 else 0.0
    trend_clicks = float(np.polyfit(range(n), wc_arr, 1)[0]) if n > 1 else 0.0

    ws_nonzero = ws_arr[ws_arr > 0]
    score_all   = float(ws_nonzero.mean()) if len(ws_nonzero) > 0 else 0.0
    score_early = float(ws_arr[:EARLY_END][ws_arr[:EARLY_END] > 0].mean()) if len(ws_arr[:EARLY_END][ws_arr[:EARLY_END] > 0]) > 0 else 0.0
    score_late  = float(ws_arr[LATE_START:][ws_arr[LATE_START:] > 0].mean()) if len(ws_arr[LATE_START:][ws_arr[LATE_START:] > 0]) > 0 else 0.0
    score_trend = round(score_late - score_early, 2)
    num_assess  = int(np.count_nonzero(ws_arr))
    late_subs   = 0
    peak_score  = float(ws_arr.max())
    score_std   = float(ws_arr.std()) if n > 1 else 0.0
    score_slope = float(np.polyfit(range(n), ws_arr, 1)[0]) if n > 1 else 0.0

    fv = pd.DataFrame([{
        'code_module':             module_id,
        'highest_education':       highest_education,
        'age_band':                age_band,
        'num_of_prev_attempts':    prev_att,
        'studied_credits':         studied_credits,
        'total_clicks':            clicks_total,
        'active_days':             active_total,
        'avg_clicks_per_window':   avg_clicks,
        'std_clicks':              std_clicks,
        'peak_clicks':             peak_clicks,
        'min_clicks':              min_clicks,
        'trend_clicks':            trend_clicks,
        'clicks_early':            clicks_early,
        'clicks_mid':              clicks_mid,
        'clicks_late':             clicks_late,
        'active_days_early':       active_early,
        'active_days_late':        active_late,
        'avg_score':               score_all,
        'peak_score':              peak_score,
        'score_std':               score_std,
        'num_assessments':         num_assess,
        'late_submissions':        late_subs,
        'score_early':             score_early,
        'score_late':              score_late,
        'score_trend':             score_trend,
        'score_trend_slope':       score_slope,
        'prev_attempts_x_score':   prev_att * score_all,
        'prev_attempts_x_clicks':  prev_att * clicks_total,
    }])[FEATURE_NAMES]

    perf_data = {
        'num_assessments': num_assess,
        'avg_score':       score_all,
        'late_submissions': late_subs,
        'active_days':     active_total,
        'score_trend':     score_trend,
        'clicks_late':     clicks_late,
        'window_clicks':   wc,
        'window_scores':   ws,
    }
    # حساب risk per-window
    window_risks = []
    for i in range(20):
        # بيانات تراكمية حتى النافذة i
        wc_cumul = wc_arr[:i+1]
        ws_cumul = ws_arr[:i+1]
        n_w = i + 1

        c_total = float(wc_cumul.sum())
        c_early = float(wc_arr[:min(3, n_w)].sum())
        c_mid   = float(wc_arr[3:min(14, n_w)].sum())
        c_late  = float(wc_arr[14:n_w].sum())
        a_total = int(np.count_nonzero(wc_cumul))
        a_early = int(np.count_nonzero(wc_arr[:min(3, n_w)]))
        a_late  = int(np.count_nonzero(wc_arr[14:n_w]))
        p_clk   = float(wc_cumul.max())
        m_clk   = float(wc_cumul.min())
        avg_c   = float(wc_cumul.mean())
        std_c   = float(wc_cumul.std()) if n_w > 1 else 0.0
        tr_c    = float(np.polyfit(range(n_w), wc_cumul, 1)[0]) if n_w > 1 else 0.0

        ws_nz   = ws_arr[:n_w][ws_arr[:n_w] > 0]
        s_all   = float(ws_nz.mean()) if len(ws_nz) > 0 else 0.0
        s_early = float(ws_arr[:min(3,n_w)][ws_arr[:min(3,n_w)]>0].mean()) if len(ws_arr[:min(3,n_w)][ws_arr[:min(3,n_w)]>0]) > 0 else 0.0
        s_late  = float(ws_arr[14:n_w][ws_arr[14:n_w]>0].mean()) if len(ws_arr[14:n_w][ws_arr[14:n_w]>0]) > 0 else 0.0
        s_trend = round(s_late - s_early, 2)
        n_ass   = int(np.count_nonzero(ws_arr[:n_w]))
        p_scr   = float(ws_arr[:n_w].max())
        s_std   = float(ws_arr[:n_w].std()) if n_w > 1 else 0.0
        s_slp   = float(np.polyfit(range(n_w), ws_arr[:n_w], 1)[0]) if n_w > 1 else 0.0

        fv_w = pd.DataFrame([{
            'code_module': module_id, 'highest_education': highest_education,
            'age_band': age_band, 'num_of_prev_attempts': prev_att,
            'studied_credits': studied_credits,
            'total_clicks': c_total, 'active_days': a_total,
            'avg_clicks_per_window': avg_c, 'std_clicks': std_c,
            'peak_clicks': p_clk, 'min_clicks': m_clk, 'trend_clicks': tr_c,
            'clicks_early': c_early, 'clicks_mid': c_mid, 'clicks_late': c_late,
            'active_days_early': a_early, 'active_days_late': a_late,
            'avg_score': s_all, 'peak_score': p_scr, 'score_std': s_std,
            'num_assessments': n_ass, 'late_submissions': 0,
            'score_early': s_early, 'score_late': s_late,
            'score_trend': s_trend, 'score_trend_slope': s_slp,
            'prev_attempts_x_score': prev_att * s_all,
            'prev_attempts_x_clicks': prev_att * c_total,
        }])[FEATURE_NAMES]
        window_risks.append(float(model.predict_proba(fv_w)[0][1]))

    perf_data['window_risks'] = window_risks

    return fv, perf_data

# ─── توليد التوصيات من features ───────────────────────────────────────────────
def generate_recommendations(fv, perf_data):
    recs = []
    num_assess  = fv['num_assessments'].values[0]
    avg_score   = fv['avg_score'].values[0]
    active_days = fv['active_days'].values[0]
    clicks_late = fv['clicks_late'].values[0]
    score_trend = fv['score_trend'].values[0]
    late_subs   = fv['late_submissions'].values[0]
    total_clicks= fv['total_clicks'].values[0]

    if num_assess == 0:
        recs.append("⚠️ You haven't submitted any assignments — submit them immediately.")
    elif num_assess < 5:
        recs.append(f"⚠️ Only {int(num_assess)} assignments submitted — try to complete all pending assignments.")

    if avg_score < 50 and avg_score > 0:
        recs.append(f"⚠️ Your average score is {avg_score:.1f}% — seek help from your instructor.")
    elif avg_score < 70 and avg_score > 0:
        recs.append(f"📚 Average score is {avg_score:.1f}% — review course materials to improve.")

    if active_days < 10:
        recs.append(f"⚠️ Only {int(active_days)} active days on the platform — try to log in daily.")
    elif active_days < 20:
        recs.append(f"📅 {int(active_days)} active days — increase your platform engagement.")

    if clicks_late == 0:
        recs.append("⚠️ No activity in the final weeks — the course is still ongoing, stay engaged.")
    elif clicks_late < 50:
        recs.append(f"⚠️ Low activity in final weeks ({int(clicks_late)} clicks) — increase participation.")

    if score_trend < -10:
        recs.append(f"📉 Your scores declined in the second half — focus on late assignments.")

    if late_subs > 2:
        recs.append(f"⏰ {int(late_subs)} late submissions — manage your time better.")

    if total_clicks < 100:
        recs.append("⚠️ Very low platform engagement — explore all course resources.")

    if not recs:
        recs.append("✅ Good performance! Keep up your consistent effort.")

    return recs

# ─── SHAP Explanation ─────────────────────────────────────────────────────────
def get_shap_context(feature_label, value, shap_val):
    """تفسير ذكي يعتمد على القيمة الفعلية"""
    v = float(value)
    is_risk = shap_val > 0  # هل يزيد الخطر

    contexts = {
        'Late-Stage Activity (wks 29-40)': {
            'unit': 'clicks',
            'good': v >= 100,
            'risk_ar':    f"نشاط منخفض جداً في آخر الفصل ({int(v)} ضغطة فقط)",
            'risk_en':    f"Very low activity in final weeks — only {int(v)} clicks",
            'safe_ar':    f"نشاط جيد في آخر الفصل ({int(v)} ضغطة)",
            'safe_en':    f"Good engagement in final weeks — {int(v)} clicks",
        },
        'Mid-Stage Activity (wks 7-28)': {
            'unit': 'clicks',
            'good': v >= 300,
            'risk_ar':    f"نشاط منخفض في منتصف الفصل ({int(v)} ضغطة)",
            'risk_en':    f"Low engagement in mid-course — only {int(v)} clicks",
            'safe_ar':    f"نشاط جيد في منتصف الفصل ({int(v)} ضغطة)",
            'safe_en':    f"Strong engagement in mid-course — {int(v)} clicks",
        },
        'Early-Stage Activity (wks 1-6)': {
            'unit': 'clicks',
            'good': v >= 50,
            'risk_ar':    f"نشاط منخفض في بداية الفصل ({int(v)} ضغطة)",
            'risk_en':    f"Low activity at course start — only {int(v)} clicks",
            'safe_ar':    f"نشاط جيد في بداية الفصل ({int(v)} ضغطة)",
            'safe_en':    f"Good activity at course start — {int(v)} clicks",
        },
        'Active Days - Late Stage': {
            'unit': 'days',
            'good': v >= 10,
            'risk_ar':    f"أيام نشاط قليلة في نهاية الفصل ({int(v)} يوم فقط)",
            'risk_en':    f"Only {int(v)} active days in final weeks",
            'safe_ar':    f"أيام نشاط كافية في نهاية الفصل ({int(v)} يوم)",
            'safe_en':    f"{int(v)} active days toward end of course",
        },
        'Active Days - Early Stage': {
            'unit': 'days',
            'good': v >= 5,
            'risk_ar':    f"أيام نشاط قليلة في البداية ({int(v)} يوم فقط)",
            'risk_en':    f"Only {int(v)} active days in the first 6 weeks",
            'safe_ar':    f"أيام نشاط جيدة في البداية ({int(v)} يوم)",
            'safe_en':    f"{int(v)} active days in the first 6 weeks",
        },
        'Total Active Days': {
            'unit': 'days',
            'good': v >= 30,
            'risk_ar':    f"عدد أيام نشاط منخفض طوال الفصل ({int(v)} يوم)",
            'risk_en':    f"Only {int(v)} active days throughout the course",
            'safe_ar':    f"عدد أيام نشاط جيد ({int(v)} يوم)",
            'safe_en':    f"{int(v)} active days throughout the course",
        },
        'Total Platform Clicks': {
            'unit': 'clicks',
            'good': v >= 500,
            'risk_ar':    f"تفاعل منخفض مع المنصة ({int(v)} ضغطة إجمالاً)",
            'risk_en':    f"Low platform engagement — only {int(v)} total clicks",
            'safe_ar':    f"تفاعل جيد مع المنصة ({int(v)} ضغطة إجمالاً)",
            'safe_en':    f"Good platform engagement — {int(v)} total clicks",
        },
        'Activity Trend': {
            'unit': '',
            'good': v >= 0,
            'risk_ar':    f"اتجاه تراجعي في النشاط (معدل التغيير: {v:.1f})",
            'risk_en':    f"Declining engagement over time (slope: {v:.1f})",
            'safe_ar':    f"اتجاه تصاعدي في النشاط (معدل التغيير: {v:.1f})",
            'safe_en':    f"Increasing engagement over time (slope: {v:.1f})",
        },
        'Average Score': {
            'unit': '%',
            'good': v >= 60,
            'risk_ar':    f"متوسط الدرجات منخفض ({v:.1f}%)",
            'risk_en':    f"Low average score — {v:.1f}%",
            'safe_ar':    f"متوسط الدرجات جيد ({v:.1f}%)",
            'safe_en':    f"Good average score — {v:.1f}%",
        },
        'Highest Score Achieved': {
            'unit': '%',
            'good': v >= 70,
            'risk_ar':    f"أعلى درجة محققة منخفضة ({v:.1f}%)",
            'risk_en':    f"Low peak score — best was {v:.1f}%",
            'safe_ar':    f"أعلى درجة محققة جيدة ({v:.1f}%)",
            'safe_en':    f"Strong peak score — best was {v:.1f}%",
        },
        'Score - Early Stage': {
            'unit': '%',
            'good': v >= 60,
            'risk_ar':    f"أداء ضعيف في تقييمات البداية ({v:.1f}%)",
            'risk_en':    f"Weak early performance — {v:.1f}% average",
            'safe_ar':    f"أداء جيد في تقييمات البداية ({v:.1f}%)",
            'safe_en':    f"Strong early performance — {v:.1f}% average",
        },
        'Score - Late Stage': {
            'unit': '%',
            'good': v >= 60,
            'risk_ar':    f"أداء ضعيف في تقييمات النهاية ({v:.1f}%)",
            'risk_en':    f"Weak late performance — {v:.1f}% average",
            'safe_ar':    f"أداء جيد في تقييمات النهاية ({v:.1f}%)",
            'safe_en':    f"Strong late performance — {v:.1f}% average",
        },
        'Score Trend (late - early)': {
            'unit': 'pts',
            'good': v >= 0,
            'risk_ar':    f"تراجع الدرجات بمقدار {abs(v):.1f} نقطة مقارنة بالبداية",
            'risk_en':    f"Scores dropped by {abs(v):.1f} points compared to early stage",
            'safe_ar':    f"تحسن الدرجات بمقدار {v:.1f} نقطة مقارنة بالبداية",
            'safe_en':    f"Scores improved by {v:.1f} points compared to early stage",
        },
        'Score Slope': {
            'unit': '',
            'good': v >= 0,
            'risk_ar':    f"منحنى الدرجات سلبي طوال الفصل (ميل: {v:.2f})",
            'risk_en':    f"Negative score trajectory throughout course (slope: {v:.2f})",
            'safe_ar':    f"منحنى الدرجات إيجابي طوال الفصل (ميل: {v:.2f})",
            'safe_en':    f"Positive score trajectory throughout course (slope: {v:.2f})",
        },
        'Score Consistency': {
            'unit': 'std',
            'good': v <= 20,
            'risk_ar':    f"درجات غير منتظمة (انحراف معياري: {v:.1f})",
            'risk_en':    f"Inconsistent scores — std deviation: {v:.1f}",
            'safe_ar':    f"درجات منتظمة ومتسقة (انحراف معياري: {v:.1f})",
            'safe_en':    f"Consistent scores — std deviation: {v:.1f}",
        },
        'Assignments Submitted': {
            'unit': 'assignments',
            'good': v >= 5,
            'risk_ar':    f"عدد قليل من الواجبات المقدمة ({int(v)} فقط)",
            'risk_en':    f"Only {int(v)} assignments submitted",
            'safe_ar':    f"عدد جيد من الواجبات المقدمة ({int(v)})",
            'safe_en':    f"{int(v)} assignments submitted",
        },
        'Late Submissions': {
            'unit': 'late',
            'good': v == 0,
            'risk_ar':    f"{int(v)} تسليمات متأخرة — يؤثر سلباً على الأداء",
            'risk_en':    f"{int(v)} late submissions — affects performance",
            'safe_ar':    f"لا تسليمات متأخرة — ملتزم بالمواعيد",
            'safe_en':    f"No late submissions — good time management",
        },
        'Retakes x Avg Score': {
            'unit': '',
            'good': v >= 100,
            'risk_ar':    f"محاولات متكررة مع درجات منخفضة (مؤشر: {v:.0f})",
            'risk_en':    f"Repeated attempts with low scores (index: {v:.0f})",
            'safe_ar':    f"محاولات سابقة مع أداء قوي (مؤشر: {v:.0f})",
            'safe_en':    f"Previous attempts combined with strong scores (index: {v:.0f})",
        },
        'Retakes x Total Clicks': {
            'unit': '',
            'good': v >= 500,
            'risk_ar':    f"محاولات متكررة مع تفاعل منخفض (مؤشر: {v:.0f})",
            'risk_en':    f"Repeated attempts with low engagement (index: {v:.0f})",
            'safe_ar':    f"محاولات سابقة مع تفاعل مرتفع (مؤشر: {v:.0f})",
            'safe_en':    f"Previous attempts with high engagement (index: {v:.0f})",
        },
        'Previous Attempts': {
            'unit': 'attempts',
            'good': v == 0,
            'risk_ar':    f"سجّل المقرر {int(v)} مرة سابقاً",
            'risk_en':    f"Has taken this course {int(v)} time(s) before",
            'safe_ar':    f"أول تسجيل في هذا المقرر",
            'safe_en':    f"First attempt at this course",
        },
        'Avg Clicks per Fortnight': {
            'unit': 'clicks/window',
            'good': v >= 50,
            'risk_ar':    f"متوسط نشاط منخفض في كل نافذة ({v:.0f} ضغطة/أسبوعين)",
            'risk_en':    f"Low average activity per window — {v:.0f} clicks/fortnight",
            'safe_ar':    f"متوسط نشاط جيد في كل نافذة ({v:.0f} ضغطة/أسبوعين)",
            'safe_en':    f"Good average activity per window — {v:.0f} clicks/fortnight",
        },
    }

    ctx = contexts.get(feature_label)
    if ctx:
        good = ctx['good']
        if is_risk:
            return {'ar': ctx['risk_ar'], 'en': ctx['risk_en']}
        else:
            return {'ar': ctx['safe_ar'], 'en': ctx['safe_en']}
    else:
        # تفسير افتراضي
        if is_risk:
            return {'ar': f"هذا العامل يزيد من احتمالية الخطر (القيمة: {v:.1f})",
                    'en': f"This factor increases risk probability (value: {v:.1f})"}
        else:
            return {'ar': f"هذا العامل يحمي من الخطر (القيمة: {v:.1f})",
                    'en': f"This factor reduces risk probability (value: {v:.1f})"}


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
        'Feature': [FEATURE_LABELS.get(f, f) for f in FEATURE_NAMES],
        'SHAP':    sv,
        'Value':   fv.values[0],
        'RawFeature': FEATURE_NAMES,
    })
    df_shap['abs'] = df_shap['SHAP'].abs()
    df_top = df_shap.nlargest(10, 'abs').sort_values('SHAP')

    # رسم SHAP bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#E24B4A' if v > 0 else '#4CAF50' for v in df_top['SHAP']]
    bars = ax.barh(df_top['Feature'], df_top['SHAP'], color=colors)

    for bar, (_, row) in zip(bars, df_top.iterrows()):
        val = row['Value']
        val_str = f"= {val:.1f}" if isinstance(val, float) else f"= {int(val)}"
        x = bar.get_width()
        ax.text(
            x + 0.001 if x >= 0 else x - 0.001,
            bar.get_y() + bar.get_height() / 2,
            val_str, va='center',
            ha='left' if x >= 0 else 'right',
            fontsize=8, color='gray'
        )

    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Impact on Risk Score', fontsize=9)
    ax.set_title('Red = increases risk   |   Green = decreases risk', fontsize=9, pad=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # تفسير نصي
    top_risk    = df_shap[df_shap['SHAP'] > 0.005].nlargest(3, 'SHAP')
    top_protect = df_shap[df_shap['SHAP'] < -0.005].nsmallest(3, 'SHAP')

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔴 Main risk factors:**")
        if not top_risk.empty:
            for _, row in top_risk.iterrows():
                feat    = row['Feature']
                val     = row['Value']
                val_str = f"{val:.1f}" if isinstance(val, float) else f"{int(val)}"
                ctx     = get_shap_context(feat, val, row['SHAP'])
                st.error(f"**{feat}** = `{val_str}`\n\n{ctx['ar']}\n\n_{ctx['en']}_")
        else:
            st.info("No significant risk factors.")

    with col2:
        st.markdown("**🟢 Protective factors:**")
        if not top_protect.empty:
            for _, row in top_protect.iterrows():
                feat    = row['Feature']
                val     = row['Value']
                val_str = f"{val:.1f}" if isinstance(val, float) else f"{int(val)}"
                ctx     = get_shap_context(feat, val, row['SHAP'])
                st.success(f"**{feat}** = `{val_str}`\n\n{ctx['ar']}\n\n_{ctx['en']}_")
        else:
            st.warning("⚠️ No clear protective factors for this student.")

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

        risk_row = conn.execute(f"""
            SELECT longitudinal_risk FROM Has_Prediction
            WHERE {hp_sid} = {student_id}
            AND {hp_mid} = {module_id}
            AND presentation = '{presentation}'
            AND longitudinal_risk IS NOT NULL
            LIMIT 1
        """).fetchone()

        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.markdown(f"**{module_name}** - {presentation}")
            if final_result in ['Pass', 'Distinction']:
                st.success(f"✅ Final Result: {final_result}")
            elif final_result == 'Fail':
                st.error(f"❌ Final Result: {final_result}")
            elif final_result == 'Withdrawn':
                st.warning(f"⚠️ Final Result: {final_result}")
            else:
                st.caption(f"Final result: {final_result}")
        with col2:
            if risk_row:
                risk = risk_row[0]
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

    # نجيب البيانات
    fv, perf_data = build_feature_vector(conn, student_id, module_id, presentation)

    if fv is None:
        st.info("No data available for this module.")
        conn.close()
        return

    risk_row = conn.execute(f"""
        SELECT longitudinal_risk FROM Has_Prediction
        WHERE {hp_sid} = {student_id}
        AND {hp_mid} = {module_id}
        AND presentation = '{presentation}'
        AND longitudinal_risk IS NOT NULL
        LIMIT 1
    """).fetchone()

    risk = risk_row[0] if risk_row else float(model.predict_proba(fv)[0][1])

    col1, col2, col3 = st.columns(3)
    with col1:
        risk_badge(risk)
    with col2:
        st.metric("Assignments", int(perf_data['num_assessments']))
    with col3:
        st.metric("Avg Score", f"{perf_data['avg_score']:.1f}%")

    # ── Risk Chart per window ───────────────────────────────────────────────
    st.divider()
    st.subheader("📈 Risk over time")

    window_risks = perf_data.get('window_risks', [])
    windows = list(range(1, 21))

    fig, ax = plt.subplots(figsize=(10, 3))
    colors = ['#E24B4A' if r >= 0.7 else '#EF9F27' if r >= 0.5 else '#639922'
              for r in window_risks]
    ax.bar(windows, window_risks, color=colors)
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1,
               label='Risk threshold (50%)')
    ax.set_xlabel('Window (every 2 weeks)')
    ax.set_ylabel('Risk probability')
    ax.set_ylim(0, 1)
    ax.set_xticks(windows)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── SHAP Explanation ────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔍 Why this prediction?")
    st.caption("Factors that most influenced your overall risk score.")
    show_shap_explanation(conn, student_id, module_id, presentation)

    # ── Recommendations ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("💡 Your Recommendations")

    recs = generate_recommendations(fv, perf_data)
    for rec in recs:
        st.info(rec)

    # توصيات الدكتور
    last_pred = conn.execute(f"""
        SELECT p.prediction_id FROM Has_Prediction hp
        JOIN Prediction p ON hp.prediction_id = p.prediction_id
        WHERE hp.{hp_sid} = {student_id}
        AND hp.{hp_mid} = {module_id}
        AND hp.presentation = '{presentation}'
        LIMIT 1
    """).fetchone()

    if last_pred:
        doc_recs = conn.execute(f"""
            SELECT dr.rec_text, i.name FROM Doctor_Recommendation dr
            JOIN Instructor i ON dr.instructor_id = i.instructor_id
            WHERE dr.prediction_id = {last_pred[0]}
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
        st.divider()
        cols = st.columns(min(len(courses), 4))
        for i, (mid, pres) in enumerate(courses):
            module_name = 'BBB' if mid == 0 else 'FFF'
            count = conn.execute(f"""
                SELECT COUNT(DISTINCT {sup_sid}) FROM Supervises
                WHERE {sup_iid} = {instructor_id}
                AND {sup_mid} = {mid}
                AND presentation = '{pres}'
            """).fetchone()[0]
            with cols[i % 4]:
                st.markdown(f"""
                <div style='text-align:center; padding:12px;
                            border:1px solid #ddd; border-radius:8px; margin:4px;'>
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
        risk_row = conn.execute(f"""
            SELECT longitudinal_risk FROM Has_Prediction
            WHERE {hp_sid} = {sid}
            AND {hp_mid} = {sel_mid}
            AND presentation = '{sel_pres}'
            AND longitudinal_risk IS NOT NULL
            LIMIT 1
        """).fetchone()
        if risk_row:
            risk_data.append({
                'student_id':   sid,
                'module_id':    sel_mid,
                'presentation': sel_pres,
                'risk':         risk_row[0]
            })

    if not risk_data:
        st.info("No predictions available for this course.")
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
                st.error(f"🔴 {row['risk']:.1%}")
            elif row['risk'] >= 0.5:
                st.warning(f"🟠 {row['risk']:.1%}")
            else:
                st.success(f"🟢 {row['risk']:.1%}")
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

    # نجيب البيانات
    fv, perf_data = build_feature_vector(conn, student_id, module_id, presentation)

    if fv is None:
        st.info("No data available for this student.")
        conn.close()
        return

    risk_row = conn.execute(f"""
        SELECT longitudinal_risk FROM Has_Prediction
        WHERE {hp_sid} = {student_id}
        AND {hp_mid} = {module_id}
        AND presentation = '{presentation}'
        AND longitudinal_risk IS NOT NULL
        LIMIT 1
    """).fetchone()

    risk = risk_row[0] if risk_row else float(model.predict_proba(fv)[0][1])

    # النتيجة النهائية
    sup_sid2 = get_col(conn, 'Supervises', 'student_id')
    sup_mid2 = get_col(conn, 'Supervises', 'module_id')
    final_result_row = conn.execute(f"""
        SELECT final_result FROM Supervises
        WHERE {sup_sid2} = {student_id}
        AND {sup_mid2} = {module_id}
        AND presentation = '{presentation}'
        LIMIT 1
    """).fetchone()
    final_result = final_result_row[0] if final_result_row else 'N/A'

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        risk_badge(risk)
    with col2:
        if final_result in ['Pass', 'Distinction']:
            st.success(f"✅ {final_result}")
        elif final_result == 'Fail':
            st.error(f"❌ {final_result}")
        elif final_result == 'Withdrawn':
            st.warning(f"⚠️ Withdrawn")
        else:
            st.info(f"📋 {final_result}")
    with col3:
        st.metric("Avg Score", f"{perf_data['avg_score']:.1f}%")
    with col4:
        trend = perf_data.get('score_trend', 0)
        arrow = "📈" if trend > 0 else ("📉" if trend < 0 else "➡️")
        st.metric("Score Trend", f"{arrow} {trend:+.1f}")
    with col5:
        st.metric("Assignments", int(perf_data['num_assessments']))

    # ── Risk Chart per window ───────────────────────────────────────────────
    st.divider()
    st.subheader("📈 Risk over time")

    window_risks = perf_data.get('window_risks', [])
    windows = list(range(1, 21))

    fig, ax = plt.subplots(figsize=(10, 3))
    colors = ['#E24B4A' if r >= 0.7 else '#EF9F27' if r >= 0.5 else '#639922'
              for r in window_risks]
    ax.bar(windows, window_risks, color=colors)
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1,
               label='Risk threshold (50%)')
    ax.set_xlabel('Window (every 2 weeks)')
    ax.set_ylabel('Risk probability')
    ax.set_ylim(0, 1)
    ax.set_xticks(windows)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── SHAP ───────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔍 Why this prediction?")
    st.caption("Factors that most influenced this student's risk score.")
    show_shap_explanation(conn, student_id, module_id, presentation)

    # ── AI Recommendations ──────────────────────────────────────────────────
    st.divider()
    st.subheader("💡 AI Recommendations")
    recs = generate_recommendations(fv, perf_data)
    for rec in recs:
        st.info(rec)

    # ── Doctor Note ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📝 Add your recommendation")
    note = st.text_area("Write your note", placeholder="Enter your recommendation...")

    last_pred = conn.execute(f"""
        SELECT p.prediction_id FROM Has_Prediction hp
        JOIN Prediction p ON hp.prediction_id = p.prediction_id
        WHERE hp.{hp_sid} = {student_id}
        AND hp.{hp_mid} = {module_id}
        AND hp.presentation = '{presentation}'
        LIMIT 1
    """).fetchone()

    if last_pred:
        pred_id = last_pred[0]
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
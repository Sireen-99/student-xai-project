def get_shap_context(feature_label, value, shap_val):
    """تفسير ذكي يعتمد على القيمة الفعلية"""
    v = float(value)
    is_risk = shap_val > 0

    contexts = {
        'Late-Stage Activity (wks 29-40)': {
            'risk_ar':    f"نشاط منخفض جداً في آخر الفصل ({int(v)} ضغطة فقط)",
            'risk_en':    f"Very low activity in final weeks — only {int(v)} clicks",
            'safe_ar':    f"نشاط جيد في آخر الفصل ({int(v)} ضغطة)",
            'safe_en':    f"Good engagement in final weeks — {int(v)} clicks",
        },
        'Mid-Stage Activity (wks 7-28)': {
            'risk_ar':    f"نشاط منخفض في منتصف الفصل ({int(v)} ضغطة)",
            'risk_en':    f"Low engagement in mid-course — only {int(v)} clicks",
            'safe_ar':    f"نشاط جيد في منتصف الفصل ({int(v)} ضغطة)",
            'safe_en':    f"Strong engagement in mid-course — {int(v)} clicks",
        },
        'Early-Stage Activity (wks 1-6)': {
            'risk_ar':    f"نشاط منخفض في بداية الفصل ({int(v)} ضغطة)",
            'risk_en':    f"Low activity at course start — only {int(v)} clicks",
            'safe_ar':    f"نشاط جيد في بداية الفصل ({int(v)} ضغطة)",
            'safe_en':    f"Good activity at course start — {int(v)} clicks",
        },
        'Active Days - Late Stage': {
            'risk_ar':    f"أيام نشاط قليلة في نهاية الفصل ({int(v)} يوم فقط)",
            'risk_en':    f"Only {int(v)} active days in final weeks",
            'safe_ar':    f"أيام نشاط كافية في نهاية الفصل ({int(v)} يوم)",
            'safe_en':    f"{int(v)} active days toward end of course",
        },
        'Active Days - Early Stage': {
            'risk_ar':    f"أيام نشاط قليلة في البداية ({int(v)} يوم فقط)",
            'risk_en':    f"Only {int(v)} active days in the first 6 weeks",
            'safe_ar':    f"أيام نشاط جيدة في البداية ({int(v)} يوم)",
            'safe_en':    f"{int(v)} active days in the first 6 weeks",
        },
        'Total Active Days': {
            'risk_ar':    f"عدد أيام نشاط منخفض طوال الفصل ({int(v)} يوم)",
            'risk_en':    f"Only {int(v)} active days throughout the course",
            'safe_ar':    f"عدد أيام نشاط جيد ({int(v)} يوم)",
            'safe_en':    f"{int(v)} active days throughout the course",
        },
        'Total Platform Clicks': {
            'risk_ar':    f"تفاعل منخفض مع المنصة ({int(v)} ضغطة إجمالاً)",
            'risk_en':    f"Low platform engagement — only {int(v)} total clicks",
            'safe_ar':    f"تفاعل جيد مع المنصة ({int(v)} ضغطة إجمالاً)",
            'safe_en':    f"Good platform engagement — {int(v)} total clicks",
        },
        'Activity Trend': {
            'risk_ar':    f"اتجاه تراجعي في النشاط (معدل التغيير: {v:.1f})",
            'risk_en':    f"Declining engagement over time (slope: {v:.1f})",
            'safe_ar':    f"اتجاه تصاعدي في النشاط (معدل التغيير: {v:.1f})",
            'safe_en':    f"Increasing engagement over time (slope: {v:.1f})",
        },
        'Average Score': {
            'risk_ar':    f"متوسط الدرجات منخفض ({v:.1f}%)",
            'risk_en':    f"Low average score — {v:.1f}%",
            'safe_ar':    f"متوسط الدرجات جيد ({v:.1f}%)",
            'safe_en':    f"Good average score — {v:.1f}%",
        },
        'Highest Score Achieved': {
            'risk_ar':    f"أعلى درجة محققة منخفضة ({v:.1f}%)",
            'risk_en':    f"Low peak score — best was {v:.1f}%",
            'safe_ar':    f"أعلى درجة محققة جيدة ({v:.1f}%)",
            'safe_en':    f"Strong peak score — best was {v:.1f}%",
        },
        'Score - Early Stage': {
            'risk_ar':    f"أداء ضعيف في تقييمات البداية ({v:.1f}%)",
            'risk_en':    f"Weak early performance — {v:.1f}% average",
            'safe_ar':    f"أداء جيد في تقييمات البداية ({v:.1f}%)",
            'safe_en':    f"Strong early performance — {v:.1f}% average",
        },
        'Score - Late Stage': {
            'risk_ar':    f"أداء ضعيف في تقييمات النهاية ({v:.1f}%)",
            'risk_en':    f"Weak late performance — {v:.1f}% average",
            'safe_ar':    f"أداء جيد في تقييمات النهاية ({v:.1f}%)",
            'safe_en':    f"Strong late performance — {v:.1f}% average",
        },
        'Score Trend (late - early)': {
            'risk_ar':    f"تراجع الدرجات بمقدار {abs(v):.1f} نقطة مقارنة بالبداية",
            'risk_en':    f"Scores dropped by {abs(v):.1f} points compared to early stage",
            'safe_ar':    f"تحسن الدرجات بمقدار {v:.1f} نقطة مقارنة بالبداية",
            'safe_en':    f"Scores improved by {v:.1f} points compared to early stage",
        },
        'Score Slope': {
            'risk_ar':    f"منحنى الدرجات سلبي طوال الفصل (ميل: {v:.2f})",
            'risk_en':    f"Negative score trajectory throughout course (slope: {v:.2f})",
            'safe_ar':    f"منحنى الدرجات إيجابي طوال الفصل (ميل: {v:.2f})",
            'safe_en':    f"Positive score trajectory throughout course (slope: {v:.2f})",
        },
        'Score Consistency': {
            'risk_ar':    f"درجات غير منتظمة (انحراف معياري: {v:.1f})",
            'risk_en':    f"Inconsistent scores — std deviation: {v:.1f}",
            'safe_ar':    f"درجات منتظمة ومتسقة (انحراف معياري: {v:.1f})",
            'safe_en':    f"Consistent scores — std deviation: {v:.1f}",
        },
        'Assignments Submitted': {
            'risk_ar':    f"عدد قليل من الواجبات المقدمة ({int(v)} فقط)",
            'risk_en':    f"Only {int(v)} assignments submitted",
            'safe_ar':    f"عدد جيد من الواجبات المقدمة ({int(v)})",
            'safe_en':    f"{int(v)} assignments submitted",
        },
        'Late Submissions': {
            'risk_ar':    f"{int(v)} تسليمات متأخرة — يؤثر سلباً على الأداء",
            'risk_en':    f"{int(v)} late submissions — affects performance",
            'safe_ar':    f"لا تسليمات متأخرة — ملتزم بالمواعيد",
            'safe_en':    f"No late submissions — good time management",
        },
        'Retakes x Avg Score': {
            'risk_ar':    f"محاولات متكررة مع درجات منخفضة (مؤشر: {v:.0f})",
            'risk_en':    f"Repeated attempts with low scores (index: {v:.0f})",
            'safe_ar':    f"محاولات سابقة مع أداء قوي (مؤشر: {v:.0f})",
            'safe_en':    f"Previous attempts combined with strong scores (index: {v:.0f})",
        },
        'Retakes x Total Clicks': {
            'risk_ar':    f"محاولات متكررة مع تفاعل منخفض (مؤشر: {v:.0f})",
            'risk_en':    f"Repeated attempts with low engagement (index: {v:.0f})",
            'safe_ar':    f"محاولات سابقة مع تفاعل مرتفع (مؤشر: {v:.0f})",
            'safe_en':    f"Previous attempts with high engagement (index: {v:.0f})",
        },
        'Previous Attempts': {
            'risk_ar':    f"سجّل المقرر {int(v)} مرة سابقاً",
            'risk_en':    f"Has taken this course {int(v)} time(s) before",
            'safe_ar':    f"أول تسجيل في هذا المقرر",
            'safe_en':    f"First attempt at this course",
        },
        'Avg Clicks per Fortnight': {
            'risk_ar':    f"متوسط نشاط منخفض في كل نافذة ({v:.0f} ضغطة/أسبوعين)",
            'risk_en':    f"Low average activity per window — {v:.0f} clicks/fortnight",
            'safe_ar':    f"متوسط نشاط جيد في كل نافذة ({v:.0f} ضغطة/أسبوعين)",
            'safe_en':    f"Good average activity per window — {v:.0f} clicks/fortnight",
        },
    }

    ctx = contexts.get(feature_label)

    if ctx:
        if is_risk:
            return {'ar': ctx['risk_ar'], 'en': ctx['risk_en']}
        else:
            # ✅ FIX: لو القيمة 0 مش نقول "جيد"
            if v == 0:
                return {
                    'ar': "لا توجد بيانات مسجلة لهذا المؤشر",
                    'en': "No data recorded for this indicator"
                }
            return {'ar': ctx['safe_ar'], 'en': ctx['safe_en']}
    else:
        if is_risk:
            return {
                'ar': f"هذا العامل يزيد من احتمالية الخطر (القيمة: {v:.1f})",
                'en': f"This factor increases risk probability (value: {v:.1f})"
            }
        else:
            if v == 0:
                return {
                    'ar': "لا توجد بيانات مسجلة لهذا المؤشر",
                    'en': "No data recorded for this indicator"
                }
            return {
                'ar': f"هذا العامل يحمي من الخطر (القيمة: {v:.1f})",
                'en': f"This factor reduces risk probability (value: {v:.1f})"
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
        'Feature':    [FEATURE_LABELS.get(f, f) for f in FEATURE_NAMES],
        'SHAP':       sv,
        'Value':      fv.values[0],
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
    top_risk = df_shap[df_shap['SHAP'] > 0.005].nlargest(3, 'SHAP')

    # ✅ FIX: Protective فقط لو القيمة > 0
    top_protect = df_shap[
        (df_shap['SHAP'] < -0.005) &
        (df_shap['Value'] > 0)
    ].nsmallest(3, 'SHAP')

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

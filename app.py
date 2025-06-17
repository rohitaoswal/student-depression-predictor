import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="🧠 Student Depression Prediction App",
    layout="centered",
    initial_sidebar_state="expanded"
)

logreg = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

def risk_pred(probability):
    if probability < 0.40:
        return "Low risk of depression"
    elif probability < 0.70:
        return "Moderate risk of depression"
    else:
        return "High risk of depression"

def give_suggestions(user_data, risk_category):
    suggestion_list = []

    if risk_category == 'High risk of depression':
        suggestion_list.append("⚠️ Consult a mental health professional.")
        suggestion_list.append("👪 Talk to family members or trusted friends.")
        suggestion_list.append("🧘‍♂️ Practice yoga, meditation and physical exercises.")
        
    elif risk_category == 'Moderate risk of depression':
        suggestion_list.append("🎨 Try stress-relief activities such as hobbies or meditation.")
        suggestion_list.append("🕰 Maintain a consistent sleep schedule.")
        suggestion_list.append("💬 Consider speaking to a counselor or therapist.")
        
    else:
        suggestion_list.append("✅ Keep maintaining healthy habits and stay connected.")
        suggestion_list.append("🛌 Maintain adequate sleep and a balanced lifestyle.")

    if user_data['Sleep Duration'][0] < 6:
        suggestion_list.append(f"💤 Increase sleep from {user_data['Sleep Duration'][0]} hours to at least 7–8 hours.")

    if user_data['Academic Pressure'][0] > 3:
        suggestion_list.append("📚 Use time management techniques to reduce academic stress.")

    if user_data['Dietary Habits'][0] == 2:
        suggestion_list.append("🥦 Improve diet to include more nutritious meals.")

    if user_data['Have you ever had suicidal thoughts ?'][0] == 1:
        suggestion_list.append("🚨 Seek immediate help if you have suicidal thoughts.")

    return suggestion_list

with st.sidebar:
    st.header("🧠 About This App")
    st.write("""
    This application helps predict the **risk of depression** in students based on lifestyle, academic, and psychological factors.
    
    **Note:** This is not a diagnostic tool. For any mental health concerns, please consult a licensed professional.
    """)
    st.markdown("---")
    st.write("Developed with ❤️ for student mental wellness.")

st.title("🎓 Student Depression Prediction App")

st.markdown("### ✍️ Please fill in the following details:")

age_input = st.text_input("🎂 Age")
academic_pressure_input = st.text_input("📘 Academic Pressure (0-5)")
work_pressure_input = st.text_input("💼 Work Pressure (0-5)")
cgpa_input = st.text_input("📊 CGPA (out of 10)")
study_satisfaction_input = st.text_input("📖 Study Satisfaction (0-5)")
job_satisfaction_input = st.text_input("💻 Job Satisfaction (0-5)")
sleep_input = st.text_input("😴 Sleep Duration (in hours)")
work_hours_input = st.text_input("⏱️ Work/Study Hours (per day)")
financial_stress_input = st.text_input("💸 Financial Stress (0-5)")

dietary_habits = st.selectbox("🥗 Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])
dietary_map = {"Healthy": 0, "Moderate": 1, "Unhealthy": 2}

degree = st.selectbox("🎓 Degree", ["Class 12", "Bachelors (B.Com, B.Tech, BSc, etc.)", "Masters (M.Tech, MBA, MSc, etc.)", "PhD", "Others"])
degree_map = {"Class 12": 0, "Bachelors (B.Com, B.Tech, BSc, etc.)": 1, "Masters (M.Tech, MBA, MSc, etc.)": 2, "PhD": 3, "Others": 4}

suicidal_thoughts = st.selectbox("🧠 Have you ever had suicidal thoughts?", ["No", "Yes"])
suicide_map = {"No": 0, "Yes": 1}

family_history = st.selectbox("👨‍👩‍👧 Family History of Mental Illness", ["No", "Yes"])
family_map = {"No": 0, "Yes": 1}

if st.button("🔍 Predict Depression Risk"):
    required_fields = [
        age_input, academic_pressure_input, work_pressure_input, cgpa_input,
        study_satisfaction_input, job_satisfaction_input, sleep_input,
        work_hours_input, financial_stress_input
    ]

    if not all(required_fields):
        st.error("❗ Please fill in all the numeric input fields before predicting.")
    else:
        try:
            age = int(age_input)
            academic_pressure = float(academic_pressure_input)
            work_pressure = float(work_pressure_input)
            cgpa = float(cgpa_input)
            study_satisfaction = float(study_satisfaction_input)
            job_satisfaction = float(job_satisfaction_input)
            sleep_duration = float(sleep_input)
            work_study_hours = float(work_hours_input)
            financial_stress = float(financial_stress_input)

            user_data = pd.DataFrame({
                'Age': [age],
                'Academic Pressure': [academic_pressure],
                'Work Pressure': [work_pressure],
                'CGPA': [cgpa],
                'Study Satisfaction': [study_satisfaction],
                'Job Satisfaction': [job_satisfaction],
                'Sleep Duration': [sleep_duration],
                'Dietary Habits': [dietary_map[dietary_habits]],
                'Degree': [degree_map[degree]],
                'Have you ever had suicidal thoughts ?': [suicide_map[suicidal_thoughts]],
                'Work/Study Hours': [work_study_hours],
                'Financial Stress': [financial_stress],
                'Family History of Mental Illness': [family_map[family_history]]
            })

            scaled_user_data = scaler.transform(user_data)
            probability = logreg.predict_proba(scaled_user_data)[0][1]
            result = risk_pred(probability)

            radar_fig = go.Figure()
            radar_fig.add_trace(go.Scatterpolar(
                r=user_data.values[0],
                theta=user_data.columns,
                fill='toself',
                name='Your Input Profile'
            ))
            radar_fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, max(user_data.values[0]) + 1])),
                showlegend=False,
                title="🧭 Personal Lifestyle Radar Chart"
            )
            st.plotly_chart(radar_fig)

           
            st.markdown(f"### 🎯 Predicted Probability of Depression: **{round(probability*100, 2)} %**")
            st.markdown(f"### 🧠 Risk Level: **{result}**")
            
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(probability * 100, 2),
                delta={'reference': 50},
                title={'text': "🧠 Depression Risk Level (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgreen"},
                        {'range': [40, 70], 'color': "orange"},
                        {'range': [70, 100], 'color': "red"},
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': round(probability * 100, 2),
                    }
                }
            ))
            st.plotly_chart(gauge_fig)

            st.markdown("---")
            st.markdown("### 💡 Personalized Suggestions:")
            for s in give_suggestions(user_data, result):
                st.write("•", s)

        except Exception as e:
            st.error(f"⚠️ Error occurred: {e}")

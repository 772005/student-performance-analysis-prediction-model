import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

df = pd.read_csv("student_performance_updated_1000.csv") 

st.title("📊 Student Performance Dashboard")

# Sidebar filters
study_hours = st.slider("Filter by Study Hours", 0, 10, (0, 10))
filtered_df = df[(df['Study Hours'] >= study_hours[0]) & (df['Study Hours'] <= study_hours[1])]

# Show data
st.write("Filtered Data", filtered_df)


# Scatter plot (Study Hours vs Final Grade)
st.write("📈 Performance vs Study Hours")
fig, ax = plt.subplots()
sns.scatterplot(data=filtered_df, x="Study Hours", y="FinalGrade", hue="AttendanceRate", size="PreviousGrade", ax=ax)
st.pyplot(fig)

# Bar chart (Average Grade by Attendance)
st.write("📊 Average Grade by Attendance Rate")
avg_grade = filtered_df.groupby("AttendanceRate")["FinalGrade"].mean()
st.bar_chart(avg_grade)


# Boxplot (Grades by Extracurricular Activities)
st.write("### 🎭 Grades Distribution by Extracurricular Activities")
fig, ax = plt.subplots()
sns.boxplot(data=filtered_df, x="ExtracurricularActivities", y="FinalGrade", ax=ax)
st.pyplot(fig)

# Histogram (Study Hours distribution)
st.write("⏳ Study Hours Distribution")
fig, ax = plt.subplots()
sns.histplot(filtered_df["Study Hours"], bins=10, kde=True, ax=ax)
st.pyplot(fig)

# Line Chart (Study Hours vs Average Grade)
st.write("📉 Study Hours vs Average Final Grade")
avg_by_hours = filtered_df.groupby("Study Hours")["FinalGrade"].mean()
st.line_chart(avg_by_hours)

# Summary 
st.write("📌 Key Insights")
st.write(f"- Average Final Grade: {filtered_df['FinalGrade'].mean():.2f}")
st.write(f"- Highest Grade: {filtered_df['FinalGrade'].max()}")
st.write(f"- Lowest Grade: {filtered_df['FinalGrade'].min()}")


st.write("🎯 Predict Student Final Grade")

# Load trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Input fields
study_hours_input = st.number_input("Study Hours", min_value=0, max_value=12, value=8)
attendance_input = st.number_input("Attendance Rate (%)", min_value=0, max_value=100, value=85)
previous_grade_input = st.number_input("Previous Grade", min_value=0, max_value=100, value=75)
extracurricular_input = st.selectbox("Extracurricular Activities", ["No", "Yes"])
extra_value = 1 if extracurricular_input == "Yes" else 0

# Prediction button
if st.button("Predict Final Grade"):
    sample_student = pd.DataFrame([[study_hours_input, attendance_input, previous_grade_input, extra_value]],
                                  columns=['Study Hours', 'AttendanceRate', 'PreviousGrade', 'ExtracurricularActivities'])
    
    # Scale input
    sample_scaled = scaler.transform(sample_student)
    
    # Predict
    predicted_grade = model.predict(sample_scaled)[0]
    
    # Show result
    st.success(f"✅ Predicted Final Grade: **{predicted_grade:.2f}**")

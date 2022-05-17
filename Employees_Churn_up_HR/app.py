from tkinter.messagebox import YES
import streamlit as st
import pickle
import pandas as pd
import base64
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier

st.markdown(
			"<h1 style='font-size:230%;\
						font-family:tahoma;\
						border-radius: 10px;\
						text-align:center;\
						background-color:lavender;\
						color:indigo;'>EMPLOYEE LEAVE PREDICTION \
						GROUP - 3 \
			</h1>", unsafe_allow_html=True
			)
st.markdown("""
			<div style="background-color:white;\
						border-radius: 10px;\
						padding:0px">
			<h2 style="color:black;\
					   text-align:center;\
					   font-size:180%;\
					   font-family:tahoma;">Streamlit Churn Prediction ML App\
			</h2>
			</div>
			""", unsafe_allow_html=True
			)
st.write('\n')
st.markdown("""
    		<h2 style='font-size:140%;\
						font-family:thoma;\
						background-color:;\
						text-align: left\
						border-radius: 10px;\
						color:darkblue;'>Select Your ML Model\
			</h2>
			
			""", unsafe_allow_html=True
			)

st.markdown("""
			<style>
    		[data-baseweb="select"] {
        							margin-top: -50px;
    								}
    		</style>
    		""", unsafe_allow_html=True,
			)

st.sidebar.markdown(
			"<h1 style='text-align:center;\
						color: red;font-family:tahoma;font-size:115%;'> Human is the most valuable asset.\
						Retain your employees.\
			</h1>", unsafe_allow_html=True
			)

st.sidebar.markdown(
    		f"""
    		<div>
        	<img class="emir.png" 
				 src="data:image/png;base64,{base64.b64encode(open("churn.png", "rb").read()).decode()}" 
				 width="300">
			</img>
    		</div>
    		""", unsafe_allow_html=True
			)

st.sidebar.write('\n')
st.sidebar.markdown("""
			<p style='text-align:center;\
						color: white; background-color:lightblue;'>Please Enter Employee Information\
			</p>
			""", unsafe_allow_html=True
			)
satisfaction_level = st.sidebar.slider(label = "Satisfaction Level", min_value=0.0, max_value=1.0, step=0.01, )
last_evaluation = st.sidebar.slider(label="Last Evaluation", min_value=0.0, max_value=1.0, step=0.01)
time_spend_company = st.sidebar.slider("Time Spend in the Company", min_value=0, max_value=15, step=1)
average_monthly_hours = st.sidebar.slider("Average Monthly Hours", min_value=0, max_value=350, step=1)

st.sidebar.write('\n')

number_project = st.sidebar.number_input(label="Number of Projects", min_value=1, max_value=200)

st.sidebar.write('\n')

st.sidebar.markdown("""<p style='text-align:left;color:black;font-size:90%;'>Departments</p>""", unsafe_allow_html=True)
st.sidebar.write('\n')
Departments = st.sidebar.selectbox("Departments", ['RandD', 'accounting', 'hr', 'management', 'marketing', 'product_mng',  'sales', 'support', 'technical', 'IT'])

st.sidebar.markdown("""<p style='text-align:left;color:black;font-size:90%;'>Salary</p>""", unsafe_allow_html=True)
st.sidebar.write('\n')
salary = st.sidebar.selectbox("Salary", ['low', 'medium', 'high'])

Work_accident = st.sidebar.radio("Work Accident", ("Yes", "No"))
if Work_accident=="Yes":
    Work_accident=1
else:
    Work_accident=0	
	
promotion_last_5years = st.sidebar.radio("Promotion in Last 5 Years", ("Yes", "No"))
if promotion_last_5years=="Yes":
    promotion_last_5years=1
else:
    promotion_last_5years=0	


coll_dict = {'satisfaction_level':satisfaction_level, 'last_evaluation':last_evaluation, 'number_project':number_project, 'average_montly_hours':average_monthly_hours,\
			'time_spend_company':time_spend_company, 'Work_accident':Work_accident, 'promotion_last_5years':promotion_last_5years,\
			'Departments': Departments, 'salary':salary}
columns = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company',\
            'Work_accident', 'promotion_last_5years', 'Departments_RandD', 'Departments_accounting', 'Departments_hr',\
            'Departments_management', 'Departments_marketing', 'Departments_product_mng', 'Departments_sales',\
            'Departments_support', 'Departments_technical', 'salary_low', 'salary_medium']

# dumy model
df_input = pd.DataFrame.from_dict([coll_dict])
scaler= pickle.load(open("scaler.pkl", 'rb'))
user_inputs_dumy = pd.get_dummies(df_input).reindex(columns=columns, fill_value=0)
user_inputs_transformed = scaler.transform(user_inputs_dumy)

# encoder
loaded_enc = pickle.load(open("encoder_son.pkl", 'rb'))
new_df = pd.DataFrame(df_input, index=[0])
new_df.salary = new_df.salary.map({"low":1, "medium" : 2, "high" : 3})

cat = new_df.select_dtypes("object").columns
new_df[cat] = loaded_enc.transform(new_df[cat])

selection = st.selectbox("",["Gradient Boost", "Random Forest", "KNN"])

if selection =="Gradient Boost":
	st.markdown("<p style='text-align:center; color:black; font-size:110%; background-color:#F2F3F4 ;'>\
				You selected \
				<span style='color:black;font-weight:bold'>\
				'Gradient Boost'\
				</span> model!\
				</p>", unsafe_allow_html=True
				)
	model = pickle.load(open('gradient_boosting_model.pkl', 'rb'))
	prediction= model.predict(new_df)

elif selection =="Random Forest":
	st.markdown("<p style='text-align:center; color:black; font-size:110%; background-color:#F2F3F4 ;'>\
				You selected \
				<span style='color:tomato;font-weight:bold'>\
				'Random Forest'\
				</span> model!\
				</p>", unsafe_allow_html=True
				)
	model = pickle.load(open('rf_grid_model.pkl', 'rb'))
	prediction = model.predict(new_df)
elif selection =="KNN":
	st.markdown("<p style='text-align:center; color:black; font-size:110%; background-color:#F2F3F4 ;'>\
				You selected \
				<span style='color:tomato;font-weight:bold'>\
				'KNN'\
				</span> model!\
				</p>", unsafe_allow_html=True
				)
	model = pickle.load(open('knn_final_pickle.pkl', 'rb'))
	prediction = model.predict(user_inputs_transformed)

st.sidebar.markdown("""
			<p style='text-align:center;\
						color: white; background-color:lightblue;'>Please complete the  employee information\
			</p>
			""", unsafe_allow_html=True
			)

st.write('\n')

st.markdown("""
			<center>
			<p style='font-size:140%;\
						background-color:lightblue;\
						border-radius:10px;\
						color:white;'>Employee Information\
			</p>
			</center>
			""", unsafe_allow_html=True
			)

st.dataframe(new_df)

st.write('\n')

st.write('\n')
st.markdown("""
			<center>
			<p style='font-size:140%;\
						font-family:tahoma;\
						background-color:lightblue;\
						border-radius: 10px;\
						color:white;'> Please press < PREDICT > to see the result\
			</p>
			</center>
			""", unsafe_allow_html=True
			)

col1, col2, = st.columns([1, 1.5])
if col2.button("PREDICT"):
	if prediction[0]==0:
		st.success(prediction[0])
		st.success(f'Employee will STAY :)')
		st.markdown(
    		f"""
    		<div>
        	<img class="group3.png" 
				 src="data:image/png;base64,{base64.b64encode(open("to be continued1.png", "rb").read()).decode()}" 
				 width="704">
			</img>
    		</div>
    		""", unsafe_allow_html=True
			)		
	elif prediction[0]==1:
		st.warning(prediction[0])
		st.warning(f'Employee will LEAVE :(')

		st.markdown(
    		f"""
    		<div>
        	<img class="group3.png" 
				 src="data:image/png;base64,{base64.b64encode(open("left.png", "rb").read()).decode()}" 
				 width="704">
			</img>
    		</div>
    		""", unsafe_allow_html=True
			)

pip3 freeze > requirements.txt